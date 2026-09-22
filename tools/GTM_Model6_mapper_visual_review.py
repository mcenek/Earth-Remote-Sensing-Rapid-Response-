"""Verify saved mapper scores/export parity and produce a fixed visual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from train_GTM_Model6_mapper_mars_pilot import ROOT, _metrics, _pooled_metrics, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run = ROOT/'outputs/GTM_Model6'/args.run_id
    result = json.loads((run/'GTM_Model6_mapper_mars_pilot_results.json').read_text())
    manifest = json.loads((run/'GTM_Model6_mapper_mars_pilot_manifest.json').read_text())
    rows = {r['id']:r for r in manifest['observations']}
    bundles = {}
    for method in ('mean', 'S16', 'S32', 'S64', 'S128'):
        folder = ROOT/'outputs/GTM_viewer_bundles'/f'{args.run_id}_{method}'
        exp = json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
        bundles[method] = (folder, {s['id']:s for s in exp['scenes']})
    checked = {k:[] for k in bundles}
    for r in result['observations']:
        with np.load(run/f"prediction_{r['index']:04d}.npz") as z:
            item = {'valid':z['valid'].astype(bool), 'target':z['target']}
            for method in bundles:
                p = z['probability' if method == 'mean' else method]
                m = _metrics(p, item)
                for key in ('tp','fp','fn','tn'):
                    if m[key] != r['per_method'][method][key] or m[key] != bundles[method][1][r['id']]['nativeMetrics'][key]:
                        raise ValueError(f'{method}: export/native score mismatch at {r["id"]}')
                checked[method].append(m)
    for method in bundles:
        aggregate = _pooled_metrics(checked[method])
        for key in ('tp', 'fp', 'fn', 'tn', 'iou', 'dice', 'precision', 'recall'):
            if not np.isclose(aggregate[key], result['methods'][method][key]):
                raise ValueError(f'{method}: aggregate mismatch')
    positive = [r for r in result['observations'] if rows[r['id']]['label']=='PLUME']
    negative = sorted([r for r in result['observations'] if rows[r['id']]['label']=='NO_PLUME'],
                      key=lambda r:(-r['metrics']['fp'],r['id']))
    selected = positive + negative[:2]
    width, row_height, thumb = 1180, 254, 220
    sheet = Image.new('RGB',(width,100+row_height*len(selected)),'#f5f7fa')
    draw = ImageDraw.Draw(sheet)
    draw.text((12,8),'R5: all 3 plume cases + 2 worst false positives | fixed threshold 0.50 | reused development fold',fill='#13242d')
    draw.text((12,30),'Zoom to predicted support. Outside the central 88x88 source pixels is unsupported. Cyan=reference; orange=prediction.',fill='#13242d')
    titles = ['Sentinel RGB','Reviewed reference','Four-branch mean','S128 (one branch)','Mean disagreement']
    for col,title in enumerate(titles):
        draw.text((12+col*234,60),title,fill='#13242d')
    for line, rec in enumerate(selected):
        sid=rec['id']; folder,scenes=bundles['mean']; s=scenes[sid]
        grid=json.loads((folder/s['grid']).read_text())
        valid=np.array(grid['valid']).reshape(grid['height'],grid['width']).astype(bool)
        rr,cc=np.where(valid); box=(int(cc.min()),int(rr.min()),int(cc.max())+1,int(rr.max())+1)
        rgb=Image.open(folder/s['rgb']).convert('RGBA')
        truth=Image.open(folder/s['truthImage']).convert('RGBA')
        pred=Image.open(folder/s['predictionImage']).convert('RGBA')
        s128_folder,s128_scenes=bundles['S128']
        pred128=Image.open(s128_folder/s128_scenes[sid]['predictionImage']).convert('RGBA')
        p=np.array(grid['probability']).reshape(valid.shape)>=.5
        y=np.array([v or 0 for v in grid['truth']]).reshape(valid.shape)>0
        errors=np.zeros((*valid.shape,4),np.uint8)
        errors[valid&p&y]=[0,180,90,200]
        errors[valid&p&~y]=[245,60,110,215]
        errors[valid&~p&y]=[125,70,225,215]
        images=[rgb, Image.alpha_composite(rgb,truth), Image.alpha_composite(Image.alpha_composite(rgb,truth),pred),
                Image.alpha_composite(Image.alpha_composite(rgb,truth),pred128), Image.alpha_composite(rgb,Image.fromarray(errors))]
        for col,im in enumerate(images):
            sheet.paste(im.crop(box).resize((thumb,thumb),Image.Resampling.NEAREST).convert('RGB'),(12+col*234,85+line*row_height))
        draw.text((12,310+line*row_height),f"{rows[sid]['label']} | {sid[-8:]} | mean native IoU {rec['metrics']['iou']:.1%}, FP {rec['metrics']['fp']} | S128 IoU {rec['per_method']['S128']['iou']:.1%}",fill='#13242d')
    path=run/'visual_review.png'
    sheet.save(path)
    receipt={'scope':'all plume cases + highest false-positive controls; output sanity review, not model selection',
             'run':args.run_id,'all_scene_method_comparisons_verified':len(result['observations'])*len(bundles),
             'selected_ids':[r['id'] for r in selected],'selected_rule':'all positives then two largest FP counts',
             'image_sha256':digest(path),'image':str(path),'checkpoint_sha256':digest(run/'GTM_Model6_mapper_checkpoint.pt')}
    (run/'visual_review.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
