"""Compare frozen before/after DEV outputs on identical real source arrays.

The pipeline diagnosis protocol requires every case, native recounts and visible
misses/false alarms. This saves repeatable visual review sheets and a receipt.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import GTM_Model7_reference_sensitivity as images
import GTM_Model7_train_repaired_mars_log_v3 as t


def score(c):
    return dict(**c, iou=c['tp']/max(c['tp']+c['fp']+c['fn'],1),
                precision=c['tp']/max(c['tp']+c['fp'],1),
                recall=c['tp']/max(c['tp']+c['fn'],1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before-freeze', type=Path, required=True)
    p.add_argument('--after-freeze', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = [json.loads(f.read_text()) for f in (args.before_freeze,args.after_freeze)]
    paths = [Path(f['metadata']['manifest']) for f in frozen]
    scenes = [{s['id']:s for s in json.loads(p.read_text())['scenes'] if s['split']=='dev'} for p in paths]
    if set(scenes[0])!=set(scenes[1]):
        raise ValueError('Before and after development cohorts differ')
    ordered = sorted(scenes[1].values(),key=lambda s:(s['scene_kind']!='PLUME',s['id']))
    counts = {(i,m):Counter() for i in range(2) for m in ('stable_residual','siamese')}
    negative = {(i,m):dict(scenes=0,activated=0,fp=0,pixels=0) for i,m in counts}
    records = []
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',13)
    sheets = []
    headers = ['Current','Reference','Annotation','Before residual','After residual','Before U-Net','After U-Net']
    for index,s in enumerate(ordered):
        if scenes[0][s['id']]['array_sha256'] != s['array_sha256']:
            raise ValueError('Source array changed between fits')
        file = paths[1].parent/s['array_file']
        if t.sha(file)!=s['array_sha256']:
            raise ValueError('Prepared array changed')
        d = t.load_scene(paths[1],s)
        valid = d['valid_native'].astype(bool)
        truth = d['label_native']==1
        rgb,ref = images.rgb_pair(d)
        rgb,ref = [np.repeat(np.repeat(x,2,0),2,1) for x in (rgb,ref)]
        panels = [rgb,ref,images.paint(rgb,truth&valid,(30,230,220))]
        row = dict(id=s['id'],kind=s['scene_kind'],models={})
        for m in ('stable_residual','siamese'):
            row['models'][m] = []
            for i,f in enumerate(frozen):
                choice=f['models'][m]
                cutoff=choice['threshold']
                pred=Path(f['run'])/m/f'dev_step_{choice["steps"]}'/(s['id']+'.npz')
                with np.load(pred,allow_pickle=False) as z:
                    probability=z['probability_native']
                    if not np.array_equal(z['valid_native'],valid):
                        raise ValueError('Prediction support changed')
                hit=probability>=cutoff
                c=dict(tp=int((hit&truth&valid).sum()),fp=int((hit&~truth&valid).sum()),
                       fn=int((~hit&truth&valid).sum()),tn=int((~hit&~truth&valid).sum()))
                counts[(i,m)].update(c)
                if s['scene_kind']=='NO_PLUME':
                    n=negative[(i,m)]
                    n['scenes']+=1;n['activated']+=c['fp']>0;n['fp']+=c['fp'];n['pixels']+=int(valid.sum())
                panel=images.paint(rgb,hit&valid,(255,140,30))
                panels.append(panel)
                row['models'][m].append(dict(stage=('before','after')[i],cutoff=cutoff,**score(c)))
        records.append(row)
        if index%6==0:
            sheet=Image.new('RGB',(1540,6*250+40),'#10212a')
            draw=ImageDraw.Draw(sheet)
            for j,label in enumerate(headers):draw.text((j*220+8,8),label,font=font,fill='white')
            sheets.append(sheet)
        y=(index%6)*250+35
        draw=ImageDraw.Draw(sheets[-1])
        draw.text((8,y),f'{s["scene_kind"]} | {s["satellite_time"][:10]} | {s["id"]}',font=font,fill='white')
        for j,panel in enumerate(panels):
            sheets[-1].paste(Image.fromarray(panel),(j*220+8,y+25))
        for j,(m,i) in enumerate([('stable_residual',0),('stable_residual',1),('siamese',0),('siamese',1)],3):
            c=row['models'][m][i]
            draw.text((j*220+8,y+225),f'IoU {100*c["iou"]:.1f}% | FP {c["fp"]} | t={c["cutoff"]:.2f}',font=font,fill='white')
    for i,sheet in enumerate(sheets):sheet.save(args.output/f'contact_{i:03d}.png')
    aggregate=[]
    for (i,m),c in counts.items():
        measured=score(dict(c))
        expected=frozen[i]['models'][m]['development']['pooled']
        if any(measured[k]!=expected[k] for k in ('tp','fp','fn','tn')):
            raise ValueError('Recount differs from frozen native development results')
        aggregate.append(dict(stage=('before','after')[i],model=m,**measured,no_plume=negative[(i,m)]))
    result=dict(passed=True,scope='Exposed DEV regression; not fresh validation',scenes=len(ordered),
                before_sha256=t.sha(args.before_freeze),after_sha256=t.sha(args.after_freeze),
                aggregates=aggregate,scenes_review=records,contact_sheets=len(sheets))
    t.write_json(args.output/'receipt.json',result)
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
