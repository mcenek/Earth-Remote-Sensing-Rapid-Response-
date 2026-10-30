"""Bounded TRAIN-only image-pair sensitivity review, never a test-set score.

Failure cases and case selection precede implementation in the pipeline protocol.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import GTM_Model7_train_repaired_mars_log_v3 as t


def rgb_pair(data):
    valid = data['valid'].astype(bool)
    values = np.concatenate([data[k][valid][:, [2, 1, 0]]
                             for k in ('target_image', 'reference_image')])
    lo, hi = np.percentile(values, [2, 98], axis=0)
    width = np.maximum(hi-lo, 1000)
    center = (hi+lo)/2
    images = []
    for k in ('target_image', 'reference_image'):
        x = np.clip((data[k][..., [2, 1, 0]]-(center-width/2))/width, 0, 1)
        x[~valid] = .2
        images.append((255*x).astype('uint8'))
    return images


def paint(image, mask, color):
    out = image.astype(float).copy()
    out[mask] = .45*out[mask]+.55*np.array(color)
    return out.astype('uint8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--baseline-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.freeze.read_text())
    manifest = Path(frozen['metadata']['manifest'])
    if t.sha(manifest) != frozen['metadata']['manifest_sha256']:
        raise ValueError('Changed manifest')
    base = {s['id'] for s in json.loads(args.baseline_manifest.read_text())['scenes']}
    groups = defaultdict(list)
    for s in json.loads(manifest.read_text())['scenes']:
        if s['split'] == 'train':
            groups[('original_US' if s['id'] in base else 'auxiliary', s['scene_kind'])].append(s)
    scenes = []
    for key in sorted(groups):
        scenes.extend(sorted(groups[key], key=lambda s: hashlib.sha256(s['id'].encode()).hexdigest())[:8])
    records = []
    for model_name, choice in frozen['models'].items():
        state = torch.load(choice['checkpoint'], map_location='cpu', weights_only=False)
        if t.sha(Path(choice['checkpoint'])) != choice['checkpoint_sha256']:
            raise ValueError('Changed checkpoint')
        model = t.new_model(model_name)
        model.load_state_dict(state['model'])
        cutoff = choice['threshold']
        sheets = []
        for i, scene in enumerate(scenes):
            if t.sha(manifest.parent/scene['array_file']) != scene['array_sha256']:
                raise ValueError('Changed source array')
            d = t.load_scene(manifest, scene)
            x = t.prepare(d, state['normalization'])
            same = np.concatenate([x[..., :6], x[..., :6]], -1)
            swap = np.concatenate([x[..., 6:], x[..., :6]], -1)
            scores = [t.predict(model, a) for a in (x, same, swap)]
            valid = d['valid'].astype(bool)
            changes = {}
            for key, score in zip(('same_date', 'swapped'), scores[1:]):
                changes[key] = dict(mean_abs_score_change=float(np.abs(score-scores[0])[valid].mean()),
                                    threshold_flip_fraction=float(((score>=cutoff)!=(scores[0]>=cutoff))[valid].mean()))
            records.append(dict(model=model_name, scene=scene['id'], domain='original_US' if scene['id'] in base else 'auxiliary',
                                kind=scene['scene_kind'], cutoff=cutoff, **changes))
            np.savez_compressed(args.output/f'{model_name}_{scene["id"]}.npz', original=scores[0],
                                same_date=scores[1], swapped=scores[2], valid=valid)
            if i % 8 == 0:
                sheet = Image.new('RGB', (1320, 8*240+45), '#10212a')
                draw = ImageDraw.Draw(sheet)
                for j, label in enumerate(('Current', 'Reference', 'Annotation', 'Actual prediction', 'Same-date input', 'Exchanged dates')):
                    draw.text((j*220+8, 8), label, fill='white')
                sheets.append(sheet)
            y = (i % 8)*240+30
            draw = ImageDraw.Draw(sheets[-1])
            draw.text((8, y), f'{model_name} | {records[-1]["domain"]} | {scene["scene_kind"]} | {scene["id"]}', fill='white')
            a, b = rgb_pair(d)
            images = [a,b,paint(a,(d['target']==1)&valid,(30,230,220))]
            images += [paint(a,(v>=cutoff)&valid,(255,140,30)) for v in scores]
            for j, img in enumerate(images):
                sheets[-1].paste(Image.fromarray(img).resize((200,200),Image.Resampling.NEAREST),(j*220+8,y+25))
        for i, sheet in enumerate(sheets):
            sheet.save(args.output/f'{model_name}_{i:02d}.png')
    summary = defaultdict(list)
    for row in records:
        for perturb in ('same_date','swapped'):
            summary[(row['model'],row['domain'],row['kind'],perturb)].append(row[perturb])
    result = dict(scope='TRAIN-only sensitivity diagnostic; perturbed inputs are not accuracy tests',
                  manifest_sha256=t.sha(manifest), freeze_sha256=t.sha(args.freeze), records=records,
                  summary=[dict(model=k[0],domain=k[1],kind=k[2],perturbation=k[3],scenes=len(v),
                                mean_abs_score_change=float(np.mean([x['mean_abs_score_change'] for x in v])),
                                mean_threshold_flip_fraction=float(np.mean([x['threshold_flip_fraction'] for x in v])))
                           for k,v in summary.items()])
    t.write_json(args.output/'receipt.json',result)
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    main()
