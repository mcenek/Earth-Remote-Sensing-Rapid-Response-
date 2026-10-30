"""Render every broader-DEV plume and clear case from saved predictions.

Failure controls: frozen manifest and NPZ support, fixed cutoff, no fitting or
case replacement. The broader-dev protocol requires actual visual review.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import GTM_Model7_train_repaired_broader_v7 as t
import GTM_Model7_review_repaired as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--cutoff', type=float, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=('dev', 'test'), default='dev')
    a = p.parse_args()
    m = json.loads(a.manifest.read_text(encoding='utf-8'))
    scenes = [s for s in m['scenes'] if s['split'] == a.split]
    if not scenes or (a.split == 'test' and any(s['scene_kind'] != 'NO_PLUME' for s in scenes)):
        raise ValueError('Only broader DEV or reserved no-plume TEST supported')
    scenes.sort(key=lambda s: (s['scene_kind'] != 'PLUME', s.get('country', ''), s['physical_location_id'], s['id']))
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    for start in range(0, len(scenes), 8):
        page = Image.new('RGB', (1050, 8*245+65), '#101e25')
        draw = ImageDraw.Draw(page)
        scope = 'BROADER DEVELOPMENT, previously evaluated sites.' if a.split == 'dev' else 'RESERVED NEW NO-PLUME SITES, no plume recall claim.'
        draw.text((10, 8), scope+' '+a.label+' cutoff '+str(a.cutoff), fill='white')
        for j, label in enumerate(('Current', 'Reference', 'Annotation', 'Old v5 errors, .95', 'Candidate errors')):
            draw.text((10+210*j, 33), label, fill='white')
        for row, s in enumerate(scenes[start:start+8]):
            if t.sha(a.manifest.parent/s['array_file']) != s['array_sha256']:
                raise ValueError('Array drift')
            d = t.load_scene(a.manifest, s)
            current, reference = r.balanced_rgb_pair(d, 2)
            truth, valid = d['label_native'], d['valid_native'].astype(bool)
            annotation = current.copy()
            annotation[(truth == 1) & valid] = (46, 224, 219)
            images = [current, reference, annotation]
            counts = {}
            for label, folder, cutoff in [('baseline', a.baseline, .95), ('candidate', a.candidate, a.cutoff)]:
                with np.load(folder/(s['id']+'.npz'), allow_pickle=False) as z:
                    prob = z['probability_native']
                    if not np.array_equal(z['valid_native'], valid) or prob.shape != valid.shape:
                        raise ValueError('Prediction/support mismatch')
                mask = (prob >= cutoff).astype(np.float32)
                images.append(r.v.overlay(current, mask, truth, valid)[1])
                counts[label] = r.v.counts(mask, truth, valid)
            title = f"{s['scene_kind']} | {s.get('country')} | {s['id'][5:13]}"
            y = 60+row*245
            draw.text((10, y), title, fill='white')
            for j, im in enumerate(images):
                page.paste(Image.fromarray(im).convert('RGB'), (10+210*j, y+20))
            draw.text((640, y+221), f"TP {counts['baseline']['tp']} FP {counts['baseline']['fp']}", fill='white')
            draw.text((850, y+221), f"TP {counts['candidate']['tp']} FP {counts['candidate']['fp']}", fill='white')
            records.append(dict(id=s['id'], kind=s['scene_kind'], country=s.get('country'), counts=counts))
        page.save(a.output/f'contact_{start//8:03d}.png')
    t.write_json(a.output/'receipt.json', dict(manifest_sha256=t.sha(a.manifest),
        source_sha256=t.sha(Path(__file__)), label=a.label, candidate_cutoff=a.cutoff,
        scene_count=len(records), legend='Teal overlap, pink extra prediction, purple missed annotation', records=records))
    print(json.dumps(dict(output=str(a.output), scenes=len(records), pages=(len(records)+7)//8)))


if __name__ == '__main__':
    main()
