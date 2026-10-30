"""Select one bounded recall candidate before reserved final pixels are opened.

Failure cases and ranking were written first in the U-Net recall protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/GTM_Model7_unet_recall_protocol_20261024.md'
BASELINE = ROOT / 'outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration/freeze.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(frozen):
    choice = frozen['models']['siamese']
    manifest = Path(frozen['metadata']['manifest'])
    if sha(manifest) != frozen['metadata']['manifest_sha256']:
        raise ValueError('Changed training manifest')
    totals = dict(tp=0, fp=0, fn=0, tn=0)
    recalls, negative_pixels, negative_fp, negative_activated = [], 0, 0, 0
    recovered = 0
    for scene in json.loads(manifest.read_text())['scenes']:
        if scene['split'] != 'dev':
            continue
        array = manifest.parent / scene['array_file']
        if sha(array) != scene['array_sha256']:
            raise ValueError('Changed development array')
        with np.load(array, allow_pickle=False) as data:
            label = data['label_native'].astype(bool)
            valid = data['valid_native'].astype(bool)
        file = Path(frozen['run']) / 'siamese' / f"dev_step_{choice['steps']}" / f"{scene['id']}.npz"
        with np.load(file, allow_pickle=False) as data:
            prob = data['probability_native']
            if not np.array_equal(data['valid_native'], valid):
                raise ValueError('Different prediction support')
        hit = prob >= choice['threshold']
        counts = dict(tp=int((hit & label & valid).sum()), fp=int((hit & ~label & valid).sum()),
                      fn=int((~hit & label & valid).sum()), tn=int((~hit & ~label & valid).sum()))
        for key in totals:
            totals[key] += counts[key]
        if scene['scene_kind'] == 'PLUME':
            recall = counts['tp'] / max(counts['tp'] + counts['fn'], 1)
            recalls.append(recall)
            recovered += recall >= .1
        else:
            negative_pixels += int(valid.sum())
            negative_fp += counts['fp']
            negative_activated += counts['fp'] > 0
    tp, fp, fn = [totals[k] for k in ('tp', 'fp', 'fn')]
    metrics = dict(**totals, iou=tp / max(tp+fp+fn, 1), precision=tp / max(tp+fp, 1),
                   recall=tp / max(tp+fn, 1), negative_pixel_fpr=negative_fp / max(negative_pixels, 1),
                   mean_positive_scene_recall=float(np.mean(recalls)), recovered_positive_scenes=int(recovered),
                   positive_scenes=len(recalls), activated_negative_scenes=negative_activated)
    metrics['qualified'] = (metrics['precision'] >= .5 and metrics['recall'] >= .3 and
                            metrics['iou'] >= .2 and metrics['negative_pixel_fpr'] <= .001 and
                            recovered >= len(recalls)/2)
    return metrics


def rank(row):
    m, c = row['metrics'], row['choice']
    return (m['mean_positive_scene_recall'], m['iou'], -m['negative_pixel_fpr'],
            -c['steps'], c['threshold'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, action='append', required=True)
    parser.add_argument('--reservation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) != 2 or args.output.exists():
        raise ValueError('Exactly two frozen candidates and a new output folder required')
    reservation = json.loads(args.reservation.read_text())
    if not reservation['metadata_only'] or reservation['source_pixels_read']:
        raise ValueError('Final sites must still be reserved metadata only')
    options = []
    for path in [BASELINE, *args.candidate]:
        frozen = json.loads(path.read_text())
        if path != BASELINE and frozen['metadata']['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('Candidate protocol differs')
        if sha(frozen['models']['siamese']['checkpoint']) != frozen['models']['siamese']['checkpoint_sha256']:
            raise ValueError('Candidate weights changed')
        options.append(dict(path=str(path.resolve()), freeze_sha256=sha(path),
                            arm=frozen['metadata'].get('arm'), choice=frozen['models']['siamese'],
                            metrics=measure(frozen)))
    eligible = [r for r in options if r['metrics']['qualified']]
    if not eligible:
        raise ValueError('No eligible model, including baseline')
    winner = max(eligible, key=rank)
    args.output.mkdir(parents=True)
    source = Path(winner['path'])
    (args.output / 'freeze.json').write_bytes(source.read_bytes())
    receipt = dict(schema='GTM_Model7_recall_selection_v1', passed=True, protocol_sha256=sha(PROTOCOL),
                   source_sha256=sha(__file__), reservation_path=str(args.reservation.resolve()),
                   reservation_sha256=sha(args.reservation), final_pixels_opened=False,
                   selected_arm=winner['arm'], selected_freeze_sha256=sha(source),
                   improved_development_objective=winner['path'] != str(BASELINE.resolve()),
                   candidates=options, final_predictions_used=False)
    (args.output / 'selection_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
