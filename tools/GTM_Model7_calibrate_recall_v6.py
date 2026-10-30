"""Freeze bounded recall candidates on exposed DEV, never on new final sites.

Failure cases and selection precede code in GTM_Model7_unet_recall_protocol_20261024.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_repaired_inference as inference

PROTOCOL = Path(__file__).resolve().parents[1] / 'docs/GTM_Model7_unet_recall_protocol_20261024.md'
CUTOFFS = [round(i / 20, 3) for i in range(1, 20)] + [.975, .99, .995]
KEYS = ('tp', 'fp', 'fn', 'tn')


def counts(p, truth, valid, cutoff):
    prediction, label = p[valid] >= cutoff, truth[valid] == 1
    return dict(tp=int((prediction & label).sum()), fp=int((prediction & ~label).sum()),
                fn=int((~prediction & label).sum()), tn=int((~prediction & ~label).sum()))


def ratios(c):
    tp, fp, fn = (c[k] for k in ('tp', 'fp', 'fn'))
    return dict(**c, iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1))


def rank(row):
    return row['mean_positive_scene_recall'], row['pooled']['iou'], -row['negative_pixel_fpr'], -row['step'], row['cutoff']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((run / 'metadata.json').read_text())
    receipt = json.loads((run / 'receipt.json').read_text())
    replay = json.loads((run / 'e2e_receipt.json').read_text())
    inference.check_implementation(meta)
    t = inference.t
    if (not replay['passed'] or replay['manifest_sha256'] != meta['manifest_sha256'] or
            replay['trainer_sha256'] != meta['trainer_sha256'] or
            t.sha(run / 'metadata.json') != receipt['metadata_sha256']):
        raise ValueError('A matching completed training replay is required')
    path = Path(meta['manifest'])
    if t.sha(path) != meta['manifest_sha256']:
        raise ValueError('Dataset changed')
    manifest = json.loads(path.read_text())
    scenes = [s for s in manifest['scenes'] if s['split'] == 'dev']
    if not scenes or not any(s['scene_kind'] == 'PLUME' for s in scenes) or not any(s['scene_kind'] == 'NO_PLUME' for s in scenes):
        raise ValueError('Both development classes are required')
    t.set_branch(meta['config']['branch'])
    data = {}
    for scene in scenes:
        if t.sha(path.parent / scene['array_file']) != scene['array_sha256']:
            raise ValueError('DEV array changed')
        data[scene['id']] = t.load_scene(path, scene)
    choices, candidates, selected_replays = {}, {}, {}
    baseline_path = Path(__file__).resolve().parents[1] / 'outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration/freeze.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    if meta['normalization'] != baseline['metadata']['normalization']:
        raise ValueError('Normalizer differs from retained control')
    for model_name in inference.MODELS:
        if model_name == 'stable_residual':
            choices[model_name] = baseline['models'][model_name]
            candidates[model_name] = {'retained_control': True, 'baseline_freeze_sha256': t.sha(baseline_path)}
            selected_replays[model_name] = {'retained_control': True, 'training_e2e': 'verified by copied-source checkpoint identity'}
            continue
        rows = []
        milestones = receipt['models'][model_name]['milestones']
        for step_key, milestone in milestones.items():
            step = int(step_key)
            checkpoint = run / model_name / f'step_{step}.pt'
            if t.sha(checkpoint) != milestone['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            aggregate = [{k: 0 for k in KEYS} for _ in CUTOFFS]
            neg_fp, neg_pixels = np.zeros(len(CUTOFFS), np.int64), 0
            recovered, activated = np.zeros(len(CUTOFFS), int), np.zeros(len(CUTOFFS), int)
            recall_sums = np.zeros(len(CUTOFFS), float)
            positives = negatives = 0
            for scene in scenes:
                sample = data[scene['id']]
                file = run / model_name / f'dev_step_{step}' / (scene['id'] + '.npz')
                expected = milestone['dev']['scenes'][scene['id']]
                if t.sha(file) != expected['probability_sha256']:
                    raise ValueError('Saved DEV scores changed')
                with np.load(file, allow_pickle=False) as z:
                    probability = z['probability_native']
                    if (probability.shape != sample['label_native'].shape or
                            not np.isfinite(probability).all() or
                            ((probability < 0) | (probability > 1)).any() or
                            not np.array_equal(z['valid_native'], sample['valid_native']) or
                            not np.array_equal(probability, np.repeat(np.repeat(z['probability20'], t.native_factor(), 0), t.native_factor(), 1))):
                        raise ValueError('Malformed saved probability/support')
                if counts(probability, sample['label_native'], sample['valid_native'], .5) != expected['native_confusion']:
                    raise ValueError('Original native counts changed')
                positive = scene['scene_kind'] == 'PLUME'
                positives += positive
                negatives += not positive
                if not positive:
                    neg_pixels += int(sample['valid_native'].sum())
                for index, cutoff in enumerate(CUTOFFS):
                    c = counts(probability, sample['label_native'], sample['valid_native'], cutoff)
                    for key in KEYS:
                        aggregate[index][key] += c[key]
                    if positive:
                        recall_sums[index] += c['tp'] / max(c['tp'] + c['fn'], 1)
                        recovered[index] += c['tp'] / max(c['tp'] + c['fn'], 1) >= .1
                    else:
                        neg_fp[index] += c['fp']
                        activated[index] += c['fp'] > 0
            for index, cutoff in enumerate(CUTOFFS):
                r = ratios(aggregate[index])
                fpr = int(neg_fp[index]) / max(neg_pixels, 1)
                qualified = (r['precision'] >= .5 and r['recall'] >= .3 and r['iou'] >= .2 and
                             fpr <= .001 and recovered[index] >= positives / 2)
                rows.append(dict(step=step, cutoff=cutoff, pooled=r, negative_pixel_fpr=fpr,
                                 recovered_positive_scenes=int(recovered[index]), positive_scenes=positives,
                                 activated_negative_scenes=int(activated[index]), negative_scenes=negatives,
                                 mean_positive_scene_recall=float(recall_sums[index] / positives),
                                 qualified=bool(qualified)))
        qualified = [row for row in rows if row['qualified']]
        selected = max(qualified or rows, key=rank)
        checkpoint = run / model_name / f'step_{selected["step"]}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['normalization'] != meta['normalization'] or saved['steps'] != selected['step']:
            raise ValueError('Checkpoint preprocessing differs')
        model = t.new_model(model_name)
        model.load_state_dict(saved['model'])
        maximum = 0.0
        for scene in scenes:
            sample = data[scene['id']]
            actual = t.predict(model, t.prepare(sample, meta['normalization']))
            file = run / model_name / f'dev_step_{selected["step"]}' / (scene['id'] + '.npz')
            with np.load(file, allow_pickle=False) as z:
                error = float(np.max(np.abs(actual - z['probability20'])))
            maximum = max(maximum, error)
            if error > t.REPLAY_ATOL:
                raise ValueError('Selected DEV checkpoint replay differs')
        choices[model_name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=t.sha(checkpoint),
                                  steps=selected['step'], threshold=selected['cutoff'],
                                  qualified_development_gate=selected['qualified'], development=selected)
        candidates[model_name] = rows
        selected_replays[model_name] = dict(scenes=len(scenes), maximum_probability_error=maximum)
        print(json.dumps({'model': model_name, 'selection': selected, 'replay_error': maximum}), flush=True)
    output.mkdir(parents=True)
    frozen = dict(schema='GTM_Model7_repaired_freeze_v1', selection_kind='DEV_scene_recall_calibration_v6',
                  run=str(run), metadata=meta, training_receipt_sha256=t.sha(run / 'receipt.json'),
                  training_replay_sha256=t.sha(run / 'e2e_receipt.json'),
                  selection='Fixed DEV cutoff grid; strict unchanged gate, then mean positive-scene native recall / pooled IoU / lower clear FPR / earlier step / higher cutoff. Retained residual control is not recalibrated.',
                  threshold=.5, models=choices, test_predictions_used_for_selection=False,
                  retained_control_freeze_sha256=t.sha(baseline_path), calibration_protocol_sha256=t.sha(PROTOCOL), calibration_source_sha256=t.sha(Path(__file__)))
    t.write_json(output / 'freeze.json', frozen)
    t.write_json(output / 'candidates.json', dict(cutoffs=CUTOFFS, models=candidates,
                 manifest_sha256=t.sha(path), test_array_access='none'))
    t.write_json(output / 'verification.json', dict(passed=True, fresh_process=True,
                 selected_checkpoint_replays=selected_replays, freeze_sha256=t.sha(output / 'freeze.json'),
                 test_array_access='none'))


if __name__ == '__main__':
    main()
