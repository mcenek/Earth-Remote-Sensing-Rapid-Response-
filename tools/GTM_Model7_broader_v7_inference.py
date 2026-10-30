"""Replay frozen models on broader, exposed DEV without changing fit identity.

Failure modes precede implementation in the broader-dev model protocol.
This entry point deliberately cannot open a final TEST split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_repaired_inference as inference


def run(freeze_path: Path, evaluation_path: Path, output: Path, verify: bool):
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    meta = frozen['metadata']
    inference.check_implementation(meta)
    t = inference.t
    t.set_branch(meta['config']['branch'])
    training_path = Path(meta['manifest'])
    if t.sha(training_path) != meta['manifest_sha256']:
        raise ValueError('Frozen fit manifest changed')
    training = json.loads(training_path.read_text(encoding='utf-8'))
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    scenes = [s for s in evaluation['scenes'] if s['split'] == 'dev']
    fit = [s for s in training['scenes'] if s['split'] == 'train']
    if not scenes or {s['scene_kind'] for s in scenes} != {'PLUME', 'NO_PLUME'}:
        raise ValueError('Both DEV classes required')
    for field in ('id', 'physical_location_id', 'target_scene_id', 'reference_scene_id'):
        fit_values = {s[field] for s in fit}
        if any(s[field] in fit_values for s in scenes):
            raise ValueError('Fit/DEV identity overlap: ' + field)
    fit_acq = {s[k] for s in fit for k in ('target_scene_id', 'reference_scene_id')}
    dev_acq = {s[k] for s in scenes for k in ('target_scene_id', 'reference_scene_id')}
    fit_hashes = {a['sha256'] for s in fit for a in s['sources']
                  if a['role'] in ('target_image', 'reference_image')}
    dev_hashes = {a['sha256'] for s in scenes for a in s['sources']
                  if a['role'] in ('target_image', 'reference_image')}
    if fit_acq & dev_acq or fit_hashes & dev_hashes:
        raise ValueError('Fit/DEV acquisition or spectral content overlap')
    for s in scenes:
        if t.sha(evaluation_path.parent / s['array_file']) != s['array_sha256']:
            raise ValueError('DEV array changed: ' + s['id'])
    identities = dict(freeze_sha256=t.sha(freeze_path), manifest_sha256=t.sha(evaluation_path),
                      evaluation_manifest_sha256=t.sha(evaluation_path),
                      training_manifest_sha256=t.sha(training_path), split='dev')
    if verify:
        receipt = json.loads((output / 'receipt.json').read_text(encoding='utf-8'))
        if any(receipt[k] != v for k, v in identities.items()):
            raise ValueError('Saved inference identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        receipt = dict(schema='GTM_Model7_broader_dev_inference_v1', **identities,
                       role='exposed_development_selection_not_independent_test',
                       inference_source_sha256=t.sha(Path(__file__)), models={})
    maximum = 0.0
    for key, choice in frozen['models'].items():
        checkpoint = Path(choice['checkpoint'])
        if t.sha(checkpoint) != choice['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['normalization'] != meta['normalization'] or saved['steps'] != choice['steps']:
            raise ValueError('Checkpoint/preprocessing mismatch')
        model = t.new_model(key)
        model.load_state_dict(saved['model'])
        t.THRESHOLD = choice.get('threshold', frozen['threshold'])
        if not verify:
            score = t.evaluate(model, evaluation_path, scenes, meta['normalization'], 'dev', output/key)
            receipt['models'][key] = dict(**score, checkpoint=str(checkpoint),
                checkpoint_sha256=choice['checkpoint_sha256'], threshold=t.THRESHOLD,
                qualified_development_gate=choice['qualified_development_gate'])
        else:
            for scene in scenes:
                sample = t.load_scene(evaluation_path, scene)
                actual = t.predict(model, t.prepare(sample, meta['normalization']))
                path = output/key/(scene['id']+'.npz')
                expected = receipt['models'][key]['scenes'][scene['id']]
                if t.sha(path) != expected['probability_sha256']:
                    raise ValueError('Saved DEV prediction changed')
                with np.load(path, allow_pickle=False) as z:
                    error = float(np.max(np.abs(actual-z['probability20'])))
                    maximum = max(maximum, error)
                    if error > t.REPLAY_ATOL or not np.array_equal(z['valid20'], sample['valid']):
                        raise ValueError('DEV probability/support replay failed')
                    native = np.repeat(np.repeat(actual, t.native_factor(), 0), t.native_factor(), 1)
                    if not np.array_equal(z['probability_native'], native) or not np.array_equal(z['valid_native'], sample['valid_native']):
                        raise ValueError('Native grid replay failed')
                    if t.confusion(native, sample['label_native'], sample['valid_native']) != expected['native_confusion']:
                        raise ValueError('Native confusion replay failed')
        print(json.dumps(dict(model=key, dev_scenes=len(scenes), replay=verify)), flush=True)
    if verify:
        t.write_json(output/'replay.json', dict(passed=True, fresh_process=True, **identities,
            receipt_sha256=t.sha(output/'receipt.json'), maximum_probability_error=maximum,
            scenes_per_model=len(scenes)))
    else:
        t.write_json(output/'receipt.json', receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--evaluation-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    run(a.freeze.resolve(), a.evaluation_manifest.resolve(), a.output.resolve(), a.verify)


if __name__ == '__main__':
    main()
