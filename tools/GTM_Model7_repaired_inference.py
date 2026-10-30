"""Freeze repaired-model choices, save actual predictions, and replay them.

Failure and visual acceptance preceded implementation in
docs/GTM_Model7_dataset_repair_acceptance.md. Selection uses DEV only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired as t

MODELS = ('stable_residual', 'siamese')


def check_implementation(meta):
    global t
    if meta.get('trainer_path'):
        trainer_path = Path(meta['trainer_path']).resolve()
        if (trainer_path.parent != Path(__file__).resolve().parent or
                not trainer_path.name.startswith('GTM_Model7_train_repaired')):
            raise ValueError('Unexpected training implementation location')
        if t.sha(trainer_path) != meta['trainer_sha256']:
            raise ValueError('Versioned trainer source changed')
        spec = importlib.util.spec_from_file_location('repaired_frozen_trainer', trainer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        t = module
    for source, field in ((t.HERE, 'trainer_sha256'), (t.ORIGINAL, 'original_sha256'),
                          (t.STABLE, 'stable_source_sha256'), (t.PROTOCOL, 'protocol_sha256')):
        if t.sha(source) != meta[field]:
            raise ValueError('Frozen training implementation changed: ' + str(source))
    # Versioned trainers can reuse a prior implementation. Freeze that code too.
    if 'v3_source_sha256' in meta:
        dependency = getattr(t, 'V3', None)
        if dependency is None or t.sha(dependency) != meta['v3_source_sha256']:
            raise ValueError('Frozen v3 training dependency changed')
    if 'v4_source_sha256' in meta:
        dependency = getattr(t, 'V4', None)
        if dependency is None or t.sha(dependency) != meta['v4_source_sha256']:
            raise ValueError('Frozen v4 training dependency changed')
    if 'v5_source_sha256' in meta:
        dependency = getattr(t, 'V5', None)
        if dependency is None or t.sha(dependency) != meta['v5_source_sha256']:
            raise ValueError('Frozen v5 training dependency changed')
    # New versioned trainers can have explicitly frozen architecture variants.
    # Restore their configuration before constructing a checkpoint's model.
    if hasattr(t, 'configure'):
        t.configure(meta)


def freeze(run: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(destination)
    meta = json.loads((run / 'metadata.json').read_text())
    receipt = json.loads((run / 'receipt.json').read_text())
    e2e = json.loads((run / 'e2e_receipt.json').read_text())
    check_implementation(meta)
    if (not e2e['passed'] or e2e['manifest_sha256'] != meta['manifest_sha256'] or
            e2e['trainer_sha256'] != meta['trainer_sha256']):
        raise ValueError('A matching successful training replay is required')
    if t.sha(run / 'metadata.json') != receipt['metadata_sha256']:
        raise ValueError('Training metadata changed')
    manifest = Path(meta['manifest'])
    if t.sha(manifest) != meta['manifest_sha256']:
        raise ValueError('Dataset changed')
    entries = {}
    for key in MODELS:
        report = receipt['models'][key]
        step = report['selection']['selected_step']
        qualified = step is not None
        if step is None:
            # Preserve the failed candidate for visual diagnosis without promoting it.
            step = max((int(s) for s in report['milestones']), key=lambda s: (
                report['milestones'][str(s)]['dev']['positive']['iou'] or 0,
                -(report['milestones'][str(s)]['dev']['negative']['false_positive_area_rate'] or 0), -s))
        checkpoint = run / key / f'step_{step}.pt'
        expected = report['milestones'][str(step)]['checkpoint_sha256']
        if t.sha(checkpoint) != expected:
            raise ValueError('Selected checkpoint changed')
        entries[key] = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=expected,
                            steps=step, qualified_development_gate=qualified,
                            development=report['milestones'][str(step)]['dev'])
    result = dict(schema='GTM_Model7_repaired_freeze_v1', run=str(run.resolve()),
                  metadata=meta, training_receipt_sha256=t.sha(run / 'receipt.json'),
                  training_replay_sha256=t.sha(run / 'e2e_receipt.json'),
                  selection='DEV Pareto rule; unqualified candidates shown as failed diagnostics',
                  threshold=0.5, models=entries, test_predictions_used_for_selection=False)
    t.write_json(destination, result)
    print(json.dumps({'freeze': str(destination), 'models': {
        k: {'step': v['steps'], 'qualified': v['qualified_development_gate']} for k, v in entries.items()}}))


def infer(frozen_path: Path, split: str, output: Path, verify: bool):
    frozen = json.loads(frozen_path.read_text())
    meta = frozen['metadata']
    manifest_path = Path(meta['manifest'])
    if t.sha(manifest_path) != meta['manifest_sha256']:
        raise ValueError('Frozen dataset changed')
    check_implementation(meta)
    t.ACTIVE_BRANCH = meta['config']['branch']
    manifest = json.loads(manifest_path.read_text())
    scenes = [s for s in manifest['scenes'] if s['split'] == split]
    if not scenes:
        raise ValueError('Empty evaluation split')
    for scene in scenes:
        if t.sha(manifest_path.parent / scene['array_file']) != scene['array_sha256']:
            raise ValueError('Evaluation array changed before inference: ' + scene['id'])
    norm = meta['normalization']
    if not verify:
        output.mkdir(parents=True, exist_ok=False)
        receipt = dict(schema='GTM_Model7_repaired_inference_v1', freeze_sha256=t.sha(frozen_path),
                       manifest_sha256=meta['manifest_sha256'], split=split,
                       threshold=.5, models={})
    else:
        receipt = json.loads((output / 'receipt.json').read_text())
        if receipt['freeze_sha256'] != t.sha(frozen_path) or receipt['split'] != split:
            raise ValueError('Inference identity differs')
    maximum = 0.0
    for key, choice in frozen['models'].items():
        t.THRESHOLD = choice.get('threshold', frozen['threshold'])
        checkpoint = Path(choice['checkpoint'])
        if t.sha(checkpoint) != choice['checkpoint_sha256']:
            raise ValueError('Frozen weights changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['normalization'] != norm or saved['steps'] != choice['steps']:
            raise ValueError('Checkpoint preprocessing or update count differs')
        model = t.new_model(key)
        model.load_state_dict(saved['model'])
        folder = output / key
        if not verify:
            measured = t.evaluate(model, manifest_path, scenes, norm, split, folder)
            receipt['models'][key] = dict(**measured, checkpoint=str(checkpoint),
                                         threshold=t.THRESHOLD,
                                         checkpoint_sha256=choice['checkpoint_sha256'],
                                         qualified_development_gate=choice['qualified_development_gate'])
        else:
            for scene in scenes:
                path = manifest_path.parent / scene['array_file']
                if t.sha(path) != scene['array_sha256']:
                    raise ValueError('Array changed')
                data = t.load_scene(manifest_path, scene)
                p = t.predict(model, t.prepare(data, norm))
                file = folder / (scene['id'] + '.npz')
                expected = receipt['models'][key]['scenes'][scene['id']]
                if t.sha(file) != expected['probability_sha256']:
                    raise ValueError('Saved prediction changed')
                with np.load(file, allow_pickle=False) as z:
                    error = float(np.max(np.abs(p - z['probability20'])))
                    maximum = max(maximum, error)
                    if error > t.REPLAY_ATOL or not np.array_equal(z['valid20'], data['valid']):
                        raise ValueError('Prediction replay differs')
                    native = np.repeat(np.repeat(p, t.native_factor(), 0), t.native_factor(), 1)
                    if t.confusion(native, data['label_native'], data['valid_native']) != expected['native_confusion']:
                        raise ValueError('Replay native counts differ')
        print(json.dumps({'model': key, 'split': split, 'replay': verify, 'scenes': len(scenes)}), flush=True)
    if verify:
        t.write_json(output / 'replay.json', dict(passed=True, scenes_per_model=len(scenes),
                     receipt_sha256=t.sha(output / 'receipt.json'), maximum_probability_error=maximum,
                     freeze_sha256=t.sha(frozen_path), fresh_process=True))
    else:
        t.write_json(output / 'receipt.json', receipt)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    f = commands.add_parser('freeze')
    f.add_argument('--run', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('infer')
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--split', choices=('dev', 'test'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.run.resolve(), args.output.resolve())
    else:
        infer(args.freeze.resolve(), args.split, args.output.resolve(), args.verify)


if __name__ == '__main__':
    main()
