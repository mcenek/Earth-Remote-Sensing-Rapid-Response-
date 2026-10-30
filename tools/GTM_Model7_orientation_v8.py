"""Real broader-DEV D4 stability audit, with frozen weights and no fitting.

Failure cases were declared in orientation_v8_protocol before implementation.
Only DEV rows of the pinned combined TRAIN/DEV manifest may be opened.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_repaired_inference as factory
import GTM_Model7_calibrate_broader_v7 as calibration

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'docs/GTM_Model7_orientation_v8_protocol_20261024.md'
MANIFEST_SHA = '1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a'
FREEZE_SHA = '8a676661a15285637d7490a27204f2e1c0b5d39f0eb5c94cc0b27655304b6223'


def inverse(a, code):
    result = np.rot90(a, -(code >> 1), axes=(0, 1))
    return (np.flip(result, axis=1) if code & 1 else result).copy()


def rank(row):
    return (row['mean_positive_site_recall'], row['mean_positive_scene_recall'],
            row['pooled']['iou'], -row['clear_pixel_fpr'], -row['step'], row['cutoff'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--identity-inference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    a = parser.parse_args()
    torch.set_num_threads(2)
    frozen = json.loads(a.freeze.read_text())
    factory.check_implementation(frozen['metadata'])
    t = factory.t
    if t.sha(a.freeze) != FREEZE_SHA or t.sha(a.manifest) != MANIFEST_SHA:
        raise ValueError('Only the pinned v7 selection and broader DEV manifest are allowed')
    manifest = json.loads(a.manifest.read_text())
    if any(s['split'] not in ('train', 'dev') for s in manifest['scenes']):
        raise ValueError('No final evaluation splits allowed')
    scenes = sorted((s for s in manifest['scenes'] if s['split'] == 'dev'), key=lambda s: s['id'])
    if len(scenes) != 198 or sum(s['scene_kind'] == 'PLUME' for s in scenes) != 46:
        raise ValueError('Broader development membership changed')
    choice = frozen['models']['siamese']
    checkpoint = Path(choice['checkpoint'])
    if t.sha(checkpoint) != choice['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['normalization'] != frozen['metadata']['normalization'] or saved['steps'] != 1200:
        raise ValueError('Checkpoint preprocessing/update identity changed')
    model = t.new_model('siamese')
    model.load_state_dict(saved['model'])
    model.eval()
    t.set_branch('mars_repaired')
    norm = saved['normalization']
    identity_receipt = json.loads((a.identity_inference/'receipt.json').read_text())
    if identity_receipt['freeze_sha256'] != FREEZE_SHA or identity_receipt['evaluation_manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Identity comparison is not the frozen v7 DEV inference')
    pins = dict(source_sha256=t.sha(Path(__file__)), protocol_sha256=t.sha(PROTOCOL),
                freeze_sha256=FREEZE_SHA, manifest_sha256=MANIFEST_SHA,
                checkpoint_sha256=choice['checkpoint_sha256'],
                identity_receipt_sha256=t.sha(a.identity_inference/'receipt.json'))
    if a.verify:
        receipt = json.loads((a.output/'receipt.json').read_text())
        if any(receipt[k] != v for k, v in pins.items()):
            raise ValueError('Audit identity changed before replay')
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output/'scores').mkdir()
    # Real-data preflight, before the full 198-case inference.
    preflight = []
    coord = np.arange(10000).reshape(100, 100)
    transforms = t.v5.v4.v3.old.d4
    for kind in ('PLUME', 'NO_PLUME'):
        scene = next(s for s in scenes if s['scene_kind'] == kind)
        if t.sha(a.manifest.parent/scene['array_file']) != scene['array_sha256']:
            raise ValueError('Preflight source drift')
        data = t.load_scene(a.manifest, scene)
        features = t.prepare(data, norm)
        for code in range(8):
            permutation = transforms(coord, code)
            if len(np.unique(permutation)) != 10000 or not np.array_equal(inverse(permutation, code), coord):
                raise ValueError('D4 coordinate inversion failed')
            moved = {key: transforms(value, code) if isinstance(value, np.ndarray) and value.shape[:2] == (100, 100)
                     else value for key, value in data.items()}
            if not np.allclose(t.prepare(moved, norm), transforms(features, code), atol=1e-7, rtol=0):
                raise ValueError('Preprocessing does not commute with D4')
            for key in ('target_image', 'reference_image', 'target', 'valid'):
                if not np.array_equal(inverse(transforms(data[key], code), code), data[key]):
                    raise ValueError('Real source/mask roundtrip failed')
        preflight.append(dict(scene=scene['id'], kind=kind, passed=True))
    if not a.verify:
        t.write_json(a.output/'preflight.json', dict(passed=True, examples=preflight, **pins))
    probabilities, identity_probabilities, samples, records = {}, {}, {}, {}
    member_times = np.zeros(8)
    max_identity_error = max_replay_error = inverse_error = 0.0
    for number, scene in enumerate(scenes):
        sid = scene['id']
        if t.sha(a.manifest.parent/scene['array_file']) != scene['array_sha256']:
            raise ValueError('DEV array changed: '+sid)
        data = t.load_scene(a.manifest, scene)
        features = t.prepare(data, norm)
        valid = data['valid'].astype(bool)
        members = []
        for code in range(8):
            started = time.perf_counter()
            raw = t.predict(model, transforms(features, code))
            member_times[code] += time.perf_counter()-started
            restored = inverse(raw, code)
            scatter = np.empty_like(raw).reshape(-1)
            scatter[transforms(coord, code).reshape(-1)] = raw.reshape(-1)
            error = float(np.max(np.abs(restored-scatter.reshape(raw.shape))))
            inverse_error = max(inverse_error, error)
            if error != 0:
                raise ValueError('Independent coordinate scatter disagrees with inverse')
            members.append(restored)
        members = np.stack(members)
        mean = members.mean(axis=0, dtype=np.float32)
        deviation = members.std(axis=0, dtype=np.float32)
        with np.load(a.identity_inference/'siamese'/(sid+'.npz'), allow_pickle=False) as old:
            expected = identity_receipt['models']['siamese']['scenes'][sid]['probability_sha256']
            if t.sha(a.identity_inference/'siamese'/(sid+'.npz')) != expected:
                raise ValueError('Frozen identity prediction changed')
            error = float(np.max(np.abs(old['probability20']-members[0])))
            max_identity_error = max(max_identity_error, error)
            if error > 1e-6 or not np.array_equal(old['valid20'], valid):
                raise ValueError('Identity member does not replay frozen prediction')
        if not np.isfinite(members).all() or members.min() < 0 or members.max() > 1:
            raise ValueError('Malformed orientation probabilities')
        native = np.repeat(np.repeat(mean, 2, 0), 2, 1)
        file = a.output/'scores'/(sid+'.npz')
        if a.verify:
            if t.sha(file) != receipt['records'][sid]['probability_sha256']:
                raise ValueError('Saved orientation score changed')
            with np.load(file, allow_pickle=False) as z:
                error = max(float(np.max(np.abs(z['members20']-members))), float(np.max(np.abs(z['probability20']-mean))))
                max_replay_error = max(max_replay_error, error)
                if error > 1e-6 or not np.array_equal(z['valid20'], valid) or not np.array_equal(z['probability_native'], native):
                    raise ValueError('Fresh-process orientation replay failed')
        else:
            np.savez_compressed(file, members20=members, probability20=mean, std20=deviation,
                                probability_native=native, valid20=valid, valid_native=data['valid_native'])
        probabilities[sid] = native
        identity_probabilities[sid] = np.repeat(np.repeat(members[0], 2, 0), 2, 1)
        samples[sid] = data
        records[sid] = dict(probability_sha256=t.sha(file),
                            mean_orientation_std=float(deviation[valid].mean()),
                            max_orientation_std=float(deviation[valid].max()),
                            threshold_member_disagreement=float(((members[:, valid] >= .7).any(0) &
                                                                 ~(members[:, valid] >= .7).all(0)).mean()))
        if number % 25 == 0:
            print(json.dumps(dict(scene=number+1, total=len(scenes), replay=a.verify)), flush=True)
    baseline = calibration.measure(identity_probabilities, samples, scenes, .7, 1200)
    fixed = calibration.measure(probabilities, samples, scenes, .7, 1200)
    candidates = [calibration.measure(probabilities, samples, scenes, cutoff, 1200) for cutoff in calibration.CUTOFFS]
    eligible = [row for row in candidates if row['qualified']]
    selected = max(eligible, key=rank) if eligible else fixed
    improved = (selected['qualified'] and selected['mean_positive_site_recall'] > baseline['mean_positive_site_recall'] and
                selected['pooled']['iou'] >= baseline['pooled']['iou'] and selected['clear_pixel_fpr'] <= baseline['clear_pixel_fpr'])
    measurement = dict(identity=baseline, mean_fixed_cutoff=fixed, mean_selected=selected,
                       eligible_cutoffs=len(eligible), improved_development_gate=bool(improved), candidates=candidates)
    if a.verify:
        old_measurement = json.loads((a.output/'measurement.json').read_text())
        if measurement != old_measurement:
            raise ValueError('Independent native counts/selection changed')
        t.write_json(a.output/'replay.json', dict(passed=True, fresh_cpu_process=True, **pins,
            observations=len(scenes), members_per_observation=8, maximum_probability_error=max_replay_error,
            maximum_identity_error=max_identity_error, inverse_scatter_error=inverse_error))
    else:
        t.write_json(a.output/'measurement.json', measurement)
        t.write_json(a.output/'receipt.json', dict(passed=True, **pins,
            scope='Exposed broader DEV only; orientation averaging is inference, not training',
            records=records, observations=len(scenes), members_per_observation=8,
            maximum_identity_error=max_identity_error, inverse_scatter_error=inverse_error,
            seconds_per_member=member_times.tolist(), identity_seconds=float(member_times[0]),
            eight_view_seconds=float(member_times.sum()), torch_threads=2,
            test_arrays_opened=False, downloaded_bytes=0))
    print(json.dumps(dict(passed=True, replay=a.verify, baseline_iou=baseline['pooled']['iou'],
        mean_fixed_iou=fixed['pooled']['iou'], selected_cutoff=selected['cutoff'],
        eligible_cutoffs=len(eligible), improved_development_gate=bool(improved),
        identity_seconds=float(member_times[0]), eight_view_seconds=float(member_times.sum()))), flush=True)


if __name__ == '__main__':
    main()
