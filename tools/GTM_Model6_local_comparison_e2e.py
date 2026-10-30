"""Public-command acceptance for the offline CPU comparison; written before implementation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'tools/GTM_Model6_local_comparison.py'


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def fixture(root, fault=None):
    cache, r5 = root / 'cache', root / 'r5'
    cache.mkdir(parents=True); r5.mkdir()
    rows, arrays = [], {}
    for i in range(4):
        row = dict(id=f'fixture-{i}', dataset='MARS', label='PLUME', site=f'site-{i}',
                   group=f'group-{i}', partition=0 if i < 2 else 1,
                   target_scene=f'acquisition-{i}', reference_scene=f'reference-{i}',
                   crs='EPSG:32613', affine=[10, 0, 500000, 0, -10, 3500000], shape=[16, 16],
                   hashes={'image': f'image-hash-{i}'}, fully_labeled=True, date='2023-01-01',
                   lat=31.0 + i, lon=-103.0)
        if fault == 'leak' and i == 2: row['group'] = 'group-0'
        rows.append(row)
        target = np.zeros((16, 16), np.float32); target[:8] = 1
        arrays[f'{i}_x'] = np.zeros((17, 16, 16), np.float32)
        arrays[f'{i}_target'] = target
        arrays[f'{i}_valid'] = np.ones((16, 16), bool)
        arrays[f'{i}_rgb'] = np.ones((16, 16, 3), np.float32)
    if fault == 'nonfinite': arrays['0_x'][0, 0, 0] = np.nan
    if fault == 'no_eval_targets':
        for i in (2, 3): del arrays[f'{i}_target']
    np.savez_compressed(cache / 'GTM_Model6_plume_inputs.npz', **arrays)
    write(cache / 'GTM_Model6_plume_manifest.json', {'observations': rows})
    write(r5 / 'GTM_Model6_mapper_mars_pilot_manifest.json', {
        'holdout_fold': 1, 'training_records': 2, 'held_records': 2,
        'training_observations': rows[:2], 'observations': rows[2:]})
    records = []
    for index, row in enumerate(rows[2:]):
        target = np.zeros((16, 16), np.float32); target[:8] = 1
        if fault == 'target_mismatch' and index == 0: target[0, 0] = 0
        score = np.full((16, 16), .5, np.float32)
        np.savez_compressed(r5 / f'prediction_{index:04d}.npz', probability=score,
                            target=target, valid=np.ones_like(target, np.uint8),
                            context_support=np.ones_like(target, np.uint8),
                            **{name: score for name in ('S16', 'S32', 'S64', 'S128')})
        counts = dict(tp=128, fp=128, fn=0, tn=0)
        records.append({'index': index, 'id': row['id'], 'metrics': counts,
                        'per_method': {name: counts for name in ('mean', 'S16', 'S32', 'S64', 'S128')}})
    write(r5 / 'GTM_Model6_mapper_mars_pilot_results.json', {'observations': records})
    if fault == 'metadata':
        changed = json.loads((cache / 'GTM_Model6_plume_manifest.json').read_text())
        changed['observations'][0]['hashes']['image'] = 'changed'
        write(cache / 'GTM_Model6_plume_manifest.json', changed)
    return cache, r5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    records = []

    def invoke(name, cache, r5, output, expect=0, phase='all'):
        command = [sys.executable, str(CLI), '--cache-dir', str(cache), '--r5-dir', str(r5),
                   '--output', str(output), '--phase', phase, '--minutes', '1']
        result = subprocess.run(command, capture_output=True, text=True, timeout=75)
        passed = result.returncode == expect if expect == 0 else result.returncode != 0
        records.append(dict(name=name, command=command, exit_code=result.returncode,
                            passed=passed, stdout=result.stdout, stderr=result.stderr))
        if not passed: raise AssertionError(records[-1])

    cache, r5 = fixture(root / 'fixture')
    invoke('balanced zero-feature oracle', cache, r5, root / 'oracle')
    model = json.loads((root / 'oracle/models.json').read_text())
    for value in model['models'].values():
        assert np.array_equal(value['weights'], np.zeros(len(value['weights'])))
        assert value['intercept'] == 0
    result = json.loads((root / 'oracle/results.json').read_text())
    for name in ('linear16', 'swir1'):
        assert {k: result['methods'][name][k] for k in ('tp','fp','fn','tn')} == dict(tp=256, fp=256, fn=0, tn=0)
    records.append(dict(name='independent oracle counts and parameters', passed=True))
    invoke('deterministic replay', cache, r5, root / 'replay')
    assert model['models'] == json.loads((root / 'replay/models.json').read_text())['models']
    with np.load(root / 'oracle/prediction_0000.npz') as first, np.load(root / 'replay/prediction_0000.npz') as second:
        assert all(np.array_equal(first[key], second[key]) for key in first.files)
    records.append(dict(name='identical parameters and predictions', passed=True))
    before = hashlib.sha256((root / 'oracle/results.json').read_bytes()).hexdigest()
    invoke('existing output refusal', cache, r5, root / 'oracle', expect=1)
    assert before == hashlib.sha256((root / 'oracle/results.json').read_bytes()).hexdigest()
    for fault in ('leak', 'metadata', 'nonfinite', 'target_mismatch'):
        c, r = fixture(root / fault, fault)
        invoke('reject ' + fault, c, r, root / ('rejected_' + fault), expect=1)
    c, r = fixture(root / 'without_labels', 'no_eval_targets')
    invoke('fit without evaluation targets', c, r, root / 'fit_without_labels', phase='fit')
    assert model['models'] == json.loads((root / 'fit_without_labels/models.json').read_text())['models']
    records.append(dict(name='evaluation targets cannot change fitted parameters', passed=True))
    receipt = dict(result='passed', checks=records, check_count=len(records),
                   cli_sha256=hashlib.sha256(CLI.read_bytes()).hexdigest())
    write(root / 'e2e.json', receipt)
    print(json.dumps({'result': 'passed', 'checks': len(records), 'artifact': str(root / 'e2e.json')}))


if __name__ == '__main__':
    main()
