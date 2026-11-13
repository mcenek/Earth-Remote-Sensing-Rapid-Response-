"""Rehearse calibration/evaluation with fabricated scores, never imagery.

python tools/rehearse_temporal_evaluation.py --output-dir <new-directory>
The scores are hand-built fixtures, not outputs of a fitted methane model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'EarthRemoteSensingRapidResponse'))
from temporal_attention_evaluation import calibrate, evaluate_pair, dense_metrics


def fixture(split: str) -> dict:
    rows = []
    for group in range(4):
        for label in (0, 1):
            score = .2 + .5*label + .01*group
            rows.append({'sample_id': f'SYNTHETIC-{split}-{group}-{label}',
                         'group_id': f'SYNTHETIC-{split}-site-{group}',
                         'label': label, 'valid': True,
                         'cnn': score, 'temporal_attention': score})
    rows.append({'sample_id': f'SYNTHETIC-{split}-invalid',
                 'group_id': f'SYNTHETIC-{split}-invalid-site',
                 'label': 1, 'valid': False, 'cnn': None, 'temporal_attention': None})
    return {'split': split, 'samples': rows}


def write_json(path: Path, data: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'EarthRemoteSensingRapidResponse/temporal_attention_evaluation.py']
    write_json(args.output_dir / 'configuration.json', {
        'scope': 'fabricated score fixtures only; no accuracy claim',
        'real_data_opened': False, 'real_test_outcomes_opened': False,
        'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'bootstrap_replicates': 200, 'bootstrap_seed': 20260907,
    })
    try:
        validation, evaluation = fixture('validation'), fixture('test')
        calibration = calibrate(validation, max_fpr=0)
        report = evaluate_pair(evaluation, calibration, replicates=200)
        intervals = report['paired_delta_bootstrap']['intervals']
        assert intervals and all(v['lower'] == v['upper'] == 0 for v in intervals.values())
        assert report['abstained_scenes'] == 1
        dense = dense_metrics(np.array([[.8, .6], [.1, np.nan]]),
                              np.array([[1., 0.], [0., np.nan]]),
                              np.array([[True, True], [True, False]]), annotation_available=True)
        assert dense['tp'] == 1 and dense['fp'] == 1 and dense['iou'] == .5
        write_json(args.output_dir / 'calibration.json', calibration)
        write_json(args.output_dir / 'evaluation.json', report)
        write_json(args.output_dir / 'dense.json', dense)
    except Exception as error:
        write_json(args.output_dir / 'status.json', {'status': 'failed', 'error': str(error)})
        raise
    write_json(args.output_dir / 'status.json', {'status': 'completed', 'accuracy_claim': False})
    print('Synthetic calibration, paired evaluation and dense metrics completed; no real accuracy claim')


if __name__ == '__main__':
    main()
