"""Exercise the staged experiment on generated tensors; never reads imagery.

Usage: python tools/rehearse_temporal_experiment.py --output-dir <new-directory>
The exclusive output directory preserves configuration, source hashes, status,
and both histories. This is a workflow diagnostic, not a methane benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'EarthRemoteSensingRapidResponse'))
from temporal_attention_candidate import matched_pair
from temporal_attention_experiment import fit_two_stage
from temporal_attention_training import normalize_physical_reflectance


#### SYNTHETIC INPUTS WITH MATERIAL MISSING-DATA CASES ####

def make_batch(split: str, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    size, count = 32, 4
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    plume = torch.zeros(count, 1, size, size)
    plume[0, 0] = ((x - 12) ** 2 + (y - 16) ** 2 < 36).float()
    plume[2, 0] = ((x - 20) ** 2 + (y - 12) ** 2 < 25).float()
    reference = .4 + .02 * torch.randn(count, 6, size, size, generator=generator)
    target = reference.clone()
    target[:, 5:6] -= .15 * plume
    target_valid = torch.ones(count, 1, size, size, dtype=torch.bool)
    reference_valid = target_valid.clone()
    reference_valid[1] = False
    target_valid[3] = False  # exclude this scene from training/evaluation
    physics = F.avg_pool2d(reference[:, 5:6] - target[:, 5:6], 2)
    physics_valid = F.avg_pool2d((target_valid & reference_valid).float(), 2) == 1
    plume[2] = float('nan')  # positive with scene label but no dense annotation
    return {
        'split': split,
        'sample_ids': [f'synthetic-{split}-{i}' for i in range(count)],
        'inputs': {
            'target': normalize_physical_reflectance(target, target_valid),
            'reference': normalize_physical_reflectance(reference, reference_valid),
            'sensor_index': torch.tensor([0, 0, 1, 1]),
            'target_valid': target_valid, 'reference_valid': reference_valid,
            'native_physics': physics, 'native_physics_valid': physics_valid,
            'physics_channel_names': ('synthetic_swir_deficit',),
        },
        'plume': plume,
        'pixel_truth_available': torch.tensor([True, True, False, True]),
        'scene_labels': torch.tensor([1., 0., 1., 0.]),
    }


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


#### EXCLUSIVE, BOUNDED RUN RECORD ####

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        'scope': 'synthetic workflow diagnostic only',
        'seed': 20260907, 'device': 'cpu', 'threads': 2,
        'dense_epochs': 3, 'scene_epochs': 3, 'lr': .001,
        'real_data_opened': False, 'test_data_opened': False,
        'torch_version': torch.__version__,
    }
    files = [Path(__file__)] + [ROOT / 'EarthRemoteSensingRapidResponse' / name for name in
             ('temporal_attention_candidate.py', 'temporal_attention_training.py',
              'temporal_attention_experiment.py')]
    configuration['source_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    write_json(args.output_dir / 'configuration.json', configuration)
    try:
        torch.set_num_threads(configuration['threads'])
        torch.manual_seed(configuration['seed'])
        torch.use_deterministic_algorithms(True)
        train = [make_batch('train', 1)]
        validation = [make_batch('validation', 2)]
        cnn, attention = matched_pair(('synthetic_swir_deficit',))
        results = {}
        for name, model in [('cnn', cnn), ('temporal_attention', attention)]:
            results[name] = fit_two_stage(
                model, train, validation, dense_epochs=configuration['dense_epochs'],
                scene_epochs=configuration['scene_epochs'], lr=configuration['lr'])
            write_json(args.output_dir / f'{name}.json', results[name])
        status = {'status': 'completed', 'scope': configuration['scope'],
                  'models': list(results), 'accuracy_claim': False}
    except Exception as error:
        write_json(args.output_dir / 'status.json',
                   {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
        raise
    write_json(args.output_dir / 'status.json', status)
    print(json.dumps(status))


if __name__ == '__main__':
    main()
