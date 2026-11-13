"""Bounded CPU fitting diagnostic; no satellite files, network or holdouts.

Fits four deliberately easy fixed synthetic examples. Results establish only
trainability of the wiring, never methane accuracy or attention superiority.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'EarthRemoteSensingRapidResponse'))
from temporal_attention_candidate import matched_pair
from temporal_attention_training import dense_objective


def main() -> None:
    report_path = ROOT / 'reports/experiments/temporal_attention_synthetic_smoke.json'
    if report_path.exists():
        raise FileExistsError('Preserve the existing diagnostic; no implicit repeat')
    torch.set_num_threads(2)
    torch.manual_seed(20260905)
    torch.use_deterministic_algorithms(True)
    names = ('synthetic_swir_deficit',)
    batch, size = 4, 32
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    plume = torch.zeros(batch, 1, size, size)
    plume[0, 0] = ((xx - 12) ** 2 / 36 + (yy - 14) ** 2 / 64 < 1).float()
    plume[2, 0] = ((xx - 21) ** 2 / 25 + (yy - 19) ** 2 / 49 < 1).float()
    reference = 0.4 + 0.02 * torch.randn(batch, 6, size, size)
    target = reference.clone()
    target[:, 5:6] -= 0.15 * plume
    inputs = dict(target=target, reference=reference,
                  sensor_index=torch.tensor([0, 0, 1, 1]),
                  target_valid=torch.ones(batch, 1, size, size, dtype=torch.bool),
                  reference_valid=torch.ones(batch, 1, size, size, dtype=torch.bool),
                  native_physics=F.avg_pool2d(reference[:, 5:6] - target[:, 5:6], 2),
                  native_physics_valid=torch.ones(batch, 1, size//2, size//2, dtype=torch.bool),
                  physics_channel_names=names)
    scene_labels = torch.tensor([1., 0., 1., 0.])
    cnn, hybrid = matched_pair(names)
    results = {}
    for name, model in [('cnn', cnn), ('temporal_attention', hybrid)]:
        start = time.perf_counter()
        dense_parameters = [p for n, p in model.named_parameters()
                            if not n.startswith(('scene_encoder.', 'scene_head.'))]
        scene_parameters = [p for n, p in model.named_parameters()
                            if n.startswith(('scene_encoder.', 'scene_head.'))]
        optimizer = torch.optim.AdamW(dense_parameters, lr=1e-3, weight_decay=1e-4)
        with torch.no_grad():
            initial = float(dense_objective(model(**inputs)['mask_logits'], plume,
                                            inputs['target_valid'])['loss'])
        for _ in range(24):
            model.zero_grad(set_to_none=True)
            loss = dense_objective(model(**inputs)['mask_logits'], plume,
                                   inputs['target_valid'])['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        for p in dense_parameters:
            p.requires_grad_(False)
        with torch.no_grad():
            output = model(**inputs)
            dense_final = float(dense_objective(output['mask_logits'], plume,
                                               inputs['target_valid'])['loss'])
            scene_initial = float(F.binary_cross_entropy_with_logits(output['scene_logits'], scene_labels))
        optimizer = torch.optim.AdamW(scene_parameters, lr=1e-3, weight_decay=1e-4)
        for _ in range(24):
            model.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(**inputs)['scene_logits'], scene_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scene_parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            output = model(**inputs)
            scene_final = float(F.binary_cross_entropy_with_logits(output['scene_logits'], scene_labels))
            predicted = output['mask_logits'] >= 0
            truth = plume.bool()
            iou = float((predicted & truth).sum() / (predicted | truth).sum().clamp_min(1))
        results[name] = {'parameters': sum(p.numel() for p in model.parameters()),
                         'dense_initial_loss': initial, 'dense_final_loss': dense_final,
                         'scene_initial_loss': scene_initial, 'scene_final_loss': scene_final,
                         'training_only_iou': iou,
                         'dense_learns': dense_final < initial * 0.8,
                         'scene_learns': scene_final < scene_initial * 0.8,
                         'elapsed_seconds': round(time.perf_counter() - start, 3)}
    files = [Path(__file__), ROOT / 'EarthRemoteSensingRapidResponse/temporal_attention_candidate.py',
             ROOT / 'EarthRemoteSensingRapidResponse/temporal_attention_training.py']
    report = {'scope': 'synthetic fitting-only wiring diagnostic; no real accuracy claim',
              'seed': 20260905, 'device': 'cpu', 'threads': 2,
              'torch_version': torch.__version__, 'examples': 4, 'image_shape': [32, 32],
              'dense_steps': 24, 'scene_steps': 24, 'learning_rate': 0.001,
              'shared_initial_weights_equal': True, 'real_data_opened': False,
              'holdouts_opened': False, 'models': results,
              'passed': all(r['dense_learns'] and r['scene_learns'] for r in results.values()),
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
