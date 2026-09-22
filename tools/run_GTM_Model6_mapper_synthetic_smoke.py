"""Synthetic-only fitting smoke for the context-aligned Model6 mapper."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "GTM_Model6"))
from GTM_Model6_mapper import (  # noqa: E402
    CONTEXT_WINDOWS,
    MultiScaleMapper,
    multiscale_mapper_loss,
)


def _centre_crop(values: torch.Tensor, size: int) -> torch.Tensor:
    top = (values.shape[-2] - size) // 2
    left = (values.shape[-1] - size) // 2
    return values[..., top : top + size, left : left + size]


def main() -> None:
    output = ROOT / "reports" / "experiments" / "GTM_Model6_mapper_synthetic_smoke_2026-09-22_v3.json"
    if output.exists():
        raise FileExistsError(f"preserving existing diagnostic: {output}")
    torch.set_num_threads(2)
    torch.manual_seed(20260922)
    torch.use_deterministic_algorithms(True)
    batch, full_size, target_size, channels = 2, 128, 16, 16
    yy, xx = torch.meshgrid(torch.arange(full_size), torch.arange(full_size), indexing="ij")
    plume = torch.zeros(batch, 1, full_size, full_size)
    plume[0, 0] = (((xx - 57) ** 2) / 220 + ((yy - 65) ** 2) / 480 < 1).float()
    plume[1, 0] = (((xx - 71) ** 2) / 270 + ((yy - 58) ** 2) / 190 < 1).float()
    enhancement = plume * (2.0 + 0.25 * torch.sin(xx.float() / 4.0))
    full_inputs = torch.randn(batch, channels, full_size, full_size) * 0.05
    full_inputs[:, 0:1] += enhancement
    contexts = {f"S{window}": _centre_crop(full_inputs, window) for window in CONTEXT_WINDOWS}
    plume_target = _centre_crop(plume, target_size)
    enhancement_target = _centre_crop(enhancement, target_size)
    plume_support = torch.ones_like(plume_target, dtype=torch.bool)
    enhancement_support = plume_support.clone()
    model = MultiScaleMapper(input_channels=channels, base_channels=8, prediction_window=target_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    with torch.no_grad():
        initial_output = model(contexts)
        initial = multiscale_mapper_loss(
            initial_output,
            plume_target,
            plume_support,
            enhancement_target,
            enhancement_support,
        )
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = multiscale_mapper_loss(
            model(contexts),
            plume_target,
            plume_support,
            enhancement_target,
            enhancement_support,
        )["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        final_output = model(contexts)
        final = multiscale_mapper_loss(
            final_output,
            plume_target,
            plume_support,
            enhancement_target,
            enhancement_support,
        )
        prediction = final_output["mask_logits"] >= 0
        truth = plume_target.bool()
        intersection = (prediction & truth).sum()
        union = (prediction | truth).sum().clamp_min(1)
        iou = float(intersection / union)
        enhancement_mae = float(
            (
                final_output["enhancement_log1p"]
                - torch.sign(enhancement_target) * torch.log1p(torch.abs(enhancement_target))
            )
            .abs()
            .mean()
        )
    source_files = [Path(__file__), ROOT / "research" / "GTM_Model6" / "GTM_Model6_mapper.py"]
    report = {
        "scope": "synthetic fitting-only context-contract diagnostic; no real accuracy claim",
        "contract_version": 2,
        "seed": 20260922,
        "device": "cpu",
        "threads": 2,
        "examples": batch,
        "context_shapes": {name: [batch, channels, window, window] for name, window in zip((f"S{w}" for w in CONTEXT_WINDOWS), CONTEXT_WINDOWS)},
        "prediction_shape": [batch, 1, target_size, target_size],
        "input_channels": channels,
        "steps": 24,
        "real_data_opened": False,
        "holdouts_opened": False,
        "branches": [f"S{window}" for window in CONTEXT_WINDOWS],
        "fusion_mode": "logit_mean",
        "branch_supervision": "mean branch losses plus fused loss",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": float(initial["loss"]),
        "final_loss": float(final["loss"]),
        "final_training_only_iou": iou,
        "final_training_only_enhancement_log1p_mae": enhancement_mae,
        "plume_supported_pixels": int(final["plume_supported_pixels"]),
        "enhancement_supported_pixels": int(final["enhancement_supported_pixels"]),
        "learns": float(final["loss"]) < float(initial["loss"]) * 0.8,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    }
    report["passed"] = bool(report["learns"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
