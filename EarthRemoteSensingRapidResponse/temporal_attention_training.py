"""Dense supervision contracts for the prospective temporal model comparison.

Label support is separate from inference QA. A positive scene without a dense
annotation is eligible for scene training, never for all-background pixel loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def normalize_physical_reflectance(
    reflectance: torch.Tensor, sensor_valid: torch.Tensor
) -> torch.Tensor:
    """Normalize already product-correct reflectance, never loader-scaled DN.

    The caller must convert its source product before this function. Keeping
    physical values until this point avoids the legacy DN/5000 preclip at 2.
    Invalid pixels map to zero and remain explicitly invalid in model QA.
    """
    if reflectance.ndim != 4 or reflectance.shape[1] != 6:
        raise ValueError('Physical reflectance must be Bx6xHxW')
    if sensor_valid.shape != reflectance[:, :1].shape or sensor_valid.dtype != torch.bool:
        raise ValueError('Sensor validity must be boolean Bx1xHxW')
    if not reflectance.is_floating_point():
        raise ValueError('Physical reflectance must be floating point')
    if not torch.isfinite(reflectance[sensor_valid.expand_as(reflectance)]).all():
        raise ValueError('Physical reflectance must be finite where valid')
    safe = torch.where(sensor_valid, reflectance, torch.zeros_like(reflectance))
    normalized = safe.clamp(0, 1.5) * (2 / 1.5) - 1
    return torch.where(sensor_valid, normalized, torch.zeros_like(normalized))


def dense_objective(
    logits: torch.Tensor,
    plume: torch.Tensor,
    label_support: torch.Tensor,
    *,
    bce_weight: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Average BCE per supervised scene; Dice only on annotated positive scenes.

    All tensors have shape Bx1xHxW. Support must combine annotation availability
    with valid evaluation pixels outside the model. Unsupported targets may be
    NaN. Empty support yields differentiable zero, not fabricated supervision.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Dense logits must be Bx1xHxW')
    if plume.shape != logits.shape or label_support.shape != logits.shape:
        raise ValueError('Dense target/support shapes must exactly match logits')
    if label_support.dtype != torch.bool:
        raise ValueError('Label support must be boolean')
    if not 0 <= bce_weight <= 1:
        raise ValueError('BCE weight must be in [0, 1]')
    if not torch.isfinite(logits).all():
        raise ValueError('Dense logits must be finite')
    active = plume[label_support]
    if not torch.isfinite(active).all() or not ((active == 0) | (active == 1)).all():
        raise ValueError('Supported plume targets must be finite binary labels')
    safe = torch.where(label_support, plume, torch.zeros_like(plume)).to(logits.dtype)
    weight = label_support.to(logits.dtype)
    counts = weight.flatten(1).sum(1)
    supervised = counts > 0
    zero = logits.sum() * 0
    per_scene_bce = (
        F.binary_cross_entropy_with_logits(logits, safe, reduction='none') * weight
    ).flatten(1).sum(1) / counts.clamp_min(1)
    bce = per_scene_bce[supervised].mean() if supervised.any() else zero
    probability = logits.sigmoid() * weight
    target = safe * weight
    positive = target.flatten(1).sum(1) > 0
    intersection = (probability * target).flatten(1).sum(1)
    denominator = (probability + target).flatten(1).sum(1)
    dice_per_scene = 1 - (2 * intersection + 1) / (denominator + 1)
    dice = dice_per_scene[positive].mean() if positive.any() else zero
    return {
        'loss': bce_weight * bce + (1 - bce_weight) * dice,
        'bce': bce,
        'dice': dice,
        'supervised_scenes': supervised.sum(),
    }


def annotation_support(
    sensor_valid: torch.Tensor, pixel_truth_available: torch.Tensor
) -> torch.Tensor:
    """Build loss-only support; never pass this result as a model QA feature."""
    if sensor_valid.ndim != 4 or sensor_valid.shape[1] != 1:
        raise ValueError('Sensor validity must be Bx1xHxW')
    if pixel_truth_available.shape != (sensor_valid.shape[0],):
        raise ValueError('Pixel truth availability must have one value per scene')
    if sensor_valid.dtype != torch.bool or pixel_truth_available.dtype != torch.bool:
        raise ValueError('Validity and annotation availability must be boolean')
    return sensor_valid & pixel_truth_available[:, None, None, None]
