"""Context-aligned multiscale Sentinel-to-EMIT mapper contract.

Each branch receives an actual native context window: S16, S32, S64 or S128
pixels. A branch returns a shared 16x16 prediction tile from the centre of
that context, so a caller can slide the four context windows over a scene and
stitch the tiles without fabricating missing context. The fixed merger is an
arithmetic mean of branch logits; probabilities are obtained only after that
mean. Branch parameters are independent, and ``multiscale_mapper_loss``
supervises each branch as well as the fused output.

The second head predicts signed-log quantitative EMIT enhancement, but it must
remain disabled until the data manifest has finite, time-aligned quantitative
targets and valid support. This module contains no real-data result claim.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_ID = "GTM_Model6_multiscale_sentinel_emit_mapper"
SCHEMA_VERSION = 2
CONTEXT_WINDOWS = (16, 32, 64, 128)
PREDICTION_WINDOW = 16


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def signed_log1p(values: torch.Tensor) -> torch.Tensor:
    """Compress signed ppm*m targets without turning negative retrievals into zero."""

    return torch.sign(values) * torch.log1p(torch.abs(values))


class _Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.body(values) + self.skip(values))


class _Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1)
        self.block = _Block(output_channels, output_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(values))


class _Up(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, skip_channels, 2, stride=2)
        self.block = _Block(skip_channels * 2, skip_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[-2:] != skip.shape[-2:]:
            values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((values, skip), dim=1))


class ScaleBranch(nn.Module):
    """One independently parameterized context-specific dense branch."""

    def __init__(
        self,
        input_channels: int = 16,
        context_window: int = 32,
        base_channels: int = 16,
        prediction_window: int = PREDICTION_WINDOW,
    ) -> None:
        super().__init__()
        if context_window not in CONTEXT_WINDOWS:
            raise ValueError(f"context_window must be one of {CONTEXT_WINDOWS}")
        if prediction_window < 1 or prediction_window > context_window:
            raise ValueError("prediction_window must be positive and fit inside context_window")
        if input_channels < 1 or base_channels < 4:
            raise ValueError("input_channels and base_channels must be positive")
        self.context_window = int(context_window)
        self.prediction_window = int(prediction_window)
        self.depth = int(math.log2(context_window // 16)) + 1
        widths = [min(base_channels * (2**level), 128) for level in range(self.depth + 1)]
        self.stem = _Block(input_channels, widths[0])
        self.downs = nn.ModuleList(
            [_Down(widths[level], widths[level + 1]) for level in range(self.depth)]
        )
        self.bottleneck = _Block(widths[-1], widths[-1])
        self.ups = nn.ModuleList(
            [_Up(widths[level + 1], widths[level]) for level in reversed(range(self.depth))]
        )
        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        self.enhancement_head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        if values.ndim != 4 or values.shape[1] < 1:
            raise ValueError("values must be BxCxHxW")
        if values.shape[-2:] != (self.context_window, self.context_window):
            raise ValueError(
                f"S{self.context_window} requires an exact {self.context_window}x{self.context_window} context; "
                f"got {tuple(values.shape[-2:])}"
            )
        skips = [self.stem(values)]
        dense = skips[0]
        for layer in self.downs:
            dense = layer(dense)
            skips.append(dense)
        dense = self.bottleneck(dense)
        for layer, skip in zip(self.ups, reversed(skips[:-1])):
            dense = layer(dense, skip)
        return {
            "mask_logits": self.mask_head(dense),
            "enhancement_log1p": self.enhancement_head(dense),
        }


def _centre_crop(values: torch.Tensor, size: int) -> torch.Tensor:
    height, width = values.shape[-2:]
    if height < size or width < size:
        raise ValueError(f"prediction window {size} exceeds branch output {(height, width)}")
    top = (height - size) // 2
    left = (width - size) // 2
    return values[..., top : top + size, left : left + size]


class MultiScaleMapper(nn.Module):
    """Independent context branches with a fixed mean of branch logits.

    ``values`` must be a mapping from S16/S32/S64/S128 to tensors with the
    exact corresponding spatial shape. The output is a common 16x16 centre
    tile. Keeping context preparation outside the model makes edge support,
    tiling stride and georeferencing explicit to the data pipeline.
    """

    def __init__(
        self,
        input_channels: int = 16,
        base_channels: int = 16,
        prediction_window: int = PREDICTION_WINDOW,
    ) -> None:
        super().__init__()
        if prediction_window < 1 or prediction_window > min(CONTEXT_WINDOWS):
            raise ValueError("prediction_window must fit inside the smallest context window")
        self.input_channels = int(input_channels)
        self.prediction_window = int(prediction_window)
        self.fusion_mode = "logit_mean"
        self.branches = nn.ModuleDict(
            {
                f"S{window}": ScaleBranch(
                    input_channels=input_channels,
                    context_window=window,
                    base_channels=base_channels,
                    prediction_window=prediction_window,
                )
                for window in CONTEXT_WINDOWS
            }
        )

    def forward(
        self,
        values: Mapping[str, torch.Tensor] | torch.Tensor,
        branch: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(values, Mapping):
            raise TypeError(
                "MultiScaleMapper requires a mapping of exact S16/S32/S64/S128 contexts; "
                "a single tensor cannot prove distinct native context"
            )
        names = [branch] if branch is not None else list(self.branches)
        if branch is not None and branch not in self.branches:
            raise ValueError(f"unknown branch {branch!r}")
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name in names:
            if name not in values:
                raise ValueError(f"missing context {name}")
            context = values[name]
            if context.ndim != 4 or context.shape[1] != self.input_channels:
                raise ValueError(f"{name} must be Bx{self.input_channels}xHxW")
            full = self.branches[name](context)
            outputs[name] = {
                "mask_logits": _centre_crop(full["mask_logits"], self.prediction_window),
                "enhancement_log1p": _centre_crop(full["enhancement_log1p"], self.prediction_window),
            }
        if branch is not None:
            return outputs[branch]
        return {
            "mask_logits": torch.stack([item["mask_logits"] for item in outputs.values()]).mean(0),
            "enhancement_log1p": torch.stack([item["enhancement_log1p"] for item in outputs.values()]).mean(0),
            "branches": outputs,
            "fusion_mode": self.fusion_mode,
            "prediction_window": self.prediction_window,
            "context_windows": CONTEXT_WINDOWS,
        }


def _check_support(
    values: torch.Tensor,
    reference: torch.Tensor,
    support: torch.Tensor,
    name: str,
) -> None:
    if values.shape != reference.shape or values.ndim != 4 or values.shape[1] != 1:
        raise ValueError(f"{name} and target must both be Bx1xHxW")
    if support.shape != values.shape or support.dtype is not torch.bool:
        raise ValueError(f"{name}_support must be boolean and match the target")
    if bool((~torch.isfinite(reference) & support).any()):
        raise ValueError(f"finite {name} targets are required where support is true")


def _supported_mean(values: torch.Tensor, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = support.sum()
    if int(count) == 0:
        return values.sum() * 0.0, count.to(dtype=values.dtype)
    return values.masked_select(support).mean(), count.to(dtype=values.dtype)


def masked_mapper_loss(
    outputs: Mapping[str, torch.Tensor],
    plume_target: torch.Tensor,
    plume_support: torch.Tensor,
    enhancement_target: torch.Tensor,
    enhancement_support: torch.Tensor,
    *,
    mask_weight: float = 1.0,
    enhancement_weight: float = 1.0,
    positive_weight: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Compute support-aware fused losses while ignoring unknown pixels."""

    mask_logits = outputs["mask_logits"]
    enhancement = outputs["enhancement_log1p"]
    _check_support(mask_logits, plume_target, plume_support, "plume")
    _check_support(enhancement, enhancement_target, enhancement_support, "enhancement")
    if bool(((plume_target < 0) | (plume_target > 1))[plume_support].any()):
        raise ValueError("plume_target must be in [0, 1] on supported pixels")
    safe_plume = torch.where(plume_support, plume_target, torch.zeros_like(plume_target))
    pixel_bce = F.binary_cross_entropy_with_logits(
        mask_logits,
        safe_plume,
        reduction="none",
        pos_weight=mask_logits.new_tensor(float(positive_weight)),
    )
    bce, plume_count = _supported_mean(pixel_bce, plume_support)
    probabilities = mask_logits.sigmoid()
    intersection = (probabilities * safe_plume * plume_support).sum()
    prediction_mass = (probabilities * plume_support).sum()
    target_mass = (safe_plume * plume_support).sum()
    if int(plume_count) == 0:
        dice = mask_logits.sum() * 0.0
    else:
        dice = 1.0 - (2.0 * intersection + 1.0) / (prediction_mass + target_mass + 1.0)
    safe_enhancement = signed_log1p(
        torch.where(enhancement_support, enhancement_target, torch.zeros_like(enhancement_target))
    )
    regression_values = F.smooth_l1_loss(enhancement, safe_enhancement, reduction="none")
    regression, enhancement_count = _supported_mean(regression_values, enhancement_support)
    total = float(mask_weight) * (bce + dice) + float(enhancement_weight) * regression
    return {
        "loss": total,
        "mask_bce": bce,
        "mask_dice": dice,
        "enhancement_smooth_l1": regression,
        "plume_supported_pixels": plume_count,
        "enhancement_supported_pixels": enhancement_count,
    }


def multiscale_mapper_loss(
    outputs: Mapping[str, Any],
    plume_target: torch.Tensor,
    plume_support: torch.Tensor,
    enhancement_target: torch.Tensor,
    enhancement_support: torch.Tensor,
    *,
    branch_weight: float = 1.0,
    fusion_weight: float = 1.0,
    enhancement_weight: float = 1.0,
    positive_weight: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Supervise each independent branch and the declared logit-mean fusion."""

    fused = masked_mapper_loss(
        outputs,
        plume_target,
        plume_support,
        enhancement_target,
        enhancement_support,
        enhancement_weight=enhancement_weight,
        positive_weight=positive_weight,
    )
    branches = outputs.get("branches")
    if not isinstance(branches, Mapping) or not branches:
        return fused
    branch_losses = [
        masked_mapper_loss(
            item,
            plume_target,
            plume_support,
            enhancement_target,
            enhancement_support,
            enhancement_weight=enhancement_weight,
            positive_weight=positive_weight,
        )
        for item in branches.values()
    ]
    branch_loss = torch.stack([item["loss"] for item in branch_losses]).mean()
    total = float(fusion_weight) * fused["loss"] + float(branch_weight) * branch_loss
    result = dict(fused)
    result["loss"] = total
    result["branch_loss"] = branch_loss
    result["fusion_loss"] = fused["loss"]
    return result
