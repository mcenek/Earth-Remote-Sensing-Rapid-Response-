from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "GTM_Model6"))
from GTM_Model6_mapper import (  # noqa: E402
    CONTEXT_WINDOWS,
    MultiScaleMapper,
    masked_mapper_loss,
    multiscale_mapper_loss,
    signed_log1p,
)


def _contexts(batch: int = 2, channels: int = 16):
    return {
        f"S{window}": torch.zeros(batch, channels, window, window)
        for window in CONTEXT_WINDOWS
    }


def _targets(size: int = 16):
    plume = torch.zeros(2, 1, size, size)
    plume[:, :, size // 4 : size // 2, size // 4 : size // 2] = 1
    enhancement = plume * 2
    support = torch.ones_like(plume, dtype=torch.bool)
    return plume, support, enhancement


def test_signed_log1p_preserves_sign():
    values = torch.tensor([-3.0, 0.0, 3.0])
    transformed = signed_log1p(values)
    assert transformed[0] < 0 < transformed[2]
    assert transformed[1] == 0


def test_branches_receive_distinct_exact_contexts_and_fuse_logits():
    model = MultiScaleMapper(base_channels=8, prediction_window=16)
    seen: dict[str, tuple[int, int]] = {}
    hooks = []
    for name, branch in model.branches.items():
        hooks.append(branch.register_forward_pre_hook(lambda module, args, key=name: seen.__setitem__(key, tuple(args[0].shape[-2:]))))
    output = model(_contexts())
    for hook in hooks:
        hook.remove()
    assert list(model.branches) == ["S16", "S32", "S64", "S128"]
    assert seen == {f"S{window}": (window, window) for window in CONTEXT_WINDOWS}
    assert output["mask_logits"].shape == (2, 1, 16, 16)
    assert output["enhancement_log1p"].shape == (2, 1, 16, 16)
    assert torch.allclose(
        output["mask_logits"],
        torch.stack([output["branches"][name]["mask_logits"] for name in model.branches]).mean(0),
    )
    assert output["fusion_mode"] == "logit_mean"


def test_context_contract_rejects_insufficient_or_missing_context():
    model = MultiScaleMapper(base_channels=8)
    contexts = _contexts()
    contexts["S128"] = torch.zeros(2, 16, 32, 32)
    with pytest.raises(ValueError, match="S128 requires an exact"):
        model(contexts)
    contexts = _contexts()
    del contexts["S64"]
    with pytest.raises(ValueError, match="missing context S64"):
        model(contexts)
    with pytest.raises(TypeError, match="mapping of exact"):
        model(torch.zeros(2, 16, 32, 32))


def test_unknown_support_does_not_change_loss_and_empty_counts_are_real_zero():
    model = MultiScaleMapper(base_channels=8)
    output = model(_contexts())
    plume, support, enhancement = _targets()
    unknown = support.clone()
    unknown[:, :, :4, :4] = False
    changed_plume = plume.clone()
    changed_enhancement = enhancement.clone()
    changed_plume[:, :, :4, :4] = 1 - changed_plume[:, :, :4, :4]
    changed_enhancement[:, :, :4, :4] = 999
    first = masked_mapper_loss(output, plume, unknown, enhancement, unknown)["loss"]
    second = masked_mapper_loss(output, changed_plume, unknown, changed_enhancement, unknown)["loss"]
    assert torch.allclose(first, second)
    empty = torch.zeros_like(support)
    empty_result = masked_mapper_loss(output, plume, empty, enhancement, empty)
    assert empty_result["plume_supported_pixels"] == 0
    assert empty_result["enhancement_supported_pixels"] == 0
    assert empty_result["loss"] == 0


def test_nonfinite_supported_target_is_rejected_but_outside_support_is_ignored():
    model = MultiScaleMapper(base_channels=8)
    output = model(_contexts())
    plume, support, enhancement = _targets()
    enhancement[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite enhancement"):
        masked_mapper_loss(output, plume, support, enhancement, support)
    partial = support.clone()
    partial[:, :, :2, :2] = False
    changed = enhancement.clone()
    changed[:, :, :2, :2] = float("nan")
    result = masked_mapper_loss(output, plume, partial, changed, partial)
    assert torch.isfinite(result["loss"])


def test_branchwise_loss_declares_independent_supervision():
    model = MultiScaleMapper(base_channels=8)
    output = model(_contexts())
    plume, support, enhancement = _targets()
    result = multiscale_mapper_loss(output, plume, support, enhancement, support)
    assert "branch_loss" in result and "fusion_loss" in result
    assert torch.isfinite(result["loss"])
