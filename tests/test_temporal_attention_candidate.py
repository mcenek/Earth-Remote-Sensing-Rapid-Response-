"""Synthetic contracts only; no dataset or performance evaluation."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EarthRemoteSensingRapidResponse"))
from temporal_attention_candidate import ChannelNorm, TemporalAttentionCandidate, matched_pair


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(h=17, w=19, batch=2):
    return dict(target=torch.randn(batch, 6, h, w),
                reference=torch.randn(batch, 6, h, w),
                sensor_index=torch.arange(batch) % 2,
                target_valid=torch.ones(batch, 1, h, w, dtype=torch.bool),
                reference_valid=torch.ones(batch, 1, h, w, dtype=torch.bool),
                native_physics=torch.randn(batch, 2, max(1, h // 2), max(1, w // 2)),
                native_physics_valid=torch.ones(batch, 1, max(1, h // 2), max(1, w // 2), dtype=torch.bool),
                physics_channel_names=("mbmp", "signed_swir_change"))


def model(attention=False):
    return TemporalAttentionCandidate(("mbmp", "signed_swir_change"), attention=attention)


@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (17, 19), (32, 24)])
def test_shapes_finite_and_explicit_support(attention, shape):
    data = inputs(*shape)
    data["target_valid"][1] = False
    data["reference_valid"][1] = False
    with torch.no_grad():
        result = model(attention)(**data)
    assert result["mask_logits"].shape == (2, 1, *shape)
    assert result["scene_logits"].shape == (2,)
    assert torch.equal(result["support"], data["target_valid"])
    assert all(torch.isfinite(x).all() for x in result.values())


@pytest.mark.parametrize("attention", [False, True])
def test_invalid_reference_and_physics_values_cannot_leak(attention):
    data = inputs()
    data["reference_valid"][..., :10, :] = False
    data["native_physics_valid"][..., :3, :] = False
    net = model(attention).eval()
    with torch.no_grad():
        before = net(**data)
        data["reference"] = data["reference"].masked_fill(~data["reference_valid"], float("nan"))
        data["native_physics"] = data["native_physics"].masked_fill(~data["native_physics_valid"], float("inf"))
        after = net(**data)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_matched_weights_and_no_reference_exact_identity():
    cnn, hybrid = matched_pair(("mbmp", "signed_swir_change"))
    for name, value in cnn.state_dict().items():
        torch.testing.assert_close(value, hybrid.state_dict()[name], rtol=0, atol=0)
    extras = set(hybrid.state_dict()) - set(cnn.state_dict())
    assert extras and all(name.startswith("attention.") for name in extras)
    data = inputs()
    data["reference_valid"].zero_()
    data["reference"].fill_(float("nan"))
    with torch.no_grad():
        a, b = cnn(**data), hybrid(**data)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_scene_gradients_isolated_and_attention_can_learn():
    net = model(True)
    out = net(**inputs())
    out["scene_logits"].sum().backward()
    for name, parameter in net.named_parameters():
        if name.startswith(("scene_encoder.", "scene_head.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None, name
    net.zero_grad(set_to_none=True)
    net(**inputs())["mask_logits"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.attention.parameters())
    assert all(p.grad is None for p in net.scene_encoder.parameters())


@pytest.mark.parametrize("ids", [torch.tensor([-1, 0]), torch.tensor([0, 2]), torch.tensor([0., 1.])])
def test_sensor_bounds_and_type(ids):
    data = inputs()
    data["sensor_index"] = ids
    with pytest.raises(ValueError, match="sensor_index"):
        model()(**data)


@pytest.mark.parametrize("fault", ["names", "channels", "mask", "nonfinite"])
def test_native_physics_contract(fault):
    data = inputs()
    if fault == "names":
        data["physics_channel_names"] = tuple(reversed(data["physics_channel_names"]))
    elif fault == "channels":
        data["native_physics"] = data["native_physics"][:, :1]
    elif fault == "mask":
        data["native_physics_valid"] = data["native_physics_valid"].float()
    else:
        data["native_physics"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model()(**data)


def test_channel_normalization_does_not_use_canvas_statistics():
    norm = ChannelNorm(32)
    x = torch.randn(2, 32, 3, 5)
    padded = torch.nn.functional.pad(x, (0, 7, 0, 5))
    torch.testing.assert_close(norm(x), norm(padded)[..., :3, :5], rtol=0, atol=0)
