from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'EarthRemoteSensingRapidResponse'))
from temporal_attention_training import annotation_support, dense_objective, normalize_physical_reflectance


def test_physical_reflectance_bright_values_remain_distinct():
    physical = torch.tensor([0., 0.5, 1., 1.5, 2.]).reshape(1, 1, 1, 5).expand(1, 6, 1, 5)
    valid = torch.ones(1, 1, 1, 5, dtype=torch.bool)
    result = normalize_physical_reflectance(physical, valid)
    torch.testing.assert_close(result[0, 0, 0], torch.tensor([-1., -1/3, 1/3, 1., 1.]))
    assert result[0, 0, 0, 2] < result[0, 0, 0, 3]


def test_missing_dense_truth_has_zero_pixel_gradient():
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    support = annotation_support(valid, torch.tensor([True, False]))
    plume = torch.zeros_like(logits)
    plume[1] = float('nan')
    result = dense_objective(logits, plume, support)
    result['loss'].backward()
    assert torch.count_nonzero(logits.grad[0]) == 16
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert result['supervised_scenes'].item() == 1


def test_all_missing_truth_returns_differentiable_zero():
    logits = torch.zeros(1, 1, 3, 3, requires_grad=True)
    result = dense_objective(logits, torch.full_like(logits, float('nan')),
                             torch.zeros_like(logits, dtype=torch.bool))
    result['loss'].backward()
    assert result['loss'].item() == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_equal_scene_weight_despite_different_supported_area():
    logits = torch.zeros(2, 1, 4, 4, requires_grad=True)
    support = torch.ones_like(logits, dtype=torch.bool)
    support[1, :, 1:] = False
    loss = dense_objective(logits, torch.zeros_like(logits), support)['loss']
    loss.backward()
    torch.testing.assert_close(logits.grad[0].sum(), logits.grad[1].sum())


def test_background_has_bce_but_no_positive_dice():
    logits = torch.zeros(1, 1, 4, 4)
    result = dense_objective(logits, torch.zeros_like(logits),
                             torch.ones_like(logits, dtype=torch.bool))
    assert result['bce'] > 0
    assert result['dice'] == 0


def test_supported_nan_and_broadcast_masks_are_rejected():
    logits = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError, match='binary'):
        dense_objective(logits, torch.full_like(logits, float('nan')),
                        torch.ones_like(logits, dtype=torch.bool))
    with pytest.raises(ValueError, match='shapes'):
        dense_objective(logits, torch.zeros_like(logits), torch.ones(1, dtype=torch.bool))
