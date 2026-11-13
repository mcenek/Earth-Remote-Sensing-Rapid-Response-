import json
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EarthRemoteSensingRapidResponse"))
from temporal_attention_experiment import _metric, fit_two_stage


class TinyTemporal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_weight = torch.nn.Parameter(torch.tensor(0.0))
        self.scene_encoder = torch.nn.Linear(1, 1)
        self.scene_head = torch.nn.Linear(1, 1)

    def forward(self, x):
        mask = x * self.dense_weight
        scene = self.scene_head(self.scene_encoder(x.mean((2, 3)))).squeeze(1)
        return {"mask_logits": mask, "scene_logits": scene, "support": x.bool()}


def batch(split, ids, *, annotated=True, label=1.0, plume_value=1.0, x=None):
    x = torch.ones(len(ids), 1, 2, 2) if x is None else x
    return {"split": split, "sample_ids": ids, "inputs": {"x": x},
            "plume": torch.full_like(x, plume_value),
            "pixel_truth_available": torch.full((len(ids),), annotated),
            "scene_labels": torch.full((len(ids),), label)}


def test_two_stage_returns_json_history_and_restores_eval_state():
    model = TinyTemporal()
    model.dense_weight.requires_grad = False
    result = fit_two_stage(model, [batch("train", ["a"])], [batch("validation", ["b"])], dense_epochs=2, scene_epochs=2)
    json.dumps(result)
    assert result["best_dense_epoch"] >= 1
    assert result["best_scene_epoch"] >= 1
    assert model.training is False
    assert model.dense_weight.requires_grad is False


@pytest.mark.parametrize("kwargs", [
    {"train_ids": ["same"], "validation_ids": ["same"]},
    {"train_ids": ["a", "a"], "validation_ids": ["b"]},
])
def test_partition_ids_are_audited(kwargs):
    with pytest.raises(ValueError, match="sample_ids"):
        fit_two_stage(model=TinyTemporal(), train_batches=[batch("train", kwargs["train_ids"])],
                      validation_batches=[batch("validation", kwargs["validation_ids"])])


def test_no_scene_support_is_rejected():
    class Invalid(TinyTemporal):
        def forward(self, x):
            out = super().forward(x)
            out["support"] = torch.zeros_like(x, dtype=torch.bool)
            return out

    with pytest.raises(ValueError, match="eligible train or validation"):
        fit_two_stage(Invalid(), [batch("train", ["a"])], [batch("validation", ["b"])])


def test_test_and_swapped_split_labels_are_rejected():
    with pytest.raises(ValueError, match="test data"):
        fit_two_stage(TinyTemporal(), [batch("test", ["a"])], [batch("validation", ["b"])])
    with pytest.raises(ValueError, match="train_batches"):
        fit_two_stage(TinyTemporal(), [batch("validation", ["a"])], [batch("validation", ["b"])])


def test_missing_train_annotations_fails_before_weights_change():
    model = TinyTemporal()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match="eligible train or validation dense"):
        fit_two_stage(model, [batch("train", ["a"], annotated=False)], [batch("validation", ["b"])])
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_missing_annotation_and_invalid_support_are_excluded_from_counters():
    model = TinyTemporal()
    invalid = torch.zeros(1, 1, 2, 2)
    train = [batch("train", ["a", "b"], annotated=True,
                   x=torch.cat((torch.ones(1, 1, 2, 2), invalid)))]
    train.append(batch("train", ["unannotated"], annotated=False, plume_value=float('nan')))
    # The second scene has no QA support; its annotation must not be counted.
    result = fit_two_stage(model, train, [batch("validation", ["c"])])
    assert result["counters"]["dense_train_scenes"] == 1
    assert result["counters"]["scene_train_scenes"] == 2
    assert result["counters"]["train_missing_annotations"] == 1


def test_metric_combines_unequal_batches_by_scene():
    stats = [
        {"bce_values": torch.tensor([1.0]), "dice_values": torch.tensor([0.0])},
        {"bce_values": torch.tensor([3.0, 5.0, 7.0]), "dice_values": torch.tensor([1.0, 1.0])},
    ]
    value, scenes, positives = _metric(stats)
    assert scenes == 4 and positives == 3
    assert value == pytest.approx(0.7 * 4.0 + 0.3 * (2.0 / 3.0))


def test_best_dense_checkpoint_is_restored_and_scene_stage_cannot_change_it():
    torch.manual_seed(7)
    first = TinyTemporal()
    second = copy.deepcopy(first)
    train, validation = [batch("train", ["a"], label=1.0, plume_value=1.0)], [batch("validation", ["b"], label=0.0, plume_value=0.0)]
    one = fit_two_stage(first, train, validation, dense_epochs=3, scene_epochs=1, lr=0.2)
    three = fit_two_stage(second, train, validation, dense_epochs=3, scene_epochs=3, lr=0.2)
    assert one["best_dense_epoch"] == 1
    # The first Adam step moves the scalar from 0 to +lr; retaining epoch 3
    # would leave it substantially larger even if both runs retained it.
    assert first.dense_weight.item() == pytest.approx(0.2, abs=1e-6)
    # Scene epochs must not alter dense parameters selected by dense validation.
    assert first.dense_weight.item() == pytest.approx(second.dense_weight.item(), abs=1e-7)
