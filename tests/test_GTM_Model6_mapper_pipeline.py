"""Regression checks for native prediction support and export/scoring parity."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_GTM_Model6_mapper_mars_pilot as pilot
import GTM_Model0_local_viewer as viewer
from GTM_Model6_mapper_data import TRANSFORMS, eligible_centers, extract_patch, sample_batch, transformed_parent


def parent(positive=True, at_edge=False):
    y, x = np.indices((200, 200))
    mask = np.zeros((200, 200), np.float32)
    if positive:
        mask[1:4, 1:4] = 1 if at_edge else 0
        if not at_edge:
            mask[91:105, 97:107] = 1
    features = np.stack([x, y, mask, *[x / 200 for _ in range(13)]]).astype(np.float32)
    return {'x': features, 'target': mask, 'valid': np.ones((200, 200), bool), 'row': {'fully_labeled': True}}


@pytest.mark.parametrize('angle,mirror', TRANSFORMS)
def test_all_transforms_share_label_coordinates_and_observed_context(angle, mirror):
    view = transformed_parent(parent(), angle, mirror)
    centers = eligible_centers(view[1], view[2], view[3], positive=True)
    assert len(centers)
    contexts, target, support = extract_patch(view, centers[len(centers)//2])
    for size in (16, 32, 64, 128):
        offset = (size-16)//2
        centre = contexts[f'S{size}'][:, offset:offset+16, offset:offset+16]
        assert torch.allclose(centre, contexts['S16']), 'context crops must share the same source coordinates'
    # Third feature is a continuous interpolation of the source mask. Interior
    # pixels must match the nearest-neighbour target under every transform.
    interior = (contexts['S16'][2:3] > .99) | (contexts['S16'][2:3] < .01)
    assert torch.equal((contexts['S16'][2:3] > .5)[interior], target.bool()[interior])
    assert (target.bool() & support).any()


def test_positive_sampler_does_not_clamp_an_unreachable_plume_to_background():
    view = transformed_parent(parent(at_edge=True), 0, False)
    assert len(eligible_centers(view[1], view[2], view[3], positive=True)) == 0


def test_balanced_sampling_guarantees_positive_targets_and_valid_contexts():
    _, target, support = sample_batch([parent(), parent(False)], np.random.default_rng(123), 4)
    assert int((target.bool() & support).flatten(1).any(1).sum()) == 2


def test_unsupported_predictions_cannot_change_native_scores():
    item = {"valid": np.ones((20, 20), bool), "target": np.zeros((20, 20))}
    item["target"][9, 9] = 1
    support = np.zeros((20, 20), bool)
    support[8:12, 8:12] = True
    p = np.ones((20, 20))
    p[support] = 0
    metrics = pilot._metrics(p, item, evaluation_support=support)
    assert metrics["fp"] == 0 and metrics["tn"] == 15 and metrics["fn"] == 1
    assert metrics["evaluated_pixels"] == 16
    assert item["valid"].all(), "scoring must not mutate the input validity"


class CentreSignal:
    def __call__(self, contexts):
        return {"mask_logits": contexts["S16"][:, :1]}


@pytest.mark.parametrize("shape", [(200, 200), (201, 205), (128, 128)])
def test_tiling_preserves_native_coordinates_and_actual_margin(shape):
    y, x = np.indices(shape)
    signal = ((x + 2 * y) / 100).astype("float32")
    item = {"x": np.broadcast_to(signal, (16, *shape)).copy()}
    probability, support = pilot._dense_predict(CentreSignal(), item, "cpu")
    expected = np.zeros(shape, bool)
    expected[56:shape[0]-56, 56:shape[1]-56] = True
    assert np.array_equal(support, expected)
    assert np.allclose(probability[support], 1 / (1 + np.exp(-signal[support])), atol=1e-6)


def test_dense_prediction_refuses_insufficient_context():
    with pytest.raises(ValueError, match="largest native context"):
        pilot._dense_predict(CentreSignal(), {"x": np.zeros((16, 127, 200))}, "cpu")


def test_viewer_resolves_probability_asset_and_detects_missing_file(tmp_path, monkeypatch):
    ui, bundles = tmp_path / "ui", tmp_path / "bundles"
    ui.mkdir()
    bundle = bundles / "test"
    bundle.mkdir(parents=True)
    (ui / "GTM_Model0_registry.json").write_text(json.dumps({"experiments": []}))
    (bundle / "probability.png").write_bytes(b"fixture")
    (bundle / "experiment.json").write_text(json.dumps({"id": "test", "scenes": [{"probabilityImage": "probability.png"}]}))
    monkeypatch.setattr(viewer, "UI", ui)
    monkeypatch.setattr(viewer, "BUNDLES", bundles)
    registry = viewer.registry()
    assert registry["experiments"][0]["scenes"][0]["probabilityImage"] == "bundle-assets/test/probability.png"
    (bundle / "probability.png").unlink()
    assert "Missing asset" in viewer.registry()["local"]["warnings"][0]
