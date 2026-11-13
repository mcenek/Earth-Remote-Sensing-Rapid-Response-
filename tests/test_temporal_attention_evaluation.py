import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "EarthRemoteSensingRapidResponse"))
from temporal_attention_evaluation import calibrate, dense_metrics, evaluate_pair


def bundle(split, rows):
    return {"split": split, "samples": rows}


def row(sample_id, group_id, label, cnn, temporal_attention, valid=True):
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "label": label,
        "valid": valid,
        "cnn": cnn,
        "temporal_attention": temporal_attention,
    }


def calibration():
    return calibrate(bundle("validation", [
        row("v0", "vg0", 0, 0.1, 0.1),
        row("v1", "vg1", 1, 0.9, 0.9),
    ]), max_fpr=0.5)


def test_calibration_keeps_ties_together_and_uses_validation_fpr():
    result = calibrate(bundle("validation", [
        row("v0", "vg0", 0, 0.8, 0.2),
        row("v1", "vg1", 1, 0.8, 0.1),
        row("v2", "vg2", 1, 0.7, 0.9),
        row("v3", "vg3", 0, 0.1, 0.0),
    ]), max_fpr=0.5)
    assert result["models"]["cnn"]["rule"] == {"kind": "threshold", "threshold": 0.7, "comparator": ">="}
    assert result["models"]["cnn"]["validation"]["recall"] == pytest.approx(1.0)


def test_zero_recall_is_reject_all_and_json_safe():
    result = calibrate(bundle("validation", [
        row("v0", "vg0", 0, 0.9, 0.8), row("v1", "vg1", 1, 0.1, 0.2)
    ]), max_fpr=0.0)
    assert result["models"]["cnn"]["rule"] == {"kind": "reject_all"}
    json.dumps(result, allow_nan=False)


def test_explicit_split_and_validation_only_contract():
    with pytest.raises(ValueError, match="explicit validation"):
        calibrate(bundle("test", [row("x", "g", 0, 0.1, 0.1)]), max_fpr=0.0)
    with pytest.raises(ValueError, match="explicit test"):
        evaluate_pair(bundle("validation", [row("x", "g", 0, 0.1, 0.1)]), calibration())


def test_malformed_calibration_rejects_before_evaluation():
    cal = calibration()
    del cal['validation_ids']
    with pytest.raises(ValueError, match='identities'):
        evaluate_pair(bundle('test', [row('t', 'tg', 0, .1, .2)]), cal)


@pytest.mark.parametrize("field", ["sample_id", "group_id"])
def test_evaluation_rejects_validation_overlap(field):
    cal = calibration()
    value = cal["validation_ids"][0] if field == "sample_id" else cal["validation_groups"][0]
    test_row = row("t0", "tg0", 0, 0.2, 0.2)
    test_row[field] = value
    with pytest.raises(ValueError, match="overlaps validation"):
        evaluate_pair(bundle("test", [test_row]), cal)


def test_identical_models_have_zero_paired_intervals():
    cal = calibration()
    test = bundle("test", [
        row("t0", "tg0", 0, 0.1, 0.1), row("t1", "tg0", 1, 0.9, 0.9),
        row("t2", "tg1", 0, 0.2, 0.2), row("t3", "tg1", 1, 0.8, 0.8),
    ])
    result = evaluate_pair(test, cal, replicates=20, seed=3)
    intervals = result["paired_delta_bootstrap"]["intervals"]
    assert result["paired_delta_bootstrap"]["groups"] == 2
    assert all(interval == {"lower": 0.0, "upper": 0.0} for interval in intervals.values())


def test_class_degenerate_bootstrap_is_flagged_unavailable():
    cal = calibration()
    test = bundle("test", [
        row("t0", "tg0", 0, 0.1, 0.1), row("t1", "tg0", 0, 0.2, 0.2),
        row("t2", "tg1", 1, 0.9, 0.9), row("t3", "tg1", 1, 0.8, 0.8),
    ])
    bootstrap = evaluate_pair(test, cal, replicates=100, seed=2)["paired_delta_bootstrap"]
    assert bootstrap["degenerate_replicates"] > 0
    assert bootstrap["intervals"] is None
    assert "Class-degenerate" in bootstrap["unavailable_reason"]


def test_single_class_evaluation_reports_null_ap():
    result = evaluate_pair(bundle("test", [
        row("t0", "tg0", 1, 0.9, 0.8), row("t1", "tg1", 1, 0.7, 0.6)
    ]), calibration())
    assert result["models"]["cnn"]["average_precision"] is None
    assert result["models"]["temporal_attention"]["average_precision"] is None
    assert result["paired_delta_bootstrap"]["intervals"] is None


def test_invalid_rows_are_excluded_and_counted_by_class():
    result = evaluate_pair(bundle("test", [
        row("t0", "tg0", 0, 0.1, 0.1), row("t1", "tg1", 1, 0.9, 0.9),
        row("t2", "tg2", 1, None, None, valid=False),
        row("t3", "tg3", 0, None, None, valid=False),
    ]), calibration())
    assert result["valid_scenes"] == 2 and result["abstained_scenes"] == 2
    assert result["positive_scenes_abstained"] == 1
    assert result["negative_scenes_abstained"] == 1


def test_dense_missing_and_invalid_qa_pixels_are_excluded():
    probability = np.array([[np.nan, 2.0], [0.2, 0.0]])
    truth = np.array([[3, 2], [0, 0]])
    valid = np.array([[False, False], [True, True]], dtype=bool)
    assert dense_metrics(probability, truth, valid, annotation_available=False)["eligible"] is False
    result = dense_metrics(probability, truth, valid, annotation_available=True)
    assert result["eligible"] is True and result["valid_pixels"] == 2
    assert result["iou"] is None


def test_dense_bad_shapes_and_supported_nonbinary_values_reject():
    with pytest.raises(ValueError, match="equal HxW"):
        dense_metrics(np.zeros((2, 2)), np.zeros((2, 1)), np.ones((2, 2), dtype=bool), annotation_available=True)
    with pytest.raises(ValueError, match="binary"):
        dense_metrics(np.zeros((1, 2)), np.array([[0, 2]]), np.ones((1, 2), dtype=bool), annotation_available=True)


def test_ap_matches_sklearn_under_reordering_and_ties():
    cal = calibration()
    rows = [
        row("t0", "tg0", 0, 0.5, 0.2), row("t1", "tg0", 1, 0.5, 0.9),
        row("t2", "tg1", 1, 0.2, 0.5), row("t3", "tg1", 0, 0.1, 0.5),
    ]
    first = evaluate_pair(bundle("test", rows), cal, replicates=10)
    second = evaluate_pair(bundle("test", list(reversed(rows))), cal, replicates=10)
    expected = average_precision_score([r["label"] for r in rows], [r["cnn"] for r in rows])
    assert first["models"]["cnn"]["average_precision"] == pytest.approx(expected)
    assert second["models"]["cnn"]["average_precision"] == pytest.approx(expected)
