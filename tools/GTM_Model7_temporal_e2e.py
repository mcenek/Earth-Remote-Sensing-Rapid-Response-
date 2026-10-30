"""End-to-end verification of both completed real temporal training runs.

This file and its failure cases were written before the trainer implementation.
It verifies artifacts from actual fitted models, not an isolated unit fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


TRAINER = Path(__file__).with_name("GTM_Model7_train_temporal.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invoke(*args: str, accept_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([sys.executable, str(TRAINER), *args],
                            capture_output=True, text=True, timeout=180)
    if (result.returncode == 0) == accept_failure:
        raise AssertionError(f"Unexpected command status {result.returncode}: {args}\n"
                             f"stdout={result.stdout}\nstderr={result.stderr}")
    return result


def compare_metrics(probability: np.ndarray, target: np.ndarray,
                    support: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    predicted = probability[support] >= threshold
    truth = target[support] == 1
    return {"tp": int(np.sum(predicted & truth)),
            "fp": int(np.sum(predicted & ~truth)),
            "fn": int(np.sum(~predicted & truth)),
            "tn": int(np.sum(~predicted & ~truth))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "GTM_Model7_temporal_cohort_v1"
    tests = [scene for scene in manifest["scenes"] if scene["split"] == "test"]
    assert tests and {scene["split"] for scene in manifest["scenes"]} == {"train", "dev", "test"}
    summary: dict = {"schema": "GTM_Model7_temporal_e2e_v1", "manifest_sha256": sha(manifest_path),
                     "runs": {}, "checks": []}

    invoke("validate", "--manifest", str(manifest_path))
    summary["checks"].append("original_manifest_valid")
    original_fingerprint = json.loads(invoke("fingerprint", "--manifest", str(manifest_path)).stdout)
    output_parent = args.output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="temporal_e2e_", dir=output_parent) as temp:
        temp_path = Path(temp).resolve()
        if not temp_path.is_relative_to(output_parent) or not temp_path.name.startswith("temporal_e2e_"):
            raise ValueError("Unsafe E2E staging path")
        # Keep the sealed cohort immutable: stage a copy for failure mutations.
        for scene in manifest["scenes"]:
            original = manifest_path.parent / scene["array_file"]
            staged = temp_path / scene["array_file"]
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, staged)
        altered = json.loads(manifest_path.read_text(encoding="utf-8"))
        altered["scenes"][0]["array_sha256"] = "0" * 64
        bad_manifest = temp_path / "bad.json"
        try:
            bad_manifest.write_text(json.dumps(altered), encoding="utf-8")
            invoke("validate", "--manifest", str(bad_manifest), accept_failure=True)
        finally:
            bad_manifest.unlink(missing_ok=True)
        summary["checks"].append("changed_array_hash_rejected")

        altered = json.loads(manifest_path.read_text(encoding="utf-8"))
        altered["scenes"][0]["sources"][0]["sha256"] = "0" * 64
        bad_manifest = temp_path / "bad.json"
        try:
            bad_manifest.write_text(json.dumps(altered), encoding="utf-8")
            invoke("validate", "--manifest", str(bad_manifest), accept_failure=True)
        finally:
            bad_manifest.unlink(missing_ok=True)
        summary["checks"].append("changed_source_hash_rejected")

        altered = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_scene = next(scene for scene in altered["scenes"] if scene["split"] == "train")
        test_scene = next(scene for scene in altered["scenes"] if scene["split"] == "test")
        test_scene["group_id"] = train_scene["group_id"]
        bad_manifest = temp_path / "bad.json"
        try:
            bad_manifest.write_text(json.dumps(altered), encoding="utf-8")
            invoke("validate", "--manifest", str(bad_manifest), accept_failure=True)
        finally:
            bad_manifest.unlink(missing_ok=True)
        summary["checks"].append("split_overlap_rejected")

        altered = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = next(scene for scene in altered["scenes"] if scene["split"] == "train")
        with np.load(temp_path / changed["array_file"], allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        py, px = np.argwhere(arrays["valid"].astype(bool))[0]
        arrays["target_image"][py, px, 0] = np.nan
        corrupt_npz = temp_path / "nonfinite.npz"
        bad_manifest = temp_path / "bad.json"
        try:
            np.savez_compressed(corrupt_npz, **arrays)
            changed["array_file"] = corrupt_npz.name
            changed["array_sha256"] = sha(corrupt_npz)
            bad_manifest.write_text(json.dumps(altered), encoding="utf-8")
            invoke("validate", "--manifest", str(bad_manifest), accept_failure=True)
        finally:
            bad_manifest.unlink(missing_ok=True)
            corrupt_npz.unlink(missing_ok=True)
        summary["checks"].append("nonfinite_supported_input_rejected")

        altered = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = next(scene for scene in altered["scenes"]
                       if scene["split"] == "test" and scene["scene_kind"] == "PLUME")
        original_npz = temp_path / changed["array_file"]
        with np.load(original_npz, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        assert np.any(arrays["label_native"] == 1), "Sentinel scene has no positive label to remove"
        arrays["label_native"] = np.zeros_like(arrays["label_native"])
        arrays["target"] = np.zeros_like(arrays["target"])
        sentinel_npz = temp_path / "sentinel.npz"
        sentinel_manifest = temp_path / "sentinel.json"
        try:
            np.savez_compressed(sentinel_npz, **arrays)
            assert sha(sentinel_npz) != sha(original_npz), "Test-label sentinel did not change the array"
            changed["array_file"] = sentinel_npz.name
            changed["array_sha256"] = sha(sentinel_npz)
            sentinel_manifest.write_text(json.dumps(altered), encoding="utf-8")
            altered_fingerprint = json.loads(invoke("fingerprint", "--manifest", str(sentinel_manifest)).stdout)
            assert altered_fingerprint == original_fingerprint, "Test-label sentinel changed training fingerprint"
        finally:
            sentinel_manifest.unlink(missing_ok=True)
            sentinel_npz.unlink(missing_ok=True)
        summary["checks"].append("test_label_sentinel_cannot_change_training_fingerprint")

    common_config = common_schedule = None
    for run in args.runs:
        run = run.resolve()
        receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
        model = receipt["model"]
        assert model in {"residual", "siamese"} and model not in summary["runs"]
        assert receipt["manifest_sha256"] == summary["manifest_sha256"]
        for name, expected in receipt["snapshot_sha256"].items():
            assert sha(run / "source" / name) == expected, f"Changed source snapshot: {name}"
        assert receipt["snapshot_sha256"]["manifest.json"] == summary["manifest_sha256"]
        assert receipt["training_fingerprint"] == original_fingerprint["training_fingerprint"]
        assert receipt["config"]["threshold"] == 0.5
        assert receipt["config"]["gradient_norm_cap"] == 1.0
        if common_config is None:
            common_config = receipt["config"]
            common_schedule = receipt["draw_schedule_sha256"]
        else:
            assert receipt["config"] == common_config, "Comparison settings differ"
            assert receipt["draw_schedule_sha256"] == common_schedule, "Training crops differ"
        assert receipt["checkpoint_sha256"]["best"] == receipt["checkpoint_sha256"]["final"]
        assert receipt["checkpoint_selection"].startswith("none;")
        assert receipt["split_counts"]["test"] == len(tests)
        assert (run / "initial.pt").is_file() and (run / "best.pt").is_file() and (run / "final.pt").is_file()
        assert (run / "learning_curve.json").is_file()
        pooled = dict(tp=0, fp=0, fn=0, tn=0)
        positive_controls = negative_controls = 0
        for scene in tests:
            prediction_path = run / "predictions" / f"{scene['id']}.npz"
            assert prediction_path.is_file(), prediction_path
            with np.load(prediction_path, allow_pickle=False) as z:
                p20 = z["probability20"]
                p_native = z["probability_native"]
                valid20 = z["valid20"]
            with np.load(manifest_path.parent / scene["array_file"], allow_pickle=False) as z:
                truth = z["label_native"]
                support = z["valid_native"].astype(bool)
            assert p20.shape == (100, 100) and p20.dtype == np.float32
            assert p_native.shape == (200, 200) and p_native.dtype == np.float32
            assert valid20.shape == (100, 100)
            assert np.isfinite(p20[valid20]).all()
            assert np.all((p20[valid20] >= 0) & (p20[valid20] <= 1))
            assert np.array_equal(p_native, np.repeat(np.repeat(p20, 2, axis=0), 2, axis=1), equal_nan=True)
            metric = compare_metrics(p_native, truth, support)
            for key in pooled:
                pooled[key] += metric[key]
            indexed = receipt["test_scenes"][scene["id"]]
            assert {key: indexed[key] for key in pooled} == metric
            if np.any(truth[support] == 1):
                positive_controls += 1
            else:
                negative_controls += 1
                assert indexed["negative_any_activation"] == bool(metric["fp"])
                assert abs(indexed["negative_fp_area_rate"] - metric["fp"] / int(support.sum())) < 1e-12
        assert positive_controls and negative_controls
        assert receipt["pooled_test"]["confusion"] == pooled
        invoke("replay", "--manifest", str(manifest_path), "--run", str(run),
               "--checkpoint", "best")
        summary["runs"][model] = {"receipt_sha256": sha(run / "receipt.json"),
                                  "test_scenes": len(tests), "pooled_confusion": pooled,
                                  "fresh_process_replay": "passed"}
    assert set(summary["runs"]) == {"residual", "siamese"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"e2e": "passed", "artifact": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
