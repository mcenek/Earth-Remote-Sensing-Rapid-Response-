"""Frozen CNN 2 inference and fresh-process replay on prepared real unseen scenes.

Failure cases and the predeclared cohort are documented in
docs/GTM_Model6_unseen_evaluation_20261023.md. This tool never fits a model.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import torch

from GTM_Model6_train_real_emit import (Branch, REFERENCE, apply_norm, inference,
                                         sha, write_json)


SCHEMA = "GTM_Model6_real_unseen_cohort_v1"
OUTPUT_SCHEMA = "GTM_Model6_real_unseen_inference_v1"
SNAPSHOT_SOURCES = (Path(__file__), Path(__file__).with_name("GTM_Model6_train_real_emit.py"), REFERENCE)


def checked_local_array(folder: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing relative array_file")
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or path.suffix.lower() != ".npz":
        raise ValueError("array_file must be an NPZ inside the manifest folder")
    return path


def training_state(run: Path):
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "GTM_Model6_real_cnn2_fit_v1" or receipt.get("stage") != "CNN 2":
        raise ValueError("Expected accepted CNN 2 fitting receipt")
    if receipt.get("config", {}).get("threshold") != 0.5:
        raise ValueError("Accepted threshold is not 0.50")
    if receipt.get("source_hashes") != {p.name: sha(p) for p in SNAPSHOT_SOURCES[1:]}:
        raise ValueError("Training or Branch source differs from accepted receipt")
    for name, digest in receipt["source_snapshots"].items():
        if sha(run / "source_snapshots" / name) != digest:
            raise ValueError("Training snapshot differs from accepted receipt")
    checkpoint = run / receipt["final"]["path"]
    if sha(checkpoint) != receipt["final"]["sha256"]:
        raise ValueError("Accepted final checkpoint differs")
    if sha(Path(receipt["manifest_path"])) != receipt["manifest_sha256"]:
        raise ValueError("Training manifest differs")
    if sha(Path(receipt["array_path"])) != receipt["array_sha256"]:
        raise ValueError("Training array differs")
    if sha(run / receipt["prediction"]["path"]) != receipt["prediction"]["sha256"]:
        raise ValueError("Training prediction archive differs")
    for source in json.loads(Path(receipt["manifest_path"]).read_text())["sources"]:
        path = Path(source["path"])
        if path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
            raise ValueError(f"Training acquisition source differs: {path}")
    files = {str(p.resolve()): sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
    return receipt, checkpoint, files


def cohort(path: Path, fit: dict, fit_receipt_hash: str):
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema") != SCHEMA or doc.get("synthetic") is not False or
            doc.get("role") != "unseen_diagnostic" or doc.get("downloads") != 0):
        raise ValueError("Expected frozen real unseen cohort manifest")
    if doc.get("training_receipt_sha256") != fit_receipt_hash or doc.get("training_checkpoint_sha256") != fit["final"]["sha256"]:
        raise ValueError("Cohort pins a different fitting run")
    scenes = doc.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Unseen cohort needs scenes")
    fit_manifest = json.loads(Path(fit["manifest_path"]).read_text(encoding="utf-8"))
    fit_satellite = fit_manifest["satellite"]["id"]
    fit_sources = {s["sha256"] for s in fit_manifest["sources"]}
    ids, groups, asset_hashes = set(), set(), set()
    for scene in scenes:
        sid = scene.get("id")
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("Duplicate or absent scene ID")
        ids.add(sid)
        group = scene.get("group_id")
        if not isinstance(group, str) or not group or group in groups:
            raise ValueError("Duplicate or absent independent location group")
        groups.add(group)
        if scene.get("independent_of_training") is not True or scene.get("satellite_id") == fit_satellite:
            raise ValueError(f"Scene is not independent of fitting: {sid}")
        if scene.get("reference_kind") not in ("reviewed_binary_annotation", "continuous_retrieval_context"):
            raise ValueError(f"Unrecognized reference kind: {sid}")
        if scene.get("reference_mapping") not in ("repeat_2", "none"):
            raise ValueError(f"Unrecognized reference mapping: {sid}")
        if (scene["reference_kind"] == "reviewed_binary_annotation") != (scene["reference_mapping"] == "repeat_2"):
            raise ValueError(f"Reference kind/mapping conflict: {sid}")
        ap = checked_local_array(path.parent, scene.get("array_file"))
        if sha(ap) != scene.get("array_sha256") or scene["array_sha256"] == fit["array_sha256"]:
            raise ValueError(f"Scene array changed or duplicates fitting array: {sid}")
        sources = scene.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Scene lacks acquisition source hashes: {sid}")
        for source in sources:
            sp = Path(source["path"])
            if not sp.is_absolute() or sp.stat().st_size != source["bytes"] or sha(sp) != source["sha256"]:
                raise ValueError(f"Scene source changed: {sp}")
            if source["sha256"] in fit_sources:
                raise ValueError(f"Scene repeats a fitting source: {sid}")
            asset_hashes.add(source["sha256"])
    return doc


def grid(scene: dict, image_shape: tuple, target_shape: tuple):
    input_grid, reference_grid = scene["input_grid"], scene["reference_grid"]
    if input_grid["shape"] != list(image_shape) or reference_grid["shape"] != list(target_shape):
        raise ValueError("Prepared arrays differ from declared grids")
    if not input_grid.get("crs") or input_grid["crs"] != reference_grid.get("crs"):
        raise ValueError("Grid CRS mismatch")
    a, b = input_grid["affine"], reference_grid["affine"]
    if len(a) != 6 or len(b) != 6:
        raise ValueError("Expected six affine coefficients")
    if scene["reference_mapping"] == "repeat_2":
        if target_shape != (image_shape[0]*2, image_shape[1]*2):
            raise ValueError("MARS reference must be exact 2:1")
        expected = [a[0]/2, a[1]/2, a[2], a[3]/2, a[4]/2, a[5]]
        if not np.allclose(b, expected, rtol=0, atol=1e-6):
            raise ValueError("MARS reference affine is not exact half-size input grid")


def load_scene(folder: Path, scene: dict):
    file = checked_local_array(folder, scene["array_file"])
    with np.load(file, allow_pickle=False) as z:
        image = z["image"]
        valid = z["input_valid"]
        target = z["target"]
        support = z["target_valid"]
    if image.dtype != np.float32 or valid.dtype != np.bool_ or target.dtype != np.float32 or support.dtype != np.bool_:
        raise ValueError("Prepared array dtypes differ from contract")
    if image.ndim != 3 or image.shape[2] != 5 or valid.shape != image.shape[:2] or target.ndim != 2 or support.shape != target.shape:
        raise ValueError("Prepared scene shapes differ from contract")
    grid(scene, valid.shape, target.shape)
    if not np.isfinite(image[valid]).all():
        raise ValueError("Nonfinite supported input")
    if scene["reference_mapping"] == "repeat_2":
        if not np.isin(target[support], [0, 1]).all():
            raise ValueError("Reviewed reference must use 0/1 codes")
        if np.any(support & ~np.repeat(np.repeat(valid, 2, 0), 2, 1)):
            raise ValueError("Reference includes invalid input support")
    elif support.any() or not np.isnan(target).all():
        raise ValueError("Continuous context must not contain binary scoring targets")
    return image, valid, target, support


def counts(prob, target, support, threshold):
    pred = prob[support] >= threshold
    truth = target[support] == 1
    tp, fp = int(np.sum(pred & truth)), int(np.sum(pred & ~truth))
    fn, tn = int(np.sum(~pred & truth)), int(np.sum(~pred & ~truth))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, supported=int(support.sum()),
                predicted_positive=int(pred.sum()), recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None)


def summarize(fine, reference, valid, support, target, threshold, kind):
    v = fine[valid]
    summary = dict(valid_input_pixels=int(valid.sum()),
                   valid_input_positive=int(np.sum(v >= threshold)),
                   valid_input_positive_fraction=float(np.mean(v >= threshold)) if v.size else None,
                   valid_input_mean_probability=float(np.mean(v)) if v.size else None,
                   valid_input_max_probability=float(np.max(v)) if v.size else None,
                   native_reference_metrics=None, controls=None)
    if kind == "reviewed_binary_annotation":
        actual = counts(reference, target, support, threshold)
        summary["native_reference_metrics"] = actual
        summary["controls"] = dict(always_negative=counts(np.zeros_like(reference), target, support, threshold),
                                   always_positive=counts(np.ones_like(reference), target, support, threshold))
    return summary


def model_for(fit, checkpoint):
    model = Branch(fit["config"]["base_channels"]).double()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model


def evaluate_one(model, fit, scene, folder):
    image, valid, target, support = load_scene(folder, scene)
    x = apply_norm(image.astype(np.float64), valid, fit["normalization"])
    fine = inference(model, x, valid, fit["config"])
    if scene["reference_mapping"] == "repeat_2":
        reference = np.repeat(np.repeat(fine, 2, 0), 2, 1)
    else:
        reference = fine.copy()
    return dict(fine_probability=fine, input_valid=valid, reference_probability=reference,
                target=target, target_valid=support), x


def score_after_inference(arrays, x, fit, scene):
    # The target and its support never enter the model call. Score saved outputs.
    summary = summarize(arrays["fine_probability"], arrays["reference_probability"],
                        arrays["input_valid"], arrays["target_valid"], arrays["target"],
                        fit["config"]["threshold"], scene["reference_kind"])
    valid = arrays["input_valid"]
    if valid.any():
        z = x[valid]
        summary["frozen_standardized_input"] = dict(mean=[float(v) for v in z.mean(0)],
                                                      min=[float(v) for v in z.min(0)],
                                                      max=[float(v) for v in z.max(0)])
    return summary


def predict(args):
    run = args.training_run.resolve()
    fit, checkpoint, before = training_state(run)
    fit_receipt_hash = sha(run / "receipt.json")
    doc = cohort(args.manifest, fit, fit_receipt_hash)
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Prediction output directory must be new")
    model = model_for(fit, checkpoint)
    dest.mkdir(parents=True)
    snapshots = dest / "source_snapshots"
    snapshots.mkdir()
    for source in SNAPSHOT_SOURCES:
        shutil.copy2(source, snapshots / source.name)
    records = []
    for index, scene in enumerate(doc["scenes"]):
        arrays, x = evaluate_one(model, fit, scene, args.manifest.parent)
        name = f"scene_{index:03d}.npz"
        file = dest / name
        np.savez_compressed(file, **arrays)
        with np.load(file, allow_pickle=False) as saved:
            summary = score_after_inference({k: saved[k] for k in arrays}, x, fit, scene)
        records.append(dict(id=scene["id"], group_id=scene["group_id"], scene_kind=scene["scene_kind"],
                            reference_kind=scene["reference_kind"], reference_mapping=scene["reference_mapping"],
                            array_file=name, array_sha256=sha(file), summary=summary))
        print(json.dumps(dict(scene=scene["id"], summary=summary)), flush=True)
    after = {str(p.resolve()): sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
    if before != after:
        raise RuntimeError("Accepted training files changed during evaluation")
    receipt = dict(schema=OUTPUT_SCHEMA, manifest_path=str(args.manifest.resolve()),
                   manifest_sha256=sha(args.manifest), training_run=str(run),
                   training_receipt_sha256=fit_receipt_hash,
                   training_checkpoint_sha256=fit["final"]["sha256"],
                   training_file_hashes=before, evaluator_source_hashes={p.name: sha(p) for p in SNAPSHOT_SOURCES},
                   source_snapshots={p.name: sha(snapshots / p.name) for p in SNAPSHOT_SOURCES},
                   normalization_source="frozen Volusia fitting only", threshold=fit["config"]["threshold"],
                   inference_config={k: fit["config"][k] for k in ("base_channels", "inference_core", "inference_halo")},
                   reference_grid_semantics="MARS exact repeat2 onto delivered 10m labels; Georgia unscored context",
                   scenes=records, downloads=0,
                   runtime=dict(python=platform.python_version(), pytorch=torch.__version__, numpy=np.__version__,
                                dtype="float64 computation / float32 predictions"))
    write_json(dest / "receipt.json", receipt)
    print(json.dumps(dict(receipt=str(dest / "receipt.json"), scenes=len(records))), flush=True)


def replay(args):
    dest = args.output.resolve()
    saved_receipt = dest / "receipt.json"
    record = json.loads(saved_receipt.read_text(encoding="utf-8"))
    if record.get("schema") != OUTPUT_SCHEMA or record.get("manifest_sha256") != sha(args.manifest):
        raise ValueError("Prediction receipt/manifest differs")
    fit, checkpoint, before = training_state(args.training_run.resolve())
    if before != record["training_file_hashes"] or sha(args.training_run / "receipt.json") != record["training_receipt_sha256"]:
        raise ValueError("Accepted training artifacts differ from inference")
    if {p.name: sha(p) for p in SNAPSHOT_SOURCES} != record["evaluator_source_hashes"]:
        raise ValueError("Current evaluator or imported source changed")
    for name, digest in record["source_snapshots"].items():
        if sha(dest / "source_snapshots" / name) != digest:
            raise ValueError("Evaluation source snapshot changed")
    doc = cohort(args.manifest, fit, record["training_receipt_sha256"])
    if len(doc["scenes"]) != len(record["scenes"]):
        raise ValueError("Saved scene count differs")
    model = model_for(fit, checkpoint)
    results = []
    for scene, item in zip(doc["scenes"], record["scenes"]):
        file = dest / item["array_file"]
        if scene["id"] != item["id"] or sha(file) != item["array_sha256"]:
            raise ValueError("Saved prediction differs")
        arrays, x = evaluate_one(model, fit, scene, args.manifest.parent)
        summary = score_after_inference(arrays, x, fit, scene)
        with np.load(file, allow_pickle=False) as saved:
            diffs = {}
            for key in ("fine_probability", "reference_probability"):
                if not np.array_equal(np.isnan(arrays[key]), np.isnan(saved[key])):
                    raise ValueError(f"Changed prediction support: {scene['id']} {key}")
                mask = np.isfinite(arrays[key])
                diffs[key] = float(np.max(np.abs(arrays[key][mask] - saved[key][mask]))) if mask.any() else 0.0
            for key in ("input_valid", "target_valid", "target"):
                if not np.array_equal(arrays[key], saved[key], equal_nan=True):
                    raise ValueError(f"Saved support/reference differs: {scene['id']} {key}")
        if max(diffs.values()) > fit["config"]["replay_atol"] or summary != item["summary"]:
            raise ValueError(f"Replay differs for {scene['id']}: {diffs}")
        results.append(dict(id=scene["id"], max_abs_diff=diffs, counts=summary["native_reference_metrics"]))
    after = {str(p.resolve()): sha(p) for p in sorted(args.training_run.resolve().rglob("*")) if p.is_file()}
    if before != after:
        raise RuntimeError("Accepted training files changed during replay")
    verify = args.verification_output or dest.with_name(dest.name + "_verification")
    verify = verify.resolve()
    if verify.exists():
        raise FileExistsError("Verification output directory must be new")
    verify.mkdir(parents=True)
    output = dict(schema="GTM_Model6_real_unseen_replay_v1", passed=True,
                  inference_receipt_sha256=sha(saved_receipt), manifest_sha256=sha(args.manifest),
                  training_receipt_sha256=record["training_receipt_sha256"],
                  tolerance=fit["config"]["replay_atol"], scenes=results)
    write_json(verify / "replay.json", output)
    print(json.dumps(dict(replay=str(verify / "replay.json"), passed=True, scenes=len(results))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "replay"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
