"""Export saved three-CNN probabilities, never targets disguised as prediction."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from GTM_Model6_three_cnn_data import load_arrays, load_manifest, load_scenes, load_scene_arrays, sha256
from GTM_Model0_local_viewer import load_bundle
from GTM_Model0_publish_viewer import publish

PREFIX = "GTM_Model6_three_cnn"
LABELS = {"airborne": "CNN 1 | airborne reference", "emit": "CNN 2 | EMIT reference",
          "fusion": "CNN 3 | frozen-feature combination"}


#### Saved-array verification and native scores
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Saved artifact escapes the run directory")
    return path


def counts(probability, target, valid, threshold):
    if not np.any(valid):
        raise ValueError("No measured support to evaluate")
    p, t = probability[valid] >= threshold, target[valid] == 1
    tp, fp = int(np.sum(p & t)), int(np.sum(p & ~t))
    fn, tn = int(np.sum(~p & t)), int(np.sum(~p & ~t))
    ratio = lambda a, b: a / b if b else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "iou": ratio(tp, tp + fp + fn), "evaluated_pixels": tp + fp + fn + tn}


def verify_saved(row, record, run_dir, threshold):
    path = inside(run_dir, record["prediction"]["path"])
    if sha256(path) != record["prediction"]["sha256"]:
        raise ValueError("Prediction hash differs: " + row["id"])
    arrays = load_arrays(row)
    with np.load(path, allow_pickle=False) as saved:
        expected = {"probability", "valid", "native_probability", "target", "target_valid"}
        if not expected.issubset(saved.files):
            raise ValueError("Incomplete prediction arrays")
        data = {key: saved[key] for key in expected}
    shape = tuple(row["input"]["grid"]["shape"])
    native_shape = tuple(row["target"]["grid"]["shape"])
    if any(data[k].shape != shape for k in ("probability", "valid")) or any(
            data[k].shape != native_shape for k in ("native_probability", "target", "target_valid")):
        raise ValueError("Prediction grid differs from the approved manifest")
    if not np.array_equal(data["target"], arrays["target"], equal_nan=True) or not np.array_equal(
            data["target_valid"], arrays["target_valid"]):
        raise ValueError("Saved reference differs from approved measurement support")
    valid = data["valid"].astype(bool)
    probability = data["probability"]
    if np.any(valid & ~arrays["input_valid"].astype(bool)) or not np.any(valid):
        raise ValueError("Prediction claims invalid input support")
    if not np.all(np.isfinite(probability[valid])) or np.any((probability[valid] < 0) | (probability[valid] > 1)):
        raise ValueError("Saved probability must be finite and within 0 to 1")
    # Reshape only after checking the exact grid contract. No interpolation enters scoring.
    h, w = native_shape
    fy, fx = shape[0] // h, shape[1] // w
    coarse_valid = valid.reshape(h, fy, w, fx).all(axis=(1, 3))
    coarse = np.where(valid, probability, 0).reshape(h, fy, w, fx).mean(axis=(1, 3))
    native_valid = data["target_valid"].astype(bool)
    if np.any(native_valid & ~coarse_valid) or not np.allclose(
            coarse[coarse_valid], data["native_probability"][coarse_valid], atol=2e-6, rtol=0):
        raise ValueError("Native probabilities do not aggregate from the saved fine prediction")
    measured = counts(data["native_probability"], data["target"], native_valid, threshold)
    recorded = record["native_metrics"]
    if any(recorded[key] != measured[key] for key in ("tp", "fp", "fn", "tn")):
        raise ValueError("Reported native scores differ from the saved prediction")
    measured["source_valid_pixels"] = int(coarse_valid.sum())
    measured["support_fraction"] = float(native_valid.mean())
    return arrays, data, measured


#### Display grid: exact pixels for fixtures, shared Mercator for measured scenes
def display_layers(row, arrays, data, synthetic):
    shape = tuple(row["input"]["grid"]["shape"])
    if "target" in data:
        native_shape = tuple(row["target"]["grid"]["shape"])
        fy, fx = shape[0] // native_shape[0], shape[1] // native_shape[1]
        truth = np.repeat(np.repeat(data["target"], fy, axis=0), fx, axis=1)
        known = np.repeat(np.repeat(data["target_valid"], fy, axis=0), fx, axis=1).astype(bool)
    else:
        truth = np.full(shape, np.nan, dtype=np.float32)
        known = np.zeros(shape, dtype=bool)
    rgb = arrays["image"][..., [2, 1, 0]]
    valid = data["valid"].astype(bool)
    p = np.where(valid, data["probability"], np.nan)
    truth = np.where(known & valid, truth, np.nan)
    if synthetic:
        return rgb, p, truth, valid, {"coordinateSystem": "pixel", "imageSize": list(shape)}
    import rasterio.warp as warp
    from rasterio.transform import Affine, array_bounds, from_bounds
    native = Affine(*row["input"]["grid"]["affine"])
    crs = row["input"]["grid"]["crs"]
    bounds = array_bounds(*shape, native)
    mercator = warp.transform_bounds(crs, "EPSG:3857", *bounds, densify_pts=21)
    width, height = min(shape[1], 512), min(shape[0], 512)
    dest = from_bounds(*mercator, width, height)

    def project(a, nearest=False):
        out = np.full((height, width), np.nan, dtype=np.float32)
        warp.reproject(a.astype(np.float32), out, src_transform=native, src_crs=crs,
                       dst_transform=dest, dst_crs="EPSG:3857", src_nodata=np.nan,
                       dst_nodata=np.nan, resampling=warp.Resampling.nearest if nearest else warp.Resampling.bilinear)
        return out

    rgb_out = np.stack([project(np.where(valid, rgb[..., i], np.nan)) for i in range(3)], axis=-1)
    p_out, t_out = project(p), project(truth, nearest=True)
    v_out = (project(valid.astype(np.float32), nearest=True) > .5) & np.isfinite(p_out)
    west, south, east, north = warp.transform_bounds("EPSG:3857", "EPSG:4326", *mercator)
    if not all(math.isfinite(v) for v in (west, south, east, north)) or south < -85 or north > 85:
        raise ValueError("Scene is outside the viewer map bounds")
    return rgb_out, p_out, t_out, v_out, {"bounds": [[south, west], [north, east]]}


def image_assets(folder, rgb, p, truth, valid, threshold):
    # One fixed reflectance stretch for all stages, never a prediction-specific stretch.
    rgb8 = (np.clip(np.nan_to_num(rgb) / .35, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb8).save(folder / "satellite.png")
    rgba = np.zeros((*p.shape, 4), dtype=np.uint8)
    rgba[np.isfinite(truth) & (truth == 1)] = [0, 225, 230, 180]
    Image.fromarray(rgba).save(folder / "reference.png")
    rgba[:] = 0
    rgba[valid & (p >= threshold)] = [255, 135, 35, 220]
    Image.fromarray(rgba).save(folder / "prediction.png")


def export_run(manifest_path, run_dir, stage, output_dir, bundles_dir=None):
    manifest, run_dir = load_manifest(manifest_path), Path(run_dir).resolve()
    receipt = read_json(run_dir / f"{PREFIX}_{stage}_results.json")
    if receipt["manifest_sha256"] != manifest["manifest_sha256"] or receipt["stage"] != stage:
        raise ValueError("Prediction receipt belongs to another manifest or stage")
    if receipt["data_kind"] != manifest["data_kind"]:
        raise ValueError("Prediction and manifest data kinds differ")
    checkpoint = inside(run_dir, receipt["checkpoint"]["path"])
    if sha256(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from prediction receipt")
    training = read_json(run_dir / f"{PREFIX}_{stage}_training.json")
    if training["manifest_sha256"] != manifest["manifest_sha256"] or training["checkpoint"]["sha256"] != sha256(checkpoint):
        raise ValueError("Training lineage differs from prediction receipt")
    if receipt.get("training_receipt_sha256") != sha256(run_dir / f"{PREFIX}_{stage}_training.json"):
        raise ValueError("Training receipt changed after prediction")
    for name, expected in training["source_hashes"].items():
        if sha256(HERE / name) != expected:
            raise ValueError("Source code changed after training: " + name)
    for parent, lineage in training.get("parents", {}).items():
        parent_path = run_dir / f"{PREFIX}_{parent}_training.json"
        parent_receipt = read_json(parent_path)
        if sha256(parent_path) != lineage["receipt_sha256"] or sha256(
                inside(run_dir, parent_receipt["checkpoint"]["path"])) != lineage["checkpoint_sha256"]:
            raise ValueError("Frozen parent changed after fusion training")
    rows = {row["id"]: row for row in manifest["observations"]}
    synthetic = manifest["data_kind"] == "synthetic"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scenes, totals = [], {key: 0 for key in ("tp", "fp", "fn", "tn")}
    seen = set()
    threshold = receipt["threshold"]
    for index, record in enumerate(receipt["records"]):
        row = rows[record["id"]]
        if record["id"] in seen or record["role"] != row["role"] or record["group"] != row["group"]:
            raise ValueError("Duplicate or mismatched prediction membership")
        seen.add(record["id"])
        arrays, data, native = verify_saved(row, record, run_dir, threshold)
        rgb, p, truth, valid, geometry = display_layers(row, arrays, data, synthetic)
        folder = output_dir / f"scene_{index:04d}"
        folder.mkdir()
        image_assets(folder, rgb, p, truth, valid, threshold)
        values = lambda a: [float(v) if np.isfinite(v) else None for v in a.ravel()]
        grid = {"width": p.shape[1], "height": p.shape[0], "probability": values(np.where(valid, p, np.nan)),
                "truth": values(truth), "valid": valid.astype(np.uint8).ravel().tolist()}
        if not synthetic:
            grid["crs"] = "EPSG:3857"
        write_json(folder / "grid.json", grid)
        relative = folder.name + "/"
        flags = record.get("quality_flags", [])
        note = ("SYNTHETIC ENGINEERING CHECK. No measured methane or geographic location. " if synthetic else
                "Experimental satellite-only prediction. Review before use. ")
        note += ("Cyan is the reference; orange is the model. The fine prediction is averaged to the native reference grid for scores. "
                 "Fine pixel detail is not validated upscaling. Unknown reference cells remain unscored. ")
        if flags:
            note += "Quality findings: " + ", ".join(flags) + "."
        scene = {"id": row["id"], "name": ("Synthetic | " if synthetic else "") + row["id"],
                 "task": "plume_segmentation", **geometry,
                 "date": row["input"]["acquired_at"][:10], "referenceDate": row["target"]["acquired_at"][:10],
                 "rgb": relative + "satellite.png", "truthImage": relative + "reference.png",
                 "predictionImage": relative + "prediction.png", "grid": relative + "grid.json",
                 "predictionAvailable": True, "decisionThreshold": threshold,
                 "truthLabel": "Synthetic reference" if synthetic else row["target_kind"].upper() + " measured reference",
                 "truthNote": "Engineering fixture only" if synthetic else row["target"]["source"],
                 "rgbNote": "Synthetic five-band input" if synthetic else "Sentinel-2 reflectance RGB context",
                 "notes": note, "split": row["role"] + " | group " + row["group"],
                 "source": row["input"]["sha256"], "checkpointHash": receipt["checkpoint"]["sha256"],
                 "predictionHash": record["prediction"]["sha256"], "nativeMetrics": native,
                 "resolution": "20 m prediction grid; native reference " + str(row["target"]["grid"]["shape"]),
                 "dataKind": manifest["data_kind"], "qualityFlags": flags}
        scenes.append(scene)
        for key in totals:
            totals[key] += native[key]
    if not scenes:
        raise ValueError("No completed predictions to export")
    # Keep engineering runs out of the default presentation and geographic catalog.
    identifier = run_dir.name + "_" + stage
    if not identifier.startswith("GTM_Model6"):
        identifier = "GTM_Model6_" + identifier
    status = "ENGINEERING ONLY | synthetic inputs" if synthetic else "EXPERIMENTAL | visual review required"
    description = ("Saved outputs from a local software rehearsal. These are not methane research results." if synthetic else
                   "Actual saved predictions from " + LABELS[stage] + ". Native scores cover the exported cohort.")
    experiment = {"id": identifier, "name": ("Engineering | " if synthetic else "Three CNNs | ") + LABELS[stage],
                  "lifecycle": "experimental", "priority": False, "dataKind": manifest["data_kind"],
                  "architectureFamily": PREFIX, "runGroup": run_dir.name, "stage": stage,
                  "status": status, "description": description, "notes": description,
                  "decisionThreshold": threshold, "scenes": scenes, "reportUrl": "GTM_Model6_three_cnn_report.md",
                  "run": {"state": "completed", "progress": {"completed": len(scenes), "total": len(scenes), "unit": "scenes"}}}
    iou = totals["tp"] / (totals["tp"] + totals["fp"] + totals["fn"]) if totals["tp"] + totals["fp"] + totals["fn"] else None
    experiment["nativeAggregate"] = {**totals, "scenes": len(scenes), "iou": iou,
        "precision": totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None,
        "recall": totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None}
    experiment["metricScope"] = "synthetic_engineering" if synthetic else "exported_native_test_cohort"
    report = (f"# {LABELS[stage]}\n\n{status}\n\n{description}\n\n"
              f"Manifest SHA-256: `{manifest['manifest_sha256']}`\n\n"
              f"Checkpoint SHA-256: `{receipt['checkpoint']['sha256']}`\n\n"
              f"Fixed threshold: {threshold}. Native counts: {totals}. Native IoU: {iou}.\n\n"
              "Scores are recalculated from saved native arrays. Display masks repeat native reference cells or reproject them for viewing. "
              "Fine prediction pixels are experimental. They do not prove that methane detail was recovered. "
              "This export does not establish an emission origin, emission rate, or CAFO detection accuracy.\n")
    (output_dir / experiment["reportUrl"]).write_text(report, encoding="utf-8")
    # experiment.json is the last prepared file, followed by the existing atomic publisher.
    write_json(output_dir / "experiment.json", experiment)
    load_bundle(output_dir / "experiment.json", "bundle-assets/")
    publication = publish(output_dir, Path(bundles_dir)) if bundles_dir is not None else None
    return {"experiment_id": identifier, "prepared_bundle": str(output_dir), "scenes": len(scenes),
            "data_kind": manifest["data_kind"], "native_counts": totals, "native_iou": iou,
            "publication": publication, "url": f"http://127.0.0.1:8766/?catalog=all&experiment={identifier}#compare"}


def export_inference(scene_manifest_path, run_dir, stage, prediction_dir, output_dir, bundles_dir=None):
    """Publish satellite-only predictions without inventing a methane reference."""
    manifest = load_scenes(scene_manifest_path)
    run_dir, prediction_dir = Path(run_dir).resolve(), Path(prediction_dir).resolve()
    receipt = read_json(prediction_dir / f"{PREFIX}_{stage}_inference.json")
    training_path = run_dir / f"{PREFIX}_{stage}_training.json"
    training = read_json(training_path)
    if (receipt.get("schema") != "GTM_Model6_three_cnn_inference_v1" or receipt.get("stage") != stage or
            receipt.get("scene_manifest_sha256") != manifest["manifest_sha256"] or
            receipt.get("data_kind") != manifest["data_kind"] or
            receipt.get("training_receipt_sha256") != sha256(training_path) or
            receipt.get("training_manifest_sha256") != training["manifest_sha256"] or
            receipt.get("checkpoint") != training["checkpoint"]):
        raise ValueError("Inference receipt or training lineage differs")
    if sha256(inside(run_dir, training["checkpoint"]["path"])) != training["checkpoint"]["sha256"]:
        raise ValueError("Inference checkpoint changed")
    for name, expected in training["source_hashes"].items():
        if sha256(HERE / name) != expected:
            raise ValueError("Source code changed after training: " + name)
    rows = {row["id"]: row for row in manifest["observations"]}
    records = receipt["records"]
    if len(records) != len(rows) or {r["id"] for r in records} != set(rows):
        raise ValueError("Incomplete inference cohort")
    synthetic = manifest["data_kind"] == "synthetic"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    scenes = []
    for index, record in enumerate(records):
        row = rows[record["id"]]
        path = inside(prediction_dir, record["prediction"]["path"])
        if sha256(path) != record["prediction"]["sha256"]:
            raise ValueError("Inference array changed")
        arrays = load_scene_arrays(row)
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"probability", "valid"}:
                raise ValueError("Input-only inference must not contain target arrays")
            data = {key: z[key] for key in z.files}
        valid, p = data["valid"].astype(bool), data["probability"]
        if (p.shape != arrays["input_valid"].shape or not np.array_equal(valid, arrays["input_valid"].astype(bool)) or
                not np.isfinite(p[valid]).all() or np.any((p[valid] < 0) | (p[valid] > 1))):
            raise ValueError("Invalid inference probability or support")
        rgb, p, truth, valid, geometry = display_layers(row, arrays, data, synthetic)
        folder = output_dir / f"scene_{index:04d}"
        folder.mkdir()
        image_assets(folder, rgb, p, truth, valid, receipt["threshold"])
        write_json(folder / "grid.json", {"width": p.shape[1], "height": p.shape[0],
                   "probability": [float(v) if okay else None for v, okay in zip(p.ravel(), valid.ravel())],
                   "valid": valid.astype(np.uint8).ravel().tolist()})
        relative = folder.name + "/"
        note = ("SYNTHETIC ENGINEERING CHECK. " if synthetic else "Experimental satellite-only prediction. ")
        note += "No methane reference was supplied. No accuracy or source-location claim. Fine detail remains experimental."
        scenes.append({"id": row["id"], "name": ("Synthetic | " if synthetic else "") + row["id"], **geometry,
                       "task": "plume_segmentation", "dataKind": manifest["data_kind"],
                       "date": row["input"]["acquired_at"][:10], "rgb": relative + "satellite.png",
                       "predictionImage": relative + "prediction.png", "grid": relative + "grid.json",
                       "predictionAvailable": True, "decisionThreshold": receipt["threshold"],
                       "truthLabel": "Reference unavailable", "truthNote": "Input-only inference; no measured methane reference",
                       "notes": note, "split": "Inference only; unscored", "source": row["input"]["sha256"],
                       "checkpointHash": receipt["checkpoint"]["sha256"], "predictionHash": record["prediction"]["sha256"],
                       "resolution": "20 m prediction grid; no reference grid", "qualityFlags": record["quality_flags"]})
    identifier = run_dir.name + "_" + stage + "_unlabeled"
    if not identifier.startswith("GTM_Model6"):
        identifier = "GTM_Model6_" + identifier
    experiment = {"id": identifier, "name": ("Engineering | " if synthetic else "Three CNNs | ") + LABELS[stage] + " | input only",
                  "architectureFamily": PREFIX, "runGroup": run_dir.name + "_unlabeled", "stage": stage,
                  "lifecycle": "experimental", "priority": False, "dataKind": manifest["data_kind"],
                  "status": "ENGINEERING ONLY | no reference" if synthetic else "UNSCORED | no methane reference",
                  "description": "Actual saved prediction from satellite-only input. No target was required or supplied.",
                  "notes": "No methane accuracy can be measured for these scenes.", "scenes": scenes,
                  "decisionThreshold": receipt["threshold"], "reportUrl": "GTM_Model6_three_cnn_report.md"}
    (output_dir / experiment["reportUrl"]).write_text(
        "# Input-only inference\n\n" + experiment["status"] + "\n\n" + experiment["description"] +
        "\n\nCheckpoint SHA-256: " + receipt["checkpoint"]["sha256"] + "\n\nNo reference or accuracy was invented.\n", encoding="utf-8")
    write_json(output_dir / "experiment.json", experiment)
    load_bundle(output_dir / "experiment.json", "bundle-assets/")
    publication = publish(output_dir, Path(bundles_dir)) if bundles_dir is not None else None
    return {"experiment_id": identifier, "scenes": len(scenes), "publication": publication,
            "url": f"http://127.0.0.1:8766/?catalog=all&experiment={identifier}#compare"}
