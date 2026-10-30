"""Independently replay every prepared international scene from cached TIFF bytes."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training"
OLD = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json"
PROTOCOL = ROOT / "docs/GTM_Model7_cached_diversity_preparation_20261024.md"
EXPORTER = ROOT / "tools/GTM_Model7_prepare_cached_diversity.py"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_asset(source: dict) -> Path:
    path = Path(source["path"])
    if not path.is_file() or sha(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
        raise ValueError("Source identity failed: " + str(path))
    return path


def check_new(scene: dict, dest: Path) -> dict:
    source = {item["role"]: check_asset(item) for item in scene["sources"]}
    with rasterio.open(source["target_image"]) as target, rasterio.open(source["reference_image"]) as ref, \
            rasterio.open(source["target_cloud"]) as target_cloud, rasterio.open(source["reference_cloud"]) as ref_cloud:
        for ds in (target, ref):
            if ds.shape != (200, 200) or ds.count != 12 or ds.dtypes != ("uint16",) * 12 or \
                    ds.res != (10, 10) or list(ds.descriptions) != BANDS + [b + "_bg" for b in BANDS]:
                raise ValueError("Source image contract failed: " + scene["id"])
        if target.transform != ref.transform or target.crs != ref.crs:
            raise ValueError("Image grid mismatch: " + scene["id"])
        for cloud in (target_cloud, ref_cloud):
            if cloud.shape != target.shape or cloud.transform != target.transform or cloud.crs != target.crs or \
                    cloud.count != 1 or cloud.dtypes[0] != "uint8":
                raise ValueError("Cloud grid mismatch: " + scene["id"])
        a = np.moveaxis(target.read([1, 2, 3, 4, 5, 6]), 0, -1).astype(np.float32)
        b = np.moveaxis(ref.read([1, 2, 3, 4, 5, 6]), 0, -1).astype(np.float32)
        ca, cb = target_cloud.read(1), ref_cloud.read(1)
        if not np.isin(ca, [0, 1, 2, 3, 4]).all() or not np.isin(cb, [0, 1, 2, 3, 4]).all():
            raise ValueError("Unknown cloud code: " + scene["id"])
        native = (ca == 0) & (cb == 0) & np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & \
                 (a > 0).all(-1) & (b > 0).all(-1)
        for ds, image in ((target, a), (ref, b)):
            if ds.nodata is not None:
                native &= (image != ds.nodata).all(-1)
        valid = native.reshape(100, 2, 100, 2).all((1, 3))
        if valid.mean() < .90:
            raise ValueError("Bidate support below gate: " + scene["id"])
        valid_native = np.repeat(np.repeat(valid, 2, 0), 2, 1)
        csv_row = scene["official_csv_row"]
        expected_affine = Affine(*(float(csv_row["transform_" + k]) for k in "abcdef"))
        if target.transform != expected_affine or target.crs.to_string() != csv_row["crs"]:
            raise ValueError("Official CSV/source grid mismatch: " + scene["id"])
        if ref.crs.to_string() != scene["reference_source_row"]["crs"]:
            raise ValueError("Reference pinned/source CRS mismatch: " + scene["id"])
        for image, name in ((a, "target"), (b, "reference")):
            green, nir, swir = image[:, :, 1], image[:, :, 3], image[:, :, 4]
            score = (green - swir) / np.maximum(green + swir, 1)
            snow = (green > 3000) & (nir > 1000) & (score > .4) & valid_native
            if float(snow.sum() / valid_native.sum()) > .10:
                raise ValueError("Frozen snow-like gate failed for " + name + ": " + scene["id"])
        label = np.zeros((200, 200), dtype=np.uint8)
        if scene["scene_kind"] == "PLUME":
            with rasterio.open(source["plume_mask"]) as mask:
                if mask.shape != target.shape or mask.transform != target.transform or mask.crs != target.crs or \
                        mask.count != 1 or mask.dtypes[0] != "uint8":
                    raise ValueError("Annotation grid failed: " + scene["id"])
                label = mask.read(1)
                if not np.isin(label, [0, 1]).all() or not label.any():
                    raise ValueError("Annotation content failed: " + scene["id"])
        coarse = label.reshape(100, 2, 100, 2).max((1, 3)).astype(np.float32)
        if scene["scene_kind"] == "PLUME" and not np.any((coarse == 1) & valid):
            raise ValueError("No supported annotation: " + scene["id"])
        expected = {"target_image": a.reshape(100, 2, 100, 2, 6).mean((1, 3)),
                    "reference_image": b.reshape(100, 2, 100, 2, 6).mean((1, 3)),
                    "valid": valid, "target": coarse,
                    "label_native": label, "valid_native": valid_native}
        path = dest / scene["array_file"]
        if sha(path) != scene["array_sha256"]:
            raise ValueError("Prepared array hash failed: " + scene["id"])
        with np.load(path, allow_pickle=False) as npz:
            if set(npz.files) != set(expected) or any(not np.array_equal(npz[k], value) for k, value in expected.items()):
                raise ValueError("Independent source replay differed: " + scene["id"])
        return {"id": scene["id"], "array_sha256": scene["array_sha256"],
                "supported_native_positive_pixels": int(np.count_nonzero(label & valid_native)),
                "valid_20m_cells": int(valid.sum()), "source_hashes": {k: sha(v) for k, v in source.items()}}


def run(dest: Path) -> dict:
    manifest_path = dest / "manifest.json"
    combined_path = dest / "combined_manifest.json"
    supplement = json.loads(manifest_path.read_text(encoding="utf-8"))
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    if supplement["old_manifest"]["sha256"] != sha(OLD) or len(old["scenes"]) != 565:
        raise ValueError("Frozen manifest changed")
    if combined["schema"] != "GTM_Model7_temporal_cohort_v1" or len(combined["scenes"]) != len(old["scenes"]) + len(supplement["scenes"]):
        raise ValueError("Combined manifest shape failed")
    old_records = combined["scenes"][:565]
    for before, after in zip(old["scenes"], old_records):
        restored = dict(after)
        restored["array_file"] = before["array_file"]
        if restored != before or sha(dest / after["array_file"]) != before["array_sha256"] or \
                sha(OLD.parent / before["array_file"]) != before["array_sha256"]:
            raise ValueError("Old scene or array changed: " + before["id"])
    if combined["scenes"][565:] != supplement["scenes"]:
        raise ValueError("Supplement does not match combined manifest")
    old_eval = [s for s in old["scenes"] if s["split"] in ("dev", "test")]
    eval_sites = {s["physical_location_id"] for s in old_eval}
    eval_acqs = {s[k] for s in old_eval for k in ("target_scene_id", "reference_scene_id")}
    eval_hashes = {x["sha256"] for s in old_eval for x in s["sources"]
                   if x["role"] in ("target_image", "reference_image")}
    eval_points = [(s["latitude"], s["longitude"]) for s in old_eval]
    from math import radians, sin, cos, asin, sqrt
    def km(a, b):
        y1, y2 = radians(a[0]), radians(b[0])
        q = sin((y2-y1)/2)**2 + cos(y1)*cos(y2)*sin(radians(b[1]-a[1])/2)**2
        return 12742*asin(min(1, sqrt(q)))
    records = []
    site_counts = collections.defaultdict(lambda: collections.Counter())
    for scene in supplement["scenes"]:
        if scene["split"] != "train" or scene["source_level"] != "L1C" or scene["country"] == "United States of America" or \
                scene["source_row"]["research_role"] != "development_training" or \
                scene["reference_source_row"]["research_role"] != "development_training":
            raise ValueError("Auxiliary source role failed: " + scene["id"])
        target_time = datetime.fromisoformat(scene["satellite_time"])
        ref_time = datetime.fromisoformat(scene["reference_time"])
        hours = (target_time - ref_time).total_seconds() / 3600
        if target_time.date() == ref_time.date() or not 24 <= hours <= 1440 or \
                abs(hours - scene["target_minus_reference_hours"]) > 1e-8:
            raise ValueError("Reference chronology failed: " + scene["id"])
        if scene["physical_location_id"] in eval_sites or \
                scene["target_scene_id"] in eval_acqs or scene["reference_scene_id"] in eval_acqs or \
                any(km((scene["latitude"], scene["longitude"]), p) < 25 for p in eval_points) or \
                any(x["sha256"] in eval_hashes for x in scene["sources"]
                    if x["role"] in ("target_image", "reference_image")):
            raise ValueError("Old DEV/TEST exclusion failed: " + scene["id"])
        site_counts[scene["physical_location_id"]][scene["scene_kind"]] += 1
        records.append(check_new(scene, dest))
    if len(site_counts) > 100 or any(v["PLUME"] < 1 or v["PLUME"] > 2 or v["NO_PLUME"] > 4
                                  for v in site_counts.values()):
        raise ValueError("Site quota failed")
    array_bytes = sum((dest / s["array_file"]).stat().st_size for s in combined["scenes"])
    if array_bytes >= 1 << 30:
        raise ValueError("Prepared arrays exceed 1 GiB")
    return {"schema": "GTM_Model7_cached_diversity_e2e_v1", "old_manifest_sha256": sha(OLD),
            "old_entries_and_arrays_identical": len(old_records), "new_reconstructed_scenes": len(records),
            "positive_sites": sum(v["PLUME"] > 0 for v in site_counts.values()),
            "new_scene_counts": dict(collections.Counter(s["scene_kind"] for s in supplement["scenes"])),
            "countries": dict(collections.Counter(s["country"] for s in supplement["scenes"])),
            "combined_manifest_sha256": sha(combined_path), "supplement_manifest_sha256": sha(manifest_path),
            "exporter_sha256": sha(EXPORTER), "protocol_sha256": sha(PROTOCOL),
            "all_prepared_array_bytes": array_bytes, "downloads": 0, "new_scene_replay": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    dest = args.output.resolve()
    receipt = run(dest)
    path = dest / "e2e_receipt.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != receipt:
            raise ValueError("Repeat run differs from saved receipt")
    else:
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "new_scene_replay"}, indent=2))


if __name__ == "__main__":
    main()
