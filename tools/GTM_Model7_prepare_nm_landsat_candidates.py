"""Audit two pinned New Mexico Landsat pairs without merging them into Sentinel training.

The source and exclusion protocol was fixed first in
docs/GTM_Model7_landsat_candidate_preparation_20261024.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
MARS = ROOT.parent / ("Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/"
                      "Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L")
OUT = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/nm_landsat_candidates"
OLD = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json"
PROTOCOL = ROOT / "docs/GTM_Model7_landsat_candidate_preparation_20261024.md"
PAIRS = (("0f41981f-2751-4bba-ab02-3035ef8df2c8", "204a3ea0-15ec-45c1-8436-ba2cada2708b"),
         ("18e171df-18a2-4316-aee8-8064a2221042", "fb5870e5-7e4e-470e-89b9-c535ed9ab27c"))
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
GRID_FIELDS = ["crs", "transform_a", "transform_b", "transform_c", "transform_d",
               "transform_e", "transform_f", "width", "height"]
SNOW_GATE = {"green_b03_gt_dn": 3000, "nir_b05_gt_dn": 1000,
             "green_minus_swir_b06_over_sum_gt": .4, "date_fraction_gt": .10}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def km(a: tuple, b: tuple) -> float:
    y1, y2 = math.radians(a[0]), math.radians(b[0])
    z = math.sin((y2-y1)/2)**2 + math.cos(y1)*math.cos(y2)*math.sin(math.radians(b[1]-a[1])/2)**2
    return 12742*math.asin(min(1, math.sqrt(z)))


def load_rows():
    ids = {x for pair in PAIRS for x in pair}
    official = {}
    with (MARS / "train.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sid = row["id_loc_image"]
            if sid in ids:
                if sid in official:
                    raise ValueError("Duplicate official TRAIN ID: " + sid)
                official[sid] = row
    pinned = {}
    with (MARS / "paper_v3_mixed_samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in ids:
                if sid in pinned:
                    raise ValueError("Duplicate pinned mixed-manifest ID: " + sid)
                pinned[sid] = row
    if set(official) != ids or set(pinned) != ids:
        raise ValueError("Pinned ID missing from official TRAIN CSV or mixed manifest")
    for sid in ids:
        row, pin = official[sid], pinned[sid]
        assets = {a["role"]: a for a in pin["assets"]}
        expected = "PLUME" if row["isplume"] == "True" else "NO_PLUME"
        if (row["split_name"] != "train_2023" or row["country"] != "United States of America" or
                row["satellite"] not in ("LC08", "LC09") or
                pin["split"] != "development_training" or pin["research_role"] != "development_training" or
                pin["sensor_family"] != "Landsat" or pin["product_level"] != "Collection 2 TOA" or
                pin["country"] != row["country"] or pin["satellite"] != row["satellite"] or
                pin["physical_location_id"] != "PB_30" or pin["label_state"] != expected or
                pin["observability"] != row["observability"] or row["observability"] != "clear" or
                float(pin["percentage_clear"]) != float(row["percentage_clear"]) or
                float(row["percentage_clear"]) < 90 or
                abs(float(pin["latitude"]) - float(row["lat"])) > 1e-9 or
                abs(float(pin["longitude"]) - float(row["lon"])) > 1e-9 or
                pin["height"] != 200 or pin["width"] != 200 or
                pin["band_order"] != BANDS + [f"{b}_bg" for b in BANDS] or
                pin["pixel_truth_available"] is not (expected == "PLUME") or
                assets["image"]["path"] != row["s2path"] or
                assets["cloud_mask"]["path"] != row["cloudmaskpath"] or
                (expected == "PLUME" and assets["plume_mask"]["path"] != row["plumepath"])):
            raise ValueError("Official TRAIN / pinned metadata disagreement: " + sid)
    return official, pinned


def checked_asset(asset: dict, role: str) -> dict:
    path = (MARS / asset["path"]).resolve()
    if not path.is_relative_to(MARS.resolve()) or not path.is_file() or path.stat().st_size != asset["size"]:
        raise ValueError("Missing, escaped, or resized source: " + str(path))
    h = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f"blob {asset['size']}\0".encode())
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
            git.update(part)
    digest = h.hexdigest()
    upstream = digest if asset["remote_oid_type"] == "sha256_lfs" else git.hexdigest() \
        if asset["remote_oid_type"] == "git_blob_sha1" else None
    if upstream != asset["remote_oid"]:
        raise ValueError("Pinned upstream identity changed: " + str(path))
    return {"role": role, "path": str(path), "sha256": digest, "bytes": asset["size"],
            "source_url": asset["source_url"], "remote_oid": asset["remote_oid"],
            "remote_oid_type": asset["remote_oid_type"], "upstream_identity_verified": True}


def affine(row: dict) -> Affine:
    return Affine(*(float(row["transform_" + letter]) for letter in "abcdef"))


def read_pair(sid: str, rid: str, official: dict, pinned: dict) -> tuple[dict, list[dict], dict]:
    row, ref_row = official[sid], official[rid]
    pin, ref_pin = pinned[sid], pinned[rid]
    if row["id_location"] != ref_row["id_location"] or tuple(row[x] for x in GRID_FIELDS) != \
            tuple(ref_row[x] for x in GRID_FIELDS):
        raise ValueError("Official same-site exact-grid rule failed")
    a, b = ({x["role"]: x for x in p["assets"]} for p in (pin, ref_pin))
    sources = [checked_asset(a["image"], "target_image"), checked_asset(a["cloud_mask"], "target_cloud"),
               checked_asset(a["plume_mask"], "plume_mask"), checked_asset(b["image"], "reference_image"),
               checked_asset(b["cloud_mask"], "reference_cloud")]
    paths = {x["role"]: x["path"] for x in sources}
    with rasterio.open(paths["target_image"]) as t, rasterio.open(paths["reference_image"]) as r, \
            rasterio.open(paths["target_cloud"]) as tc, rasterio.open(paths["reference_cloud"]) as rc, \
            rasterio.open(paths["plume_mask"]) as ld:
        for image, csv_row in ((t, row), (r, ref_row)):
            if (image.shape != (200, 200) or image.count != 12 or image.res != (10, 10) or
                    image.dtypes != ("uint16",)*12 or image.transform != affine(csv_row) or
                    str(image.crs) != csv_row["crs"] or list(image.descriptions) != BANDS + [f"{b}_bg" for b in BANDS]):
                raise ValueError("Landsat delivered image contract failed")
        if t.transform != r.transform or t.crs != r.crs:
            raise ValueError("Target/reference grid differs")
        for ds in (tc, rc, ld):
            if ds.shape != t.shape or ds.transform != t.transform or ds.crs != t.crs or ds.count != 1 or ds.dtypes[0] != "uint8":
                raise ValueError("Cloud or annotation grid/dtype failed")
        target = np.moveaxis(t.read([1, 2, 3, 4, 5, 6]), 0, -1).astype(np.float32)
        reference = np.moveaxis(r.read([1, 2, 3, 4, 5, 6]), 0, -1).astype(np.float32)
        cloud_t, cloud_r, label = tc.read(1), rc.read(1), ld.read(1)
        if not np.isin(cloud_t, [0, 1, 2, 3, 4]).all() or not np.isin(cloud_r, [0, 1, 2, 3, 4]).all():
            raise ValueError("Unknown cloud code")
        if not np.isin(label, [0, 1]).all() or not label.any():
            raise ValueError("Producer positive annotation invalid")
        valid = (cloud_t == 0) & (cloud_r == 0) & np.isfinite(target).all(-1) & \
                np.isfinite(reference).all(-1) & (target > 0).all(-1) & (reference > 0).all(-1)
        for ds, image in ((t, target), (r, reference)):
            if ds.nodata is not None:
                valid &= (image != ds.nodata).all(-1)
        fraction = float(valid.mean())
        if fraction < .90 or not np.any(label & valid):
            raise ValueError("Less than 90% bidate support or no supported positive")
        snow = {}
        for name, image in (("target", target), ("reference", reference)):
            green, nir, swir = image[:, :, 1], image[:, :, 3], image[:, :, 4]
            score = (green - swir) / np.maximum(green + swir, 1)
            snow[name] = float(((green > 3000) & (nir > 1000) & (score > .4) & valid).sum() / valid.sum())
            if snow[name] > .10:
                raise ValueError("Frozen snow-like gate failed: " + name)
        arrays = {"target_image": target, "reference_image": reference,
                  "label_native": label, "valid_native": valid}
        qa = {"delivered_grid": {"shape": [200, 200], "affine": list(t.transform)[:6], "crs": str(t.crs),
                                 "delivered_pixel_size_m": 10},
              "native_sensor_resolution_m": 30, "valid_fraction": fraction,
              "supported_positive_pixels": int(np.count_nonzero(label & valid)),
              "cloud_code_counts": {"target": dict(zip(*[list(map(int, x)) for x in np.unique(cloud_t, return_counts=True)])),
                                    "reference": dict(zip(*[list(map(int, x)) for x in np.unique(cloud_r, return_counts=True)]))},
              "snow_like_fraction": snow}
        return arrays, sources, qa


def exclusion(sid: str, rid: str, official: dict, pinned: dict, old: dict, sources: list[dict]) -> dict:
    eval_scenes = [s for s in old["scenes"] if s["split"] in ("dev", "test")]
    point = (float(official[sid]["lat"]), float(official[sid]["lon"]))
    nearest = min(((km(point, (s["latitude"], s["longitude"])), s["id"], s["split"])
                   for s in eval_scenes), key=lambda x: x[0])
    sites = {s["physical_location_id"] for s in eval_scenes}
    acqs = {s[k] for s in eval_scenes for k in ("target_scene_id", "reference_scene_id")}
    hashes = {x["sha256"] for s in eval_scenes for x in s["sources"]
              if x["role"] in ("target_image", "reference_image")}
    reasons = []
    if official[sid]["id_location"] in sites or official[rid]["id_location"] in sites or \
            pinned[sid]["physical_location_id"] in sites or pinned[rid]["physical_location_id"] in sites:
        reasons.append("old_dev_test_site")
    for pin in (pinned[sid], pinned[rid]):
        if pin["target_scene_id"] in acqs or pin["reference_scene_id"] in acqs:
            reasons.append("old_dev_test_acquisition")
    if any(s["sha256"] in hashes for s in sources if s["role"] in ("target_image", "reference_image")):
        reasons.append("old_dev_test_spectral_hash")
    if nearest[0] < 25:
        reasons.append("within_25km_old_dev_test")
    return {"eligible_for_training_export": not reasons, "reasons": sorted(set(reasons)),
            "nearest_old_dev_test_km": nearest[0], "nearest_old_dev_test_scene_id": nearest[1],
            "nearest_old_dev_test_split": nearest[2], "distance_threshold_km": 25}


def previews(dest: Path, sid: str, arrays: dict) -> dict:
    folder = dest / "previews"
    folder.mkdir(exist_ok=True)
    t, r = arrays["target_image"], arrays["reference_image"]
    # Shared percentile bounds preserve target/reference radiometric comparisons.
    colors = [2, 1, 0]
    low = np.percentile(np.concatenate((t[:, :, colors], r[:, :, colors]), axis=0), 2, axis=(0, 1))
    high = np.percentile(np.concatenate((t[:, :, colors], r[:, :, colors]), axis=0), 98, axis=(0, 1))
    result = {}
    for name, image in (("target", t), ("reference", r)):
        rgb = np.clip((image[:, :, colors] - low) / np.maximum(high-low, 1), 0, 1)
        path = folder / f"{sid}_{name}.png"
        Image.fromarray((rgb * 255).astype(np.uint8), "RGB").save(path)
        result[name] = entry(path)
    label = arrays["label_native"]
    support = arrays["valid_native"]
    display = np.zeros((200, 200, 3), dtype=np.uint8)
    display[support] = (35, 45, 52)
    display[label.astype(bool) & support] = (255, 115, 62)
    display[label.astype(bool) & ~support] = (150, 55, 45)
    path = folder / f"{sid}_annotation_support.png"
    Image.fromarray(display, "RGB").save(path)
    result["annotation_support"] = entry(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Candidate destination must be new: " + str(dest))
    old_hash = sha(OLD)
    old = json.loads(OLD.read_text(encoding="utf-8"))
    if old_hash != "925f951a610e5d61f02b76914887fa46907d3a648a227ad46f7bf0853c4ac53d" or len(old["scenes"]) != 565:
        raise ValueError("Frozen old manifest changed")
    official, pinned = load_rows()
    dest.mkdir(parents=True)
    candidates = []
    for sid, rid in PAIRS:
        target, ref = pinned[sid], pinned[rid]
        a = datetime.fromisoformat(target["target_datetime"])
        b = datetime.fromisoformat(ref["target_datetime"])
        hours = (a-b).total_seconds()/3600
        if (target["label_state"] != "PLUME" or ref["label_state"] != "NO_PLUME" or
                official[sid]["id_location"] != official[rid]["id_location"] or
                a.date() == b.date() or not 24 <= hours <= 1440):
            raise ValueError("Candidate labels, site or chronology failed: " + sid)
        arrays, sources, qa = read_pair(sid, rid, official, pinned)
        block = exclusion(sid, rid, official, pinned, old, sources)
        array = None
        if block["eligible_for_training_export"]:
            folder = dest / "arrays"
            folder.mkdir(exist_ok=True)
            path = folder / f"{sid}.npz"
            np.savez_compressed(path, **arrays)
            array = entry(path)
        preview = previews(dest, sid, arrays)
        candidates.append({"target_observation_id": sid, "reference_observation_id": rid,
                           "physical_location_id": official[sid]["id_location"],
                           "producer_location_id": target["physical_location_id"],
                           "country": official[sid]["country"], "sensor_family": "Landsat",
                           "satellite": official[sid]["satellite"], "product_level": target["product_level"],
                           "band_order": BANDS, "target_time": target["target_datetime"],
                           "reference_time": ref["target_datetime"], "offset_hours": hours,
                           "original_target_release_reference_scene_id": target["reference_scene_id"],
                           "target_scene_id": target["target_scene_id"], "reference_scene_id": ref["target_scene_id"],
                           "target_official_train_row": official[sid], "reference_official_train_row": official[rid],
                           "target_pinned_row": {k: v for k, v in target.items() if k != "paper_baseline"},
                           "reference_pinned_row": {k: v for k, v in ref.items() if k != "paper_baseline"},
                           "sources": sources, "raster_quality": qa, "old_evaluation_exclusion": block,
                           "training_array": array, "previews": preview,
                           "historical_project_exposure": "unknown"})
    if sha(OLD) != old_hash:
        raise ValueError("Old manifest changed during candidate audit")
    manifest = {"schema": "GTM_Model7_landsat_candidate_v1", "downloads": 0,
                "role": "separate_source_diagnostic_not_sentinel_training",
                "old_mars_manifest": entry(OLD), "source_csv": entry(MARS / "train.csv"),
                "pinned_source_manifest": entry(MARS / "paper_v3_mixed_samples.jsonl"),
                "protocol": entry(PROTOCOL), "exporter": entry(Path(__file__)),
                "native_landsat_resolution_m": 30, "delivered_raster_grid_m": 10,
                "frozen_snow_like_gate": SNOW_GATE,
                "producer_label_limitations": "Binary source plume/no-plume classes accepted as supplied; no independent human relabeling. Two observations share one Permian site and cannot establish independent regional transfer.",
                "candidates": candidates,
                "summary": {"official_train_plume_observations": len(candidates),
                            "physical_sites": len({c["physical_location_id"] for c in candidates}),
                            "qa_passed": len(candidates),
                            "blocked_from_fitting": sum(not c["old_evaluation_exclusion"]["eligible_for_training_export"] for c in candidates),
                            "training_arrays": sum(c["training_array"] is not None for c in candidates)}}
    save(dest / "manifest.json", manifest)
    receipt = {"schema": "GTM_Model7_landsat_candidate_preparation_receipt_v1",
               "manifest": entry(dest / "manifest.json"), "protocol": entry(PROTOCOL),
               "old_manifest_sha256_unchanged": sha(OLD) == old_hash,
               "candidate_outcomes": [{"id": c["target_observation_id"],
                                        "valid_fraction": c["raster_quality"]["valid_fraction"],
                                        "nearest_old_dev_test_km": c["old_evaluation_exclusion"]["nearest_old_dev_test_km"],
                                        "excluded": not c["old_evaluation_exclusion"]["eligible_for_training_export"]}
                                       for c in candidates], "summary": manifest["summary"], "downloads": 0}
    save(dest / "receipt.json", receipt)
    print(json.dumps({"output": str(dest), "summary": manifest["summary"],
                      "candidate_outcomes": receipt["candidate_outcomes"]}, indent=2))


if __name__ == "__main__":
    main()
