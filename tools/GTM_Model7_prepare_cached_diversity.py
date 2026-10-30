"""Prepare a bounded cached international MARS L1C training supplement.

The selection and failure protocol is docs/GTM_Model7_cached_diversity_preparation_20261024.md.
No network, evaluation raster, or model output is used.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "tools/GTM_Model7_prepare_mars_repaired.py"
CLEAR_PATH = ROOT / "tools/GTM_Model7_prepare_mars_clear.py"
PROTOCOL = ROOT / "docs/GTM_Model7_cached_diversity_preparation_20261024.md"
OLD = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json"
OUT = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training"
spec = importlib.util.spec_from_file_location("mars_diversity_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)
clear_spec = importlib.util.spec_from_file_location("mars_diversity_clear", CLEAR_PATH)
clear = importlib.util.module_from_spec(clear_spec)
assert clear_spec.loader is not None
clear_spec.loader.exec_module(clear)

MAX_SITES = 100
MAX_POS = 2
MAX_NEG = 4
MAX_ARRAY_BYTES = 1 << 30
SEED = "GTM_Model7_cached_diversity_20261024_seed_1"


def stable(raw: str) -> str:
    return hashlib.sha256(f"{SEED}:{raw}".encode()).hexdigest()


def grid_key(row: dict) -> tuple:
    return row["id_location"], tuple(row[k] for k in base.GRID_FIELDS)


def roles(row: dict) -> dict:
    return {item["role"]: item for item in row["assets"]}


def asset_present(item: dict) -> bool:
    path = (base.MARS / item["path"]).resolve()
    return path.is_relative_to(base.MARS.resolve()) and path.is_file() and path.stat().st_size == item["size"]


def source_consistent(original: dict, pinned: dict) -> bool:
    expected = "PLUME" if original["isplume"] == "True" else "NO_PLUME"
    a = roles(pinned)
    return (pinned["sample_id"] == original["id_loc_image"] and
            pinned["split"] == "development_training" and
            pinned["research_role"] == "development_training" and
            pinned["sensor_family"] == "Sentinel-2" and pinned["product_level"] == "L1C" and
            pinned["country"] == original["country"] and pinned["satellite"] == original["satellite"] and
            pinned["label_state"] == expected and pinned["observability"] == original["observability"] and
            float(pinned["percentage_clear"]) == float(original["percentage_clear"]) and
            abs(float(pinned["latitude"]) - float(original["lat"])) < 1e-9 and
            abs(float(pinned["longitude"]) - float(original["lon"])) < 1e-9 and
            pinned["height"] == 200 and pinned["width"] == 200 and
            pinned["band_order"] == base.FULL_BANDS and
            pinned["pixel_truth_available"] is (expected == "PLUME") and
            a["image"]["path"] == original["s2path"] and
            a["cloud_mask"]["path"] == original["cloudmaskpath"] and
            (expected == "NO_PLUME" or a["plume_mask"]["path"] == original["plumepath"]))


def expected_affine(row: dict) -> Affine:
    return Affine(*(float(row[k]) for k in base.GRID_FIELDS[1:7]))


def verify_csv_grid(paths: dict, target: dict, reference: dict) -> None:
    for role, row in (("target_image", target), ("reference_image", reference)):
        with rasterio.open(paths[role]) as ds:
            if (ds.transform != expected_affine(row) or str(ds.crs) != row["crs"] or
                    ds.width != int(row["width"]) or ds.height != int(row["height"])):
                raise ValueError("source_csv_grid_mismatch_" + role)


def load_inventory():
    original = {}
    with (base.MARS / "train.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["country"] == "United States of America" or row["satellite"] not in ("S2A", "S2B", "S2C"):
                continue
            if row["id_loc_image"] in original:
                raise ValueError("Duplicate official train ID")
            original[row["id_loc_image"]] = row
    pinned = {}
    with (base.MARS / "paper_v3_mixed_samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in original:
                if sid in pinned:
                    raise ValueError("Duplicate pinned ID: " + sid)
                pinned[sid] = row
    if set(original) != set(pinned):
        raise ValueError("Official train and pinned international Sentinel inventories differ")
    for sid in original:
        if not source_consistent(original[sid], pinned[sid]):
            raise ValueError("Official CSV/pinned metadata disagreement: " + sid)
    return original, pinned


def old_evaluation(manifest: dict) -> dict:
    scenes = [s for s in manifest["scenes"] if s["split"] in ("dev", "test")]
    return {"scenes": scenes, "sites": {s["physical_location_id"] for s in scenes},
            "acquisitions": {s[k] for s in scenes for k in ("target_scene_id", "reference_scene_id")},
            "spectral_hashes": {x["sha256"] for s in scenes for x in s["sources"]
                                if x["role"] in ("target_image", "reference_image")},
            "points": [(s["latitude"], s["longitude"]) for s in scenes]}


def prohibited(sid: str, ref_sid: str, original: dict, pinned: dict, evaluation: dict) -> str | None:
    target = original[sid]
    ref = original[ref_sid]
    if target["id_location"] in evaluation["sites"] or ref["id_location"] in evaluation["sites"]:
        return "old_dev_test_site"
    for row in (pinned[sid], pinned[ref_sid]):
        if row["target_scene_id"] in evaluation["acquisitions"] or row["reference_scene_id"] in evaluation["acquisitions"]:
            return "old_dev_test_acquisition"
    for row in (target, ref):
        if any(base.kilometers((float(row["lat"]), float(row["lon"])), p) < 25 for p in evaluation["points"]):
            return "within_25km_old_dev_test"
    return None


def country_site_order(positive_sites: dict, original: dict) -> list[str]:
    countries = collections.defaultdict(list)
    for site, ids in positive_sites.items():
        countries[original[ids[0]]["country"]].append(site)
    for sites in countries.values():
        sites.sort(key=stable)
    order = []
    keys = sorted(countries, key=stable)
    for i in range(max(map(len, countries.values()), default=0)):
        order.extend(countries[c][i] for c in keys if i < len(countries[c]))
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Supplement destination exists: " + str(dest))
    old_sha = base.sha(OLD)
    old = json.loads(OLD.read_text(encoding="utf-8"))
    if old_sha != "925f951a610e5d61f02b76914887fa46907d3a648a227ad46f7bf0853c4ac53d" or len(old["scenes"]) != 565:
        raise ValueError("Frozen 565-scene manifest identity changed")
    original, pinned = load_inventory()
    evaluation = old_evaluation(old)
    audit = collections.Counter()
    by_grid = collections.defaultdict(list)
    pos_sites = collections.defaultdict(list)
    for sid, row in original.items():
        pin = pinned[sid]
        if pin["label_state"] not in ("PLUME", "NO_PLUME"):
            audit["metadata_other_label"] += 1
            continue
        if row["observability"] != "clear" or float(row["percentage_clear"]) < 90:
            audit["metadata_not_clear_90"] += 1
            continue
        a = roles(pin)
        required = ["image", "cloud_mask"] + (["plume_mask"] if pin["label_state"] == "PLUME" else [])
        if not all(asset_present(a[k]) for k in required):
            audit["metadata_uncached_or_bad_size"] += 1
            continue
        if pin["label_state"] == "NO_PLUME":
            by_grid[grid_key(row)].append(sid)
        else:
            pos_sites[row["id_location"]].append(sid)
        audit["metadata_eligible_" + pin["label_state"]] += 1
    for ids in by_grid.values():
        ids.sort(key=lambda sid: (pinned[sid]["target_datetime"], sid))
    for ids in pos_sites.values():
        ids.sort(key=stable)
    dest.mkdir(parents=True)
    (dest / "arrays" / "new").mkdir(parents=True)
    attempts_file = (dest / "reference_attempts.jsonl").open("w", encoding="utf-8")
    new_scenes = []
    accepted_sites = []
    arrays_bytes = 0
    stop = False

    def export(sid: str, scene_kind: str) -> bool:
        nonlocal arrays_bytes, stop
        target_row = original[sid]
        target_pin = pinned[sid]
        target_time = datetime.fromisoformat(target_pin["target_datetime"])
        refs = []
        for rid in by_grid[grid_key(target_row)]:
            if rid == sid:
                continue
            ref_time = datetime.fromisoformat(pinned[rid]["target_datetime"])
            hours = (target_time - ref_time).total_seconds() / 3600
            if 24 <= hours <= 1440 and target_time.date() != ref_time.date():
                refs.append((hours, rid))
        refs.sort()
        if not refs:
            audit["no_earlier_clear_same_grid_reference"] += 1
            return False
        for hours, rid in refs:
            reason = prohibited(sid, rid, original, pinned, evaluation)
            payload = None
            sources = []
            fraction = None
            snow_target = snow_reference = None
            if reason is None:
                try:
                    payload, sources, reason, fraction = base.raster_pair(sid, rid, pinned, original,
                                                                           {k: roles(v) for k, v in ((sid, target_pin), (rid, pinned[rid]))})
                    if reason is None:
                        arrays, grids = payload
                        paths = {s["role"]: Path(s["path"]) for s in sources}
                        verify_csv_grid(paths, target_row, original[rid])
                        snow_target = clear.snow_fraction(paths["target_image"], arrays["valid_native"])
                        snow_reference = clear.snow_fraction(paths["reference_image"], arrays["valid_native"])
                        if max(snow_target, snow_reference) > clear.GATE["date_fraction_gt"]:
                            reason = "snow_like_fraction_gt_10pct_target_or_reference"
                        elif any(s["sha256"] in evaluation["spectral_hashes"] for s in sources
                                 if s["role"] in ("target_image", "reference_image")):
                            reason = "old_dev_test_spectral_hash"
                except (ValueError, KeyError, rasterio.errors.RasterioError) as exc:
                    reason = "source_or_raster_contract:" + str(exc)
            attempt = {"sample_id": sid, "reference_observation_id": rid, "country": target_row["country"],
                       "physical_location_id": target_row["id_location"], "scene_kind": scene_kind,
                       "target_time": target_pin["target_datetime"],
                       "reference_time": pinned[rid]["target_datetime"], "offset_hours": hours,
                       "bidate_valid_fraction": fraction, "target_snow_like_fraction": snow_target,
                       "reference_snow_like_fraction": snow_reference, "reason": reason or "accepted"}
            attempts_file.write(json.dumps(attempt, allow_nan=False) + "\n")
            audit["reference_attempt_" + (reason or "accepted").split(":")[0]] += 1
            if reason:
                continue
            arrays, (coarse_grid, native_grid) = payload
            path = dest / "arrays" / "new" / f"MARS_{sid}.npz"
            np.savez_compressed(path, **arrays)
            size = path.stat().st_size
            if arrays_bytes + size > MAX_ARRAY_BYTES:
                path.unlink()
                audit["array_budget_stop"] += 1
                stop = True
                return False
            arrays_bytes += size
            scene = {"id": f"MARS_{sid}", "name": f"{scene_kind} | {target_pin['target_datetime'][:10]} | {sid[:8]}",
                     "split": "train", "group_id": "international_site_" + target_row["id_location"],
                     "physical_location_id": target_row["id_location"], "scene_kind": scene_kind,
                     "source_level": "L1C", "country": target_row["country"], "sensor_family": "Sentinel-2",
                     "original_release_role": "development_training", "latitude": float(target_row["lat"]),
                     "longitude": float(target_row["lon"]), "target_scene_id": target_pin["target_scene_id"],
                     "reference_scene_id": pinned[rid]["target_scene_id"], "reference_observation_id": rid,
                     "original_release_reference_scene_id": target_pin["reference_scene_id"],
                     "satellite_id": target_pin["target_scene_id"], "satellite_time": target_pin["target_datetime"],
                     "reference_time": pinned[rid]["target_datetime"], "target_minus_reference_hours": hours,
                     "input_grid": coarse_grid, "reference_grid": native_grid, "reference_input_grid": coarse_grid,
                     "clear_fraction": fraction, "target_snow_like_fraction": snow_target,
                     "reference_snow_like_fraction": snow_reference,
                     "array_file": str(path.relative_to(dest)).replace("\\", "/"), "array_sha256": base.sha(path),
                     "sources": sources, "source_row": {k: v for k, v in target_pin.items() if k != "paper_baseline"},
                     "official_csv_row": target_row,
                     "reference_source_row": {k: v for k, v in pinned[rid].items() if k != "paper_baseline"},
                     "source_label": scene_kind, "reference_mapping": "repeat_2",
                     "reference_kind": "reviewed_binary_annotation",
                     "notes": "Official MARS TRAIN international auxiliary example; binary producer label and earlier same-site reviewed no-plume reference. External-country training does not establish US transfer."}
            new_scenes.append(scene)
            return True
        audit["all_reference_candidates_failed_" + scene_kind] += 1
        return False

    try:
        for site in country_site_order(pos_sites, original):
            if stop or len(accepted_sites) >= MAX_SITES:
                break
            before = len(new_scenes)
            for sid in pos_sites[site]:
                if sum(s["physical_location_id"] == site and s["scene_kind"] == "PLUME" for s in new_scenes) >= MAX_POS:
                    break
                export(sid, "PLUME")
                if stop:
                    break
            if len(new_scenes) == before:
                audit["positive_site_without_passing_pair"] += 1
                continue
            accepted_sites.append(site)
            negatives = [sid for sid, row in original.items() if row["id_location"] == site and
                         pinned[sid]["label_state"] == "NO_PLUME" and
                         row["observability"] == "clear" and float(row["percentage_clear"]) >= 90]
            negatives.sort(key=stable)
            count = 0
            for sid in negatives:
                if stop or count >= MAX_NEG:
                    break
                if export(sid, "NO_PLUME"):
                    count += 1
    finally:
        attempts_file.close()
    if not new_scenes:
        raise ValueError("No cached international scenes passed the frozen contract")
    if base.sha(OLD) != old_sha:
        raise ValueError("Old manifest changed during preparation")
    old_dir = dest / "arrays" / "old"
    old_dir.mkdir()
    preserved = []
    for scene in old["scenes"]:
        source = OLD.parent / scene["array_file"]
        if base.sha(source) != scene["array_sha256"]:
            raise ValueError("Old array changed: " + scene["id"])
        target = old_dir / source.name
        shutil.copyfile(source, target)
        if base.sha(target) != scene["array_sha256"]:
            raise ValueError("Copied old array differs: " + scene["id"])
        copy = dict(scene)
        copy["array_file"] = str(target.relative_to(dest)).replace("\\", "/")
        preserved.append(copy)
    combined = dict(old)
    combined["role"] = "frozen_mars_clear_cohort_plus_official_train_international_auxiliary"
    combined["old_manifest"] = base.file_source(OLD)
    combined["auxiliary_manifest"] = "manifest.json"
    combined["quality_limitations"] = old["quality_limitations"] + [
        "International official TRAIN scenes are auxiliary examples for a US task, not evidence of US transfer.",
        "Previously evaluated US DEV/TEST remain regression checks with historical project exposure."]
    combined["scenes"] = preserved + new_scenes
    combined["split_counts"] = {split: {label: sum(s["split"] == split and s["scene_kind"] == label
                                              for s in combined["scenes"]) for label in ("PLUME", "NO_PLUME")}
                                for split in ("train", "dev", "test")}
    base.check_split(combined["scenes"])
    supplement = {"schema": "GTM_Model7_cached_diversity_v1", "downloads": 0,
                  "selection_seed": SEED, "protocol": base.file_source(PROTOCOL),
                  "exporter": base.file_source(Path(__file__)), "base_exporter": base.file_source(BASE_PATH),
                  "clear_gate_source": base.file_source(CLEAR_PATH), "old_manifest": base.file_source(OLD),
                  "source_csv": base.file_source(base.MARS / "train.csv"),
                  "source_manifest": base.file_source(base.MARS / "paper_v3_mixed_samples.jsonl"),
                  "source_pool": "official TRAIN, non-US Sentinel-2 L1C only",
                  "metadata_audit": dict(audit), "accepted_positive_sites": len(accepted_sites),
                  "country_counts": dict(collections.Counter(s["country"] for s in new_scenes)),
                  "positive_site_country_counts": dict(collections.Counter(
                      next(s["country"] for s in new_scenes if s["physical_location_id"] == site) for site in accepted_sites)),
                  "scene_counts": dict(collections.Counter(s["scene_kind"] for s in new_scenes)),
                  "array_bytes": arrays_bytes, "old_array_bytes_copied": sum((old_dir / Path(s["array_file"]).name).stat().st_size for s in old["scenes"]),
                  "reference_attempt_log": "reference_attempts.jsonl", "scenes": new_scenes,
                  "human_limitations": ["Producer binary plume and no-plume labels are accepted as supplied; no independent human re-review was performed.",
                                        "Exact-grid matching and image-only diagnostics do not certify landmark registration, illumination, or methane causality."]}
    base.save(dest / "manifest.json", supplement)
    base.save(dest / "combined_manifest.json", combined)
    receipt = {"schema": "GTM_Model7_cached_diversity_preparation_receipt_v1",
               "supplement_manifest": base.file_source(dest / "manifest.json"),
               "combined_manifest": base.file_source(dest / "combined_manifest.json"),
               "reference_attempt_log": base.file_source(dest / "reference_attempts.jsonl"),
               "old_manifest_sha256_unchanged": base.sha(OLD) == old_sha,
               "old_scene_count": len(preserved), "new_scene_count": len(new_scenes),
               "new_positive_site_count": len(accepted_sites), "new_array_bytes": arrays_bytes,
               "all_array_bytes": arrays_bytes + supplement["old_array_bytes_copied"],
               "split_counts": combined["split_counts"], "metadata_audit": dict(audit), "downloads": 0}
    base.save(dest / "receipt.json", receipt)
    print(json.dumps({"destination": str(dest), "old_scenes": len(preserved), "new_scenes": len(new_scenes),
                      "positive_sites": len(accepted_sites), "countries": supplement["positive_site_country_counts"],
                      "bytes": receipt["all_array_bytes"], "counts": combined["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
