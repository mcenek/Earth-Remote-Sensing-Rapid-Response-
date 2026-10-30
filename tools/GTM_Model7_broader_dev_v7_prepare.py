"""Freeze zero fresh TEST availability, then assemble a disclosed broader DEV from cached evaluated scenes."""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
from pathlib import Path

from GTM_Model7_broader_dev_v7_inventory import ROOT, OUT, TRAIN, ext

SOURCES = [
    (ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json", {"dev", "test"}),
    (ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/external_data/manifest.json", {"test"}),
    (ROOT / "outputs/GTM_Model7_unet_recall_20261024/fresh_data/manifest.json", {"test"}),
]
PROTOCOL = ROOT / "docs/GTM_Model7_broader_dev_v7_protocol_20261024.md"


def stable(value):
    return hashlib.sha256(("broader-dev-v7-20261024:" + value).encode()).hexdigest()


def save_new(path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ids(row):
    return {row["id"].removeprefix("MARS_"), row.get("reference_observation_id")}


def acqs(row):
    return {row.get("target_scene_id"), row.get("reference_scene_id")}


def spectral(row):
    return {s["sha256"] for s in row.get("sources", []) if s.get("role") in ("target_image", "reference_image")}


def freeze():
    inventory_path = OUT / "inventory.json"
    inv = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inv["fresh_site_count"] or inv["fresh_positive_count"]:
        raise ValueError("New fresh sites exist; reserve them before DEV selection")
    test = {"schema": "GTM_Model7_broader_dev_v7_test_reservation_v1", "metadata_only": True,
            "source_pixels_read": False, "prediction_results_read": False, "downloads": 0,
            "inventory": {"path": str(inventory_path), "sha256": ext.base.sha(inventory_path)},
            "reserved_sites": [], "reserved_positive": [], "reserved_no_plume": [],
            "reason": "No certifiable fresh positive TEST site: 35 of 43 candidate sites occur in fitted/evaluated cohort manifests; seven others are within 25 km of those sites; the remaining Libya site shares a target acquisition with consumed fresh-data evaluation. Preparation-only manifest membership was not counted as model exposure."}
    test_path = OUT / "test_reservation.json"
    if test_path.exists():
        if json.loads(test_path.read_text(encoding="utf-8")) != test:
            raise ValueError("Existing TEST reservation differs")
    else:
        save_new(test_path, test)
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    train_rows = [r for r in train["scenes"] if r["split"] == "train"]
    train_sites = {r["physical_location_id"] for r in train_rows}
    train_points = [(r["latitude"], r["longitude"]) for r in train_rows]
    train_ids = set().union(*(ids(r) for r in train_rows))
    train_acqs = set().union(*(acqs(r) for r in train_rows))
    train_hashes = set().union(*(spectral(r) for r in train_rows))
    source_receipts, pool, rejects = [], [], collections.Counter()
    for path, splits in SOURCES:
        obj = json.loads(path.read_text(encoding="utf-8"))
        source_receipts.append({"path": str(path.relative_to(ROOT)), "sha256": ext.base.sha(path), "splits": sorted(splits)})
        for row in obj["scenes"]:
            if row.get("split") not in splits or row.get("scene_kind") not in ("PLUME", "NO_PLUME"):
                continue
            site = row["physical_location_id"]
            point = (float(row["latitude"]), float(row["longitude"]))
            if site in train_sites or any(ext.base.kilometers(point,p) < 25 for p in train_points):
                rejects["train_site_or_25km"] += 1
                continue
            if ids(row) & train_ids or acqs(row) & train_acqs or spectral(row) & train_hashes:
                rejects["train_observation_acquisition_or_hash"] += 1
                continue
            if row.get("source_level") != "L1C" or row.get("sensor_family", "Sentinel-2") != "Sentinel-2":
                rejects["source_level"] += 1
                continue
            source_array = path.parent / row["array_file"]
            if not source_array.is_file() or ext.base.sha(source_array) != row["array_sha256"]:
                rejects["source_array_missing_or_hash"] += 1
                continue
            pool.append({"source_manifest": str(path.relative_to(ROOT)), "source_split": row["split"],
                         "id": row["id"], "reference_id": row["reference_observation_id"],
                         "source_array": str(source_array), "array_sha256": row["array_sha256"],
                         "scene_kind": row["scene_kind"], "site": site,
                         "country": row.get("country") or row.get("official_csv_row", {}).get("country") or row.get("source_row", {}).get("country") or "Unknown",
                         "latitude": point[0], "longitude": point[1],
                         "target_acquisition": row["target_scene_id"], "reference_acquisition": row["reference_scene_id"],
                         "spectral_hashes": sorted(spectral(row))})
    # Deduplicate rows by identity and select sites in country round robin. Positive-first within each site.
    by_site = collections.defaultdict(list)
    for record in pool:
        by_site[record["site"]].append(record)
    countries = collections.defaultdict(list)
    for site, rows in by_site.items():
        countries[rows[0]["country"]].append(site)
    for values in countries.values():
        values.sort(key=stable)
    order = [countries[c][i] for i in range(max(map(len,countries.values()),default=0))
             for c in sorted(countries,key=stable) if i < len(countries[c])]
    selected, selected_ids, selected_acqs, selected_hashes, selected_points = [], set(), set(), set(), []
    for site in order:
        rows = sorted(by_site[site], key=lambda r: (r["scene_kind"] != "PLUME", stable(r["id"])))
        counts = collections.Counter()
        prior_site_acqs = selected_acqs.copy()
        prior_site_hashes = selected_hashes.copy()
        point = (rows[0]["latitude"], rows[0]["longitude"])
        if any(ext.base.kilometers(point,p) < 25 for p in selected_points):
            rejects["selected_site_25km"] += len(rows)
            continue
        picked = False
        for rec in rows:
            kind = rec["scene_kind"]
            if counts[kind] >= 4:
                rejects["per_site_quota"] += 1
                continue
            rec_ids = {rec["id"].removeprefix("MARS_"), rec["reference_id"]}
            rec_acqs = {rec["target_acquisition"], rec["reference_acquisition"]}
            rec_hashes = set(rec["spectral_hashes"])
            # A same-site reference may be reused; target and acquisition sharing is otherwise expected within site.
            if rec["id"].removeprefix("MARS_") in selected_ids:
                rejects["duplicate_target"] += 1
                continue
            if rec_acqs & prior_site_acqs or rec_hashes & prior_site_hashes:
                rejects["cross_site_acquisition_or_hash"] += 1
                continue
            selected.append(rec)
            counts[kind] += 1
            selected_ids.update(rec_ids)
            selected_acqs.update(rec_acqs)
            selected_hashes.update(rec_hashes)
            picked = True
        if picked:
            selected_points.append(point)
    if not any(x["scene_kind"] == "PLUME" for x in selected):
        raise ValueError("No positive DEV rows")
    freeze_obj = {"schema": "GTM_Model7_broader_dev_v7_frozen_selection_v1", "metadata_only": True,
                  "source_pixels_read": False, "downloads": 0, "protocol_sha256": ext.base.sha(PROTOCOL),
                  "train_manifest_sha256": ext.base.sha(TRAIN), "source_manifests": source_receipts,
                  "test_reservation_sha256": ext.base.sha(OUT / "test_reservation.json"),
                  "candidate_count": len(pool), "selected_count": len(selected),
                  "selected_positive": sum(x["scene_kind"] == "PLUME" for x in selected),
                  "selected_no_plume": sum(x["scene_kind"] == "NO_PLUME" for x in selected),
                  "selected_sites": len({x["site"] for x in selected}),
                  "selected_countries": len({x["country"] for x in selected}),
                  "rejections": dict(rejects), "historical_exposure": "Every DEV row has already appeared in a saved evaluation manifest; no pristine DEV claim.",
                  "selection": selected}
    save_new(OUT / "frozen_selection.json", freeze_obj)
    print(json.dumps({k:v for k,v in freeze_obj.items() if k not in ("selection","source_manifests")}, indent=2))


def prepare():
    frozen_path = OUT / "frozen_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["train_manifest_sha256"] != ext.base.sha(TRAIN) or frozen["protocol_sha256"] != ext.base.sha(PROTOCOL):
        raise ValueError("Frozen source changed")
    for src in frozen["source_manifests"]:
        if ext.base.sha(ROOT / src["path"]) != src["sha256"]:
            raise ValueError("Source manifest changed")
    manifest_path = OUT / "dev_data/manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.parent.mkdir(exist_ok=True)
    arrays = manifest_path.parent / "arrays"
    arrays.mkdir(exist_ok=True)
    lookup = {}
    for src in frozen["source_manifests"]:
        path = ROOT / src["path"]
        for row in json.loads(path.read_text(encoding="utf-8"))["scenes"]:
            lookup[(src["path"], row["id"])] = row
    scenes = []
    for item in frozen["selection"]:
        source_path = Path(item["source_array"])
        if ext.base.sha(source_path) != item["array_sha256"]:
            raise ValueError("Source array changed")
        dest = arrays / source_path.name
        if dest.exists():
            raise FileExistsError(dest)
        os.link(source_path, dest)
        row = copy.deepcopy(lookup[(item["source_manifest"], item["id"])])
        row["split"] = "dev"
        row["array_file"] = "arrays/" + dest.name
        row["country"] = item["country"]
        row["broader_dev_provenance"] = {"source_manifest": item["source_manifest"], "original_split": item["source_split"], "prior_evaluation_exposure": True}
        scenes.append(row)
    manifest = {"schema": "GTM_Model7_temporal_cohort_v1", "synthetic": False,
                "role": "broader_development_reused_evaluated_sites", "downloads": 0,
                "bands": ["B02","B03","B04","B08","B11","B12"],
                "input_channels": {"target_image": ["B02","B03","B04","B08","B11","B12"], "reference_image": ["B02","B03","B04","B08","B11","B12"]},
                "training_manifest": {"path": str(TRAIN), "sha256": ext.base.sha(TRAIN)},
                "selection": {"path": str(frozen_path), "sha256": ext.base.sha(frozen_path)},
                "test_reservation": {"path": str(OUT / "test_reservation.json"), "sha256": ext.base.sha(OUT / "test_reservation.json")},
                "protocol": {"path": str(PROTOCOL), "sha256": ext.base.sha(PROTOCOL)},
                "quality_limitations": ["DEV rows were previously evaluated and are used for model development; they are not independent final TEST evidence."],
                "split_counts": {"dev": dict(collections.Counter(r["scene_kind"] for r in scenes))},
                "scenes": scenes}
    save_new(manifest_path, manifest)
    train_source = json.loads(TRAIN.read_text(encoding="utf-8"))
    train_rows = [copy.deepcopy(r) for r in train_source["scenes"] if r["split"] == "train"]
    if len(train_rows) != 958 or collections.Counter(r["scene_kind"] for r in train_rows) != {"PLUME":268,"NO_PLUME":690}:
        raise ValueError("Frozen TRAIN count changed")
    train_arrays = arrays / "train"
    train_arrays.mkdir(exist_ok=True)
    for row in train_rows:
        source_array = TRAIN.parent / row["array_file"]
        if ext.base.sha(source_array) != row["array_sha256"]:
            raise ValueError("Frozen TRAIN array mismatch")
        target_array = train_arrays / source_array.name
        if target_array.exists():
            raise FileExistsError(target_array)
        os.link(source_array, target_array)
        row["array_file"] = "arrays/train/" + target_array.name
    combined = copy.deepcopy(manifest)
    combined["role"] = "frozen_train_plus_reused_broader_dev"
    combined["scenes"] = train_rows + scenes
    combined["split_counts"]["train"] = dict(collections.Counter(r["scene_kind"] for r in train_rows))
    combined_path = manifest_path.parent / "combined_manifest.json"
    save_new(combined_path, combined)
    print(json.dumps({"dev_manifest": str(manifest_path), "dev_sha256": ext.base.sha(manifest_path),
                      "combined_manifest": str(combined_path), "combined_sha256": ext.base.sha(combined_path),
                      "counts": combined["split_counts"], "dev_sites": len({r["physical_location_id"] for r in scenes}),
                      "dev_countries": len({r["country"] for r in scenes})}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "prepare"))
    args = parser.parse_args()
    if args.mode == "freeze": freeze()
    else: prepare()
