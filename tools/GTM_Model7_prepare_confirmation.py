"""Prepare frozen official-test confirmation data after model configuration freeze.

Failure cases and the metadata-only selection were recorded before this exporter
in docs/GTM_Model7_confirmation_protocol.md. Never run this script before the
recovery configuration is frozen: running it reads sealed official-test rasters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from GTM_Model7_prepare_temporal import (
    BANDS,
    FULL_BANDS,
    MARS,
    ROOT,
    digest,
    grid,
    source,
    source_arrays,
    temporal_offset,
)

RUN = ROOT / "outputs/GTM_Model7_recovery_20261023"
SELECTION = RUN / "confirmation_selection.json"
OLD = ROOT / "outputs/GTM_Model7_temporal_20261023/data"
PROTOCOL = ROOT / "docs/GTM_Model7_confirmation_protocol.md"
DEFAULT_OUTPUT = RUN / "confirmation_data"


def distance_km(left: dict, right: dict) -> float:
    lat1, lat2 = math.radians(left["latitude"]), math.radians(right["latitude"])
    delta_lon = math.radians(left["longitude"] - right["longitude"])
    h = (math.sin((lat1 - lat2) / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2)
    return 12742 * math.asin(min(1, math.sqrt(h)))


def validate_selection(selection: dict, old: dict) -> list[dict]:
    if (selection["schema"] != "GTM_Model7_confirmation_selection_v1" or
            selection["selection_seed"] != "20261023" or
            selection["old_cohort_count"] != 203):
        raise ValueError("Unexpected frozen selection contract")
    for key, expected in (("old_manifest", OLD / "manifest.json"),
                          ("official_test_csv", MARS / "test.csv"),
                          ("sealed_test_metadata", MARS / "paper_v3_sealed_test_samples.jsonl"),
                          ("protocol", PROTOCOL)):
        receipt = selection["source_files"][key]
        if (Path(receipt["path"]).resolve() != expected.resolve() or
                expected.stat().st_size != receipt["bytes"] or
                digest(expected) != receipt["sha256"]):
            raise ValueError("Frozen metadata source changed: " + key)
    rows = selection["selected_rows"]
    if ([r["sample_id"] for r in rows] != selection["selected_sample_ids"] or
            len(rows) != 27 or len({r["sample_id"] for r in rows}) != 27 or
            Counter(r["label_state"] for r in rows) != {"PLUME": 3, "NO_PLUME": 24}):
        raise ValueError("Selection IDs or counts changed")
    test_by_id = {row["id_loc_image"]: row for row in csv.DictReader(
        (MARS / "test.csv").open(newline="", encoding="utf-8"))}
    official_meta = {row["sample_id"]: row for row in (
        json.loads(line) for line in (MARS / "paper_v3_sealed_test_samples.jsonl").open(encoding="utf-8"))}
    old_scenes = old["scenes"]
    old_acquisitions = {s[field] for s in old_scenes
                        for field in ("target_scene_id", "reference_scene_id")}
    positive_sites = {r["physical_location_id"] for r in rows if r["label_state"] == "PLUME"}
    if len(positive_sites) != 3:
        raise ValueError("Expected three positive sites")
    for row in rows:
        sample_id = row["sample_id"]
        if official_meta.get(sample_id) != row:
            raise ValueError("Selected metadata differs from pinned release: " + sample_id)
        original = test_by_id.get(sample_id)
        if original is None:
            raise ValueError("Selected row absent from official test.csv: " + sample_id)
        assets = {a["role"]: a for a in row["assets"]}
        is_positive = row["label_state"] == "PLUME"
        if (row["country"] != "United States of America" or
                row["satellite"] not in ("S2A", "S2B") or row["product_level"] != "L1C" or
                row["observability"] != "clear" or row["physical_location_id"] not in positive_sites or
                row["band_order"] != FULL_BANDS or row["split"] != "sealed_paper_test" or
                (original["isplume"] == "True") != is_positive or
                original["s2path"] != assets["image"]["path"] or
                original["cloudmaskpath"] != assets["cloud_mask"]["path"] or
                (is_positive and original["plumepath"] != assets["plume_mask"]["path"]) or
                not Path(original["s2path"]).name.startswith(original["id_location"] + "_") or
                abs(float(original["lat"]) - row["latitude"]) > 1e-5 or
                abs(float(original["lon"]) - row["longitude"]) > 1e-5):
            raise ValueError("Selected row violates official metadata: " + sample_id)
        if min(distance_km(row, scene) for scene in old_scenes) <= 100:
            raise ValueError("Confirmation site within 100 km of old cohort: " + sample_id)
        if (row["target_scene_id"] in old_acquisitions or
                row["reference_scene_id"] in old_acquisitions):
            raise ValueError("Confirmation acquisition overlaps old cohort: " + sample_id)
    return rows


def prepare(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("Confirmation output already exists: " + str(destination))
    old = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    rows = validate_selection(selection, old)
    # The first official-test raster is opened only after all metadata checks.
    destination.mkdir(parents=True)
    scenes = []
    for previous in old["scenes"]:
        if previous["split"] == "test":
            continue
        current = dict(previous)
        previous_array = OLD / previous["array_file"]
        new_array = destination / previous["array_file"]
        if digest(previous_array) != previous["array_sha256"]:
            raise ValueError("Old train/dev array changed: " + previous["id"])
        shutil.copy2(previous_array, new_array)
        scenes.append(current)
    exclusions = []
    old_acquisitions = {s[field] for s in old["scenes"]
                        for field in ("target_scene_id", "reference_scene_id")}
    for row in rows:
        sid = "MARS_" + row["sample_id"]
        payload, sources, reason, clear_fraction = source_arrays(row)
        if reason:
            exclusions.append({"sample_id": row["sample_id"], "reason": reason,
                               "clear_fraction": clear_fraction})
            continue
        arrays, input_grid, reference_grid = payload
        # source_arrays checks the raster contract and pinned upstream identity.
        np.savez_compressed(destination / (sid + ".npz"), **arrays)
        reference_time, offset_hours = temporal_offset(row)
        scenes.append({
            "id": sid,
            "name": f"{row['label_state']} | {row['target_datetime'][:10]} | {row['sample_id'][:8]}",
            "split": "test",
            "group_id": "confirm_" + hashlib.sha256(row["physical_location_id"].encode()).hexdigest()[:16],
            "source_group_id": row["group_id"],
            "physical_location_id": row["physical_location_id"],
            "scene_kind": row["label_state"],
            "source_level": "L1C",
            "latitude": row["latitude"], "longitude": row["longitude"],
            "target_scene_id": row["target_scene_id"],
            "reference_scene_id": row["reference_scene_id"],
            "satellite_id": row["target_scene_id"],
            "satellite_time": row["target_datetime"],
            "reference_time": reference_time,
            "target_minus_reference_hours": offset_hours,
            "input_grid": input_grid, "reference_grid": reference_grid,
            "clear_fraction": clear_fraction,
            "array_file": sid + ".npz",
            "array_sha256": digest(destination / (sid + ".npz")),
            "sources": sources,
            "source_row": row,
            "source_label": row["label_state"],
            "reference_mapping": "repeat_2",
            "reference_kind": "reviewed_binary_annotation",
            "notes": "Official sealed test scene selected by metadata before recovery configuration freeze."
        })
    if exclusions:
        # Do not replace poor-quality confirmation observations based on rasters.
        (destination / "quality_exclusions.json").write_text(
            json.dumps(exclusions, indent=2) + "\n", encoding="utf-8")
    # A quality exclusion shortens the frozen set. Keep it visible and never
    # replace it with a model-dependent selection.
    train_dev = [s for s in scenes if s["split"] != "test"]
    test = [s for s in scenes if s["split"] == "test"]
    if any(distance_km(t, s) <= 100 for t in test for s in train_dev):
        raise ValueError("Confirmation geographic overlap with fit/dev")
    if any(t[field] in {s[k] for s in train_dev for k in
                        ("target_scene_id", "reference_scene_id")}
           for t in test for field in ("target_scene_id", "reference_scene_id")):
        raise ValueError("Confirmation acquisition overlap with fit/dev")
    if any(t[field] in old_acquisitions for t in test for field in
           ("target_scene_id", "reference_scene_id")):
        raise ValueError("Confirmation acquisition overlap with old test")
    scenes.sort(key=lambda s: ({"train": 0, "dev": 1, "test": 2}[s["split"]], s["id"]))
    counts = {split: dict(Counter(s["scene_kind"] for s in scenes if s["split"] == split))
              for split in ("train", "dev", "test")}
    manifest = {**{key: value for key, value in old.items()
                   if key not in ("scenes", "split_counts", "native_positive_pixels",
                                  "coarse_positive_pixels", "quality_exclusions",
                                  "quality_exclusion_rows", "selection_rule", "role",
                                  "source_csv", "source_manifest", "graph_summary",
                                  "partition_components", "source_summary", "quality_limitations")},
                "role": "official_test_geographic_confirmation_after_config_freeze",
                "selection_rule": "Frozen metadata-only selection in confirmation_selection.json; three >100 km official-test positive sites plus eight clear reviewed negatives per site by SHA-256 seed 20261023",
                "selection_file": source(SELECTION),
                "confirmation_protocol": source(PROTOCOL),
                "old_train_dev_manifest": source(OLD / "manifest.json"),
                "source_csv": source(MARS / "test.csv"),
                "source_manifest": source(MARS / "paper_v3_sealed_test_samples.jsonl"),
                "quality_limitations": [
                    "No reference-date cloud mask is supplied; only radiometry is verified for the reference.",
                    "This official test split is consumed once after recovery configuration freeze.",
                    "Kansas has test_only_site=false and is not claimed to be untouched by all earlier project work.",
                    "The old internal test was already exposed and is diagnostic only."],
                "split_counts": counts,
                "quality_exclusions": dict(Counter(e["reason"] for e in exclusions)),
                "quality_exclusion_rows": exclusions,
                "native_positive_pixels": {split: sum(int(np.load(destination / s["array_file"])["label_native"].sum())
                                                       for s in scenes if s["split"] == split and s["scene_kind"] == "PLUME")
                                           for split in ("train", "dev", "test")},
                "coarse_positive_pixels": {split: sum(int(np.load(destination / s["array_file"])["target"].sum())
                                                       for s in scenes if s["split"] == split and s["scene_kind"] == "PLUME")
                                           for split in ("train", "dev", "test")},
                "scenes": scenes}
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {"schema": "GTM_Model7_confirmation_preparation_receipt_v1",
               "manifest": source(destination / "manifest.json"),
               "selection": source(SELECTION), "protocol": source(PROTOCOL),
               "exporter": source(Path(__file__)), "split_counts": counts,
               "quality_exclusions": exclusions, "downloads": 0,
               "official_test_rasters_opened_after_config_freeze": True}
    (destination / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "counts": counts,
                      "quality_exclusions": exclusions, "downloads": 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.output.resolve())
