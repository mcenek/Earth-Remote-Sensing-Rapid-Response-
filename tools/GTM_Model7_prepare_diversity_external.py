"""Export frozen, metadata-selected external MARS L1C observations.

The selection JSON must be sealed before this program reads source pixels. See
docs/GTM_Model7_diversity_external_evaluation_20261024.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json"
TRAIN_SHA = "b7d44b43e113baa98d61353750a8721d8297ab6a9d94fd6e74ca0ae50da75836"
PROTOCOL = ROOT / "docs/GTM_Model7_diversity_external_evaluation_20261024.md"
OUT = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/external_data"
POOL = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/unseen_audit/weight_unseen_positive_pool.json"
SELECTION = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/unseen_audit/frozen_external_selection.json"
SEED = "GTM_Model7_external_metadata_20261024_seed_1"
BASE_PATH = ROOT / "tools/GTM_Model7_prepare_mars_repaired.py"
CACHED_PATH = ROOT / "tools/GTM_Model7_prepare_cached_diversity.py"
CLEAR_PATH = ROOT / "tools/GTM_Model7_prepare_mars_clear.py"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


base = module("external_mars_base", BASE_PATH)
cached = module("external_cached_contract", CACHED_PATH)
clear = module("external_snow_gate", CLEAR_PATH)


def load_inventory() -> tuple[dict, dict]:
    original = {}
    for name in ("train.csv", "val.csv", "test.csv"):
        with (base.MARS / name).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                sid = row["id_loc_image"]
                if sid in original:
                    raise ValueError("Duplicate official observation: " + sid)
                original[sid] = row
    pinned = {}
    with (base.MARS / "paper_v3_mixed_samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid in original:
                if sid in pinned:
                    raise ValueError("Duplicate pinned observation: " + sid)
                pinned[sid] = row
    if set(original) != set(pinned):
        raise ValueError("Official CSV and pinned inventory differ")
    return original, pinned


def forbidden(train: dict) -> dict:
    scenes = train["scenes"]
    return {
        "ids": {s["id"].removeprefix("MARS_") for s in scenes} | {s.get("reference_observation_id") for s in scenes},
        "sites": {s["physical_location_id"] for s in scenes},
        "acquisitions": {s[k] for s in scenes for k in ("target_scene_id", "reference_scene_id")},
        "hashes": {x["sha256"] for s in scenes for x in s["sources"] if x["role"] in ("target_image", "reference_image")},
        "points": [(float(s["latitude"]), float(s["longitude"])) for s in scenes],
    }


def check_metadata(sid: str, original: dict, pinned: dict, exclusion: dict, role: str) -> None:
    if sid not in original or sid not in pinned:
        raise ValueError("Unknown frozen " + role + " ID: " + sid)
    row, pin = original[sid], pinned[sid]
    expected = "PLUME" if row["isplume"] == "True" else "NO_PLUME"
    assets = {a["role"]: a for a in pin["assets"]}
    if (row["satellite"] not in ("S2A", "S2B", "S2C") or pin["sensor_family"] != "Sentinel-2" or
            pin["product_level"] != "L1C" or pin["band_order"] != base.FULL_BANDS or
            pin["label_state"] != expected or pin["country"] != row["country"] or
            pin["satellite"] != row["satellite"] or pin["observability"] != row["observability"] or
            assets["image"]["path"] != row["s2path"] or assets["cloud_mask"]["path"] != row["cloudmaskpath"] or
            (expected == "PLUME" and assets["plume_mask"]["path"] != row["plumepath"])):
        raise ValueError("Official/pinned L1C source contract failed: " + sid)
    if role == "reference" and expected != "NO_PLUME":
        raise ValueError("Reference is not explicit NO_PLUME: " + sid)
    if pin["observability"] != "clear" or float(row["percentage_clear"]) < 90:
        raise ValueError("Frozen observation lacks producer clear 90%: " + sid)
    if sid in exclusion["ids"] or row["id_location"] in exclusion["sites"] or any(
            base.kilometers((float(row["lat"]), float(row["lon"])), point) < 25 for point in exclusion["points"]):
        raise ValueError("Combined-cohort ID/site/25km exclusion failed: " + sid)
    if pin["target_scene_id"] in exclusion["acquisitions"] or pin["reference_scene_id"] in exclusion["acquisitions"]:
        raise ValueError("Combined-cohort acquisition exclusion failed: " + sid)
    required = ("image", "cloud_mask") + (("plume_mask",) if expected == "PLUME" and role == "target" else ())
    if not all(cached.asset_present(assets[k]) for k in required):
        raise ValueError("Frozen source asset missing or wrong size: " + sid)


def references(sid: str, original: dict, pinned: dict, exclusion: dict) -> list[tuple[float, str]]:
    target, when = original[sid], datetime.fromisoformat(pinned[sid]["target_datetime"])
    result = []
    for rid, row in original.items():
        if (rid == sid or row["id_location"] != target["id_location"] or
                row["split_name"] != target["split_name"] or base.grid_key(row) != base.grid_key(target)):
            continue
        pin = pinned[rid]
        if pin["label_state"] != "NO_PLUME" or pin["observability"] != "clear" or float(row["percentage_clear"]) < 90:
            continue
        ref_time = datetime.fromisoformat(pin["target_datetime"])
        hours = (when - ref_time).total_seconds() / 3600
        if not (24 <= hours <= 1440 and when.date() != ref_time.date()):
            continue
        try:
            check_metadata(rid, original, pinned, exclusion, "reference")
        except ValueError:
            continue
        result.append((hours, rid))
    return sorted(result)


def stable(raw: str) -> str:
    return hashlib.sha256(f"{SEED}:{raw}".encode()).hexdigest()


def official_priority(name: str) -> int:
    return {"official_validation": 0, "official_test": 0, "official_train": 1}[name]


def select_from_pool(pool_path: Path, selection_path: Path) -> dict:
    """Freeze all target/reference IDs using metadata only. No raster access."""
    if selection_path.exists():
        raise FileExistsError("Selection identity must be new: " + str(selection_path))
    if base.sha(TRAIN) != TRAIN_SHA:
        raise ValueError("Frozen combined manifest changed")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    if pool["frozen_manifest_sha256"] != TRAIN_SHA or not pool["metadata_only"]:
        raise ValueError("Corrected metadata pool identity failed")
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    exclusion = forbidden(train)
    original, pinned = load_inventory()
    rejected = collections.Counter()
    groups = collections.defaultdict(list)
    for item in pool["positive_candidates"]:
        sid = item["target_id"]
        try:
            check_metadata(sid, original, pinned, exclusion, "target")
            if pinned[sid]["label_state"] != "PLUME" or original[sid]["id_location"] != item["physical_site_id"]:
                raise ValueError("Candidate label/site metadata mismatch")
            refs = references(sid, original, pinned, exclusion)
            if not refs or item["reference_id"] not in [rid for _, rid in refs]:
                raise ValueError("Candidate reference metadata mismatch")
        except (ValueError, KeyError) as exc:
            rejected[str(exc).split(":")[0]] += 1
            continue
        groups[item["physical_site_id"]].append((item, refs))
    if not groups:
        raise ValueError("No positive sites independently survived combined-cohort metadata checks")
    # Prefer official validation/test roles, then interleave countries and sites.
    countries = collections.defaultdict(list)
    for site, values in groups.items():
        values.sort(key=lambda x: (official_priority(x[0]["official_source_role"]), stable(x[0]["target_id"])))
        country = values[0][0]["state_country"]
        countries[country].append(site)
    for sites in countries.values():
        sites.sort(key=lambda site: (official_priority(groups[site][0][0]["official_source_role"]), stable(site)))
    country_order = sorted(countries, key=stable)
    ordered_sites = []
    for index in range(max(map(len, countries.values()))):
        ordered_sites.extend(site for country in country_order
                             if index < len(countries[country]) for site in [countries[country][index]])
    # Positives are prioritized by release role within each selected site.
    selected, site_summary = [], []
    for site in ordered_sites:
        if len(site_summary) >= 10:
            break
        values = groups[site]
        positives = []
        for item, refs in values:
            if len(positives) >= 2:
                break
            positives.append({"sample_id": item["target_id"], "scene_kind": "PLUME",
                              "official_role": item["official_source_role"],
                              "reference_ids": [rid for _, rid in refs],
                              "historical_project_exposure": "historical metadata match; not project-naive",
                              "historical_metadata_matches": item.get("metadata_exposure_exclusions", []),
                              "historical_site_receipts": item.get("historical_site_exposure_receipts", [])})
        negative_options = []
        for sid, row in original.items():
            if row["id_location"] != site or pinned[sid]["label_state"] != "NO_PLUME":
                continue
            try:
                check_metadata(sid, original, pinned, exclusion, "target")
            except (ValueError, KeyError):
                continue
            refs = references(sid, original, pinned, exclusion)
            if refs:
                role = {"train_2023": "official_train", "val_2023": "official_validation",
                        "test_2023": "official_test"}.get(row["split_name"], row["split_name"])
                negative_options.append((official_priority(role) if role in ("official_train", "official_validation", "official_test") else 2,
                                         stable(sid), sid, role, refs))
        negative_options.sort()
        negatives = [{"sample_id": sid, "scene_kind": "NO_PLUME", "official_role": role,
                      "reference_ids": [rid for _, rid in refs],
                      "historical_project_exposure": "not established; unseen by new weights only",
                      "historical_metadata_matches": [], "historical_site_receipts": []}
                     for _, _, sid, role, refs in negative_options[:4]]
        if not positives:
            continue
        selected.extend(positives + negatives)
        site_summary.append({"physical_site_id": site, "country": values[0][0]["state_country"],
                             "positive_ids": [x["sample_id"] for x in positives],
                             "negative_ids": [x["sample_id"] for x in negatives],
                             "empty_negative_quota": max(0, 4 - len(negatives))})
    if not selected:
        raise ValueError("Country/site round-robin selected no observations")
    result = {"schema": "GTM_Model7_diversity_external_selection_v1", "metadata_only": True,
              "source_pixels_read": False, "downloads": 0, "selection_seed": SEED,
              "training_manifest_sha256": TRAIN_SHA, "positive_pool": base.file_source(pool_path),
              "protocol": base.file_source(PROTOCOL), "candidate_counts": {
                  "positive_pool_pairs": len(pool["positive_candidates"]), "independently_eligible_positive_sites": len(groups),
                  "selected_sites": len(site_summary), "selected_positive": sum(x["scene_kind"] == "PLUME" for x in selected),
                  "selected_no_plume": sum(x["scene_kind"] == "NO_PLUME" for x in selected)},
              "metadata_rejections": dict(rejected), "sites": site_summary, "candidates": selected,
              "historical_exposure_rule": "Historical project metadata matches are disclosed, not current-weight exclusions. No project-naive claim."}
    base.save(selection_path, result)
    if base.sha(TRAIN) != TRAIN_SHA or base.sha(pool_path) != result["positive_pool"]["sha256"]:
        raise ValueError("Selection source changed while freezing IDs")
    return {"selection": str(selection_path), "sha256": base.sha(selection_path), **result["candidate_counts"],
            "sites": site_summary}


def selected_records(selection: dict) -> list[dict]:
    records = selection.get("candidates", selection.get("selected_candidates", selection.get("selected", [])))
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen selection needs a nonempty candidates list")
    for record in records:
        if not isinstance(record, dict) or not record.get("sample_id"):
            raise ValueError("Candidate needs sample_id")
    ids = [x["sample_id"] for x in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Frozen target IDs repeat")
    return records


def prepare(selection_path: Path, dest: Path) -> dict:
    if dest.exists():
        raise FileExistsError("External output must be new: " + str(dest))
    if base.sha(TRAIN) != TRAIN_SHA:
        raise ValueError("Frozen combined training manifest changed")
    selection_sha = base.sha(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records = selected_records(selection)
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    exclusion = forbidden(train)
    original, pinned = load_inventory()
    site_counts = collections.defaultdict(collections.Counter)
    sites_positive = set()
    plan = []
    for record in records:
        sid = record["sample_id"]
        check_metadata(sid, original, pinned, exclusion, "target")
        kind = pinned[sid]["label_state"]
        if record.get("scene_kind", kind) != kind:
            raise ValueError("Frozen target label role changed: " + sid)
        site = original[sid]["id_location"]
        site_counts[site][kind] += 1
        if kind == "PLUME":
            sites_positive.add(site)
        if site_counts[site][kind] > (2 if kind == "PLUME" else 4):
            raise ValueError("Frozen site quota exceeded: " + site)
        eligible = references(sid, original, pinned, exclusion)
        if not eligible:
            plan.append((record, [], "no_earlier_clear_same_grid_reference"))
            continue
        frozen_refs = record.get("reference_ids", record.get("reference_candidates"))
        if frozen_refs is None:
            raise ValueError("Frozen candidate lacks complete metadata reference_ids: " + sid)
        frozen_refs = [x.get("sample_id", x.get("id")) if isinstance(x, dict) else x for x in frozen_refs]
        if frozen_refs != [rid for _, rid in eligible]:
            raise ValueError("Frozen reference order/inventory differs from metadata: " + sid)
        plan.append((record, eligible, None))
    if len(sites_positive) > 10:
        raise ValueError("More than ten frozen positive sites")
    # Seal the complete metadata plan before opening any TIFF, annotation or array.
    dest.mkdir(parents=True)
    base.save(dest / "selection_receipt.json", {"selection": base.file_source(selection_path),
              "training_manifest_sha256": TRAIN_SHA, "candidate_ids": [r["sample_id"] for r in records],
              "metadata_plan": [{"sample_id": r["sample_id"], "reference_ids": [rid for _, rid in refs],
                                 "metadata_reason": reason} for r, refs, reason in plan]})
    scenes, attempts = [], []
    for record, refs, no_ref_reason in plan:
        sid = record["sample_id"]
        row, pin = original[sid], pinned[sid]
        kind = pin["label_state"]
        if no_ref_reason:
            attempts.append({"sample_id": sid, "reason": no_ref_reason})
            continue
        for hours, rid in refs:
            reason = None
            payload = sources = fraction = None
            snow_target = snow_reference = None
            try:
                assets = {key: {a["role"]: a for a in pinned[key]["assets"]} for key in (sid, rid)}
                payload, sources, reason, fraction = base.raster_pair(sid, rid, pinned, original, assets)
                if reason is None:
                    arrays, (coarse_grid, native_grid) = payload
                    paths = {x["role"]: Path(x["path"]) for x in sources}
                    cached.verify_csv_grid(paths, row, original[rid])
                    if any(x["sha256"] in exclusion["hashes"] for x in sources
                           if x["role"] in ("target_image", "reference_image")):
                        reason = "combined_cohort_spectral_hash"
                    else:
                        snow_target = clear.snow_fraction(paths["target_image"], arrays["valid_native"])
                        snow_reference = clear.snow_fraction(paths["reference_image"], arrays["valid_native"])
                        if max(snow_target, snow_reference) > clear.GATE["date_fraction_gt"]:
                            reason = "snow_like_fraction_gt_10pct_target_or_reference"
            except (ValueError, KeyError, OSError) as exc:
                reason = "source_or_raster_contract:" + str(exc)
            attempts.append({"sample_id": sid, "reference_observation_id": rid,
                             "offset_hours": hours, "reason": reason or "accepted",
                             "bidate_valid_fraction": fraction, "target_snow_like_fraction": snow_target,
                             "reference_snow_like_fraction": snow_reference})
            if reason:
                continue
            path = dest / "arrays" / f"MARS_{sid}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, **arrays)
            scene = {"id": f"MARS_{sid}", "name": f"{kind} | {pin['target_datetime'][:10]} | {sid[:8]}",
                     "split": "test", "group_id": "external_site_" + row["id_location"],
                     "physical_location_id": row["id_location"], "scene_kind": kind, "source_level": "L1C",
                     "country": row["country"], "sensor_family": "Sentinel-2",
                     "original_release_role": pin["research_role"], "latitude": float(row["lat"]),
                     "longitude": float(row["lon"]), "target_scene_id": pin["target_scene_id"],
                     "reference_scene_id": pinned[rid]["target_scene_id"], "reference_observation_id": rid,
                     "original_release_reference_scene_id": pin["reference_scene_id"],
                     "satellite_id": pin["target_scene_id"], "satellite_time": pin["target_datetime"],
                     "reference_time": pinned[rid]["target_datetime"], "target_minus_reference_hours": hours,
                     "input_grid": coarse_grid, "reference_grid": native_grid, "reference_input_grid": coarse_grid,
                     "clear_fraction": fraction, "target_snow_like_fraction": snow_target,
                     "reference_snow_like_fraction": snow_reference,
                     "array_file": str(path.relative_to(dest)).replace("\\", "/"), "array_sha256": base.sha(path),
                     "sources": sources, "source_row": {k: v for k, v in pin.items() if k != "paper_baseline"},
                     "official_csv_row": row,
                     "reference_source_row": {k: v for k, v in pinned[rid].items() if k != "paper_baseline"},
                     "source_label": kind, "reference_mapping": "repeat_2",
                     "reference_kind": "reviewed_binary_annotation",
                     "historical_project_exposure": record.get("historical_project_exposure", "unknown; unseen by new weights only"),
                     "notes": "Metadata-selected external MARS L1C evaluation; existing project exposure is disclosed separately."}
            scenes.append(scene)
            break
    if base.sha(TRAIN) != TRAIN_SHA or base.sha(selection_path) != selection_sha:
        raise ValueError("Training or selection identity changed during export")
    manifest = {"schema": "GTM_Model7_temporal_cohort_v1", "synthetic": False,
                "role": "metadata_frozen_external_evaluation", "downloads": 0, "bands": base.BANDS,
                "input_channels": {"target_image": base.BANDS, "reference_image": base.BANDS},
                "training_manifest": base.file_source(TRAIN), "selection": base.file_source(selection_path),
                "protocol": base.file_source(PROTOCOL), "preparation_source": base.file_source(Path(__file__)),
                "source_csv": [base.file_source(base.MARS / name) for name in ("train.csv", "val.csv", "test.csv")],
                "source_manifest": base.file_source(base.MARS / "paper_v3_mixed_samples.jsonl"),
                "target_20m_rule": "binary max over each 2x2 native annotation block",
                "native_scoring_rule": "repeat each 20m probability into its 2x2 native 10m block; score only valid_native",
                "reference_selection": "metadata-frozen eligible references nearest by time then ID, first passing source and fixed QA",
                "quality_limitations": ["Historical project exposure is scene-specific; unseen by new weights does not imply project-untouched.",
                                        "International sites do not establish national US performance."],
                "split_counts": {"test": dict(collections.Counter(s["scene_kind"] for s in scenes))},
                "attempts": attempts, "scenes": scenes}
    base.save(dest / "manifest.json", manifest)
    receipt = {"schema": "GTM_Model7_diversity_external_preparation_v1", "passed": bool(scenes),
               "manifest": base.file_source(dest / "manifest.json"), "training_manifest_sha256": TRAIN_SHA,
               "selection_sha256": selection_sha, "candidate_count": len(records),
               "accepted_count": len(scenes), "counts": manifest["split_counts"],
               "attempt_counts": dict(collections.Counter(x["reason"].split(":")[0] for x in attempts)),
               "downloads": 0}
    base.save(dest / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-from-pool", type=Path)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.select_from_pool:
        print(json.dumps(select_from_pool(args.select_from_pool.resolve(), args.selection.resolve()), indent=2))
    else:
        print(json.dumps(prepare(args.selection.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
