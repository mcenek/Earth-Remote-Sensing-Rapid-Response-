"""Replay the frozen October 1 cached-pair data-contract audit, offline.

This deliberately does not decode rasters, convert radiometry, acquire data,
import model libraries, or authorize training. Exit 0 means the cache matches
the frozen evidence, even when scientific readiness is blocked. Exit 2 means
an integrity/contract failure; exit 3 refuses an existing or unsafe output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/GTM_Model0_hf_pilot_20260925"
PINS = ROOT / "reports/research/assets/GTM_Model6_data_contract_20261001/input_pins.json"
FOLLOWUP = "emit_followup/"
PROVIDER = "UNEP-IMEO__MARS-Hyperspectral-EMIT-v2025/originals/"
R5 = "outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r5/"
LEDGER = "outputs/cafo_pilot_2026-09-11/range_cache/budget.json"
BANDS = ("blue", "green", "red", "swir16", "swir22")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            digest.update(block)
    return digest.hexdigest()


def confined(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Non-portable input path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Input escapes its root: {relative}")
    return result


def audit(repo, cache, pins_path):
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    report = {"contract_date": pins["contract_date"], "scope": pins["scope"],
              "pins_sha256": sha(pins_path), "command_sha256": sha(Path(__file__)),
              "status": "integrity_error", "training_ready": False,
              "raster_pixels_recomputed": False, "network_requests": 0,
              "checks": [], "errors": []}
    paths = {}
    for record in pins["files"]:
        if record["scope"] not in ("repo", "cache"):
            raise ValueError("Unknown input scope")
        path = confined(repo if record["scope"] == "repo" else cache, record["path"])
        paths[(record["scope"], record["path"])] = path
        check = {"scope": record["scope"], "path": record["path"],
                 "expected_sha256": record["sha256"]}
        if not path.is_file():
            check["status"] = "missing"
        else:
            check.update(actual_sha256=sha(path), bytes=path.stat().st_size)
            check["status"] = "verified" if (check["actual_sha256"] == record["sha256"]
                                                and check["bytes"] == record["bytes"]) else "hash_mismatch"
        report["checks"].append(check)
    if any(c["status"] != "verified" for c in report["checks"]):
        report["errors"].append("One or more inputs differ from the frozen evidence. No readiness approval.")
        return report

    def load(relative, scope="cache"):
        return json.loads(paths[(scope, relative)].read_text(encoding="utf-8"))

    selection = load(FOLLOWUP + "native_selection.json")
    review = load(FOLLOWUP + "review/native_review.json")
    acquisition = load(FOLLOWUP + "sentinel/acquisition.json")
    preflight = load(FOLLOWUP + "sentinel/preflight.json")
    swir = load(FOLLOWUP + "swir_20260930/receipt.json")
    prior_readiness = load(FOLLOWUP + "swir_20260930/input_readiness.json")
    ledger = load(LEDGER, "repo")
    ledger_hash = sha(paths[("repo", LEDGER)])
    item = next(i for i in load(FOLLOWUP + "catalog_queries/08.json")["response"]["features"]
                if i["id"] == acquisition["sentinel_id"])
    properties = item["properties"]
    selected_ids = {s["row"]["id"] for s in selection["samples"]}
    with paths[("cache", PROVIDER + "csvs/train_t_v2025A.csv")].open(encoding="utf-8", newline="") as stream:
        rows = {r["id"]: r for r in csv.DictReader(stream) if r["id"] in selected_ids}
    if len(rows) != 2:
        raise ValueError("The pinned training CSV must contain both selected observations")
    samples = []
    for selected in selection["samples"]:
        sid = selected["row"]["id"]
        native = next(s for s in review["samples"] if s["id"] == sid)
        receipt = load(FOLLOWUP + f"native/{sid}/receipt.json")
        catalog = load(FOLLOWUP + f"native_catalogs/{sid}.json")
        if receipt["revision"] != pins["dataset_revision"] or catalog["revision"] != pins["dataset_revision"]:
            raise ValueError("Native evidence is from a different provider revision")
        for key in ("tile", "tile_date", "country", "isplume"):
            if rows[sid][key] != selected["row"][key]:
                raise ValueError(f"Selection differs from original training CSV: {sid}, {key}")
        rasters = native["rasters"]
        samples.append({
            "id": sid, "role": selected["role"], "country": rows[sid]["country"],
            "acquisition_utc": rows[sid]["tile_date"], "declared_isplume": rows[sid]["isplume"],
            "observed_positive_pixels": native["positive_pixels"],
            "observed_total_pixels": native["total_pixels"],
            "observation_source": "Pinned prior native raster audit; pixels not decoded in this replay",
            "native_resolution_m": rasters["plumemask"]["resolution_m"],
            "mask_nodata": rasters["plumemask"]["nodata"], "zero_is_reviewed_negative": False,
            "retrieval_headers": {k: {field: rasters[k][field] for field in ("descriptions", "units", "nodata")}
                                  for k in ("mf", "wmf", "mag1c")},
            "physical_units_confirmed": False, "quantitative_qa_confirmed": False,
            "provider_declared_clear_percent": float(rows[sid]["percentage_clear"]),
            "provider_rgb_constant": native["rgb_constant"],
            "provider_rgb_validity_usable_as_methane_qa": False,
            "native_folder_files": sorted(PurePosixPath(f["path"]).name for f in catalog["files"]),
            "native_grids_equal_in_prior_audit": native["all_native_grids_equal"],
            "sentinel_matches": len(selected["sentinel"]),
            "five_target_bands_available": list(BANDS) if selected["sentinel"] else [],
            "reference_date_available": False, "training_ready": False,
        })
    if prior_readiness["training_ready"] or prior_readiness["complete_temporal_input"]:
        raise ValueError("Unexpected promotion in the historical readiness record")
    if swir["scene"] != item["id"]:
        raise ValueError("SWIR and RGB receipts refer to different Sentinel observations")
    report.update(
        status="verified_cache_scientifically_blocked",
        dataset_revision=pins["dataset_revision"],
        raster_files_verified=sum(c["path"].endswith(".tif") for c in report["checks"]),
        samples=samples,
        radiometry={
            "sentinel_id": item["id"], "collection": item["collection"],
            "processing_baseline": properties["s2:processing_baseline"],
            "processing_software": properties["processing:software"],
            "boa_offset_applied": properties["earthsearch:boa_offset_applied"],
            "assets": {k: {"href": item["assets"][k]["href"],
                            "scale": item["assets"][k]["raster:bands"][0]["scale"],
                            "offset": item["assets"][k]["raster:bands"][0]["offset"]} for k in BANDS},
            "conversion_approved": False, "stored_pixels_changed": False,
            "decision": "Exact-asset provenance is required to resolve the flag/asset-offset contradiction. Apply neither candidate conversion in this audit.",
        },
        pairing={"volusia_offset_minutes": acquisition["offset_minutes"],
                 "crop_clear_fraction_from_prior_scl_audit": preflight["clear_fraction"],
                 "annotation_clear_fraction_from_prior_scl_audit": preflight["positive_area_clear_fraction"],
                 "black_oak_query_matched": load(FOLLOWUP + "catalog_queries/01.json")["response"]["numberMatched"],
                 "selection": "Development only; Volusia crop extent was chosen using the positive annotation"},
        gates={
            "cached_input_inspection": True, "radiometry_approved": False,
            "mask_background_contract": False, "quantitative_units_and_qa": False,
            "temporal_mapper_input_complete": False, "paired_reviewed_negative": False,
            "independent_evaluation_cohort": False, "finer_methane_truth": False,
            "verified_source_origin_and_wind": False,
        },
        gate_scope="Units/quantitative QA gate the enhancement head. Temporal reference gates the proposed temporal model. Background validity and reviewed controls also gate supervised mask training; none is inferred from scene names.",
        budget={**ledger, "remaining_bytes": ledger["limit_bytes"] - ledger["charged_bytes"],
                "sha256": ledger_hash, "changed": False},
        preserved_r5={"status": load(R5 + "GTM_Model6_mapper_mars_pilot_results.json", "repo")["status"],
                      "aggregate": load(R5 + "GTM_Model6_mapper_mars_pilot_results.json", "repo")["aggregate"],
                      "checkpoint_sha256": sha(paths[("repo", R5 + "GTM_Model6_mapper_checkpoint.pt")]),
                      "new_inference": False},
    )
    if sha(paths[("repo", LEDGER)]) != ledger_hash:
        raise ValueError("The shared ledger changed during the audit")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE)
    parser.add_argument("--pins", type=Path, default=PINS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, cache, pins, output = (p.resolve() for p in (args.repo_root, args.cache_root, args.pins, args.output))
    if output.exists() or output.is_relative_to(cache) or output == pins:
        print("Refusing existing report or output inside the evidence cache.", file=sys.stderr)
        return 3
    try:
        pinned_paths = json.loads(pins.read_text(encoding="utf-8"))["files"]
        if any(output == confined(repo if p["scope"] == "repo" else cache, p["path"]) for p in pinned_paths):
            print("Refusing to overwrite a protected input.", file=sys.stderr)
            return 3
        result = audit(repo, cache, pins)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        result = {"status": "integrity_error", "training_ready": False,
                  "errors": [f"{type(error).__name__}: {error}"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "training_ready": False, "report": str(output)}))
    return 0 if result["status"] == "verified_cache_scientifically_blocked" else 2


if __name__ == "__main__":
    sys.exit(main())
