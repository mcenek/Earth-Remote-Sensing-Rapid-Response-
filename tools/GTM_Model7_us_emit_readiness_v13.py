"""Bounded metadata-only US EMIT/Sentinel pairing search.

Freeze a deterministic plan before querying. Never fetch rasters, masks, or
model predictions. Existing screened EMIT training metadata is the only input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import urllib.parse

from GTM_Model6_emit_2025_stage import solar_rank
from GTM_Model0_hf_bounded_stage import Budget as SharedBudget, LEDGER, request_bytes

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "outputs/GTM_Model0_hf_pilot_20260925/emit_followup"
OUT = ROOT / "outputs/GTM_Model7_us_transfer_v13_20261026/readiness"
REVISION = "4e6c50999eb430fa33abc9488b71cd1a0dcd7991"
REPO = "UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025"
STAC = "https://earth-search.aws.element84.com/v1/search"
MAX_REQUESTS = 120
MAX_BYTES = 20_000_000
PER_RESPONSE = 49_152
SPECTRAL = {"blue", "green", "red", "nir", "swir16", "swir22"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def freeze():
    candidates = PILOT / "eligible_training_candidates.json"
    prior = PILOT / "query_plan.json"
    audit = PILOT / "overlap_audit.json"
    rows = json.loads(candidates.read_text(encoding="utf-8"))
    old = {r["id"] for r in json.loads(prior.read_text(encoding="utf-8"))["rows"]}
    selected = sorted((r for r in rows if r["country"] == "United States of America"
                       and r["isplume"] == "True" and r["tile_date"].startswith("2025")
                       and not r["exclusions"] and r["id"] not in old
                       and float(r["nearest_inventory_km"]) > 25), key=solar_rank)
    if len(rows) != 240 or len(selected) != 50 or len({r["id"] for r in selected}) != 50:
        raise ValueError("Screened US pool changed; review before catalog access")
    plan = {"status": "frozen_before_queries", "source_kind": "provider training metadata only",
            "source_sha256": sha(candidates), "overlap_audit_sha256": sha(audit),
            "prior_query_plan_sha256": sha(prior), "prior_query_ids_excluded": sorted(old),
            "order": "2025 declared-positive rows, excluding prior 12 queries, solar_rank then ID",
            "eligibility": "prior identity/25km/clear>=90 audit, not certified untouched benchmark",
            "rows": selected, "max_requests": MAX_REQUESTS, "max_response_bytes_total": MAX_BYTES,
            "max_response_bytes_each": PER_RESPONSE, "sentinel_window_minutes": 60,
            "maximum_scene_cloud_pct": 20,
            "reference_window": "prior 1 to 60 days; catalog availability only, not methane-free truth",
            "native_target_policy": "folder listing only; no file bytes or pixel reads"}
    plan["shared_ledger"] = str(LEDGER)
    plan["note"] = "v2 binds metadata reservations to the pre-existing 100 MiB shared ledger"
    save(OUT / "plan_v2.json", plan)
    return plan


def point_on_segment(p, a, b):
    cross = (p[0]-a[0])*(b[1]-a[1])-(p[1]-a[1])*(b[0]-a[0])
    return abs(cross) < 1e-10 and min(a[0],b[0])-1e-10 <= p[0] <= max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10 <= p[1] <= max(a[1],b[1])+1e-10


def in_ring(p, ring):
    inside = False
    for a, b in zip(ring, ring[1:]):
        if point_on_segment(p, a, b):
            return True
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if p[0] < x:
                inside = not inside
    return inside


def contains(geometry, p):
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        return False
    return any(in_ring(p, poly[0]) and not any(in_ring(p, hole) for hole in poly[1:]) for poly in polygons)


class Budget:
    def __init__(self):
        self.requests = 0
        self.bytes = 0
        self.shared = SharedBudget(LEDGER, 100 * 1024 * 1024)
        self.before = self.shared.snapshot()

    def get(self, url, payload=None, cap=PER_RESPONSE):
        if self.requests >= MAX_REQUESTS:
            raise RuntimeError("metadata request cap reached")
        if self.shared.snapshot()["remaining_bytes"] < cap:
            raise RuntimeError("shared metadata ledger lacks reservation")
        self.requests += 1
        body, _ = request_bytes(url, self.shared, cap,
                                data=json.dumps(payload).encode() if payload else None)
        self.bytes += len(body)
        if self.bytes > MAX_BYTES:
            raise RuntimeError("metadata byte cap reached")
        return json.loads(body), hashlib.sha256(body).hexdigest()


def stac_query(budget, row, reference=False, collection=None):
    when = time(row["tile_date"])
    lo, hi = ((when-timedelta(days=60), when-timedelta(days=1)) if reference
              else (when-timedelta(hours=1), when+timedelta(hours=1)))
    payload = {"collections": [collection] if collection else ["sentinel-2-l1c", "sentinel-2-l2a"],
               "intersects": {"type": "Point", "coordinates": [float(row["location_lon"]), float(row["location_lat"])]},
               "datetime": f"{lo.isoformat()}/{hi.isoformat()}", "limit": 2 if collection else 4}
    data, digest = budget.get(STAC, payload, cap=131_072 if reference else PER_RESPONSE)
    matched = data.get("context", {}).get("matched", data.get("numberMatched"))
    items = []
    p = (float(row["location_lon"]), float(row["location_lat"]))
    for item in data.get("features", []):
        props = item.get("properties", {})
        dt = time(props["datetime"])
        offset = (dt-when).total_seconds()/60
        cloud = props.get("eo:cloud_cover")
        assets = item.get("assets", {})
        footprint = contains(item.get("geometry", {}), p)
        bands = sorted(SPECTRAL & set(assets))
        valid = footprint and cloud is not None and cloud <= 20 and len(bands) == 6
        items.append({"id": item.get("id"), "collection": item.get("collection"),
                      "datetime": props["datetime"], "offset_minutes": round(offset, 3),
                      "cloud_pct": cloud, "site_inside_scene_footprint": footprint,
                      "spectral_assets_present": bands, "spectral_complete": len(bands) == 6,
                      "valid_metadata_match": valid,
                      "asset_urls": {k: assets[k].get("href") for k in SPECTRAL if k in assets},
                      "scl_available": "scl" in assets,
                      "stac_item_url": f"https://earth-search.aws.element84.com/v1/collections/{item.get('collection')}/items/{urllib.parse.quote(item.get('id', ''))}"})
    return {"query_sha256": digest, "matched_count_catalog": matched,
            "returned_count": len(data.get("features", [])),
            "possibly_truncated": matched is None or matched > len(data.get("features", [])),
            "items": items}


def native_catalog(budget, row):
    folder = row["folder_name"]
    if "/" in folder or "\\" in folder or ".." in folder:
        raise ValueError("unsafe native folder")
    path = f"EMIT_2025/{folder}"
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{urllib.parse.quote(path)}?recursive=false&expand=false&limit=100"
    data, digest = budget.get(url, cap=32_768)
    wanted = {"plumemask.tif", "mag1c.tif", "mf.tif", "wmf.tif"}
    files = {}
    for f in data:
        name = f["path"].rsplit("/", 1)[-1]
        if f.get("type") == "file" and name in wanted:
            files[name] = {"bytes": f.get("size"), "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(f['path'])}"}
    return {"catalog_sha256": digest, "catalog_url": url, "files": files,
            "has_native_mask_and_retrieval": "plumemask.tif" in files and any(n in files for n in ("mag1c.tif", "mf.tif", "wmf.tif"))}


def run():
    plan_path = OUT / "plan_v2.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else freeze()
    if plan["status"] != "frozen_before_queries" or len(plan["rows"]) != 50:
        raise ValueError("Invalid frozen plan")
    if sha(PILOT / "eligible_training_candidates.json") != plan["source_sha256"]:
        raise ValueError("Candidate file changed")
    budget = Budget()
    results = []
    for index, row in enumerate(plan["rows"]):
        cache = OUT / "target_queries" / f"{index:02d}.json"
        if cache.exists():
            target = json.loads(cache.read_text(encoding="utf-8"))
        else:
            try:
                target = stac_query(budget, row)
            except (ValueError, RuntimeError) as error:
                target = {"status": "metadata_query_incomplete", "error": f"{type(error).__name__}: {error}",
                          "items": [], "returned_count": 0, "possibly_truncated": True}
            save(cache, target)
        good = [i for i in target["items"] if i["valid_metadata_match"]]
        results.append({"row_id": row["id"], "tile": row["tile"], "location_name": row["location_name"],
                        "lat": float(row["location_lat"]), "lon": float(row["location_lon"]),
                        "emit_datetime": row["tile_date"], "nearest_prior_inventory_km": row["nearest_inventory_km"],
                        "target": target, "qualifying_current_count": len(good)})
        print(f"current {index+1}/50 {row['id']} matches={len(good)} requests={budget.requests}", flush=True)
    for result in results:
        if not result["qualifying_current_count"]:
            continue
        row = next(r for r in plan["rows"] if r["id"] == result["row_id"])
        if budget.requests+2 > MAX_REQUESTS:
            result["followup_status"] = "budget_exhausted"
            continue
        native_cache = OUT / "native_catalogs" / f"{result['row_id']}.json"
        result["native"] = json.loads(native_cache.read_text(encoding="utf-8")) if native_cache.exists() else native_catalog(budget, row)
        if not native_cache.exists():
            save(native_cache, result["native"])
        ref_cache = OUT / "reference_queries" / f"{result['row_id']}.json"
        if ref_cache.exists():
            result["reference"] = json.loads(ref_cache.read_text(encoding="utf-8"))
        else:
            try:
                result["reference"] = stac_query(budget, row, reference=True)
            except (ValueError, RuntimeError) as error:
                result["reference"] = {"status": "metadata_query_incomplete", "error": f"{type(error).__name__}: {error}",
                                       "items": [], "returned_count": 0, "possibly_truncated": True}
            save(ref_cache, result["reference"])
        result["followup_status"] = "complete"
    summary = {"status": "metadata_readiness_only", "plan_sha256": sha(plan_path),
               "source_rows": len(plan["rows"]), "requests": budget.requests, "received_bytes": budget.bytes,
               "shared_ledger_before": budget.before, "shared_ledger_after": budget.shared.snapshot(),
               "rows_with_qualifying_current": sum(bool(r["qualifying_current_count"]) for r in results),
               "rows_with_native_mask_retrieval": sum(r.get("native", {}).get("has_native_mask_and_retrieval", False) for r in results),
               "limits": ["EMIT provider class and folder names are metadata, not inspected pixel truth",
                          "Point containment proves only source coordinate within Sentinel footprint; native EMIT mask footprint remains unverified",
                          "Scene cloud percentage is not local plume-area cloud QA",
                          "Earlier Sentinel availability does not certify a methane-free reference",
                          "Existing known-site exclusion follows prior audit; new 2025 provider validation/test inventories are unavailable"],
               "results": results}
    save(OUT / "result.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "results"}, indent=2))


def resolve_incomplete():
    prior = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((OUT / "plan_v2.json").read_text(encoding="utf-8"))
    incomplete = [r for r in prior["results"] if r["target"].get("status") == "metadata_query_incomplete"]
    if len(incomplete) != 1:
        raise ValueError("Expected exactly one incomplete current catalog row")
    row = next(r for r in plan["rows"] if r["id"] == incomplete[0]["row_id"])
    budget = Budget()
    parts = {collection: stac_query(budget, row, collection=collection)
             for collection in ("sentinel-2-l1c", "sentinel-2-l2a")}
    result = {"status": "supplemental_catalog_completion", "row_id": row["id"],
              "plan_sha256": sha(OUT / "plan_v2.json"), "earlier_result_sha256": sha(OUT / "result.json"),
              "parts": parts, "qualifying_current": [i for p in parts.values() for i in p["items"] if i["valid_metadata_match"]],
              "requests": budget.requests, "received_bytes": budget.bytes,
              "shared_ledger_before": budget.before, "shared_ledger_after": budget.shared.snapshot()}
    save(OUT / "incomplete_completion.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "parts"}, indent=2))


def complete_followup():
    prior = json.loads((OUT / "incomplete_completion.json").read_text(encoding="utf-8"))
    plan = json.loads((OUT / "plan_v2.json").read_text(encoding="utf-8"))
    row = next(r for r in plan["rows"] if r["id"] == prior["row_id"])
    budget = Budget()
    native = native_catalog(budget, row)
    try:
        reference = stac_query(budget, row, reference=True)
    except (ValueError, RuntimeError) as error:
        reference = {"status": "metadata_query_incomplete", "error": f"{type(error).__name__}: {error}",
                     "items": [], "returned_count": 0, "possibly_truncated": True}
    result = {"status": "supplemental_followup", "row_id": row["id"],
              "completion_sha256": sha(OUT / "incomplete_completion.json"),
              "native": native, "reference": reference, "requests": budget.requests,
              "received_bytes": budget.bytes, "shared_ledger_before": budget.before,
              "shared_ledger_after": budget.shared.snapshot()}
    save(OUT / "incomplete_followup.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("native", "reference")}, indent=2))


def consolidate_offline():
    """Bind saved metadata receipts, with no network or raster access."""
    plan_path = OUT / "plan_v2.json"
    result_path = OUT / "result.json"
    completion_path = OUT / "incomplete_completion.json"
    followup_path = OUT / "incomplete_followup.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    initial = json.loads(result_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    followup = json.loads(followup_path.read_text(encoding="utf-8"))
    if initial["plan_sha256"] != sha(plan_path) or completion["plan_sha256"] != sha(plan_path):
        raise ValueError("Plan hash mismatch")
    if completion["earlier_result_sha256"] != sha(result_path) or followup["completion_sha256"] != sha(completion_path):
        raise ValueError("Supplement chain mismatch")
    sources = {"eligible_training_candidates.json": (PILOT / "eligible_training_candidates.json", plan["source_sha256"]),
               "overlap_audit.json": (PILOT / "overlap_audit.json", plan["overlap_audit_sha256"]),
               "query_plan.json": (PILOT / "query_plan.json", plan["prior_query_plan_sha256"])}
    for name, (path, expected) in sources.items():
        if sha(path) != expected:
            raise ValueError(f"Source changed: {name}")
    rows = {r["row_id"]: r for r in initial["results"]}
    if len(rows) != 50 or set(rows) != {r["id"] for r in plan["rows"]}:
        raise ValueError("Missing or duplicate row")
    if completion["row_id"] != followup["row_id"]:
        raise ValueError("Supplemental row mismatch")
    resolved_id = completion["row_id"]
    summaries = []
    for index, row in enumerate(plan["rows"]):
        prior = rows[row["id"]]
        target_path = OUT / "target_queries" / f"{index:02d}.json"
        if not target_path.exists() or json.loads(target_path.read_text(encoding="utf-8")) != prior["target"]:
            raise ValueError("Target receipt mismatch")
        target_parts = completion["parts"] if row["id"] == resolved_id else {"combined": prior["target"]}
        current = [i for part in target_parts.values() for i in part["items"] if i["valid_metadata_match"]]
        if row["id"] == resolved_id:
            native = followup["native"]
            reference = followup["reference"]
            native_receipt = followup_path
            reference_receipt = followup_path
        elif current:
            native = prior["native"]
            reference = prior["reference"]
            native_receipt = OUT / "native_catalogs" / f"{row['id']}.json"
            reference_receipt = OUT / "reference_queries" / f"{row['id']}.json"
            if json.loads(native_receipt.read_text(encoding="utf-8")) != native:
                raise ValueError("Native listing mismatch")
            if json.loads(reference_receipt.read_text(encoding="utf-8")) != reference:
                raise ValueError("Reference receipt mismatch")
        else:
            native = None
            reference = None
            native_receipt = None
            reference_receipt = None
        when = time(row["tile_date"])
        stamp = datetime.strptime(row["tile"].split("_")[4], "%Y%m%dT%H%M%S").replace(tzinfo=when.tzinfo)
        if abs((stamp-when).total_seconds()) > 1:
            raise ValueError("EMIT tile UTC timestamp mismatch")
        for item in current:
            actual_offset = (time(item["datetime"])-when).total_seconds()/60
            if abs(actual_offset-item["offset_minutes"]) > .001 or abs(actual_offset) > 60:
                raise ValueError("Sentinel UTC offset mismatch")
            if not item["site_inside_scene_footprint"] or item["cloud_pct"] > 20:
                raise ValueError("Footprint/cloud flag mismatch")
            if set(item["spectral_assets_present"]) != SPECTRAL or set(item["asset_urls"]) != SPECTRAL:
                raise ValueError("Six-band STAC asset contract mismatch")
            if not all(item["asset_urls"].values()):
                raise ValueError("Empty STAC spectral URL")
        refs = [i for i in reference["items"] if i["valid_metadata_match"]] if reference else []
        for item in refs:
            days = (when-time(item["datetime"])).total_seconds()/86400
            if not 1 <= days <= 60 or set(item["spectral_assets_present"]) != SPECTRAL:
                raise ValueError("Reference UTC/band mismatch")
        native_summary = None
        if native:
            expected_folder = f"EMIT_2025/{row['folder_name']}"
            if urllib.parse.quote(expected_folder) not in native["catalog_url"]:
                raise ValueError("Native folder mismatch")
            if set(native["files"]) != {"plumemask.tif", "mag1c.tif", "mf.tif", "wmf.tif"}:
                raise ValueError("Native listing lacks expected four files")
            if not all(str(v["bytes"]).isdigit() and v["bytes"] > 0 and expected_folder in urllib.parse.unquote(v["url"])
                       for v in native["files"].values()):
                raise ValueError("Native size/URL mismatch")
            native_summary = {"catalog_response_sha256": native["catalog_sha256"],
                              "listing_receipt_sha256": sha(native_receipt),
                              "files": {name: {**info, "upstream_object_oid": None,
                                               "object_sha256": None,
                                               "hash_status": "not retained in cached compact listing; verify upstream before download"}
                                        for name, info in native["files"].items()},
                              "total_listed_bytes": sum(v["bytes"] for v in native["files"].values())}
        summaries.append({"row_id": row["id"], "emit_tile": row["tile"],
                          "emit_datetime": row["tile_date"], "location_name": row["location_name"],
                          "lat": float(row["location_lat"]), "lon": float(row["location_lon"]),
                          "nearest_prior_inventory_km": row["nearest_inventory_km"],
                          "status": "current_metadata_match" if current else "no_qualifying_current_metadata_match",
                          "target_receipt_sha256": sha(target_path),
                          "supplemental_target_sha256": sha(completion_path) if row["id"] == resolved_id else None,
                          "current": current, "qualifying_references_in_first_page": refs,
                          "reference_page_incomplete": reference["possibly_truncated"] if reference else None,
                          "reference_receipt_sha256": sha(reference_receipt) if reference_receipt else None,
                          "native": native_summary})
    matched = [r for r in summaries if r["current"]]
    if len(matched) != 6 or len({r["emit_datetime"][:10] for r in matched}) != 4:
        raise ValueError("Expected six rows/four overpass dates")
    ledger = SharedBudget(LEDGER, 100 * 1024 * 1024).snapshot()
    if ledger["remaining_bytes"] != followup["shared_ledger_after"]["remaining_bytes"]:
        raise ValueError("Shared ledger changed since search; record new state separately")
    canonical = {"status": "canonical_metadata_only", "rows_searched": 50,
                 "rows_with_current_metadata_match": 6, "rows_without_current_metadata_match": 44,
                 "emit_overpass_dates": sorted({r["emit_datetime"][:10] for r in matched}),
                 "sha256": {"plan_v2.json": sha(plan_path), "result.json": sha(result_path),
                            "incomplete_completion.json": sha(completion_path),
                            "incomplete_followup.json": sha(followup_path),
                            **{name: expected for name, (_, expected) in sources.items()}},
                 "latest_shared_ledger": ledger,
                 "verification": {"source_and_receipt_hashes": "passed",
                                  "UTC_timestamps_and_offsets": "passed",
                                  "point_inside_scene_footprint": "query-time recorded boolean true for each match; original STAC geometry was not retained for independent offline recomputation",
                                  "six_band_support": "six named asset URLs present in STAC metadata; raster bytes not opened",
                                  "native_object_hashes": "not retained in compact catalog listings; must be checked from upstream before acquisition"},
                 "rows": summaries}
    save(OUT / "consolidated_result.json", canonical)
    chosen = {"EMIT_L1B_RAD_001_20250731T162709_2521211_016_USA_w7z4jseci",
              "EMIT_L1B_RAD_001_20250729T180213_2521012_017_WM_Security_Landfill",
              "EMIT_L1B_RAD_001_20250728T171552_2520911_024_Okeechobee_landfill"}
    selected = [r for r in matched if r["row_id"] in chosen]
    if len(selected) != 3 or len({r["emit_datetime"][:10] for r in selected}) != 3:
        raise ValueError("Selected acquisition events not independent by date")
    plan2 = {"status": "future_acquisition_plan_only_no_download_authorized", "canonical_sha256": sha(OUT / "consolidated_result.json"),
             "selected_events": [{"row_id": r["row_id"], "emit_datetime": r["emit_datetime"],
                                  "location_name": r["location_name"], "current": r["current"],
                                  "earlier_reference_candidates_first_page": r["qualifying_references_in_first_page"],
                                  "native_objects": r["native"]["files"],
                                  "native_four_object_bytes": r["native"]["total_listed_bytes"]} for r in selected],
             "native_object_count": 12,
             "native_listed_bytes_total": sum(r["native"]["total_listed_bytes"] for r in selected),
             "native_file_hash_limit": "per-object upstream hashes absent from retained compact listings; source listing response SHA and receipt hashes in canonical result; verify official OIDs before any transfer",
             "sentinel_ranged_crop_bytes": None,
             "sentinel_crop_estimate_reason": "native mask footprint and local supported crop have not been inspected; no defensible ranged-crop transfer bound yet",
             "other_matched_rows": [{"row_id": r["row_id"], "status": "retain_as_not_ready_no_acquisition_now"} for r in matched if r["row_id"] not in chosen],
             "future_clear_label_cohort": {"original_screened_US_control_rows": 176,
                                            "status": "metadata pool only; timing, independent sites, native zero-mask semantics and local cloud still need review",
                                            "selection": "prospective separate control cohort before training or accuracy evaluation"},
             "budget": {"existing_shared_ledger_remaining_bytes": ledger["remaining_bytes"],
                        "additional_256MiB_question_pending": True,
                        "authorized_to_spend_more": False}}
    save(OUT / "acquisition_plan.json", plan2)
    print(json.dumps({"canonical": str(OUT / "consolidated_result.json"),
                      "acquisition_plan": str(OUT / "acquisition_plan.json"),
                      "matched": len(matched), "selected": len(selected),
                      "native_listed_bytes_total": plan2["native_listed_bytes_total"],
                      "ledger_remaining": ledger["remaining_bytes"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "query", "resolve-incomplete", "complete-followup", "consolidate-offline"])
    args = parser.parse_args()
    if args.mode == "freeze":
        print(json.dumps({"rows": len(freeze()["rows"]), "plan": str(OUT / "plan_v2.json")}))
    elif args.mode == "query":
        run()
    elif args.mode == "resolve-incomplete":
        resolve_incomplete()
    elif args.mode == "complete-followup":
        complete_followup()
    else:
        consolidate_offline()
