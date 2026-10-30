"""Offline asset and acquisition preflight for the three US EMIT events.

Reads only saved JSON and source code. No network client or raster decoder is
created, and the shared transfer ledger is read without charging it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "outputs/GTM_Model7_us_transfer_v13_20261026/readiness"
LEDGER = ROOT / "outputs/cafo_pilot_2026-09-11/range_cache/budget.json"
BANDS = ("blue", "green", "red", "nir", "swir16", "swir22")
EVENTS = (
    ("EMIT_L1B_RAD_001_20250731T162709_2521211_016_USA_w7z4jseci",
     "S2B_17RLP_20250731_0_L1C", "S2A_17RLP_20250728_0_L1C"),
    ("EMIT_L1B_RAD_001_20250729T180213_2521012_017_WM_Security_Landfill",
     "S2B_15RTP_20250729_0_L1C", "S2C_15RTP_20250724_0_L1C"),
    ("EMIT_L1B_RAD_001_20250728T171552_2520911_024_Okeechobee_landfill",
     "S2A_17RNL_20250728_1_L1C", "S2C_17RNL_20250726_0_L1C"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def asset(url: str, product: str):
    parsed = urlsplit(url)
    if product == "L1C":
        if parsed.scheme != "s3" or parsed.netloc != "sentinel-s2-l1c" or not parsed.path.endswith(".jp2"):
            raise ValueError("Unexpected L1C source URL")
        https = "https://sentinel-s2-l1c.s3.amazonaws.com" + parsed.path
        return {"catalog_url": url, "normalized_https_url": https,
                "media_type_from_suffix": "image/jp2", "product_level": "L1C_TOA_raw_DN",
                "bytes": None, "content_hash": None,
                "bounded_reader_status": "unsupported_by_existing_charged_HTTPRangeReader"}
    if parsed.scheme != "https" or parsed.netloc != "sentinel-cogs.s3.us-west-2.amazonaws.com" or not parsed.path.endswith(".tif"):
        raise ValueError("Unexpected L2A COG URL")
    return {"catalog_url": url, "normalized_https_url": url,
            "media_type_from_suffix": "image/tiff", "product_level": "L2A_BOA_COG",
            "bytes": None, "content_hash": None,
            "bounded_reader_status": "host_and_TIFF_suffix_accepted_by_existing_reader_pending_header_probe"}


def item_record(item):
    product = "L1C" if item["collection"] == "sentinel-2-l1c" else "L2A"
    if set(item["asset_urls"]) != set(BANDS) or not all(item["asset_urls"].values()):
        raise ValueError("Missing six-band asset URL")
    return {"scene_id": item["id"], "collection": item["collection"],
            "datetime_utc": item["datetime"], "emit_offset_minutes": item["offset_minutes"],
            "catalog_scene_cloud_pct": item["cloud_pct"],
            "site_inside_STAC_scene_geometry_recorded": item["site_inside_scene_footprint"],
            "SCL_asset_listed": item["scl_available"],
            "SCL_URL": None,
            "SCL_URL_reason": "compact STAC receipt retained availability flag, not SCL href",
            "spectral_assets": {band: asset(item["asset_urls"][band], product) for band in BANDS}}


def main():
    canonical_path = READY / "consolidated_result.json"
    plan_path = READY / "acquisition_plan.json"
    original_path = READY / "result.json"
    canonical, plan = read(canonical_path), read(plan_path)
    original = read(original_path)
    if plan["canonical_sha256"] != digest(canonical_path):
        raise ValueError("Acquisition plan/canonical hash mismatch")
    ledger = read(LEDGER)
    remaining = int(ledger["limit_bytes"]) - int(ledger["charged_bytes"])
    if remaining != canonical["latest_shared_ledger"]["remaining_bytes"]:
        raise ValueError("Ledger differs from canonical; audit current state before use")
    by_id = {r["row_id"]: r for r in canonical["rows"]}
    original_by_id = {r["row_id"]: r for r in original["results"]}
    selected_ids = {r["row_id"] for r in plan["selected_events"]}
    if selected_ids != {r[0] for r in EVENTS}:
        raise ValueError("Selected event IDs changed")
    events = []
    for row_id, current_id, reference_id in EVENTS:
        row = by_id[row_id]
        current = next(i for i in row["current"] if i["id"] == current_id)
        reference = next(i for i in row["qualifying_references_in_first_page"] if i["id"] == reference_id)
        if not (current["valid_metadata_match"] and reference["valid_metadata_match"]):
            raise ValueError("Selected current/reference metadata gate failed")
        native = row["native"]
        if set(native["files"]) != {"plumemask.tif", "mag1c.tif", "mf.tif", "wmf.tif"}:
            raise ValueError("Unexpected native object listing")
        alternative = [i for i in original_by_id[row_id]["target"]["items"]
                       if i["collection"] == "sentinel-2-l2a"]
        events.append({"row_id": row_id, "emit_datetime_utc": row["emit_datetime"],
                       "site_lat_lon": [row["lat"], row["lon"]],
                       "nearest_known_location_km": row["nearest_prior_inventory_km"],
                       "native": {"listing_response_sha256": native["catalog_response_sha256"],
                                  "listing_receipt_sha256": native["listing_receipt_sha256"],
                                  "objects": native["files"],
                                  "listed_bytes": native["total_listed_bytes"]},
                       "same_domain_current": item_record(current),
                       "same_domain_reference": item_record(reference),
                       "current_L2A_catalog_alternative": [item_record(i) for i in alternative],
                       "native_footprint_status": "unknown_until_plumemask_header_and_mask_review",
                       "local_cloud_and_alignment_status": "unknown_until_native_mask_extent_and_S2_crop_review",
                       "reference_label_status": "earlier_Sentinel_scene_available_not_verified_NO_PLUME"})
    tool_paths = {name: ROOT / "tools" / name for name in (
        "bounded_http_range.py", "acquire_cafo_cog_pilot.py",
        "acquire_stanford_large_controlled_release_l1c_crops.py",
        "build_emit_v002_l1c_pairs.py")}
    report = {"status": "offline_preflight_only_no_transfer", "canonical_sha256": digest(canonical_path),
              "acquisition_plan_sha256": digest(plan_path),
              "original_result_sha256": digest(original_path),
              "source_tool_sha256": {name: digest(path) for name, path in tool_paths.items()},
              "shared_ledger": {**ledger, "remaining_bytes": remaining},
              "events": events,
              "current_valid_route": "Native EMIT metadata/annotation review first; then L1C target and L1C earlier reference with a new ledger-enforced JP2 window reader or a separately bounded exact-product route. Do not substitute L2A into the existing L1C model input.",
              "reader_findings": {
                  "existing_HTTPRangeReader": "Allows only HTTPS sentinel-cogs.s3.us-west-2.amazonaws.com TIFF paths, requires HTTP 206, Content-Range, ETag and charges fixed blocks to shared Budget.",
                  "existing_crop_asset": "tifffile tiled-GeoTIFF decoder; cannot decode L1C JP2.",
                  "existing_L1C_GDAL_paths": "Normalize s3://sentinel-s2-l1c to public HTTPS and use rasterio/GDAL remote JP2 reads; their internal requests are not charged or bounded by the shared Budget. Do not run them for this acquisition.",
                  "L2A_COG_alternative": "Eligible for existing charged TIFF range machinery after header probe, but BOA L2A is a different radiometric domain from v12 L1C TOA and is not an interchangeable model input."},
              "unknowns": ["No Sentinel object Content-Length, ETag, block map or defensible crop-byte estimate in cached metadata.",
                           "No original STAC geometries or per-band georeferencing retained; query-time point-containment flags are true but native footprint overlap is unverified.",
                           "Native mask code semantics, positive pixels, valid support, CRS, affine transform and retrieval units remain uninspected.",
                           "SCL URLs are not retained; catalog scene cloud is not local crop cloud. L1C has no SCL.",
                           "Earlier reference is only a Sentinel date, not verified methane-free truth.",
                           "Per-object upstream content hashes were not retained in compact native listings."],
              "ordered_future_stops": [
                  "0. Wait for the pending 256 MiB decision. Freeze one shared-ledger ceiling and per-attempt cap; never reset the current 100 MiB ledger or transfer under its 130,938-byte remainder.",
                  "1. Before any data bytes, reacquire/verify native listing OIDs against the pinned revision, estimate exact listed bytes, and reserve them in the authorized ledger.",
                  "2. Transfer only the selected native annotation and retrieval objects, verify upstream hashes, inspect mask/retrieval grid, valid codes, units and full positive extent. Stop if labels or footprint are unusable; do not choose a replacement by appearance.",
                  "3. From native extent, define the complete Sentinel crop plus fixed margin and check WGS84/site, native CRS/affine and six-band target/reference coverage. Stop if support does not intersect or grids cannot align.",
                  "4. Implement/test a ledger-enforced JP2 header/window reader, or stage exact L1C source assets through a separately bounded route. Probe HTTP range/size/ETag and calculate a conservative transfer ceiling before any six-band crop. Stop on unbounded GDAL reads, unknown ceiling or insufficient remaining budget.",
                  "5. Acquire L1C current and earlier L1C reference with source IDs frozen here. Obtain matched L2A SCL only for local cloud QA under the same budget, retaining L1C as the spectral input. Stop if the annotated area or reference fails local cloud/registration/support checks.",
                  "6. Visually inspect native EMIT annotation/retrieval against both real Sentinel dates. Keep all failures. Build a separate timed and independently grouped clear-label cohort from 176 screened controls before any accuracy claim or fitting."]}
    if len(events) != 3 or sum(e["native"]["listed_bytes"] for e in events) != 1_652_419:
        raise ValueError("Native byte census changed")
    save(READY / "asset_preflight_v2.json", report)
    print(json.dumps({"events": len(events), "native_bytes": sum(e["native"]["listed_bytes"] for e in events),
                      "ledger_remaining": remaining, "output": str(READY / "asset_preflight_v2.json")}, indent=2))


if __name__ == "__main__":
    main()
