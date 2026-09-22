"""Build a local-only, geometry-aware readiness manifest for cached native pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom, transform_bounds
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "Earth-Remote-Sensing-Rapid-Response-"
PAIR_ROOT = SIBLING / "EarthRemoteSensingRapidResponse" / "Data Collection" / "s2_emit_pairs" / "emit-v002-2026-07"
NATIVE_ROOT = SIBLING / "EarthRemoteSensingRapidResponse" / "Data Collection" / "EMIT_Plumes" / "emit-v002-authenticated-2026-07"
STAMPS = ("20241020T170504_003677", "20241130T180310_003723", "20250922T204933_003374")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def timing_pass(lag_hours: float, threshold: float = 6.0) -> bool:
    return abs(lag_hours) <= threshold


def native_paths(stamp: str) -> dict[str, Path]:
    stamp_only, plume = stamp.split("_")[:2]
    enh_dir = next((NATIVE_ROOT / "CH4ENH").glob(f"EMIT_L2B_CH4ENH_002_{stamp_only}_*"))
    plm_dir = NATIVE_ROOT / f"EMIT_L2B_CH4PLM_002_{stamp_only}_{plume}"
    return {"enh": next(enh_dir.glob("*CH4ENH*.tif")), "sens": next(enh_dir.glob("*CH4SENS*.tif")), "uncert": next(enh_dir.glob("*CH4UNCERT*.tif")), "plm": next(plm_dir.glob("*.tif"))}


def product_meta(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as ds:
        tags = ds.tags()
        units = tags.get("UNITS") or tags.get("Units") or tags.get("units")
        if units is None:
            with path.open("rb") as handle:
                header = handle.read(512 * 1024).decode("latin1", errors="ignore")
            match = re.search(r'<Item name="(?:UNITS|Units)"[^>]*>(.*?)</Item>', header, re.S)
            units = match.group(1).strip() if match else None
        return {"path": str(path), "sha256": sha256(path), "shape": [ds.height, ds.width], "count": ds.count, "dtype": ds.dtypes[0], "crs": ds.crs.to_string() if ds.crs else None, "transform": list(ds.transform), "nodata": ds.nodata, "units": units, "units_source": "GDAL metadata item" if units else "unavailable", "bounds": list(ds.bounds)}


def grid_signature(meta: dict[str, Any]) -> tuple[Any, ...]:
    return (meta["crs"], tuple(meta["shape"]), tuple(round(float(x), 12) for x in meta["transform"]))


def _read_masked(path: Path, shapes: list[dict[str, Any]], dst_crs: str, bounds: tuple[float, float, float, float]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(path) as ds:
        transformed = [transform_geom(dst_crs, ds.crs, shape, precision=12) for shape in shapes]
        left, bottom, right, top = transform_bounds(dst_crs, ds.crs, *bounds, densify_pts=21)
        raw_window = from_bounds(left, bottom, right, top, ds.transform)
        col0 = max(0, math.floor(raw_window.col_off))
        row0 = max(0, math.floor(raw_window.row_off))
        col1 = min(ds.width, math.ceil(raw_window.col_off + raw_window.width))
        row1 = min(ds.height, math.ceil(raw_window.row_off + raw_window.height))
        window = Window(col0, row0, max(0, col1 - col0), max(0, row1 - row0))
        values = ds.read(1, window=window, masked=False)
        transform = ds.window_transform(window)
        inside = geometry_mask(transformed, out_shape=values.shape, transform=transform, invert=True)
        valid = inside & np.isfinite(values)
        if ds.nodata is not None:
            valid &= values != ds.nodata
        meta = product_meta(path)
        meta["window"] = [int(window.row_off), int(window.col_off), int(window.height), int(window.width)]
        meta["window_transform"] = list(transform)
        return meta, values, valid, inside


def peak_in_s2_footprint(peak: tuple[float, float], s2_crs: str, bounds: tuple[float, float, float, float]) -> bool:
    point = {"type": "Point", "coordinates": [peak[1], peak[0]]}
    projected = transform_geom("EPSG:4326", s2_crs, point, precision=12)["coordinates"]
    return bounds[0] <= projected[0] <= bounds[2] and bounds[1] <= projected[1] <= bounds[3]


def inspect_scene(scene: dict[str, Any], paths: dict[str, Path], emit_time: datetime, peak: tuple[float, float], threshold: float) -> dict[str, Any]:
    s2_path = SIBLING / Path(scene["stack"])
    mask_path = SIBLING / Path(scene["mask"])
    s2_meta, mask_meta = product_meta(s2_path), product_meta(mask_path)
    with rasterio.open(s2_path) as ds:
        footprint = [{"type": "Polygon", "coordinates": [[[ds.bounds.left, ds.bounds.bottom], [ds.bounds.left, ds.bounds.top], [ds.bounds.right, ds.bounds.top], [ds.bounds.right, ds.bounds.bottom], [ds.bounds.left, ds.bounds.bottom]]]}]
        bounds, s2_crs = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top), ds.crs.to_string()
    native_meta = {name: product_meta(path) for name, path in paths.items()}
    grid_ok = len({grid_signature(native_meta[n]) for n in ("enh", "sens", "uncert")}) == 1
    common_support = native_pixels = 0
    products: dict[str, Any] = {}
    if grid_ok:
        arrays, masks, geometry_masks = {}, {}, {}
        for name in ("enh", "sens", "uncert"):
            products[name], arrays[name], masks[name], geometry_masks[name] = _read_masked(paths[name], footprint, s2_crs, bounds)
        same_window = all(products[n]["window"] == products["enh"]["window"] and products[n]["window_transform"] == products["enh"]["window_transform"] for n in ("sens", "uncert"))
        if same_window:
            common_support, native_pixels = int((masks["enh"] & masks["sens"] & masks["uncert"]).sum()), int(geometry_masks["enh"].sum())
        else:
            grid_ok = False
    else:
        products.update(native_meta)
    products["plm"], plm_values, plm_valid, _ = _read_masked(paths["plm"], footprint, s2_crs, bounds)
    positive_value = plm_valid & (plm_values > 0)
    lag = (parse_time(scene["datetime"]) - emit_time).total_seconds() / 3600
    peak_in = peak_in_s2_footprint(peak, s2_crs, bounds)
    reasons = []
    if not timing_pass(lag, threshold): reasons.append("timing_exceeds_threshold")
    if not grid_ok: reasons.append("native_enh_sens_uncert_grid_mismatch")
    if common_support == 0: reasons.append("no_common_native_support")
    if not peak_in: reasons.append("native_peak_outside_s2_footprint")
    stack_hash = sha256(s2_path)
    mask_hash = sha256(mask_path)
    uncertainty_nonnegative = None
    sensitivity_finite = None
    if grid_ok and common_support:
        common = masks["enh"] & masks["sens"] & masks["uncert"]
        uncertainty_nonnegative = round(float((arrays["uncert"][common] >= 0).mean()), 6)
        sensitivity_finite = round(float(np.isfinite(arrays["sens"][common]).mean()), 6)
    if stack_hash != scene.get("stack_sha256"): reasons.append("stack_hash_mismatch")
    if mask_hash != scene.get("mask_sha256"): reasons.append("mask_hash_mismatch")
    # PLM is continuous enhancement, not a binary plume mask. Its positive-value
    # count is descriptive only and cannot gate scene QA or define a target.
    return {"scene_id": scene["scene_id"], "datetime": scene["datetime"], "lag_hours": round(lag, 3), "timing_threshold_hours": threshold, "strict_timing_pass": timing_pass(lag, threshold), "s2": {"crs": s2_meta["crs"], "shape": s2_meta["shape"], "transform": s2_meta["transform"], "stack": str(s2_path), "stack_sha256_actual": stack_hash, "stack_sha256_manifest": scene.get("stack_sha256"), "stack_hash_matches_manifest": stack_hash == scene.get("stack_sha256"), "mask": str(mask_path), "mask_sha256_actual": mask_hash, "mask_sha256_manifest": scene.get("mask_sha256"), "mask_hash_matches_manifest": mask_hash == scene.get("mask_sha256"), "cached_mask_positive_pixels": scene.get("mask_positive_pixels"), "cloud_pct": scene["scene_cloud_cover_pct"], "roi_clear_pct": scene["roi_clear_pct"]}, "footprint_lonlat": list(transform_bounds(s2_crs, "EPSG:4326", *bounds, densify_pts=21)), "native_grid_alignment_ok": grid_ok, "native_products": products, "common_support_pixels": common_support, "geometry_inside_pixels": native_pixels, "common_support_fraction": round(common_support / native_pixels, 6) if native_pixels else 0.0, "native_plm_positive_value_pixels": int(positive_value.sum()), "plm_interpretation": "continuous enhancement; positive values are not a binary plume mask", "native_peak_latlon": list(peak), "peak_in_s2_footprint": peak_in, "uncertainty_nonnegative_fraction": uncertainty_nonnegative, "sensitivity_finite_fraction": sensitivity_finite, "sensitivity_admissibility": "metadata-defined units; no threshold assumed", "fail_reasons": reasons}


def build_report(strict_hours: float = 6.0) -> dict[str, Any]:
    acquisitions = []
    for stamp in STAMPS:
        manifest_path = next(PAIR_ROOT.glob(f"EMIT_L2B_CH4PLM_002_{stamp}/manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = native_paths(stamp)
        with rasterio.open(paths["plm"]) as ds:
            tags = ds.tags()
            peak = (float(tags["Latitude of max concentration"]), float(tags["Longitude of max concentration"]))
        emit_time = parse_time(manifest["emit_datetime"])
        acquisitions.append({"stamp": manifest["emit_datetime"], "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "scenes": [inspect_scene(scene, paths, emit_time, peak, strict_hours) for scene in manifest["scenes"]]})
    ready = [s for a in acquisitions for s in a["scenes"] if not s["fail_reasons"]]
    return {"schema_version": 3, "tool_version": "v4", "scope": "local-only native EMIT/Sentinel readiness; six cached scenes; rasterio bounded windows; no network", "strict_timing_hours": strict_hours, "acquisitions": acquisitions, "summary": {"cached_scene_count": 6, "strict_timing_scene_count": sum(s["strict_timing_pass"] for a in acquisitions for s in a["scenes"]), "strict_engineering_pair_ready": bool(ready), "national_data_readiness_claim": False, "independent_acquisition_count": 3}}


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# GTM Model6 native pair readiness {report.get('tool_version', 'v2')}", "", f"Strict timing threshold: **{report['strict_timing_hours']} h**", "", "| Acquisition | Scene | Lag h | Grid | Common support | PLM positive-valued pixels | Peak in footprint | Status |", "|---|---|---:|---|---:|---:|---|---|"]
    for group in report["acquisitions"]:
        for row in group["scenes"]:
            status = "PASS" if not row["fail_reasons"] else "; ".join(row["fail_reasons"])
            lines.append(f"| {group['stamp']} | {row['scene_id']} | {row['lag_hours']:.3f} | {row['native_grid_alignment_ok']} | {row['common_support_fraction']:.3%} | {row['native_plm_positive_value_pixels']} | {row['peak_in_s2_footprint']} | {status} |")
    lines += ["", "Native ENH/SENS/UNCERT support is ANDed only after CRS, shape, affine, and bounded-window transform equality checks. PLM is evaluated in its own native grid through transformed geometry. Its positive-valued count is descriptive continuous enhancement, not a plume mask or a readiness gate.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-hours", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "GTM_Model6" / "GTM_Model6_native_pair_readiness_20260922_v4")
    args = parser.parse_args()
    if args.output_dir.exists(): raise SystemExit(f"Refusing to overwrite existing directory: {args.output_dir}")
    report = build_report(args.strict_hours)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "README.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": report["summary"]}, indent=2))


if __name__ == "__main__": main()
