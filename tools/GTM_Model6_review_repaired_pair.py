"""Local QA of one repaired pair; preserve raw values, never fabricate labels.

Native statistics use exact transformed crop geometry before display resampling.
Acquisition checksums and grids are verified; conflicting radiometry metadata
blocks training approval. This tool has no network calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds, xy
from rasterio.warp import reproject, transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from GTM_Model6_native_pair_readiness import _read_masked, grid_signature, native_paths, parse_time, sha256
from GTM_Model6_repair_native_pair import DEFAULT_MANIFEST, PEAK_LONLAT, ROOT, SCENE_ID

DEFAULT_CROP = ROOT / "outputs/GTM_Model6/GTM_Model6_georgia_crop_repair_20260922"


def stats(values: np.ndarray, valid: np.ndarray) -> dict:
    selected = values[valid]
    return {"valid_pixels": int(selected.size), "min": float(selected.min()) if selected.size else None,
            "max": float(selected.max()) if selected.size else None,
            "mean": float(selected.mean()) if selected.size else None,
            "negative_pixels": int((selected < 0).sum()), "positive_value_pixels": int((selected > 0).sum())}


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Acquisition checksum mismatch: {path.name}")
    return actual


def radiometry_check(properties: dict, metadata: dict, values: np.ndarray, valid: np.ndarray) -> dict:
    conflict = properties.get("earthsearch:boa_offset_applied") is True and float(metadata["offset"]) != 0.0
    literal = values[valid].astype(np.float64) * float(metadata["scale"]) + float(metadata["offset"])
    return {"boa_offset_applied": properties.get("earthsearch:boa_offset_applied"),
            "literal_stac_scale": metadata["scale"], "literal_stac_offset": metadata["offset"],
            "offset_metadata_conflict": conflict,
            "literal_offset_negative_fraction": float((literal < 0).mean()) if literal.size else None,
            "effective_reflectance_conversion_approved": False}


def common_native_support(products: dict, masks: dict, inside: dict) -> np.ndarray:
    for name in ("sens", "uncert"):
        if (grid_signature(products[name]) != grid_signature(products["enh"])
                or products[name]["window"] != products["enh"]["window"]
                or products[name]["window_transform"] != products["enh"]["window_transform"]
                or not np.array_equal(inside[name], inside["enh"])):
            raise ValueError(f"Native grid or geometry mismatch: {name}")
    return masks["enh"] & masks["sens"] & masks["uncert"]


def plm_characterization(values: np.ndarray, valid: np.ndarray) -> dict:
    # A positive enhancement value is not, by itself, a scientifically valid plume label.
    return {"interpretation": "continuous enhancement; no binary label derived",
            "binary_label_created": False, "statistics": stats(values, valid)}


def display_native(values: np.ndarray, valid: np.ndarray, meta: dict, bbox: tuple) -> np.ndarray:
    """A separate nearest-neighbor display grid; never used to compute metrics."""
    result = np.full((43, 43), np.nan, dtype=np.float32)
    source = np.where(valid, values, np.nan).astype(np.float32)
    reproject(source, result, src_transform=Affine(*meta["window_transform"][:6]), src_crs=meta["crs"],
              src_nodata=np.nan, dst_transform=from_bounds(*bbox, 43, 43), dst_crs="EPSG:32616",
              dst_nodata=np.nan, resampling=Resampling.nearest)
    return result


def render_figure(path: Path, bands: dict, arrays: dict, masks: dict, products: dict,
                  common: np.ndarray, bbox: tuple, peak_xy: tuple, maximum_xy: tuple, lag: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rgb = np.stack([bands[n].astype(np.float32) for n in ("red", "green", "blue")], axis=-1)
    valid_rgb = np.all(rgb != 0, axis=-1)
    for index in range(3):
        lo, hi = np.percentile(rgb[..., index][valid_rgb], [2, 98])
        rgb[..., index] = np.clip((rgb[..., index] - lo) / max(float(hi - lo), 1), 0, 1) ** (1 / 2.2)
    rgb[~valid_rgb] = 0.65
    extent = [bbox[0] / 1000, bbox[2] / 1000, bbox[1] / 1000, bbox[3] / 1000]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.6), layout="constrained", sharex=True, sharey=True)
    fig.suptitle(f"Georgia | Sentinel {abs(lag) * 60:.1f} minutes before EMIT | 20 October 2024\n"
                 "Observed input and target data only — no model prediction", fontsize=15)
    axes[0].imshow(rgb, extent=extent, origin="upper", interpolation="nearest")
    axes[0].set_title("Sentinel RGB · 10 m native\nRaw DN display stretch; radiometry under review", fontsize=10)
    for axis, name, title, cmap, limits in (
            (axes[1], "enh", "EMIT methane enhancement", "viridis", (-1000, 10000)),
            (axes[2], "uncert", "EMIT retrieval uncertainty", "magma", (0, 5000))):
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad("#a6a6a6")
        shown = display_native(arrays[name], common, products[name], bbox)
        artist = axis.imshow(shown, extent=extent, origin="upper", interpolation="nearest",
                             cmap=palette, vmin=limits[0], vmax=limits[1])
        axis.set_title(title + "\nNative measurements; ~60 m display cells", fontsize=10)
        fig.colorbar(artist, ax=axis, orientation="horizontal", pad=.06, shrink=.94, extend="both", label="ppm m")
    for axis in axes:
        axis.scatter([peak_xy[0] / 1000], [peak_xy[1] / 1000], marker="o", facecolors="none",
                     edgecolors="#ffe34d", s=120, linewidths=1.6)
        axis.scatter([maximum_xy[0] / 1000], [maximum_xy[1] / 1000], marker="+", c="#ff4b4b", s=80, linewidths=1.5)
        axis.set_aspect("equal")
        axis.set_xlabel("UTM 16N easting (km)")
        axis.ticklabel_format(useOffset=False)
    axes[0].set_ylabel("UTM 16N northing (km)")
    axes[0].legend(handles=[Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#a89000", color="none", label="Published maximum coordinate"),
                        Line2D([], [], marker="+", color="#dd3030", linestyle="none", label="Maximum in this native window")],
                   loc="upper center", bbox_to_anchor=(.5, -.17), frameon=False, fontsize=9)
    fig.supxlabel("All panels share the exact 2.56 km footprint in EPSG:32616. Gray = unsupported.\n"
                  "RGB uses a 2–98% per-channel stretch. Colorbar saturation is explicit; no binary plume mask or source origin is inferred.", fontsize=10)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(crop_dir: Path, output_dir: Path) -> dict:
    if output_dir.resolve() == crop_dir.resolve():
        raise ValueError("Derived QA must have its own directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = crop_dir / "report.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "crop_complete" or receipt["scene_id"] != SCENE_ID:
        raise ValueError("Expected the completed frozen Georgia acquisition")
    verify_hash(DEFAULT_MANIFEST, receipt["manifest_sha256"])
    stac_path = ROOT / receipt["stac"]["saved_path"]
    verify_hash(stac_path, receipt["stac"]["sha256"])
    item = json.loads(stac_path.read_text(encoding="utf-8"))
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    lag = (parse_time(item["properties"]["datetime"]) - parse_time(manifest["emit_datetime"])).total_seconds() / 3600
    bbox = tuple(float(x) for x in receipt["bbox_projected"])
    bands, band_meta = {}, {}
    for name, entry in receipt["assets"].items():
        path = ROOT / entry["output"]
        actual_hash = verify_hash(path, entry["sha256"])
        metadata = entry["metadata"][0]
        resolution = entry["probe"]["resolution_m"]
        expected_shape = (round((bbox[3] - bbox[1]) / resolution), round((bbox[2] - bbox[0]) / resolution))
        with rasterio.open(path) as source:
            if (source.crs.to_epsg() != 32616 or tuple(source.bounds) != bbox or source.shape != expected_shape
                    or source.transform != Affine(resolution, 0, bbox[0], 0, -resolution, bbox[3])
                    or source.nodata != metadata["nodata"] or source.dtypes[0] != entry["probe"]["dtype"]):
                raise ValueError(f"Acquired native grid or dtype mismatch: {name}")
            if name != "scl" and (source.scales[0] != metadata["scale"] or source.offsets[0] != metadata["offset"]):
                raise ValueError(f"Acquired literal STAC scale/offset mismatch: {name}")
            values = source.read(1)
            valid = np.isfinite(values) & (values != source.nodata)
            bands[name] = values
            band_meta[name] = {"sha256": actual_hash, "shape": list(source.shape), "crs": str(source.crs),
                               "transform": list(source.transform), "raw_dn": stats(values, valid)}
            if name != "scl":
                band_meta[name]["radiometry"] = radiometry_check(item["properties"], metadata, values, valid)
    clear_fraction = float(np.isin(bands["scl"], [4, 5, 6]).mean())
    if clear_fraction != receipt["strict_clear_fraction"]:
        raise ValueError("SCL clear fraction differs from acquisition")
    peak_x, peak_y = transform("EPSG:4326", "EPSG:32616", [PEAK_LONLAT[0]], [PEAK_LONLAT[1]])
    peak_xy = (peak_x[0], peak_y[0])
    peak_inside = bbox[0] <= peak_xy[0] < bbox[2] and bbox[1] <= peak_xy[1] < bbox[3]
    if not peak_inside or abs(lag) > 6:
        raise ValueError("Repaired pair fails peak-containment or timing policy")
    polygon = {"type": "Polygon", "coordinates": [[[bbox[0], bbox[1]], [bbox[0], bbox[3]],
               [bbox[2], bbox[3]], [bbox[2], bbox[1]], [bbox[0], bbox[1]]]]}
    products, arrays, masks, inside = {}, {}, {}, {}
    for name, path in native_paths("20241020T170504_003677").items():
        products[name], arrays[name], masks[name], inside[name] = _read_masked(path, [polygon], "EPSG:32616", bbox)
        products[name]["window_statistics"] = stats(arrays[name], masks[name])
    common = common_native_support(products, masks, inside)
    if not common.any():
        raise ValueError("No common native ENH/SENS/UNCERT support")
    plm = plm_characterization(arrays["plm"], masks["plm"])
    plm["units"] = products["plm"]["units"]
    same_plm_grid = (grid_signature(products["plm"]) == grid_signature(products["enh"])
                     and products["plm"]["window"] == products["enh"]["window"]
                     and products["plm"]["window_transform"] == products["enh"]["window_transform"])
    # Full source raster shapes may differ, even with exactly matching window grids.
    same_window = (products["plm"]["crs"] == products["enh"]["crs"]
                   and arrays["plm"].shape == arrays["enh"].shape
                   and products["plm"]["window_transform"] == products["enh"]["window_transform"])
    plm["same_full_grid_as_enh"] = same_plm_grid
    plm["same_window_grid_as_enh"] = same_window
    if same_window:
        shared = masks["plm"] & masks["enh"]
        plm["equal_to_enh_on_common_valid_pixels"] = bool(np.array_equal(arrays["plm"][shared], arrays["enh"][shared])) if shared.any() else None
    max_index = np.unravel_index(np.where(common, arrays["enh"], -np.inf).argmax(), common.shape)
    max_x, max_y = xy(Affine(*products["enh"]["window_transform"][:6]), *max_index)
    mx, my = transform(products["enh"]["crs"], "EPSG:32616", [max_x], [max_y])
    maximum_xy = (mx[0], my[0])
    figure = output_dir / "GTM_Model6_georgia_repaired_pair_scientific.png"
    render_figure(figure, bands, arrays, masks, products, common, bbox, peak_xy, maximum_xy, lag)
    report = {"schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
              "tool": "GTM_Model6_review_repaired_pair.py", "source_sha256": sha256(Path(__file__)),
              "status": "geometry_verified_radiometry_review_required", "training_approved": False,
              "network_used": False, "scope": "one near-time observed pair; not an independent training cohort",
              "acquisition_receipt_sha256": sha256(receipt_path), "stac_sha256": sha256(stac_path),
              "manifest_sha256": sha256(DEFAULT_MANIFEST), "scene_id": SCENE_ID,
              "sentinel_minus_emit_hours": lag, "crop_bbox_epsg32616": list(bbox), "peak_in_crop": peak_inside,
              "metadata_peak_lonlat": list(PEAK_LONLAT), "metadata_peak_xy": list(peak_xy),
              "native_window_maximum_xy": list(maximum_xy),
              "maximum_to_metadata_peak_m": float(np.linalg.norm(np.array(maximum_xy) - peak_xy)),
              "strict_scl_clear_fraction": clear_fraction, "bands": band_meta, "native": products,
              "common_native_pixels": int(common.sum()), "geometry_native_pixels": int(inside["enh"].sum()),
              "common_native_fraction": float(common.sum() / inside["enh"].sum()),
              "plm": plm, "figure": str(figure.relative_to(ROOT)),
              "remaining_gates": ["Resolve already-applied BOA flag versus literal nonzero raster offset",
                                  "Define native enhancement/uncertainty/sensitivity QA and target semantics",
                                  "Acquire or identify a suitable Sentinel temporal reference",
                                  "Independent grouped positive and reviewed-negative acquisitions"]}
    (output_dir / "GTM_Model6_georgia_repaired_pair_qa.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Repaired Georgia pair: local QA v2", "",
             "**Geometry verified; quantitative training not approved.** This is observed input/target data, not a prediction.", "",
             f"Sentinel precedes EMIT by {abs(lag)*60:.4f} minutes. The exact 2.56 km UTM footprint contains the published maximum coordinate. Native common support: {common.sum()} / {inside['enh'].sum()} pixels; SCL clear fraction: {clear_fraction:.1%}.", "",
             "Every crop SHA matches the frozen acquisition receipt. ENH/SENS/UNCERT native CRS, full grid, window transform and geometry agree before combining support. Display reprojection is separate from native measurements.", "",
             "CH4PLM is continuous enhancement in ppm m, with negative and positive values. PLM > 0 was not used as a plume outline, target, or quality mask. The acquisition peak is not a verified emission origin.", "",
             "Sentinel STAC declares both earthsearch:boa_offset_applied=true and raster offset=-0.1. The raw DN is preserved; the literal tags are not an approved conversion. RGB uses raw DN percentiles solely for visibility. The prior double-offset/clipped RGB display is invalidated.", "",
             "Provider context: [EarthSearch offset discussion](https://github.com/Element84/earth-search/discussions/26) and [current provider README](https://github.com/Element84/earth-search/blob/main/README.md). These do not by themselves reconcile this frozen legacy item's flags.", "",
             f"![Aligned observed data]({figure.name})", "", "Next gates: " + "; ".join(report["remaining_gates"]) + ".", ""]
    (output_dir / "GTM_Model6_georgia_repaired_pair_qa.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-dir", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CROP / "qa_v2")
    args = parser.parse_args()
    result = run(args.crop_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "common_native_pixels", "geometry_native_pixels", "maximum_to_metadata_peak_m", "figure")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
