"""Export verified ERSRR source/reference layers and optional legacy predictions.

This exporter never derives a prediction from scene scores.  If the legacy
Keras runtime and checkpoint are available, ``--checkpoint`` can be used to
add probability/mask layers; otherwise the export remains explicitly
reference-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling


BANDS = ("B2", "B3", "B4", "B11", "B12", "EMIT_CH4")
DEFAULT_CASES = [
    "EarthRemoteSensingRapidResponse/Dataset/validation/20230203T171519_20230203T172113_T14RMU.tif",
    "EarthRemoteSensingRapidResponse/Dataset/validation/20230410T172859_20230410T174507_T13RFQ.tif",
    "EarthRemoteSensingRapidResponse/Dataset/train_test/20230304T160149_20230304T160904_T17SNV_EMIT_L2B_CH4PLM_001_20230218T181054_000663grid_1.tif",
    "EarthRemoteSensingRapidResponse/Dataset/train_test/20240729T165851_20240729T171722_T14QMG_EMIT_L2B_CH4PLM_001_20240606T151017_003236grid_1.tif",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_layer(path: Path, array: np.ndarray, profile: dict, dtype: str = "float32", nodata=None) -> None:
    p = dict(profile, count=1, dtype=dtype, compress="deflate", tiled=True)
    if nodata is not None:
        p["nodata"] = nodata
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.asarray(array, dtype=dtype), 1)


def write_rgb(path: Path, data: np.ndarray, profile: dict) -> None:
    # Reflectance display stretch is per scene/channel and is recorded in metadata.
    rgb = np.empty((3, data.shape[1], data.shape[2]), dtype="uint8")
    stretches = []
    for out, band in enumerate((2, 1, 0)):
        values = data[band].astype("float32")
        finite = values[np.isfinite(values)]
        lo, hi = (np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1.0
        rgb[out] = np.clip((values - lo) * 255.0 / (hi - lo), 0, 255).astype("uint8")
        stretches.append({"band": BANDS[band], "p02": float(lo), "p98": float(hi)})
    p = dict(profile, count=3, dtype="uint8", compress="deflate", tiled=True, nodata=None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(rgb)
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    return stretches


def compact_predict(source_data: np.ndarray, source_profile: dict, model_path: Path, config_path: Path) -> np.ndarray:
    """Run the exact compact_resunet_v1 contract at its declared 128px size."""
    import keras
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = keras.models.load_model(model_path, compile=False)
    h = w = int(config.get("inference_tile_size", 128))
    if source_data.shape[1:] != (h, w):
        raise ValueError(f"Expected native {h}x{w} inference tile, got {source_data.shape}")
    tile = np.moveaxis(source_data[:5].astype("float32"), 0, -1)
    values = np.clip(tile, 0.0, None)
    features = np.log1p(values)
    if config.get("physics_features"):
        b2, b3, b4, b11, b12 = (values[..., index] for index in range(5))
        engineered = np.stack([
            features[..., 4] - features[..., 3],
            (b12 - b11) / (b12 + b11 + 1.0),
            features[..., 3] - features[..., 2],
            features[..., 4] - features[..., 2],
            np.log1p(b11 + b12) - np.log1p(b2 + b3 + b4),
            (b4 - b2) / (b4 + b2 + 1.0),
        ], axis=-1)
        features = np.concatenate([features, engineered], axis=-1)
    mean = np.asarray(config["normalization"]["mean"], dtype="float32")
    std = np.asarray(config["normalization"]["std"], dtype="float32")
    features = (features - mean) / std
    return np.asarray(model.predict(features[None], verbose=0)[0, ..., 0], dtype="float32")


def write_prediction_display(case_dir: Path, source_profile: dict, source_data: np.ndarray, emit: np.ndarray, prob: np.ndarray, model_label: str) -> None:
    """Create a shared 128x128 EPSG:3857 display grid and compact JSON arrays."""
    src_crs = source_profile["crs"]
    src_transform = source_profile["transform"]
    left, bottom, right, top = rasterio.transform.array_bounds(source_data.shape[1], source_data.shape[2], src_transform)
    projected_left, projected_bottom, projected_right, projected_top = transform_bounds(src_crs, "EPSG:3857", left, bottom, right, top)
    width = height = 128
    dst_transform = from_bounds(projected_left, projected_bottom, projected_right, projected_top, width, height)
    dst_profile = dict(source_profile, crs="EPSG:3857", transform=dst_transform, width=width, height=height, count=1, dtype="float32", compress="deflate", tiled=True, nodata=np.nan)
    rgb = np.empty((3, source_data.shape[1], source_data.shape[2]), dtype="float32")
    for o, band in enumerate((2, 1, 0)):
        rgb[o] = source_data[band]
    rgb_dst = np.empty((3, height, width), dtype="float32")
    for o in range(3):
        reproject(rgb[o], rgb_dst[o], src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    rgb_u8 = np.clip((rgb_dst - np.nanpercentile(rgb_dst, 2, axis=(1, 2))[:, None, None]) * 255 / (np.nanpercentile(rgb_dst, 98, axis=(1, 2)) - np.nanpercentile(rgb_dst, 2, axis=(1, 2)))[:, None, None], 0, 255).astype("uint8")
    with rasterio.open(case_dir / f"display_rgb_{model_label}_3857.tif", "w", **dict(dst_profile, count=3, dtype="uint8", nodata=None)) as dst:
        dst.write(rgb_u8)
    truth_support = np.isfinite(emit) & (emit != -9999)
    truth = np.where(truth_support, (emit > 300).astype("float32"), np.nan)
    model_valid = np.isfinite(source_data[:5]).all(axis=0)
    layers = {"probability": prob, "truth": truth.astype("float32"), "valid": model_valid.astype("float32")}
    json_arrays = {}
    for name, arr in layers.items():
        dst = np.empty((height, width), dtype="float32")
        reproject(arr.astype("float32"), dst, src_transform=src_transform if arr.shape == source_data.shape[1:] else Affine(src_transform.a * 2, 0, src_transform.c, 0, src_transform.e * 2, src_transform.f), src_crs=src_crs, dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.nearest)
        write_layer(case_dir / f"display_{name}_{model_label}_3857.tif", dst, dst_profile, nodata=np.nan)
        if name == "truth":
            json_arrays[name] = [None if not np.isfinite(value) else int(value) for value in dst.ravel()]
        else:
            json_arrays[name] = dst.round(6).ravel().tolist()
    (case_dir / f"overlay_{model_label}_3857.json").write_text(json.dumps({"width": width, "height": height, "crs": "EPSG:3857", "truth_semantics": "EMIT_CH4 > 300 ppm*m where supported; null outside EMIT support; aligned paired reference, not independent truth", "valid_semantics": "finite Sentinel-2 model input pixels", **json_arrays}) + "\n", encoding="utf-8")


def export_case(root: Path, source: Path, out_root: Path, model_path: Path | None = None, config_path: Path | None = None, model_label: str = "compact") -> dict:
    with rasterio.open(source) as src:
        data = src.read(out_dtype="float32")
        profile = src.profile.copy()
        bounds = [float(x) for x in src.bounds]
        crs = src.crs.to_string() if src.crs else None
        transform = [float(x) for x in src.transform[:6]]
        descriptions = list(src.descriptions)
        nodata = src.nodata
    if data.shape[0] != 6 or data.shape[1:] != (256, 256):
        raise ValueError(f"Expected 256x256x6 source, got {data.shape}: {source}")
    sample_id = source.stem
    case_dir = out_root / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rgb_stretch = write_rgb(case_dir / "rgb.tif", data[:5], profile)
    emit = data[5]
    valid = np.isfinite(emit) & (emit != -9999)
    emit_out = np.where(valid, emit, np.nan).astype("float32")
    write_layer(case_dir / "emit_reference.tif", emit_out, profile, nodata=np.nan)
    write_layer(case_dir / "valid_mask.tif", valid.astype("uint8"), profile, dtype="uint8", nodata=0)
    wgs84 = None
    if crs and crs.upper() in {"EPSG:4326", "OGC:CRS84"}:
        wgs84 = {
            "south": bounds[1], "west": bounds[0],
            "north": bounds[3], "east": bounds[2],
            "leaflet_bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        }
    prediction_status = "not_run_no_compatible_keras_runtime"
    prediction_file = mask_file = None
    if model_path and config_path:
        probability = np.empty((256, 256), dtype="float32")
        for yy in (0, 128):
            for xx in (0, 128):
                tile_profile = dict(profile, width=128, height=128, transform=profile["transform"] * Affine.translation(xx, yy))
                probability[yy:yy + 128, xx:xx + 128] = compact_predict(data[:, yy:yy + 128, xx:xx + 128], tile_profile, model_path, config_path)
        prediction_file = f"probability_{model_label}_256_native.tif"
        mask_file = f"mask_{model_label}_256_native.tif"
        pred_profile = dict(profile, width=256, height=256)
        write_layer(case_dir / prediction_file, probability, pred_profile)
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        write_layer(case_dir / mask_file, (probability >= float(model_config["decision_threshold"])).astype("uint8"), pred_profile, dtype="uint8", nodata=0)
        write_prediction_display(case_dir, profile, data, emit, probability, model_label)
        prediction_status = "inference_completed_compact_resunet_v1"
    metadata = {
        "schema": "GTM_Model1_viewer_export_v1",
        "sample_id": sample_id,
        "source_file": str(source.relative_to(root)),
        "source_sha256": sha256(source),
        "band_order": list(BANDS),
        "shape": [256, 256],
        "crs": crs,
        "transform": transform,
        "bounds_native": {"left": bounds[0], "bottom": bounds[1], "right": bounds[2], "top": bounds[3]},
        "bounds_wgs84": wgs84,
        "rgb_file": "rgb.tif",
        "emit_reference_file": "emit_reference.tif",
        "valid_mask_file": "valid_mask.tif",
        "emit_reference_semantics": "aligned EMIT CH4 band from paired source TIFF; not independent truth",
        "truth_mask_semantics": "valid EMIT_CH4 pixels thresholded at >300 ppm*m; nodata pixels are unknown and excluded by valid_mask",
        "cohort": "validation" if "Dataset\\validation" in str(source) or "Dataset/validation" in str(source) else "train_test",
        "valid_pixel_fraction": float(valid.mean()),
        "sentinel_acquisition_date": sample_id[:8] if sample_id[:8].isdigit() else None,
        "source_band_descriptions": descriptions,
        "rgb_stretch": rgb_stretch,
        "prediction_status": prediction_status,
        "inference_strategy": "four native 128x128 tiles stitched to 256x256 source grid; approximately 20m source pixels",
        "predictionAvailable": prediction_status.startswith("inference_completed"),
        "prediction_probability_file": prediction_file,
        "prediction_mask_file": mask_file,
        "checkpoint": str(model_path) if model_path else None,
        "checkpoint_sha256": sha256(model_path) if model_path and model_path.exists() else None,
        "model_config_sha256": sha256(config_path) if config_path and config_path.exists() else None,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument("--compact-model", type=Path, default=None)
    ap.add_argument("--compact-config", type=Path, default=None)
    ap.add_argument("--model-label", default="compact")
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.output or root / "outputs" / "GTM_Model1_viewer_export").resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = args.source or DEFAULT_CASES
    rows = []
    for rel in sources:
        source = (root / rel).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        rows.append(export_case(root, source, out, args.compact_model, args.compact_config, args.model_label))
    predictions = sum(r["prediction_status"].startswith("inference_completed") for r in rows)
    manifest = {"schema": "GTM_Model1_viewer_manifest_v1", "prediction_cases": predictions, "reference_cases": len(rows), "cases": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "reference_cases": len(rows), "prediction_cases": predictions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
