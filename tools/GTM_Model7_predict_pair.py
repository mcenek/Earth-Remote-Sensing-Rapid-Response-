"""Run retained v11 on one aligned, local Sentinel pair under the MARS DN contract.

Example (PowerShell, repository root):
  .venv/Scripts/python.exe tools/GTM_Model7_predict_pair.py --current current.tif --reference earlier_clear.tif --current-cloud current_cloud.tif --reference-cloud earlier_cloud.tif --current-utc 2024-11-05T17:35:21Z --reference-utc 2024-10-23T17:24:01Z --radiometry mars_cached_l1c_dn_uint16 --output outputs/my_pair_prediction

The declared input convention is the cached MARS-S2L L1C-style uint16 DN crop:
native 200x200 at 10 m, first six bands B02/B03/B04/B08/B11/B12, DN nodata 0,
CloudSEN12 codes 0..4 with 0 clear. The program cannot convert arbitrary JP2,
L2A, offset-adjusted or radiometrically scaled imagery to this convention.
Output scores are 20 m model predictions repeated to the 10 m output grid;
this does not add spatial information. No label is needed for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import GTM_Model7_train_repaired_hard_v11 as v11  # noqa: E402

RADIOMETRY = "mars_cached_l1c_dn_uint16"
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
FREEZE = ROOT / "outputs/GTM_Model7_hard_negative_v11_20261025/selection/freeze.json"
SELECTION_RECEIPT = FREEZE.parent / "receipt.json"
TRAINING_RECEIPT = ROOT / "outputs/GTM_Model7_hard_negative_v11_20261025/fit/receipt.json"
TRAINING_REPLAY = ROOT / "outputs/GTM_Model7_hard_negative_v11_20261025/fit/e2e_receipt.json"
SUPPORTED_MODEL = "v11"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_grid(ds: rasterio.io.DatasetReader, path: Path) -> None:
    if ds.driver != "GTiff" or ds.shape != (200, 200) or ds.crs is None:
        raise ValueError(f"Expected georeferenced 200x200 GeoTIFF: {path}")
    unit = ds.crs.linear_units_factor if ds.crs.is_projected else None
    if unit is None or not math.isclose(float(unit[1]), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Expected projected CRS with metre units: {path}")
    t = ds.transform
    if (not all(math.isfinite(float(z)) for z in t[:6]) or t.a != 10 or t.b != 0 or
        t.d != 0 or t.e != -10):
        raise ValueError(f"Expected aligned north-up native 10 m grid: {path}")


def _read_image(path: Path) -> tuple[np.ndarray, dict]:
    if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected local GeoTIFF: {path}")
    with rasterio.open(path) as ds:
        _require_grid(ds, path)
        tags = {key.upper(): value.upper() for key, value in ds.tags().items()}
        for band in range(1, min(ds.count, 6) + 1):
            tags.update({key.upper(): value.upper() for key, value in ds.tags(band).items()})
        if (ds.count not in (6, 12) or set(ds.dtypes) != {"uint16"} or
            tuple(ds.descriptions[:6]) != BANDS or ds.nodata != 0 or
            any(float(x) != 1 for x in ds.scales[:6]) or
            any(float(x) != 0 for x in ds.offsets[:6]) or
            any("RADIO_ADD_OFFSET" in key or "BOA_ADD_OFFSET" in key for key in tags) or
            any("L2A" in value or "LEVEL-2A" in value for key, value in tags.items() if "PROCESSING_LEVEL" in key)):
            raise ValueError(f"Unsupported band, dtype, nodata, scale or offset contract: {path}")
        values = ds.read(indexes=(1, 2, 3, 4, 5, 6))
        profile = {"crs": ds.crs.to_string(), "transform": tuple(ds.transform)[:6],
                   "width": ds.width, "height": ds.height, "count": ds.count,
                   "dtype": "uint16", "nodata": ds.nodata,
                   "descriptions_first_six": list(ds.descriptions[:6])}
    return values, profile


def _read_cloud(path: Path) -> tuple[np.ndarray, dict]:
    if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected local cloud GeoTIFF: {path}")
    with rasterio.open(path) as ds:
        _require_grid(ds, path)
        # Historical MARS masks declare zero as NoData despite zero being clear.
        # Newly generated masks correctly declare class 4 invalid. Read raw
        # class codes in both cases; never mask clear zero pixels via metadata.
        if ds.count != 1 or ds.dtypes[0] != "uint8" or ds.nodata not in (0, 4):
            raise ValueError(f"Unsupported CloudSEN12 cloud raster: {path}")
        cloud = ds.read(1)
        profile = {"crs": ds.crs.to_string(), "transform": tuple(ds.transform)[:6],
                   "width": ds.width, "height": ds.height, "count": ds.count,
                   "dtype": "uint8", "nodata": ds.nodata}
    if not np.isin(cloud, (0, 1, 2, 3, 4)).all():
        raise ValueError(f"Unsupported cloud code: {path}")
    return cloud, profile


def _same_grid(profiles: list[dict]) -> None:
    first = profiles[0]
    if any(p["crs"] != first["crs"] or p["transform"] != first["transform"] or
           p["width"] != first["width"] or p["height"] != first["height"] for p in profiles[1:]):
        raise ValueError("Current, earlier reference and cloud grids/CRS differ")


def load_pair(current: Path, reference: Path, current_cloud: Path, reference_cloud: Path,
              radiometry: str) -> tuple[dict[str, np.ndarray], dict]:
    if radiometry != RADIOMETRY:
        raise ValueError("Only explicit verified MARS cached L1C uint16 DN convention is supported")
    paths = [p.resolve(strict=True) for p in (current, reference, current_cloud, reference_cloud)]
    if len(set(paths)) != 4:
        raise ValueError("Current/reference images and both cloud rasters must be distinct files")
    a, pa = _read_image(paths[0])
    b, pb = _read_image(paths[1])
    ca, pca = _read_cloud(paths[2])
    cb, pcb = _read_cloud(paths[3])
    _same_grid([pa, pb, pca, pcb])
    if np.array_equal(a, b):
        raise ValueError("Reference image is identical to current image")
    native_valid = ((a > 0).all(0) & (b > 0).all(0) & (ca == 0) & (cb == 0))
    valid20 = native_valid.reshape(100, 2, 100, 2).all(axis=(1, 3))
    valid_native = np.repeat(np.repeat(valid20, 2, 0), 2, 1)
    if not valid20.any() or not np.isfinite(a[:, native_valid]).all() or not np.isfinite(b[:, native_valid]).all():
        raise ValueError("Empty or nonfinite supported pair")
    # Preserve the earlier 10 m -> 20 m recipe and float32 mean used by v11.
    target20 = a.astype(np.float32).transpose(1, 2, 0).reshape(100, 2, 100, 2, 6).mean(axis=(1, 3))
    reference20 = b.astype(np.float32).transpose(1, 2, 0).reshape(100, 2, 100, 2, 6).mean(axis=(1, 3))
    data = {"target_image": target20, "reference_image": reference20,
            "valid": valid20, "valid_native": valid_native}
    if not all(np.isfinite(x).all() for x in (target20, reference20)):
        raise ValueError("Nonfinite 20 m DN features")
    receipt = {"radiometry_declaration": radiometry, "input_grid": pa,
               "inputs": {role: {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                                  "grid": profile} for role, path, profile in zip(
                                      ("current", "reference", "current_cloud", "reference_cloud"),
                                      paths, (pa, pb, pca, pcb))},
               "support_rule": "all four native pixels in each 20 m block: both dates six DN bands >0 and both CloudSEN12 codes 0",
               "valid20_pixels": int(valid20.sum()), "valid_native_pixels": int(valid_native.sum()),
               "model_pixel_size_m": 20, "output_grid_pixel_size_m": 10}
    return data, receipt


def load_retained_v11() -> tuple[torch.nn.Module, dict, dict]:
    selection = json.loads(SELECTION_RECEIPT.read_text(encoding="utf-8"))
    if not selection.get("passed") or selection.get("freeze_sha256") != sha(FREEZE) or selection.get("candidate_rows") != 88:
        raise ValueError("Retained v11 selection receipt drift")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    choice = frozen["models"]["siamese"]
    meta = frozen["metadata"]
    if (frozen["selection_kind"] != "hard_negative_F1_exposed_DEV_v11" or
        not choice["qualified_development_gate"] or choice["steps"] != 2400 or
        choice["threshold"] != .65 or frozen["threshold"] != .65 or
        not frozen["improved_development_gate"] or frozen["useful_target_met"] or
        meta["config"]["architecture"] != "v7_early_fusion" or
        meta["normalization"].get("radiometry") != "log1p_dn_date_band_median_centered"):
        raise ValueError("Retained v11 contract changed")
    if sha(Path(meta["manifest"])) != meta["manifest_sha256"]:
        raise ValueError("Frozen v11 source manifest changed")
    if (sha(TRAINING_RECEIPT) != frozen["training_receipt_sha256"] or
        sha(TRAINING_REPLAY) != frozen["training_replay_sha256"] or
        not json.loads(TRAINING_REPLAY.read_text(encoding="utf-8"))["passed"]):
        raise ValueError("Frozen v11 training/replay receipt changed")
    checkpoint = Path(choice["checkpoint"])
    if sha(checkpoint) != choice["checkpoint_sha256"]:
        raise ValueError("Retained v11 checkpoint hash mismatch")
    v11.configure(meta)
    v11.set_branch("mars_repaired")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("steps") != choice["steps"] or state.get("normalization") != meta["normalization"] or state.get("model_name") != "siamese":
        raise ValueError("Checkpoint preprocessing identity changed")
    model = v11.new_model("siamese").cpu().eval()
    model.load_state_dict(state["model"], strict=True)
    return model, meta["normalization"], {"model": "retained_v11", "freeze_sha256": sha(FREEZE),
                                           "selection_receipt_sha256": sha(SELECTION_RECEIPT),
                                           "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                                           "trainer_sha256": meta["trainer_sha256"],
                                           "normalization": meta["normalization"],
                                           "cutoff": choice["threshold"], "development_exposure": "exposed DEV only",
                                           "nationwide_accuracy_established": False}


def features(data: dict[str, np.ndarray], norm: dict) -> np.ndarray:
    valid = data["valid"].astype(bool)
    center = np.asarray(norm["center"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    if (norm.get("radiometry") != "log1p_dn_date_band_median_centered" or center.shape != (6,) or
        scale.shape != (6,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0) or
        norm.get("clip_standardized") != 8.0 or norm.get("invalid_fill") != 0.0):
        raise ValueError("Unsupported frozen normalizer")
    dates = []
    for key in ("target_image", "reference_image"):
        raw = data[key]
        transformed = np.log1p(raw[valid].astype(np.float64))
        median = np.median(transformed, axis=0)
        date = np.zeros((100, 100, 6), dtype=np.float32)
        date[valid] = np.clip((transformed - median - center) / scale, -8, 8).astype(np.float32)
        dates.append(date)
    current, prior = dates
    result = np.concatenate((current, prior, current - prior, np.abs(current - prior)), axis=-1).astype(np.float32)
    result[~valid] = 0
    if result.shape != (100, 100, 24) or not np.isfinite(result).all():
        raise ValueError("Invalid v11 model features")
    return result


def predict(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    inputs = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).float()
    with torch.inference_mode():
        p = torch.sigmoid(model(inputs))[0, 0].numpy().astype(np.float32)
    if p.shape != (100, 100) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise FloatingPointError("Invalid v11 score map")
    return p


def _write_tiff(path: Path, values: np.ndarray, crs: str, transform: tuple, *, nodata=None) -> None:
    with rasterio.open(path, "w", driver="GTiff", height=200, width=200, count=1,
                       dtype=values.dtype, crs=crs, transform=Affine(*transform),
                       compress="deflate", nodata=nodata) as ds:
        ds.write(values, 1)


def declared_utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an explicit UTC acquisition timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def run(current: Path, reference: Path, current_cloud: Path, reference_cloud: Path,
        radiometry: str, output: Path, *, current_utc: str, reference_utc: str,
        model_name: str = SUPPORTED_MODEL) -> dict:
    if model_name != SUPPORTED_MODEL:
        raise ValueError("v14 requires a separate eligible frozen selection and verified inference adapter")
    current_time = declared_utc(current_utc, "current_utc")
    reference_time = declared_utc(reference_utc, "reference_utc")
    if reference_time >= current_time:
        raise ValueError("Reference acquisition must be earlier than current acquisition")
    if output.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {output}")
    data, input_receipt = load_pair(current, reference, current_cloud, reference_cloud, radiometry)
    model, norm, model_receipt = load_retained_v11()
    x = features(data, norm)
    p20 = predict(model, x)
    repeat = predict(model, x)
    if not np.array_equal(p20, repeat):
        raise RuntimeError("CPU inference is not repeatable")
    p_native = np.repeat(np.repeat(p20, 2, 0), 2, 1)
    valid_native = data["valid_native"].astype(bool)
    mask_native = ((p_native >= model_receipt["cutoff"]) & valid_native).astype(np.uint8)
    grid = input_receipt["input_grid"]
    output.mkdir(parents=True)
    score_tiff = output / "probability.tif"
    mask_tiff = output / "mask.tif"
    valid_tiff = output / "valid.tif"
    data_file = output / "prediction.npz"
    _write_tiff(score_tiff, np.where(valid_native, p_native, np.nan).astype(np.float32), grid["crs"], grid["transform"], nodata=np.nan)
    _write_tiff(mask_tiff, mask_native, grid["crs"], grid["transform"])
    _write_tiff(valid_tiff, valid_native.astype(np.uint8), grid["crs"], grid["transform"])
    np.savez_compressed(data_file, probability20=p20, probability_native=p_native,
                        mask_native=mask_native, valid20=data["valid"].astype(np.uint8),
                        valid_native=valid_native.astype(np.uint8))
    with np.load(data_file, allow_pickle=False) as check:
        if not all(np.array_equal(check[k], v) for k, v in {
            "probability20": p20, "probability_native": p_native,
            "mask_native": mask_native, "valid20": data["valid"].astype(np.uint8),
            "valid_native": valid_native.astype(np.uint8)}.items()):
            raise RuntimeError("Saved prediction serialization changed")
    receipt = {"schema_version": 1, "scope": "local pair inference; no label used",
               "status": "complete", "input": input_receipt, "model": model_receipt,
               "acquisition_times": {"current_utc_declared": current_time.isoformat(),
                                     "reference_utc_declared": reference_time.isoformat(),
                                     "reference_before_current": True,
                                     "provenance": "caller declaration; TIFF pixels alone do not prove acquisition time"},
               "input_recipe": "native 10m first-six source DN; all-four two-date clear support; float32 2x2 mean; frozen per-date log1p median/normalizer; 24 early-fusion channels",
               "score_grid": "20m block-constant model score repeated to 10m output grid; not extra spatial detail",
               "valid_native_pixels": int(valid_native.sum()), "marked_native_pixels": int(mask_native.sum()),
               "cpu_repeat_exact": True, "script_sha256": sha(Path(__file__)),
               "outputs": {p.name: sha(p) for p in (score_tiff, mask_tiff, valid_tiff, data_file)}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current-cloud", type=Path, required=True)
    parser.add_argument("--reference-cloud", type=Path, required=True)
    parser.add_argument("--current-utc", required=True, help="Current acquisition time with explicit UTC offset")
    parser.add_argument("--reference-utc", required=True, help="Earlier reference acquisition time with explicit UTC offset")
    parser.add_argument("--radiometry", required=True, help=f"Required exact declaration: {RADIOMETRY}")
    parser.add_argument("--model", default=SUPPORTED_MODEL, choices=("v11", "v14"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.current, args.reference, args.current_cloud, args.reference_cloud,
                  args.radiometry, args.output.resolve(), current_utc=args.current_utc,
                  reference_utc=args.reference_utc, model_name=args.model)
    print(json.dumps({"output": str(args.output.resolve()), "marked_native_pixels": receipt["marked_native_pixels"],
                      "valid_native_pixels": receipt["valid_native_pixels"]}), flush=True)


if __name__ == "__main__":
    main()
