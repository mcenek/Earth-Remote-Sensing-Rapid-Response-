#!/usr/bin/env python3
"""Run the pinned MARS CloudSEN12 model on an already aligned 13-band L1C crop.

Example (no data are downloaded):
  .venv/Scripts/python.exe tools/GTM_Model7_cloud_runtime_v16.py infer \
      --input target_13band_l1c_dn.tif --output target_cloudsen12.tif \
      --radiometry l1c_dn_div10000

This entry point assumes the caller has separately established exact product
identity and the 10 m resampling recipe. It cannot construct seven absent bands
from a six-band methane crop or claim producer mask parity.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs/GTM_Model7_us_transfer_v13_20261026/readiness/cloud_runtime_v16"
WEIGHTS = ROOT / (
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07/_cloudsen12_weights/UNetMobV2_V2.pt"
)
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
WEIGHTS_SHA256 = "218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a"
PACKAGE_SHA256 = "99d10093cef4ae64b5b6ff1a66f620f201ef45f864e9aadbcad29002191286b3"
PADDING_SHA256 = "6f5964dbf9d4b56dd1dee605d0cd27d6ee75609fa0369f1f8a23995575219056"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def published_bands(reader_path: Path) -> tuple[str, ...]:
    """Extract the literal band list from the unmodified registry reader source."""
    tree = ast.parse(reader_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BANDS_S2" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError("Official georeader band constant missing")


def load_model(runtime: Path = RUNTIME, weights: Path = WEIGHTS):
    """Load official package source, with unused plotting/SAFE imports bypassed."""
    site = runtime / "site"
    if sha256(site / "cloudsen12_models/cloudsen12.py") != PACKAGE_SHA256:
        raise ValueError("CloudSEN12 package source hash changed")
    if sha256(site / "cloudsen12_models/utils_torch.py") != PADDING_SHA256:
        raise ValueError("CloudSEN12 padding source hash changed")
    if not weights.is_file() or sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Pinned CloudSEN12 weight absent or changed")
    if published_bands(site / "georeader/readers/S2_SAFE_reader.py") != BANDS:
        raise ValueError("Published Sentinel-2 band order changed")

    # The official cloudsen12_models 1.0.2 module eagerly imports plotting and
    # SAFE parsing. Its ndarray prediction path uses neither. Keep package source
    # untouched and limit the shims to those unused imports.
    plot = types.ModuleType("georeader.plot")
    def plotting_unavailable(*_args, **_kwargs):
        raise RuntimeError("Plotting is outside this isolated inference runtime")
    plot.plot_segmentation_mask = plotting_unavailable
    reader = types.ModuleType("georeader.readers.S2_SAFE_reader")
    reader.BANDS_S2_L1C = list(BANDS)
    reader.BANDS_S2_L2A = [band for band in BANDS if band != "B10"]
    old_plot = sys.modules.get("georeader.plot")
    old_reader = sys.modules.get("georeader.readers.S2_SAFE_reader")
    sys.modules["georeader.plot"] = plot
    sys.modules["georeader.readers.S2_SAFE_reader"] = reader
    sys.path.insert(0, str(site))
    prior_hf = os.environ.get("HF_HUB_OFFLINE")
    prior_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    def downloads_disabled(*_args, **_kwargs):
        raise RuntimeError("Network model downloads are disabled")
    prior_torch_download = torch.hub.load_state_dict_from_url
    try:
        torch.hub.load_state_dict_from_url = downloads_disabled
        from cloudsen12_models import cloudsen12
        import segmentation_models_pytorch as smp
        import timm
        for package in (cloudsen12, smp, timm):
            if not Path(package.__file__).resolve().is_relative_to(site.resolve()):
                raise ValueError(f"Cloud runtime imported a package outside isolation: {package.__name__}")
        if smp.__version__ != "0.3.4" or timm.__version__ != "0.9.7":
            raise ValueError("Isolated segmentation package versions changed")
        prior_cloud_download = cloudsen12.download_weights
        try:
            cloudsen12.download_weights = downloads_disabled
            model = cloudsen12.load_model_by_name(
                "UNetMobV2_V2", weights_folder=str(weights.parent), device=torch.device("cpu")
            )
        finally:
            cloudsen12.download_weights = prior_cloud_download
    finally:
        torch.hub.load_state_dict_from_url = prior_torch_download
        sys.path.remove(str(site))
        for parent_name, attribute, shim in (("georeader", "plot", plot), ("georeader.readers", "S2_SAFE_reader", reader)):
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, attribute, None) is shim:
                delattr(parent, attribute)
        for name, previous in (("georeader.plot", old_plot), ("georeader.readers.S2_SAFE_reader", old_reader)):
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, previous in (("HF_HUB_OFFLINE", prior_hf), ("TRANSFORMERS_OFFLINE", prior_transformers)):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    if tuple(model.bands) != BANDS:
        raise ValueError("Loaded model band order differs from frozen contract")
    return model


def read_input(path: Path):
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        if src.count != 13 or tuple(src.descriptions) != BANDS:
            raise ValueError("Input must have exactly 13 bands in published order with descriptions")
        if src.width != 200 or src.height != 200:
            raise ValueError("Input must be the declared 200x200 crop")
        if any(dtype != "uint16" for dtype in src.dtypes):
            raise ValueError("Input bands must be uint16 L1C digital numbers")
        if src.nodata not in (None, 0):
            raise ValueError("Input nodata must be zero or unset")
        if any(abs(scale - 1.0) > 1e-12 for scale in src.scales) or any(abs(offset) > 1e-12 for offset in src.offsets):
            raise ValueError("Input must have identity scales and zero offsets")
        metadata = [src.tags(), src.tags(ns="IMAGE_STRUCTURE")]
        metadata.extend(src.tags(index) for index in range(1, 14))
        for tags in metadata:
            for key, value in tags.items():
                key_lower, value_lower = key.lower(), str(value).lower()
                is_l2a_level = key_lower in ("processing_level", "product_level", "product_type") and value_lower in ("2a", "level-2a", "level 2a")
                if "offset" in key_lower or is_l2a_level or any(marker in value_lower for marker in ("l2a", "s2msi2a", "level-2a", "level 2a", "surface reflectance", "bottom of atmosphere", "boa_add_offset")):
                    raise ValueError("L2A, surface-reflectance, or radiometric-offset metadata is unsupported")
        if src.crs is None or not src.crs.is_projected or src.crs.linear_units not in ("metre", "meter"):
            raise ValueError("Input needs a projected CRS in metres")
        transform = src.transform
        if not (abs(transform.a - 10) < 1e-8 and abs(transform.e + 10) < 1e-8 and abs(transform.b) < 1e-8 and abs(transform.d) < 1e-8):
            raise ValueError("Input must be north-up on an aligned 10 m grid")
        if abs(transform.c / 10 - round(transform.c / 10)) > 1e-8 or abs(transform.f / 10 - round(transform.f / 10)) > 1e-8:
            raise ValueError("Input grid origin must align to 10 m coordinates")
        data = src.read()
        profile = src.profile.copy()
    invalid = np.any(data == 0, axis=0)
    if invalid.all():
        raise ValueError("Input has no valid 13-band support")
    return data, invalid, profile


def predict_mask(model, data, invalid):
    import numpy as np
    if not isinstance(data, np.ndarray) or data.shape != (13, 200, 200) or data.dtype != np.uint16:
        raise TypeError("Only a uint16 NumPy 13x200x200 L1C DN stack is supported")
    if not isinstance(invalid, np.ndarray) or invalid.shape != (200, 200) or invalid.dtype != np.bool_:
        raise TypeError("Invalid support must be a 200x200 NumPy boolean array")
    scaled = data.astype(np.float32) / 10_000.0
    scaled[:, invalid] = 0.0  # Publisher zeros all channels before neighborhood inference.
    prediction = np.asarray(model.predict(scaled), dtype=np.uint8)
    if prediction.shape != (200, 200) or not np.isin(prediction, [0, 1, 2, 3]).all():
        raise ValueError("Cloud model output violates four-class shape contract")
    prediction[invalid] = 4
    return prediction


def infer(input_path: Path, output_path: Path, receipt_path: Path, radiometry: str, *, runtime: Path = RUNTIME, weights: Path = WEIGHTS):
    import rasterio
    if radiometry != "l1c_dn_div10000":
        raise ValueError("Explicit --radiometry l1c_dn_div10000 is required")
    input_path, output_path, receipt_path = input_path.resolve(), output_path.resolve(), receipt_path.resolve()
    if output_path in (input_path, receipt_path) or receipt_path == input_path:
        raise ValueError("Input, output and receipt paths must differ")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("Output or receipt exists; overwrite is disabled")
    data, invalid, profile = read_input(input_path)
    model = load_model(runtime, weights)
    mask = predict_mask(model, data, invalid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=4, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.set_band_description(1, "CloudSEN12_UNetMobV2_V2")
    counts = Counter(int(x) for x in mask.ravel())
    receipt = {
        "schema": "GTM_Model7_cloud_runtime_v16_inference_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "CloudSEN12 class mask on caller-supplied 13-band L1C DN grid; publisher resampling parity unverified",
        "input": {"path": str(input_path), "sha256": sha256(input_path), "shape": [13, 200, 200], "bands": list(BANDS), "radiometry": radiometry},
        "model": {"name": "UNetMobV2_V2", "weights_path": str(weights.resolve()), "weights_sha256": WEIGHTS_SHA256,
                  "package": "cloudsen12-models==1.0.2", "package_source_sha256": PACKAGE_SHA256,
                  "padding_source_sha256": PADDING_SHA256, "device": "cpu", "encoder_pretrained_downloads": False},
        "output": {"path": str(output_path), "sha256": sha256(output_path), "classes": {"0":"clear","1":"thick_cloud","2":"thin_cloud","3":"cloud_shadow","4":"invalid"},
                   "counts": {str(i): counts.get(i, 0) for i in range(5)}, "valid_pixels": int((mask != 4).sum()), "invalid_pixels": int((mask == 4).sum())},
        "entrypoint_sha256": sha256(Path(__file__)),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["infer"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--radiometry", required=True)
    args = parser.parse_args()
    receipt = args.receipt or args.output.with_suffix(".receipt.json")
    result = infer(args.input, args.output, receipt, args.radiometry)
    print(json.dumps({"output": result["output"], "receipt": str(receipt.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
