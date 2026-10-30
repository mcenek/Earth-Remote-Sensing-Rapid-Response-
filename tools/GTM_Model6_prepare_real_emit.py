"""Prepare the cached Volusia measured annotation pilot, without network I/O.

Failure cases and interpretation were fixed before implementation in
docs/GTM_Model6_real_emit_pilot_20261023.md. This is a fitting diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/GTM_Model0_hf_pilot_20260925"
FOLLOW = CACHE / "emit_followup"
PINS = ROOT / "reports/research/assets/GTM_Model6_data_contract_20261001/input_pins.json"
BANDS = ("B2", "B3", "B4", "B11", "B12")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Preserve the existing prepared data")
    pins = json.loads(PINS.read_text())
    pin_map = {}
    for pin in pins["files"]:
        base = ROOT if pin["scope"] == "repo" else CACHE
        pin_map[(base / pin["path"]).resolve()] = pin
    sources = []

    def checked(path):
        path = path.resolve()
        pin = pin_map[path]
        actual = sha(path)
        if actual != pin["sha256"] or path.stat().st_size != pin["bytes"]:
            raise ValueError(f"Source changed: {path}")
        sources.append(dict(path=str(path), sha256=actual, bytes=path.stat().st_size))
        return path

    selection = json.loads(checked(FOLLOW / "native_selection.json").read_text())
    chosen = selection["samples"][0]
    if chosen["role"] != "declared_positive" or abs(chosen["sentinel"][0]["offset_minutes"]) > 15:
        raise ValueError("Unexpected frozen pair")
    native = FOLLOW / "native" / chosen["row"]["id"]
    mask_path = checked(native / "plumemask.tif")
    mf_path = checked(native / "mf.tif")
    scl_path = checked(FOLLOW / "sentinel/scl.tif")
    files = [checked(FOLLOW / "sentinel" / f"{name}.tif") for name in ("blue", "green", "red")]
    files += [checked(FOLLOW / "swir_20260930" / f"{name}.tif") for name in ("swir16", "swir22")]
    with rasterio.open(mask_path) as ds, rasterio.open(scl_path) as scl:
        if ds.crs != scl.crs or ds.res != (60, 60):
            raise ValueError("Unexpected native annotation geometry")
        # Retain complete native cells inside the cached Sentinel crop.
        c0 = math.ceil((scl.bounds.left - ds.transform.c) / 60)
        c1 = math.floor((scl.bounds.right - ds.transform.c) / 60)
        r0 = math.ceil((ds.transform.f - scl.bounds.top) / 60)
        r1 = math.floor((ds.transform.f - scl.bounds.bottom) / 60)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        target = ds.read(1, window=window).astype(np.float32)
        target_transform = ds.window_transform(window)
        crs = ds.crs
        full_positive = int((ds.read(1) == 1).sum())
        if set(np.unique(target)) != {0, 1} or int(target.sum()) != full_positive:
            raise ValueError("Crop loses annotation or changes raw mask codes")
        with rasterio.open(mf_path) as mf:
            if mf.transform != ds.transform or mf.crs != ds.crs or mf.shape != ds.shape:
                raise ValueError("Retrieval and annotation grids differ")
            retrieval = mf.read(1, window=window)
            retrieval_valid = np.isfinite(retrieval) & (retrieval != mf.nodata)
        fine_transform = target_transform * Affine.scale(1 / 3, 1 / 3)
        height, width = target.shape[0] * 3, target.shape[1] * 3
        image = np.zeros((height, width, 5), np.float32)
        valid = np.ones((height, width), bool)
        for band, path in enumerate(files):
            with rasterio.open(path) as src:
                if src.crs != crs or src.count != 1:
                    raise ValueError("Input band CRS/count differs")
                values = src.read(1)
                raw_valid = np.isfinite(values) & (values != src.nodata)
                support = np.zeros((height, width), np.float32)
                reproject(values, image[:, :, band], src_transform=src.transform, src_crs=crs,
                          dst_transform=fine_transform, dst_crs=crs, src_nodata=src.nodata,
                          dst_nodata=0, resampling=Resampling.average)
                reproject(raw_valid.astype(np.float32), support, src_transform=src.transform, src_crs=crs,
                          dst_transform=fine_transform, dst_crs=crs, resampling=Resampling.average)
                valid &= support > .99999
        scl_values = np.zeros((height, width), np.uint8)
        reproject(scl.read(1), scl_values, src_transform=scl.transform, src_crs=crs,
                  dst_transform=fine_transform, dst_crs=crs, resampling=Resampling.nearest)
        valid &= np.isin(scl_values, [4, 5, 6])
    native_input_valid = valid.reshape(target.shape[0], 3, target.shape[1], 3).all(axis=(1, 3))
    target_valid = native_input_valid & retrieval_valid
    if not np.any(target_valid & (target == 1)) or not np.any(target_valid & (target == 0)):
        raise ValueError("No supported positive or annotation-background cells")
    # A retained raw target does not turn unsupported cells into labels.
    dest.mkdir(parents=True)
    arrays = dest / "volusia.npz"
    np.savez_compressed(arrays, image=image, input_valid=valid, target=target,
                        target_valid=target_valid, retrieval=retrieval, scl=scl_values)
    manifest = dict(schema="GTM_Model6_real_annotation_pilot_v1", data_kind="measured_provider_annotation",
                    scene_id=chosen["row"]["id"], group_id="Volusia-20250601",
                    array_file=arrays.name, array_sha256=sha(arrays), bands=list(BANDS),
                    satellite=chosen["sentinel"][0], emit=chosen["row"],
                    input_grid=dict(shape=[height, width], crs=str(crs), affine=list(fine_transform)[:6]),
                    target_grid=dict(shape=list(target.shape), crs=str(crs), affine=list(target_transform)[:6]),
                    source_native_window=[r0, c0, r1-r0, c1-c0],
                    input_units="raw Sentinel L2A DN, uncalibrated; fit per-band normalization",
                    target_kind="provider-annotated plume versus weak unannotated background",
                    target_units="binary provider annotation, not concentration or flux",
                    background_policy="zero is absence of provider annotation, not verified methane-free truth",
                    support_policy="finite MF retrieval and all nine clear valid Sentinel cells; SCL 4/5/6",
                    role="fit_only", independent_test=False, synthetic=False,
                    native_resolution_m=60, fine_output_resolution_m=20,
                    validated_superresolution=False,
                    counts=dict(input_pixels=int(valid.size), input_supported=int(valid.sum()),
                                native_cells=int(target.size), native_supported=int(target_valid.sum()),
                                annotated_positive=int(((target == 1) & target_valid).sum()),
                                unannotated_background=int(((target == 0) & target_valid).sum()),
                                unknown=int((~target_valid).sum()), all_source_positive=full_positive),
                    sources=sources, pin_manifest_sha256=sha(PINS), preparation_source_sha256=sha(__file__),
                    downloads=0, limitations=["One deliberately plume-centered site; no independent evaluation",
                      "Provider background weakly labeled; no verified no-plume scene",
                      "Cross-sensor registration and plume movement not independently quantified",
                      "Radiometric offset ambiguity retained, not solved by image appearance",
                      "No airborne branch, fusion training, or validated finer methane detail"])
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("array_file", "input_grid", "target_grid", "counts", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
