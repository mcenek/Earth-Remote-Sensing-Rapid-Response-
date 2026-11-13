"""Offline Tessera scaffold smoke test with one existing Sentinel-2 crop."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform
from rasterio.windows import Window, transform as window_transform


CLI = Path(__file__).resolve().parents[1] / "cli.py"
BANDS = ("blue", "green", "red", "nir", "scl")
SIDE = 9
PINS = {
    "sdk_version": "offline-no-sdk",
    "dataset_version": "saved-sentinel2-crop-56216",
    "variant": "four-spectral-bands-padded-to-128",
    "year": 2024,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    content = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(content)
    assert path.read_bytes() == content


def centers(affine):
    return [affine * (col + 0.5, row + 0.5) for row in range(SIDE) for col in range(SIDE)]


def save_region(path, vectors, coverage, affine, crs):
    np.savez_compressed(path, vectors=vectors, coverage=coverage, transform=tuple(affine)[:6], crs=crs, **PINS)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["vectors"], vectors)
        np.testing.assert_array_equal(saved["coverage"], coverage)
        assert tuple(saved["transform"]) == tuple(affine)[:6]
        assert str(saved["crs"]) == crs
        assert all(saved[key].item() == value for key, value in PINS.items())


def main(run, crop, receipt_path):
    started = perf_counter()
    run.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with rasterio.open(crop / "blue.tif") as source:
        crs = source.crs
        # Use a source pixel center so the reprojection check has no nearest-neighbor tie.
        x, y = source.xy(source.height // 2, source.width // 2)
        native_affine = window_transform(Window(60, 60, SIDE, SIDE), source.transform)
    lon, lat = (coord[0] for coord in transform(crs, "EPSG:4326", [x], [y]))
    dlat = 10 / 111_320
    dlon = dlat / math.cos(math.radians(lat))
    affine = from_origin(lon - SIDE / 2 * dlon, lat + SIDE / 2 * dlat, dlon, dlat)
    geographic_centers = centers(affine)
    xs, ys = transform("EPSG:4326", crs, *zip(*geographic_centers))

    vectors = np.zeros((SIDE, SIDE, 128), dtype=np.float32)
    source_hashes = {}
    valid = np.ones((SIDE, SIDE), dtype=bool)
    for channel, band in enumerate(BANDS):
        path = crop / f"{band}.tif"
        pin = receipt["bands"][band]
        source_hashes[band] = sha(path)
        assert source_hashes[band] == pin["sha256"], band
        with rasterio.open(path) as source:
            assert source.crs == crs
            assert list(source.bounds) == receipt["bbox_projected"]
            assert list(source.shape) == pin["shape"]
            raw = source.read(1)
            warped = np.zeros((SIDE, SIDE), dtype=raw.dtype)
            reproject(raw, warped, src_transform=source.transform, src_crs=crs, src_nodata=0,
                      dst_transform=affine, dst_crs="EPSG:4326", dst_nodata=0,
                      resampling=Resampling.nearest)
            indexed = [raw[source.index(easting, northing)] for easting, northing in zip(xs, ys)]
            np.testing.assert_array_equal(warped.ravel(), indexed)
        if band == "scl":
            valid &= np.isin(warped, (4, 5))
        else:
            valid &= warped != pin["nodata"]
            vectors[:, :, channel] = warped * pin["scale"] + pin["offset"]

    coverage = np.where(valid, "valid", "missing")
    seeds = {"image-center": (4, 4), "image-neighbor": (5, 5), "masked-control": (0, 0),
             "water-control": (0, 1), "zero-vector-control": (0, 2), "outside-control": (9, 9)}
    coverage[0, :3] = ("missing", "water", "valid")
    vectors[0, 2] = 0
    points = []
    for point_id, (row, col) in seeds.items():
        point_lon, point_lat = affine * (col + 0.5, row + 0.5)
        points.append({"point_id": point_id, "site_id": "saved-crop-smoke", "lon": point_lon,
                       "lat": point_lat, "observation_date": receipt["datetime"][:10],
                       "geometry_type": "interior", "uncertainty_m": 0,
                       "provenance": "engineering pixel, no source or facility label"})
    write_json(run / "manifest.json", {"pins": PINS, "points": points})
    save_region(run / "normalized-region.npz", vectors, coverage, affine, "EPSG:4326")

    def cli(region, output):
        return subprocess.run([sys.executable, "-B", str(CLI), "run", "--manifest", str(run / "manifest.json"),
                               "--region", str(region), "--output", str(output), "--top-k", "3",
                               "--max-pixels", str(SIDE**2)], capture_output=True, text=True, timeout=30)

    projected = run / "projected-crs-guard.npz"
    save_region(projected, vectors, coverage, native_affine, str(crs))
    rejected = cli(projected, run / "rejected")
    assert rejected.returncode != 0 and "Only EPSG:4326 affine regions are supported" in rejected.stderr
    assert not (run / "rejected").exists()

    export = run / "exports"
    completed = cli(run / "normalized-region.npz", export)
    completed.check_returncode()
    result = json.loads((export / "result.json").read_text(encoding="utf-8"))
    expected_status = dict(zip(seeds, ("valid", "valid", "missing", "water", "invalid_vector", "out_of_bounds")))
    assert {seed["point_id"]: seed["coverage_status"] for seed in result["seeds"]} == expected_status

    # Chord distance between unit vectors orders the same neighbors as cosine similarity.
    eligible = [(r, c) for r, c in np.ndindex((SIDE, SIDE))
                if coverage[r, c] == "valid" and math.hypot(*vectors[r, c]) and (r, c) not in seeds.values()]
    expected = []
    for name in ("image-center", "image-neighbor"):
        seed = vectors[seeds[name]].astype(float)
        unit = seed / math.hypot(*seed)
        ranked = sorted(eligible, key=lambda rc: (math.dist(unit, vectors[rc].astype(float) / math.hypot(*vectors[rc])), *rc))
        expected.extend((name, f"r{r}c{c}") for r, c in ranked[:3])
    assert [(m["point_id"], m["pixel_id"]) for m in result["matches"]] == expected

    with (export / "matches.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    features = json.loads((export / "matches.geojson").read_text(encoding="utf-8"))["features"]
    assert len(csv_rows) == len(features) == len(expected) == 6
    for row, feature, match in zip(csv_rows, features, result["matches"]):
        assert row == {key: str(value) if value is not None else "" for key, value in match.items()}
        assert feature["geometry"] == {"type": "Point", "coordinates": [match["lon"], match["lat"]]}
        assert feature["properties"] == {k: v for k, v in match.items() if k not in ("lon", "lat")}

    write_json(run / "verification.json", {
        "status": "PASS_LOCAL_OFFLINE_ENGINEERING_SMOKE", "elapsed_seconds": perf_counter() - started,
        "source_receipt_sha256": sha(receipt_path), "rasterio_version": rasterio.__version__,
        "source_raster_sha256": source_hashes, "native_crs": str(crs), "scaffold_crs": "EPSG:4326",
        "native_crs_rejection_verified": True, "reprojection_pixel_band_checks": SIDE**2 * len(BANDS),
        "seed_statuses": expected_status, "expected_and_observed_rank_order": expected,
        "sha256": {str(p.relative_to(run)): sha(p) for p in (*sorted(export.iterdir()), run / "manifest.json", run / "normalized-region.npz", projected)},
        "scope": "Local saved imagery only. No Tessera embeddings, model runs, facility joins or accuracy claims.",
    })
    print(completed.stdout.strip())
    print(f"PASS {SIDE**2 * len(BANDS)} reprojection checks, 4 abstentions, 6 ranks and three exports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.crop_dir.resolve(), args.receipt.resolve())
