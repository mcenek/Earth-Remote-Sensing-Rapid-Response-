"""Check the real Cambridge tile through GeoTessera and the research CLI."""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

from geotessera import GeoTessera
import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin, rowcol
from rasterio.warp import Resampling, reproject, transform

from fetch import FILES, GRID, RUN


PINS = {"sdk_version": "0.11.0", "dataset_version": "v1.1", "variant": "cambridge", "year": 2024}
CENTER = (0.15, 52.05)
OFFSHORE = (-29.95, 52.05)
WGS84 = "EPSG:4326"
SIDE = 9


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
    assert path.read_bytes() == payload


def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("SDK must read only the checked local cache")


def patch_at_east_edge(tile, affine, crs):
    x, y = affine * (tile.shape[1] - 0.25, tile.shape[0] // 2 + 0.5)
    lon, lat = (v[0] for v in transform(crs, WGS84, [x], [y]))
    dlat = 10 / 111_320
    dlon = dlat / math.cos(math.radians(lat))
    grid = from_origin(lon - SIDE / 2 * dlon, lat + SIDE / 2 * dlat, dlon, dlat)
    centers = [grid * (col + 0.5, row + 0.5) for row, col in np.ndindex((SIDE, SIDE))]
    xs, ys = transform(WGS84, crs, *zip(*centers))
    r, c = rowcol(affine, xs, ys)
    inside = (r >= 0) & (r < tile.shape[0]) & (c >= 0) & (c < tile.shape[1])
    sampled = np.full((SIDE * SIDE, 128), np.nan, dtype=np.float32)
    sampled[inside] = tile[r[inside], c[inside]]
    vectors = sampled.reshape(SIDE, SIDE, 128)
    valid = np.isfinite(vectors).all(axis=-1) & np.any(vectors != 0, axis=-1)
    assert np.any(valid) and np.any(~inside)
    for channel in (0, 63, 127):
        expected = np.full((SIDE, SIDE), np.nan, dtype=np.float32)
        reproject(np.ascontiguousarray(tile[:, :, channel]), expected, src_transform=affine,
                  src_crs=crs, dst_transform=grid, dst_crs=WGS84, dst_nodata=np.nan,
                  resampling=Resampling.nearest)
        np.testing.assert_array_equal(vectors[:, :, channel], expected)
    return grid, vectors, valid


def main():
    download = json.loads((RUN / "download.json").read_bytes())
    assert download["charged_bytes"] <= download["cap_bytes"]
    assert [obj["path"] for obj in download["objects"]] == list(FILES.values())
    for obj in download["objects"]:
        path = RUN / obj["path"]
        assert path.stat().st_size == obj["bytes"] and digest(path) == obj["sha256"]
    assert importlib.metadata.version("geotessera") == PINS["sdk_version"]
    sys.addaudithook(no_network)
    registry = RUN / "registries/manifest.parquet"
    absence = pq.read_table(registry, columns=["year"], filters=[
        ("year", "=", 2024), ("lon", ">=", OFFSHORE[0] - 0.001), ("lon", "<=", OFFSHORE[0] + 0.001),
        ("lat", ">=", OFFSHORE[1] - 0.001), ("lat", "<=", OFFSHORE[1] + 0.001),
    ])
    assert absence.num_rows == 0
    sdk = GeoTessera(dataset_version="v1.1", dataset_variant="cambridge", embeddings_dir=RUN / "tile",
                     registry_dir=RUN / "registries", cache_dir=RUN / "sdk-cache",
                     bbox=(OFFSHORE[0], 52.00, 0.20, 52.10))
    assert (sdk.version, sdk.dataset_version, sdk.dataset_variant) == ("0.11.0", "v1.1", "cambridge")
    assert (2024, *CENTER) in sdk.registry.load_blocks_for_region((0.10, 52.00, 0.20, 52.10), 2024)
    tile, crs, affine = sdk.fetch_embedding(*CENTER, year=2024)
    mask_file = RUN / f"tile/global_0.1_degree_tiff_all/{GRID}.tiff"
    with rasterio.open(mask_file) as mask:
        assert crs == mask.crs and affine == mask.transform and tile.shape == (*mask.shape, 128)
        all_land = bool(np.all(mask.read(1) == 1))
    assert crs.is_projected and tile.dtype == np.float32

    x, y = transform(WGS84, crs, [CENTER[0]], [CENTER[1]])
    r, c = rowcol(affine, x[0], y[0])
    native_pixels = [(r, c), (r - 1, c + 2), (r + 3, c - 2)]
    centers = [affine * (col + 0.5, row + 0.5) for row, col in native_pixels]
    points = list(zip(*transform(crs, WGS84, *zip(*centers))))
    samples, metadata = sdk.sample_embeddings_at_points(points + [OFFSHORE], year=2024,
                                                         include_metadata=True, auto_download=False)
    assert samples.shape == (4, 128) and np.isnan(samples[-1]).all()
    assert [(m["pixel_row"], m["pixel_col"]) for m in metadata[:3]] == native_pixels
    tile_dir = RUN / f"tile/global_0.1_degree_representation/2024/{GRID}"
    raw = np.load(tile_dir / f"{GRID}.npy", mmap_mode="r", allow_pickle=False)
    scale = np.load(tile_dir / f"{GRID}_scales.npy", mmap_mode="r", allow_pickle=False)
    for n, rc in enumerate(native_pixels):
        expected = raw[rc].astype(np.float32) * np.float32(scale[rc])
        assert np.isfinite(expected).all() and np.any(expected != 0)
        np.testing.assert_array_equal(samples[n], expected)
        np.testing.assert_array_equal(tile[rc], expected)
    print(f"GeoTessera {sdk.version}: {tile.shape}, {crs}, three exact point samples and one no-coverage point", flush=True)

    grid, vectors, valid = patch_at_east_edge(tile, affine, crs)
    inputs = RUN / "verification"
    inputs.mkdir(exist_ok=False)
    status = np.where(valid, "valid", "missing")
    first_valid = tuple(map(int, np.argwhere(valid)[0]))
    first_missing = tuple(map(int, np.argwhere(~valid)[0]))
    seeds = {"land": first_valid, "missing": first_missing, "outside": (SIDE, SIDE)}
    records = []
    for name, (row, col) in seeds.items():
        lon, lat = grid * (col + 0.5, row + 0.5)
        records.append({"point_id": name, "site_id": "public-cambridge-smoke", "lon": lon, "lat": lat,
                        "observation_date": "2024-01-01", "geometry_type": "interior", "uncertainty_m": 0,
                        "provenance": "public annual embedding, engineering test point"})
    write_json(inputs / "manifest.json", {"pins": PINS, "points": records})
    region = inputs / "region.npz"
    np.savez_compressed(region, vectors=vectors, coverage=status, transform=tuple(grid)[:6], crs=WGS84, **PINS)
    with np.load(region, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["vectors"], vectors)
        np.testing.assert_array_equal(saved["coverage"], status)
        assert tuple(saved["transform"]) == tuple(grid)[:6]
        assert saved["crs"].item() == WGS84 and all(saved[key].item() == value for key, value in PINS.items())
    cli = Path(__file__).resolve().parents[2] / "cli.py"
    exports = inputs / "exports"
    subprocess.run([sys.executable, "-B", str(cli), "run", "--manifest", str(inputs / "manifest.json"),
                    "--region", str(region), "--output", str(exports), "--max-pixels", str(SIDE**2),
                    "--top-k", "3"], check=True, timeout=45)
    result = json.loads((exports / "result.json").read_bytes())
    states = {seed["point_id"]: seed["coverage_status"] for seed in result["seeds"]}
    assert states == {"land": "valid", "missing": "missing", "outside": "out_of_bounds"}
    seed = vectors[first_valid].astype(float)
    unit = seed / math.hypot(*seed)
    candidates = [(r, c) for r, c in np.ndindex(valid.shape) if valid[r, c] and (r, c) != first_valid]
    ranked = sorted(candidates, key=lambda rc: (math.dist(unit, vectors[rc].astype(float) / math.hypot(*vectors[rc])), *rc))
    matches = result["matches"]
    assert [(m["point_id"], m["pixel_id"]) for m in matches] == [("land", f"r{r}c{c}") for r, c in ranked[:3]]
    features = json.loads((exports / "matches.geojson").read_bytes())["features"]
    assert [f["geometry"]["coordinates"] for f in features] == [[m["lon"], m["lat"]] for m in matches]
    files = [inputs / "manifest.json", region, *sorted(exports.iterdir())]
    write_json(inputs / "report.json", {
        "status": "PASS_REAL_TESSERA_ONE_TILE_SMOKE", "pins": PINS, "tile_center_lon_lat": CENTER,
        "native_crs": str(crs), "native_transform": list(affine)[:6], "tile_shape": list(tile.shape),
        "native_landmask_all_land": all_land, "exact_point_samples": len(native_pixels),
        "independent_values_checked": len(native_pixels) * 128, "offshore_no_coverage_lon_lat": OFFSHORE,
        "scaffold_grid_crs": WGS84, "warp_comparisons": SIDE**2 * 3,
        "patch_valid_cells": int(np.count_nonzero(valid)), "seed_statuses": states,
        "verified_rank_order": [m["pixel_id"] for m in matches],
        "dataset_body_bytes": sum(obj["bytes"] for obj in download["objects"]),
        "dataset_charged_bytes": download["charged_bytes"], "download_receipt_sha256": digest(RUN / "download.json"),
        "artifact_sha256": {p.relative_to(RUN).as_posix(): digest(p) for p in files},
    })
    print("PASS native vectors, WGS84 normalization, ranked matches, and missing/outside abstention")


if __name__ == "__main__":
    main()
