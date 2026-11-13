#### Reproducible offline Tessera fixture and export CLI ####

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scaffold import FixtureProvider, Region, run_offline


MATCH_FIELDS = ("point_id", "site_id", "seed_lon", "seed_lat", "observation_date", "geometry_type", "uncertainty_m", "spatial_group", "year", "variant", "rank", "pixel_id", "lon", "lat", "cosine_similarity", "distance_m", "coverage_status", "provenance", "provenance_key")


def fixture():
    manifest = {"sdk_version": "synthetic-1", "dataset_version": "synthetic-1", "variant": "synthetic128d", "year": 2024, "source": "local synthetic vectors only"}
    vectors = np.zeros((2, 3, 128), dtype=np.float32)
    vectors[0, 0, 0] = 1.0  # seed A
    vectors[0, 1, 0] = 0.9
    vectors[0, 1, 1] = 0.1
    vectors[0, 2, 1] = 1.0
    vectors[1, 0, 0] = 0.8
    vectors[1, 0, 1] = 0.2
    vectors[1, 1, 1] = 1.0  # seed B
    vectors[1, 2, 1] = 1.0
    region = Region(vectors, np.full((2, 3), "valid", dtype="<U16"), (0.01, 0, -93.0, 0, -0.01, 42.0), "EPSG:4326", 2024, "synthetic128d", "synthetic-1", "synthetic-1")
    points = [
        {"point_id": "source-A", "site_id": "site-A", "lon": -92.995, "lat": 41.995, "observation_date": "2024-06-01", "geometry_type": "verified_source", "uncertainty_m": 5, "provenance": "synthetic-demo", "spatial_group": "group-A"},
        {"point_id": "interior-B", "site_id": "site-B", "lon": -92.985, "lat": 41.985, "observation_date": "2024-06-02", "geometry_type": "interior", "uncertainty_m": 15, "provenance": "synthetic-demo", "spatial_group": "group-B"},
    ]
    return manifest, points, region


def save_fixture(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    manifest, points, region = fixture()
    (output / "manifest.json").write_text(json.dumps({"pins": manifest, "points": points}, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "synthetic_region.npz", vectors=region.vectors, coverage=region.coverage, transform=np.asarray(region.transform), crs=region.crs, year=region.year, variant=region.variant, sdk_version=region.sdk_version, dataset_version=region.dataset_version)


def run_files(manifest_path: Path, region_path: Path, output: Path, *, top_k: int = 2, max_pixels: int = 1000):
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(region_path, allow_pickle=False) as saved:
        region = Region(saved["vectors"], saved["coverage"], tuple(saved["transform"].tolist()), str(saved["crs"].item()), int(saved["year"].item()), str(saved["variant"].item()), str(saved["sdk_version"].item()), str(saved["dataset_version"].item()))
    result = run_offline(source["points"], source["pins"], FixtureProvider(region, source["pins"]), top_k=top_k, max_pixels=max_pixels)
    write_exports(result, output)
    return result


def write_exports(result: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "matches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS)
        writer.writeheader()
        writer.writerows(result["matches"])
    geo = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [m["lon"], m["lat"]]}, "properties": {k: v for k, v in m.items() if k not in ("lon", "lat")}} for m in result["matches"]]}
    (output / "matches.geojson").write_text(json.dumps(geo, indent=2) + "\n", encoding="utf-8")


def demo(output: Path):
    save_fixture(output)
    result = run_files(output / "manifest.json", output / "synthetic_region.npz", output, top_k=2, max_pixels=6)
    # Read exports back so the CLI itself checks its promised result.
    parsed = json.loads((output / "result.json").read_text(encoding="utf-8"))
    with (output / "matches.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    parsed_geo = json.loads((output / "matches.geojson").read_text(encoding="utf-8"))
    assert parsed["matches"][0]["pixel_id"] == "r0c1"
    assert (parsed["matches"][0]["lon"], parsed["matches"][0]["lat"]) == (-92.985, 41.995)
    assert parsed["matches"][0]["provenance"] == "synthetic-demo"
    assert len(csv_rows) == len(parsed["matches"]) == len(parsed_geo["features"]) == 4
    print(f"PASS offline demo: {len(parsed['seeds'])} seeds, {len(parsed['matches'])} matches, exports at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline synthetic Tessera research scaffold")
    parser.add_argument("command", choices=["demo", "run"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--region", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "demo_output")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output)
    else:
        if args.manifest is None or args.region is None:
            parser.error("run requires --manifest and --region")
        result = run_files(args.manifest, args.region, args.output, top_k=args.top_k, max_pixels=args.max_pixels)
        print(f"PASS offline run: {len(result['seeds'])} seeds, {len(result['matches'])} matches, exports at {args.output}")
