#### Source-raster pixel points to WGS84; offline geometry only ####

from __future__ import annotations

import argparse
import csv
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path


FIELDS = ("point_id", "scene_id", "site_id", "observation_date", "geometry_type",
          "pixel_row", "pixel_col", "coordinate_convention", "lon", "lat",
          "uncertainty_m", "uncertainty_status", "provenance", "source_raster",
          "source_raster_sha256", "source_crs", "source_transform", "sample_status")
GEOMETRIES = {"centroid", "interior", "verified_source"}
CONVENTIONS = {"pixel_index", "fractional_index"}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _coordinate(row: dict, key: str, integer: bool) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    if integer and int(value) != value:
        raise ValueError(f"{key} must be an integer for pixel_index")
    return float(value)


def convert(manifest_path: Path) -> dict:
    """Read each source GeoTIFF's own affine and CRS. No scene-size inference."""
    try:
        import rasterio
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise RuntimeError("This converter requires rasterio and pyproj") from exc

    raw = manifest_path.read_bytes()
    source = json.loads(raw)
    if not isinstance(source, dict) or not isinstance(source.get("points"), list) or not source["points"]:
        raise ValueError("manifest.points must be a nonempty list")
    pins = source.get("pins")
    if pins is not None:
        if not isinstance(pins, dict) or any(not pins.get(k) for k in ("sdk_version", "dataset_version", "variant", "year")) or type(pins["year"]) is not int:
            raise ValueError("pins require SDK, dataset, variant and integer year")
    records = []
    seen = set()
    for row in source["points"]:
        if not isinstance(row, dict):
            raise ValueError("Each point must be an object")
        point_id = _text(row, "point_id")
        if point_id in seen:
            raise ValueError(f"Duplicate point_id: {point_id}")
        seen.add(point_id)
        scene_id, site_id = _text(row, "scene_id"), _text(row, "site_id")
        provenance = _text(row, "provenance")
        observation_date = _text(row, "observation_date")
        try:
            observed = date.fromisoformat(observation_date)
        except ValueError as exc:
            raise ValueError(f"Invalid observation_date: {observation_date}") from exc
        if observed.isoformat() != observation_date:
            raise ValueError("observation_date must be ISO YYYY-MM-DD")
        if pins is not None and observed.year != pins["year"]:
            raise ValueError("Observation year does not match pins.year")
        geometry = _text(row, "geometry_type")
        if geometry not in GEOMETRIES:
            raise ValueError("geometry_type must be centroid, interior or verified_source")
        convention = _text(row, "coordinate_convention")
        if convention not in CONVENTIONS:
            raise ValueError("coordinate_convention must be pixel_index or fractional_index")
        pixel_row = _coordinate(row, "pixel_row", convention == "pixel_index")
        pixel_col = _coordinate(row, "pixel_col", convention == "pixel_index")
        uncertainty = row.get("uncertainty_m")
        status = row.get("uncertainty_status")
        if status == "known":
            if isinstance(uncertainty, bool) or not isinstance(uncertainty, (float, int)) or not math.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError("Known uncertainty_m must be a finite nonnegative number")
            uncertainty = float(uncertainty)
        elif status == "unknown" and uncertainty is None:
            pass
        else:
            raise ValueError("Set uncertainty_status to known with uncertainty_m, or unknown with null")
        raster_name = _text(row, "source_raster")
        raster_path = (manifest_path.parent / raster_name).resolve()
        if not raster_path.is_file():
            raise ValueError(f"Missing source raster: {raster_name}")
        with rasterio.open(raster_path) as raster:
            if raster.crs is None or raster.transform is None or raster.transform.is_identity:
                raise ValueError(f"Source raster lacks CRS or georeference: {raster_name}")
            transform = raster.transform
            if not all(math.isfinite(v) for v in tuple(transform)[:6]) or abs(transform.a * transform.e - transform.b * transform.d) < 1e-20:
                raise ValueError(f"Invalid affine transform: {raster_name}")
            if raster.count < 1 or not (0 <= pixel_row < raster.height and 0 <= pixel_col < raster.width
                                      and pixel_row + 0.5 < raster.height and pixel_col + 0.5 < raster.width):
                raise ValueError(f"Pixel outside source raster: {point_id}")
            # Coordinates use zero-based index space; integer (0,0) and fractional
            # (0.25,0.75) refer to centers after a +0.5 shift on each axis.
            x, y = transform @ (pixel_col + 0.5, pixel_row + 0.5)
            lon, lat = Transformer.from_crs(CRS.from_user_input(raster.crs), CRS.from_epsg(4326), always_xy=True).transform(x, y)
            if not all(math.isfinite(v) for v in (lon, lat)) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f"Invalid WGS84 result: {point_id}")
            # A masked/nodata source pixel cannot support a point. For fractional
            # positions, this checks the containing pixel, not interpolated values.
            from rasterio.windows import Window
            sample = raster.read(1, window=Window(math.floor(pixel_col + 0.5), math.floor(pixel_row + 0.5), 1, 1), masked=True)
            if bool(sample.mask.any()) or not math.isfinite(float(sample[0, 0])):
                raise ValueError(f"Point falls on nodata/masked pixel: {point_id}")
            records.append({"point_id": point_id, "scene_id": scene_id, "site_id": site_id,
                            "observation_date": observation_date, "geometry_type": geometry,
                            "pixel_row": pixel_row, "pixel_col": pixel_col,
                            "coordinate_convention": convention, "lon": lon, "lat": lat,
                            "uncertainty_m": uncertainty, "uncertainty_status": status,
                            "provenance": provenance, "source_raster": str(raster_path),
                            "source_raster_sha256": _sha256(raster_path),
                            "source_crs": raster.crs.to_string(),
                            "source_transform": list(tuple(transform)[:6]), "sample_status": "valid"})
    ready = []
    if pins is not None:
        for record in records:
            if record["uncertainty_status"] == "known":
                ready.append({key: record[key] for key in ("point_id", "site_id", "lon", "lat", "observation_date", "geometry_type", "uncertainty_m", "provenance")})
    return {"manifest_sha256": sha256(raw).hexdigest(), "pins": pins, "points": records,
            "scaffold_points": ready,
            "interpretation": "Coordinates identify marked plume geometry. A centroid or interior point is not a verified methane source."}


def write_exports(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    names = ("points.json", "points.csv", "points.geojson")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Use a fresh output directory; existing exports are preserved")
    (output / "points.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for point in result["points"]:
            writer.writerow({**point, "source_transform": json.dumps(point["source_transform"])})
    features = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                 "properties": {k: v for k, v in p.items() if k not in ("lon", "lat")}} for p in result["points"]]
    geo = {"type": "FeatureCollection", "manifest_sha256": result["manifest_sha256"], "features": features}
    (output / "points.geojson").write_text(json.dumps(geo, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads((output / "points.json").read_text(encoding="utf-8")) == result
    assert json.loads((output / "points.geojson").read_text(encoding="utf-8")) == geo
    with (output / "points.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == len(result["points"])
    for row, point in zip(saved, result["points"]):
        expected = {**point, "source_transform": json.dumps(point["source_transform"])}
        assert row == {key: "" if value is None else str(value) for key, value in expected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline source-raster plume pixel coordinates to WGS84")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.manifest)
    write_exports(result, args.output)
    print(f"Converted {len(result['points'])} points to {args.output}")


if __name__ == "__main__":
    main()
