#### Offline Tessera research scaffold: validation, geometry, and ranking ####

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from math import asin, cos, radians, sin, sqrt
from typing import Any

import numpy as np


COVERAGE = {"valid", "missing", "water", "out_of_bounds"}
GEOMETRY = {"verified_source", "centroid", "interior", "background"}


def pins(manifest: dict[str, Any]) -> dict[str, Any]:
    required = ("sdk_version", "dataset_version", "variant", "year")
    if any(not manifest.get(key) for key in required):
        raise ValueError("Explicit SDK, dataset, variant, and year pins are required")
    result = {key: manifest[key] for key in required}
    if not isinstance(result["year"], int) or isinstance(result["year"], bool):
        raise ValueError("year must be an integer")
    return result


def provenance_key(manifest: dict[str, Any]) -> str:
    """Stable key over every required revision pin; never reuse across versions."""
    canonical = json.dumps(pins(manifest), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def validate_points(records: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    pin = pins(manifest)
    if not records:
        raise ValueError("At least one point is required")
    seen = set()
    for row in records:
        for key in ("point_id", "site_id", "lon", "lat", "observation_date", "geometry_type", "uncertainty_m", "provenance"):
            if key not in row or row[key] is None or row[key] == "":
                raise ValueError(f"Missing point field: {key}")
        if row["point_id"] in seen:
            raise ValueError("Duplicate point_id")
        seen.add(row["point_id"])
        if not isinstance(row["point_id"], str) or not isinstance(row["site_id"], str):
            raise ValueError("IDs must be strings")
        lon, lat = float(row["lon"]), float(row["lat"])
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid lon/lat")
        observed = date.fromisoformat(row["observation_date"])
        if observed.year != pin["year"] or row.get("year", observed.year) != pin["year"]:
            raise ValueError("Mixed observation years")
        if row.get("variant", pin["variant"]) != pin["variant"]:
            raise ValueError("Mixed embedding variants")
        if row["geometry_type"] not in GEOMETRY:
            raise ValueError("Invalid geometry_type")
        if not np.isfinite(float(row["uncertainty_m"])) or float(row["uncertainty_m"]) < 0:
            raise ValueError("Invalid uncertainty_m")
    return records


@dataclass(frozen=True)
class Region:
    vectors: np.ndarray
    coverage: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str
    year: int
    variant: str
    sdk_version: str
    dataset_version: str

    def validate(self, pin: dict[str, Any]) -> None:
        if self.crs != "EPSG:4326":
            raise ValueError("Only EPSG:4326 affine regions are supported")
        if self.year != pin["year"] or self.variant != pin["variant"]:
            raise ValueError("Mixed region year or variant")
        if self.sdk_version != pin["sdk_version"] or self.dataset_version != pin["dataset_version"]:
            raise ValueError("Mixed region SDK or dataset version")
        if self.vectors.ndim != 3 or self.vectors.shape[2] != 128:
            raise ValueError("Region must have H,W,128 vectors")
        if self.coverage.shape != self.vectors.shape[:2]:
            raise ValueError("Coverage shape mismatch")
        if len(self.transform) != 6 or not np.isfinite(self.transform).all():
            raise ValueError("Invalid affine transform")
        if self.transform[1] != 0 or self.transform[3] != 0 or self.transform[0] <= 0 or self.transform[4] >= 0:
            raise ValueError("Only north-up EPSG:4326 affine is supported")
        if set(np.unique(self.coverage)) - COVERAGE:
            raise ValueError("Unknown coverage status")

    def center(self, row: int, col: int) -> tuple[float, float]:
        a, b, c, d, e, f = self.transform
        return (c + a * (col + 0.5), f + e * (row + 0.5))

    def index(self, lon: float, lat: float) -> tuple[int, int] | None:
        a, _, c, _, e, f = self.transform
        col, row = int(np.floor((lon - c) / a)), int(np.floor((lat - f) / e))
        if row < 0 or col < 0 or row >= self.vectors.shape[0] or col >= self.vectors.shape[1]:
            return None
        return row, col


def valid_vector(vector: np.ndarray, status: str) -> bool:
    return status == "valid" and vector.shape == (128,) and bool(np.isfinite(vector).all()) and float(np.linalg.norm(vector.astype(np.float64))) > 0


class FixtureProvider:
    def __init__(self, region: Region, manifest: dict[str, Any]):
        self.pin = pins(manifest)
        region.validate(self.pin)
        self.region = region

    def sample_points(self, coordinates: list[tuple[float, float]], year: int) -> list[tuple[np.ndarray | None, str, tuple[int, int] | None]]:
        if year != self.pin["year"]:
            raise ValueError("Mixed sample year")
        samples = []
        for lon, lat in coordinates:
            index = self.region.index(lon, lat)
            if index is None:
                samples.append((None, "out_of_bounds", None))
                continue
            vector = self.region.vectors[index]
            status = str(self.region.coverage[index])
            samples.append((vector if valid_vector(vector, status) else None, status if status != "valid" else ("valid" if valid_vector(vector, status) else "invalid_vector"), index))
        return samples

    def read_region(self, year: int) -> Region:
        if year != self.pin["year"]:
            raise ValueError("Mixed region year")
        return self.region


class GeoTesseraAdapter:
    """Lazy injected client. Live construction requires an explicit permission flag."""

    def __init__(self, manifest: dict[str, Any], *, allow_live: bool = False, client_factory=None):
        self.pin = pins(manifest)
        if not allow_live or client_factory is None:
            raise PermissionError("Live Tessera access requires explicit permission and a client factory")
        self.client = client_factory(self.pin)
        if any(getattr(self.client, key, None) != self.pin[key] for key in ("sdk_version", "dataset_version", "variant")):
            raise ValueError("Injected client must confirm the pinned SDK, dataset, and variant")

    def sample_points(self, coordinates: list[tuple[float, float]], year: int) -> np.ndarray:
        if year != self.pin["year"]:
            raise ValueError("Mixed sample year")
        result = np.asarray(self.client.sample_points(coordinates, year=year))
        if result.shape != (len(coordinates), 128):
            raise ValueError("SDK sample_points must return N,128")
        return result

    def read_region(self, bbox: tuple[float, float, float, float], year: int):
        if year != self.pin["year"]:
            raise ValueError("Mixed region year")
        vectors, transform, crs = self.client.read_region(bbox, year=year)
        vectors = np.asarray(vectors)
        if vectors.ndim != 3 or vectors.shape[2] != 128:
            raise ValueError("SDK read_region must return H,W,128")
        return vectors, transform, crs


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(radians, (*a, *b))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371008.8 * asin(min(1.0, sqrt(value)))


def run_offline(points: list[dict[str, Any]], manifest: dict[str, Any], provider: FixtureProvider, *, top_k: int = 3, max_pixels: int = 1000, neighborhood_radius: int = 1) -> dict[str, Any]:
    validate_points(points, manifest)
    pin = pins(manifest)
    if provider.pin != pin:
        raise ValueError("Provider pins do not match manifest")
    cache_key = provenance_key(manifest)
    region = provider.read_region(pin["year"])
    region.validate(pin)
    if top_k < 1 or max_pixels < 1 or neighborhood_radius < 0 or region.vectors.shape[0] * region.vectors.shape[1] > max_pixels:
        raise ValueError("Invalid limits or region exceeds pixel cap")
    sampled = provider.sample_points([(float(p["lon"]), float(p["lat"])) for p in points], pin["year"])
    seed_pixels = {index for _, _, index in sampled if index is not None}
    seeds, candidates = [], []
    for p, (vector, status, index) in zip(points, sampled):
        seeds.append({**p, "coverage_status": status, "pixel_row": index[0] if index else None, "pixel_col": index[1] if index else None, "provenance_key": cache_key})
    for row in range(region.vectors.shape[0]):
        for col in range(region.vectors.shape[1]):
            vector = region.vectors[row, col]
            status = str(region.coverage[row, col])
            if valid_vector(vector, status):
                candidates.append((row, col, vector, region.center(row, col)))
    results, neighborhoods = [], []
    for point, (seed, status, index) in zip(points, sampled):
        if seed is None:
            continue
        seed64 = seed.astype(np.float64)
        seed_norm = float(np.linalg.norm(seed64))
        ranked = []
        for row, col, vector, center in candidates:
            if (row, col) in seed_pixels:
                continue
            vector64 = vector.astype(np.float64)
            cosine = float(np.dot(seed64, vector64) / (seed_norm * np.linalg.norm(vector64)))
            ranked.append((-cosine, row, col, center))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        for rank, (neg_cosine, row, col, center) in enumerate(ranked[:top_k], 1):
            results.append({"point_id": point["point_id"], "site_id": point["site_id"], "seed_lon": float(point["lon"]), "seed_lat": float(point["lat"]), "observation_date": point["observation_date"], "geometry_type": point["geometry_type"], "uncertainty_m": float(point["uncertainty_m"]), "spatial_group": point.get("spatial_group"), "year": pin["year"], "variant": pin["variant"], "rank": rank, "pixel_id": f"r{row}c{col}", "lon": center[0], "lat": center[1], "cosine_similarity": -neg_cosine, "distance_m": haversine_m((float(point["lon"]), float(point["lat"])), center), "coverage_status": "valid", "provenance": point["provenance"], "provenance_key": cache_key})
        row0, col0 = index
        for row in range(max(0, row0 - neighborhood_radius), min(region.vectors.shape[0], row0 + neighborhood_radius + 1)):
            for col in range(max(0, col0 - neighborhood_radius), min(region.vectors.shape[1], col0 + neighborhood_radius + 1)):
                vector = region.vectors[row, col]
                status = str(region.coverage[row, col])
                center = region.center(row, col)
                neighborhoods.append({"point_id": point["point_id"], "site_id": point["site_id"], "spatial_group": point.get("spatial_group"), "year": pin["year"], "geometry_type": point["geometry_type"], "pixel_id": f"r{row}c{col}", "lon": center[0], "lat": center[1], "coverage_status": status if status != "valid" else ("valid" if valid_vector(vector, status) else "invalid_vector"), "vector": vector.tolist() if valid_vector(vector, status) else None, "provenance_key": cache_key})
    return {"pins": pin, "provenance_key": cache_key, "grid": {"crs": region.crs, "transform": list(region.transform), "shape": list(region.vectors.shape)}, "seeds": seeds, "matches": results, "neighborhoods": neighborhoods, "interpretation": "Cosine similarity describes annual embedding context. It is not distance, an SVM margin, methane evidence, or a physical source location."}


def point_in_polygon(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    """Small synthetic planar lon/lat containment, including boundary."""
    inside = False
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-12 and min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def synthetic_nearest(point: tuple[float, float], facilities: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [(haversine_m(point, (float(f["lon"]), float(f["lat"]))), f["facility_id"]) for f in facilities]
    distance, identity = min(matches, key=lambda x: (x[0], x[1]))
    return {"facility_id": identity, "distance_m": distance, "status": "synthetic_geometry_only"}
