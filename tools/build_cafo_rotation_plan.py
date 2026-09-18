#!/usr/bin/env python3
"""Build deterministic, geographically grouped Iowa CAFO rotations.

The source is used as a facility coordinate catalogue only.  No methane or
study labels are inferred from its fields.  The default 2 km grouping buffer
is a planning convention for the 1280 m square-crop pilot (covering its
diagonal with modest margin) and should be re-verified for larger crops.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product
from typing import Any, Iterable

DEFAULT_INPUT = Path("EarthRemoteSensingRapidResponse/Polygon_CSV_Files/iowa_cafos_2024_arcgis_api.csv")
EARTH_KM = 6371.0088


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str, name: str, line: int) -> float:
    try:
        result = float(value.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} at line {line}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Invalid {name} at line {line}: {value!r}")
    return result


def load_facilities(path: Path, *, active_only: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and de-duplicate facilities, returning records and source stats."""
    by_id: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"progid", "Latitude", "Longitude"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("CSV missing required fields: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            facility_id = (row.get("progid") or "").strip()
            if not facility_id:
                raise ValueError(f"Missing progid at line {line}")
            lat = _number(row.get("Latitude", ""), "Latitude", line)
            lon = _number(row.get("Longitude", ""), "Longitude", line)
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError(f"Coordinate out of bounds at line {line}: {lat}, {lon}")
            status = (row.get("opStatus") or "").strip()
            stats[status or "<blank>"] += 1
            old = by_id.get(facility_id)
            if old is not None and (old["latitude"] != lat or old["longitude"] != lon):
                raise ValueError(f"Conflicting coordinates for progid {facility_id}")
            if old is None:
                by_id[facility_id] = {"facility_id": facility_id, "latitude": lat, "longitude": lon,
                                      "operation_status": status, "state": (row.get("State") or "").strip()}
            elif old["operation_status"] != status:
                raise ValueError(f"Conflicting opStatus for progid {facility_id}")
    records = list(by_id.values())
    if active_only:
        records = [r for r in records if r["operation_status"].casefold() == "active"]
    records.sort(key=lambda r: r["facility_id"])
    return records, {"input_rows": sum(stats.values()), "unique_facilities": len(by_id),
                     "included_facilities": len(records), "status_counts": dict(sorted(stats.items()))}


def haversine_km(a: dict[str, Any], b: dict[str, Any]) -> float:
    p1, p2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dp = p2 - p1
    dl = math.radians(b["longitude"] - a["longitude"])
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(x)))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def geographic_components(records: list[dict[str, Any]], buffer_km: float = 2.0) -> list[list[int]]:
    if buffer_km <= 0 or not math.isfinite(buffer_km):
        raise ValueError("buffer_km must be positive")
    if buffer_km >= math.pi * EARTH_KM:
        raise ValueError('buffer_km must be smaller than half Earth circumference')
    # Cartesian chord buckets avoid longitude discontinuities and polar scaling.
    step = 2 * EARTH_KM * math.sin(buffer_km / (2 * EARTH_KM))
    buckets: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
    uf = _UnionFind(len(records))
    cells = []
    for i, rec in enumerate(records):
        lat, lon = math.radians(rec['latitude']), math.radians(rec['longitude'])
        xyz = (EARTH_KM * math.cos(lat) * math.cos(lon),
               EARTH_KM * math.cos(lat) * math.sin(lon), EARTH_KM * math.sin(lat))
        cell = tuple(math.floor(x / step) for x in xyz)
        cells.append(cell)
        buckets[cell].append(i)
    for i, rec in enumerate(records):
        for delta in product((-1, 0, 1), repeat=3):
            neighbor = tuple(c + d for c, d in zip(cells[i], delta))
            for j in buckets.get(neighbor, ()):
                if j > i and haversine_km(rec, records[j]) <= buffer_km:
                    uf.union(i, j)
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[uf.find(i)].append(i)
    return sorted((sorted(v, key=lambda i: records[i]["facility_id"]) for v in groups.values()),
                  key=lambda v: records[v[0]]["facility_id"])


def build_plan(records: list[dict[str, Any]], *, buffer_km: float = 2.0, rotations: int = 5,
               seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if rotations != 5:
        raise ValueError("CAFO rotation plan requires exactly 5 rotations")
    groups = geographic_components(records, buffer_km)
    if len(groups) < rotations:
        raise ValueError(f"At least five geographic components are required; found {len(groups)}")
    group_for: dict[int, int] = {i: number for number, group in enumerate(groups) for i in group}
    order = sorted(groups, key=lambda g: (-len(g), hashlib.sha256(f"{seed}:".encode() + records[g[0]]["facility_id"].encode()).hexdigest()))
    fold_counts = [0] * rotations
    assignments: dict[int, int] = {}
    for group in order:
        fold = min(range(rotations), key=lambda f: (fold_counts[f], f))
        fold_counts[fold] += len(group)
        for i in group:
            assignments[i] = fold
    output = []
    for i, rec in enumerate(records):
        group_id = f"iowa_cafo_group_{group_for[i] + 1:04d}"
        fold = assignments[i]
        output.append({**rec, "group_id": group_id, "fold": fold,
                       "train_rotations": [k for k in range(rotations) if k == fold],
                       "validate_rotations": [k for k in range(rotations) if k != fold]})
    output.sort(key=lambda r: r["facility_id"])
    summary = {"facility_count": len(records), "group_count": len(groups), "fold_counts": fold_counts,
               "largest_group": max((len(g) for g in groups), default=0), "buffer_km": buffer_km,
               "rotations": rotations, "seed": seed,
               "warning": ("largest geographic component may cause fold imbalance" if max((len(g) for g in groups), default=0) > len(records) / rotations * 0.5 else None),
               "rotation_counts": [{"rotation": k, "train": fold_counts[k], "validate": len(records)-fold_counts[k]} for k in range(rotations)]}
    return output, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists (exclusive output required): {output_dir}")
    output_dir.mkdir(parents=True)
    fields = ["facility_id", "group_id", "fold", "latitude", "longitude", "operation_status", "state"]
    with (output_dir / "facilities.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--buffer-km", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    records, stats = load_facilities(args.input)
    rows, summary = build_plan(records, buffer_km=args.buffer_km, seed=args.seed)
    summary.update(stats); summary["source_sha256"] = source_sha256(args.input)
    summary['status'] = 'planning_only_not_frozen_scientific_split'
    summary['coordinate_assumption'] = 'Latitude/Longitude interpreted as geographic degrees; source datum needs confirmation'
    summary['label_semantics'] = 'known facility inventory, no methane labels'
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
