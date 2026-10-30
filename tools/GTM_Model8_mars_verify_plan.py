#!/usr/bin/env python3
"""Independently verify frozen Model 8 metadata assignment invariants."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tool-diagnostics/mars-baseline-20261030"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name, expected):
    path = OUT / name
    assert digest(path) == expected, name
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def km(a, b):
    lat1, lon1 = math.radians(float(a["lat"])), math.radians(float(a["lon"]))
    lat2, lon2 = math.radians(float(b["lat"])), math.radians(float(b["lon"]))
    q = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(q))


def main():
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
    roles = {role: load(f"{role}.jsonl", plan["files"][f"{role}.jsonl"]["sha256"]) for role in ("fit", "select", "final_candidate")}
    id_sets = {k: {r["sample_id"] for r in v} for k, v in roles.items()}
    site_sets = {k: {r["site_id"] for r in v} for k, v in roles.items()}
    component_sets = {k: {r["site_component_25km"] for r in v} for k, v in roles.items()}
    acquisitions = {k: {x for r in v for x in (r["current_acquisition"], r["reference_acquisition"]) if x} for k, v in roles.items()}
    pairs = (("fit", "select"), ("fit", "final_candidate"), ("select", "final_candidate"))
    overlaps = {}
    for a, b in pairs:
        site_a = {r["site_id"]: r for r in roles[a]}
        site_b = {r["site_id"]: r for r in roles[b]}
        overlaps[f"{a}_vs_{b}"] = {"sample_ids": len(id_sets[a] & id_sets[b]), "exact_sites": len(site_sets[a] & site_sets[b]), "connected_25km_components": len(component_sets[a] & component_sets[b]), "acquisition_ids_current_or_reference": len(acquisitions[a] & acquisitions[b]), "minimum_site_center_km": min(km(x, y) for x in site_a.values() for y in site_b.values())}
    if any(any(v[k] != 0 for k in ("sample_ids", "exact_sites", "connected_25km_components", "acquisition_ids_current_or_reference")) or v["minimum_site_center_km"] <= 25.0 for v in overlaps.values()):
        raise ValueError(f"Leakage invariant failed: {overlaps}")
    for role, rows in roles.items():
        ids = json.loads((OUT / f"{role}_ids.json").read_text(encoding="utf-8"))
        if ids != [r["sample_id"] for r in rows]:
            raise ValueError(f"ID list mismatch: {role}")
    catalog_receipt = json.loads((OUT / "asset_catalog_receipt.json").read_text(encoding="utf-8"))
    if catalog_receipt["plan_sha256"] != digest(OUT / "plan.json"):
        raise ValueError("Asset catalog bound to different plan")
    if catalog_receipt["required_asset_catalog_sha256"] != digest(OUT / "required_asset_catalog.jsonl"):
        raise ValueError("Asset catalog hash mismatch")
    report = {"plan_sha256": digest(OUT / "plan.json"), "role_counts": {k: {"rows": len(v), "positive": sum(r["label"] == "PLUME" for r in v), "negative": sum(r["label"] == "NO_PLUME" for r in v), "sentinel": sum(r["satellite"].startswith("S2") for r in v), "landsat": sum(r["satellite"].startswith("LC") for r in v), "sites": len(site_sets[k]), "components": len(component_sets[k])} for k, v in roles.items()}, "overlaps": overlaps, "asset_catalog_sha256": digest(OUT / "required_asset_catalog.jsonl"), "status": "metadata_invariants_passed; raster_content_unverified"}
    (OUT / "plan_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
