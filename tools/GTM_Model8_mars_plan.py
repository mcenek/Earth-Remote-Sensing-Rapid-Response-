#!/usr/bin/env python3
"""Freeze a metadata-only MARS-S2L baseline split and bounded asset worklist.

Reads CSV and prior JSONL metadata, never raster, mask, prediction, or model bytes.
The public test reservation is new to this baseline, not a pristine external test:
the full public test metadata was used by an earlier paper benchmark in this repo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"
REPO = "UNEP-IMEO/MARS-S2L"
SEED = 20261030
MIN_KM = 25.0
DEFAULT_META = Path("EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L")
DEFAULT_OUT = Path("tool-diagnostics/mars-baseline-20261030")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def km(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    t = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(t))


def coord(row):
    return (float(row.get("latitude", row.get("lat"))), float(row.get("longitude", row.get("lon"))))


def sample_id(row):
    return row.get("sample_id", row.get("id_loc_image"))


def site_id(row):
    return row.get("physical_location_id", row.get("id_location"))


def label(row):
    return row.get("label_state", "PLUME" if row.get("isplume") == "True" else "NO_PLUME")


def source_url(path):
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(path, safe='/')}?download=true"


def asset(path, meta):
    if not path:
        return None
    if not path.startswith("data/") or ".." in Path(path).parts:
        raise ValueError(f"Unexpected asset path: {path}")
    local = meta / path
    return {"path": path, "source_url": source_url(path),
            "local_exists": local.is_file(),
            "local_bytes": local.stat().st_size if local.is_file() else None}


def compact(row, role, meta):
    assets = [asset(row[k], meta) for k in ("s2path", "cloudmaskpath", "plumepath")]
    return {"sample_id": sample_id(row), "role": role, "label": label(row),
            "official_split": row["split_name"], "site_id": site_id(row),
            "country": row["country"], "lat": row["lat"], "lon": row["lon"],
            "satellite": row["satellite"], "current_acquisition": row["tile"],
            "reference_acquisition": row["background_image_tile"],
            "plume_wkt": row["plume"], "footprint_wkt": row["footprint"],
            "crs": row["crs"], "transform": [row[f"transform_{k}"] for k in "abcdef"],
            "width": row["width"], "height": row["height"],
            "wind_u": row["wind_u"], "wind_v": row["wind_v"],
            "observability": row["observability"], "percentage_clear": row["percentage_clear"],
            "assets": [a for a in assets if a is not None]}


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for v in values:
            f.write(json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-dir", type=Path, default=DEFAULT_META)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    meta = (root / a.metadata_dir).resolve()
    out = (root / a.output_dir).resolve()
    if root not in out.parents:
        raise ValueError("Output must be inside repository")
    manifest = json.loads((meta / "metadata_manifest.json").read_text(encoding="utf-8"))
    if manifest["source"]["revision"] != REVISION:
        raise ValueError("MARS revision mismatch")
    paths = {n: meta / n for n in ("train.csv", "val.csv", "test.csv", "publication_v3_training_samples.jsonl", "publication_dev_samples.jsonl", "publication_v3_strict_samples.jsonl", "paper_v3_sealed_test_samples.jsonl")}
    for item in manifest["files"]:
        if item["path"] in paths and digest(paths[item["path"]]) != item["sha256"]:
            raise ValueError(f"Metadata hash mismatch: {item['path']}")
    official = {name: load_csv(paths[name + ".csv"]) for name in ("train", "val", "test")}
    by_id = {sample_id(r): r for rows in official.values() for r in rows}
    if len(by_id) != sum(map(len, official.values())):
        raise ValueError("Duplicate source sample IDs")
    fit_ids = {sample_id(r) for r in official["train"]}
    select_ids = {sample_id(r) for r in official["val"]}
    if fit_ids & select_ids:
        raise ValueError("Official fit/select identity overlap")
    # Every individually cached or strict-test sample is excluded, including its physical site.
    prior = list(rows_jsonl(paths["publication_dev_samples.jsonl"])) + list(rows_jsonl(paths["publication_v3_strict_samples.jsonl"]))
    exposed_ids = {sample_id(r) for r in prior}
    exposed_sites = {site_id(r) for r in prior}
    sites = {site_id(r): coord(r) for r in by_id.values()}
    parent = {s: s for s in sites}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    ordered_sites = sorted(sites)
    for n, s in enumerate(ordered_sites):
        for t in ordered_sites[n + 1:]:
            if km(sites[s], sites[t]) <= MIN_KM:
                union(s, t)
    component = {s: find(s) for s in sites}
    exposed_components = {component[s] for s in exposed_sites if s in component}
    candidates = defaultdict(lambda: {"PLUME": [], "NO_PLUME": []})
    rejected = Counter()
    for r in official["test"]:
        s = site_id(r)
        if sample_id(r) in exposed_ids or s in exposed_sites:
            rejected["historically_assessed_exact_site"] += 1
        elif not r["satellite"].startswith("S2") or r["observability"] != "clear" or float(r["percentage_clear"] or 0) < 90:
            rejected["sensor_or_quality"] += 1
        elif not r["background_image_tile"] or r["tile"] == r["background_image_tile"]:
            rejected["missing_or_same_reference"] += 1
        elif not r["s2path"] or not r["cloudmaskpath"] or (label(r) == "PLUME" and not r["plumepath"]):
            rejected["missing_required_asset_path"] += 1
        else:
            candidates[s][label(r)].append(r)
    by_component = defaultdict(lambda: {"PLUME": [], "NO_PLUME": []})
    for s, classes in candidates.items():
        for cls, rows in classes.items():
            by_component[component[s]][cls].extend(rows)
    viable = [(s, v) for s, v in by_component.items() if len(v["PLUME"]) >= 2 and len(v["NO_PLUME"]) >= 4]
    cost = Counter(component[site_id(by_id[i])] for i in fit_ids | select_ids)
    viable.sort(key=lambda pair: (cost[pair[0]], hashlib.sha256(f"{SEED}:{pair[0]}".encode()).hexdigest()))
    component_options = []
    for c, v in viable:
        component_options.append({"component": c, "positive_candidates": len(v["PLUME"]), "clear_candidates": len(v["NO_PLUME"]), "countries": sorted({r["country"] for cls in v.values() for r in cls}), "fit_rows_lost": sum(component[site_id(by_id[i])] == c for i in fit_ids), "val_rows_lost": sum(component[site_id(by_id[i])] == c for i in select_ids), "historical_assessment_nearby": c in exposed_components})
    chosen = viable[:10]
    countries = {r["country"] for _, v in chosen for cls in v.values() for r in cls}
    final = []
    for s, v in chosen:
        for cls, n in (("PLUME", 2), ("NO_PLUME", 4)):
            ordered = sorted(v[cls], key=lambda r: hashlib.sha256(f"{SEED}:{sample_id(r)}".encode()).hexdigest())
            final.extend(ordered[:n])
    if len(chosen) < 10:
        raise ValueError(f"Insufficient unexposed positive site components: {len(chosen)}; viable {len(viable)}")
    heldout_components = {s for s, _ in chosen}
    removed_fit = {i for i in fit_ids if component[site_id(by_id[i])] in heldout_components}
    removed_select = {i for i in select_ids if component[site_id(by_id[i])] in heldout_components}
    fit_ids -= removed_fit
    select_ids -= removed_select
    val_by_component = defaultdict(list)
    for i in select_ids:
        val_by_component[component[site_id(by_id[i])]].append(i)
    fit_cost = Counter(component[site_id(by_id[i])] for i in fit_ids)
    positive_val = sorted([c for c, ids in val_by_component.items() if any(label(by_id[i]) == "PLUME" for i in ids)], key=lambda c: (fit_cost[c] / max(1, sum(label(by_id[i]) == "PLUME" for i in val_by_component[c])), fit_cost[c], c))
    select_components = set()
    for c in positive_val:
        if sum(label(by_id[i]) == "PLUME" for g in select_components for i in val_by_component[g]) >= 20 and len(select_components) >= 3:
            break
        select_components.add(c)
    for c in sorted(val_by_component, key=lambda c: (fit_cost[c] / max(1, len(val_by_component[c])), fit_cost[c], c)):
        if sum(len(val_by_component[g]) for g in select_components) >= 500 and len(select_components) >= 3:
            break
        select_components.add(c)
    if sum(label(by_id[i]) == "PLUME" for g in select_components for i in val_by_component[g]) < 20:
        raise ValueError("Insufficient positive validation rows in disjoint groups")
    unused_official_val = {i for i in select_ids if component[site_id(by_id[i])] not in select_components}
    select_ids -= unused_official_val
    removed_fit_for_select = {i for i in fit_ids if component[site_id(by_id[i])] in select_components}
    fit_ids -= removed_fit_for_select
    current_ref_fit = {t for i in fit_ids | select_ids for t in (by_id[i]["tile"], by_id[i]["background_image_tile"]) if t}
    if any(r["tile"] in current_ref_fit or r["background_image_tile"] in current_ref_fit for r in final):
        raise ValueError("Heldout acquisition reused by remaining fit or selection")
    final_ids = {sample_id(r) for r in final}
    if final_ids & (fit_ids | select_ids | exposed_ids):
        raise ValueError("Exact ID leakage")
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for role, ids in (("fit", fit_ids), ("select", select_ids)):
        records[role] = [compact(by_id[i], role, meta) for i in sorted(ids)]
        for row in records[role]:
            row["site_component_25km"] = component[row["site_id"]]
        write_jsonl(out / f"{role}.jsonl", records[role])
    records["final_candidate"] = [compact(r, "final_candidate", meta) for r in sorted(final, key=sample_id)]
    for row in records["final_candidate"]:
        row["site_component_25km"] = component[row["site_id"]]
    write_jsonl(out / "final_candidate.jsonl", records["final_candidate"])
    for role in records:
        (out / f"{role}_ids.json").write_text(json.dumps([r["sample_id"] for r in records[role]], indent=2) + "\n", encoding="utf-8")
    write_jsonl(out / "removed_official_rows.jsonl", [{"sample_id": i, "official_split": by_id[i]["split_name"], "reason": "heldout_final_component" if i in removed_fit | removed_select else "select_component_exclusion" if i in removed_fit_for_select else "unused_official_validation_component", "component": component[site_id(by_id[i])]} for i in sorted(removed_fit | removed_select | removed_fit_for_select | unused_official_val)])
    unique_assets = {v["path"]: v for role in records.values() for row in role for v in row["assets"]}
    missing_assets = [v for v in unique_assets.values() if not v["local_exists"]]
    write_jsonl(out / "missing_assets.jsonl", sorted(missing_assets, key=lambda v: v["path"]))
    source_hashes = {n: digest(path) for n, path in paths.items()}
    site_overlap = {site_id(by_id[i]) for i in fit_ids} & {site_id(by_id[i]) for i in select_ids}
    prior_near = sum(component[site_id(r)] in exposed_components for r in final)
    summary = {"schema_version": 1, "planner_sha256": digest(Path(__file__)), "source": {"repository": REPO, "revision": REVISION, "sha256": source_hashes},
               "algorithm": {"seed": SEED, "minimum_site_distance_km": MIN_KM, "selection": "two positive and four clear S2 observations per 25 km connected component at ten lowest fit+select cost eligible components; remove entire selected components from official fit/select; choose bounded official-val components with >=20 positives, >=500 rows, >=3 components, remove those components from fit; unused official-val components remain excluded"},
               "counts": {"official": {k: {"rows": len(v), "positive": sum(label(r) == "PLUME" for r in v), "sites": len({site_id(r) for r in v})} for k, v in official.items()},
                          "fit": len(fit_ids), "select": len(select_ids), "select_components": len(select_components), "removed_official_train_for_final": len(removed_fit), "removed_official_val_for_final": len(removed_select), "removed_official_train_for_select_disjointness": len(removed_fit_for_select), "unused_official_val": len(unused_official_val), "fit_select_shared_sites": len(site_overlap), "final_candidate": len(final), "final_positive": sum(label(r) == "PLUME" for r in final), "final_clear": sum(label(r) == "NO_PLUME" for r in final), "final_components": len(chosen), "final_sites": len({site_id(r) for r in final}), "final_countries": sorted(countries), "eligible_both_class_components": len(viable), "rejected": dict(rejected)},
               "assets": {"unique_paths": len(unique_assets), "locally_present_paths": len(unique_assets) - len(missing_assets), "missing_paths": len(missing_assets), "locally_present_bytes": sum(v["local_bytes"] or 0 for v in unique_assets.values()), "missing_remote_bytes": "unknown_without_HF_catalog"},
               "candidate_components": component_options,
               "exposure": {"individually_exposed_prior_ids": len(exposed_ids), "prior_exposed_sites": len(exposed_sites), "final_rows_within_25km_connected_component_of_prior_assessment": prior_near, "paper_v3_sealed_public_test_covers_all_test_ids": len({sample_id(r) for r in rows_jsonl(paths["paper_v3_sealed_test_samples.jsonl"])}) == len(official["test"]), "claim": "Public-test holdout for this newly fitted local baseline; not pristine project-unexposed or independent external validation"},
               "caveats": ["This is a documented site-held-out adaptation of the official train/validation split, not exact official training.", "Previously individually assessed exact sites are excluded, but some selected connected components lie within 25 km of earlier assessments; this is NOT a project-unexposed positive test.", "Remaining official train and validation share sites; selection is not site-independent from fit.", "Metadata QA does not confirm readable imagery, raster identity, true clear reference, pixel annotation quality, or physical site independence beyond center distance.", "No threshold, architecture, or model choice may use final_candidate labels or predictions.", "All official public test labels were previously represented in the project paper benchmark; independent external validation needs newly acquired observations."]}
    summary["files"] = {n: {"sha256": digest(out / n), "rows": sum(1 for _ in (out / n).open(encoding="utf-8"))} for n in ("fit.jsonl", "select.jsonl", "final_candidate.jsonl", "missing_assets.jsonl", "removed_official_rows.jsonl")}
    summary["id_lists"] = {n: digest(out / n) for n in ("fit_ids.json", "select_ids.json", "final_candidate_ids.json")}
    (out / "plan.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
