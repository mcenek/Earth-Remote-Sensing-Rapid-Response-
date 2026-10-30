"""Prepare metadata-partitioned, two-date MARS L1C scenes from cached assets only.

Failure cases and selection rules were written first in
docs/GTM_Model7_mars_repair_protocol.md.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
MARS = ROOT.parent / ("Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/"
                     "Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L")
OUT = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_data"
PROTOCOL = ROOT / "docs/GTM_Model7_mars_repair_protocol.md"
SEED = "GTM_Model7_mars_repair_20261024_seed_1"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
FULL_BANDS = BANDS + [f"{x}_bg" for x in BANDS]
GRID_FIELDS = ["crs", "transform_a", "transform_b", "transform_c", "transform_d",
               "transform_e", "transform_f", "width", "height"]
SINGLETON = {"south_texas": "800e8506-42a8-443a-a952-a5dc89a7156d",
             "oklahoma": "f5f18f8b-c0ac-4bc2-885a-e9b83edf9bf4",
             "mississippi": "9123854a-64f6-4791-9255-63ad109e0896",
             "kansas": "53ce2910-0e90-47c7-8917-5cb2e045e074"}
NEGATIVE_LIMIT = {"train": 300, "dev": 80, "test": 80}
SITE_CAP = 8


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def kilometers(a, b):
    y1, y2 = math.radians(a[0]), math.radians(b[0])
    dy, dx = y2 - y1, math.radians(b[1] - a[1])
    z = math.sin(dy / 2)**2 + math.cos(y1) * math.cos(y2) * math.sin(dx / 2)**2
    return 12742 * math.asin(min(1, math.sqrt(z)))


class UnionFind:
    def __init__(self, names):
        self.parent = {x: x for x in names}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def load_release():
    csv_paths = [MARS / "train.csv", MARS / "test.csv"]
    originals = {}
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["country"] != "United States of America" or row["satellite"] not in ("S2A", "S2B", "S2C"):
                    continue
                sid = row["id_loc_image"]
                if sid in originals:
                    raise ValueError("Duplicate official US Sentinel sample ID: " + sid)
                originals[sid] = row
    mixed_path = MARS / "paper_v3_mixed_samples.jsonl"
    metadata = {}
    with mixed_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("country") == "United States of America" and row.get("sensor_family") == "Sentinel-2" and row.get("product_level") == "L1C":
                sid = row["sample_id"]
                if sid in metadata:
                    raise ValueError("Duplicate pinned metadata ID: " + sid)
                metadata[sid] = row
    if set(originals) != set(metadata) or len(originals) != 5793:
        raise ValueError("Official CSV and pinned mixed metadata US Sentinel inventory changed")
    for sid, csv_row in originals.items():
        row = metadata[sid]
        expected_split = "development_training" if csv_row["split_name"] == "train_2023" else "sealed_paper_test"
        roles = {a["role"]: a for a in row["assets"]}
        if (row["split"] != expected_split or row["research_role"] != expected_split or
                row["label_state"] != ("PLUME" if csv_row["isplume"] == "True" else "NO_PLUME") or
                row["observability"] != csv_row["observability"] or row["satellite"] != csv_row["satellite"] or
                row["band_order"] != FULL_BANDS or row["height"] != 200 or row["width"] != 200 or
                roles["image"]["path"] != csv_row["s2path"] or
                roles["cloud_mask"]["path"] != csv_row["cloudmaskpath"] or
                (row["label_state"] == "PLUME" and roles["plume_mask"]["path"] != csv_row["plumepath"])):
            raise ValueError("Official and pinned record disagree: " + sid)
    return originals, metadata, csv_paths, mixed_path


def components(originals, metadata):
    sites = {r["id_location"]: (float(r["lat"]), float(r["lon"])) for r in originals.values()}
    uf = UnionFind(sites)
    acquisitions = collections.defaultdict(set)
    for sid, row in originals.items():
        for key in ("target_scene_id", "reference_scene_id"):
            acquisitions[metadata[sid][key]].add(row["id_location"])
    for members in acquisitions.values():
        members = sorted(members)
        for site in members[1:]:
            uf.union(members[0], site)
    points = list(sites.items())
    proximity_edges = 0
    for i, (site_a, point_a) in enumerate(points):
        for site_b, point_b in points[i+1:]:
            if kilometers(point_a, point_b) < 25:
                uf.union(site_a, site_b)
                proximity_edges += 1
    root_sites = collections.defaultdict(list)
    for site in sites:
        root_sites[uf.find(site)].append(site)
    labels = {root: "mars25_" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]
              for root, members in root_sites.items()}
    site_component = {site: labels[uf.find(site)] for site in sites}
    rows = collections.defaultdict(list)
    for sid, original in originals.items():
        rows[site_component[original["id_location"]]].append(sid)
    return site_component, dict(rows), {"official_us_s2_rows": len(originals), "physical_sites": len(sites),
                                        "connected_components": len(rows),
                                        "shared_acquisitions_across_sites": sum(len(v) > 1 for v in acquisitions.values()),
                                        "site_pairs_within_25km": proximity_edges}


def scene_time(metadata_row):
    return datetime.fromisoformat(metadata_row["target_datetime"])


def grid_key(original):
    return (original["id_location"], tuple(original[key] for key in GRID_FIELDS))


def cached(assets, roles):
    for role in roles:
        asset = assets.get(role)
        if asset is None:
            return False
        path = (MARS / asset["path"]).resolve()
        if not path.is_relative_to(MARS.resolve()) or not path.is_file() or path.stat().st_size != asset["size"]:
            return False
    return True


def reference_pairs(originals, metadata, site_component):
    by_grid = collections.defaultdict(list)
    for sid, original in originals.items():
        by_grid[grid_key(original)].append(sid)
    assets = {sid: {a["role"]: a for a in row["assets"]} for sid, row in metadata.items()}
    times = {sid: scene_time(row) for sid, row in metadata.items()}
    source_present = {sid: cached(assets[sid], ["image", "cloud_mask"]) for sid in originals}
    clear_negative = {sid for sid, original in originals.items()
                      if metadata[sid]["label_state"] == "NO_PLUME" and
                      metadata[sid]["observability"] == "clear" and
                      float(original["percentage_clear"]) >= 90 and source_present[sid]}
    reference_by_grid = {key: [sid for sid in sids if sid in clear_negative]
                         for key, sids in by_grid.items()}
    pair = {}
    counts = collections.Counter()
    for sid, original in originals.items():
        row = metadata[sid]
        if row["observability"] != "clear" or float(original["percentage_clear"]) < 90:
            counts["producer_target_not_90pct_clear"] += 1
            continue
        required = ["image", "cloud_mask"] + (["plume_mask"] if row["label_state"] == "PLUME" else [])
        if not source_present[sid] or ("plume_mask" in required and not cached(assets[sid], ["plume_mask"])):
            counts["missing_or_bad_size_target_asset"] += 1
            continue
        target_time = times[sid]
        candidates = []
        for ref_sid in reference_by_grid[grid_key(original)]:
            if ref_sid == sid:
                continue
            ref_original = originals[ref_sid]
            if site_component[ref_original["id_location"]] != site_component[original["id_location"]]:
                continue
            hours = (target_time - times[ref_sid]).total_seconds() / 3600
            if 24 <= hours <= 60 * 24 and target_time.date() != times[ref_sid].date():
                candidates.append((hours, ref_sid))
        if not candidates:
            counts["no_earlier_clear_same_grid_negative_reference_within_60days"] += 1
            continue
        hours, ref_sid = min(candidates)
        pair[sid] = {"reference_id": ref_sid, "offset_hours": hours}
        counts["metadata_paired_" + row["label_state"]] += 1
    return pair, assets, dict(counts)


def allocate(originals, metadata, components_by_id, pairs):
    positives = {g: [sid for sid in sids if metadata[sid]["label_state"] == "PLUME"]
                 for g, sids in components_by_id.items()}
    positives = {g: ids for g, ids in positives.items() if ids}
    if sorted(map(len, positives.values())) != [1, 1, 1, 1, 12, 105]:
        raise ValueError("Positive connected-component inventory changed")
    singleton_group = {name: next(g for g, sids in components_by_id.items() if sid in sids)
                       for name, sid in SINGLETON.items()}
    main = next(g for g, ids in positives.items() if len(ids) == 105)
    west = next(g for g, ids in positives.items() if len(ids) == 12)
    allocation = {main: "train", west: "dev", singleton_group["south_texas"]: "train",
                  singleton_group["oklahoma"]: "train", singleton_group["mississippi"]: "test",
                  singleton_group["kansas"]: "test"}
    if len(allocation) != 6:
        raise ValueError("Positive component labels collided")
    negative_only = [g for g in components_by_id if g not in allocation]
    def capacity(g):
        counts = collections.Counter(originals[sid]["id_location"] for sid in components_by_id[g]
                                     if sid in pairs and metadata[sid]["label_state"] == "NO_PLUME")
        return sum(min(SITE_CAP, n) for n in counts.values())
    negative_only.sort(key=lambda g: (-capacity(g), stable(g)))
    test_cap = sum(capacity(g) for g, split in allocation.items() if split == "test")
    for g in negative_only:
        if test_cap >= NEGATIVE_LIMIT["test"]:
            break
        allocation[g] = "test"
        test_cap += capacity(g)
    remaining = [g for g in negative_only if g not in allocation]
    for g in remaining[:2]:
        allocation[g] = "dev"
    for g in remaining[2:]:
        allocation[g] = "train"
    if set(allocation) != set(components_by_id):
        raise ValueError("Not every connected component was assigned")
    return allocation, {g: capacity(g) for g in components_by_id}, singleton_group


def checked_asset(item, role):
    path = (MARS / item["path"]).resolve()
    if not path.is_relative_to(MARS.resolve()) or not path.is_file() or path.stat().st_size != item["size"]:
        raise ValueError("Missing, escaped, or changed source size: " + str(path))
    h = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
            git.update(part)
    actual = h.hexdigest()
    upstream = actual if item["remote_oid_type"] == "sha256_lfs" else git.hexdigest() \
        if item["remote_oid_type"] == "git_blob_sha1" else None
    if upstream != item["remote_oid"]:
        raise ValueError("Changed pinned upstream identity: " + str(path))
    return path, {"path": str(path), "sha256": actual, "bytes": item["size"], "role": role,
                  "source_url": item["source_url"], "upstream_identity_verified": True}


def grid(shape, affine, crs):
    return {"shape": list(shape), "affine": list(affine)[:6], "crs": str(crs)}


def raster_pair(sid, ref_sid, metadata, originals, assets):
    role_spec = [(sid, "image", "target_image"), (sid, "cloud_mask", "target_cloud"),
                 (ref_sid, "image", "reference_image"), (ref_sid, "cloud_mask", "reference_cloud")]
    if metadata[sid]["label_state"] == "PLUME":
        role_spec.append((sid, "plume_mask", "plume_mask"))
    sources, paths = [], {}
    for record_id, asset_role, output_role in role_spec:
        path, entry = checked_asset(assets[record_id][asset_role], output_role)
        paths[output_role] = path
        sources.append(entry)
    with rasterio.open(paths["target_image"]) as target_ds, \
            rasterio.open(paths["reference_image"]) as reference_ds, \
            rasterio.open(paths["target_cloud"]) as target_cloud_ds, \
            rasterio.open(paths["reference_cloud"]) as reference_cloud_ds:
        for ds in (target_ds, reference_ds):
            if (ds.shape != (200, 200) or ds.count != 12 or ds.res != (10, 10) or
                    list(ds.descriptions) != FULL_BANDS or len(set(ds.dtypes)) != 1 or
                    ds.dtypes[0] != "uint16" or str(ds.crs) != originals[sid]["crs"]):
                raise ValueError("Image band, dtype, grid, or level contract failed: " + sid)
        if reference_ds.transform != target_ds.transform or reference_ds.crs != target_ds.crs:
            raise ValueError("Selected reference differs from target grid: " + sid)
        for cloud_ds in (target_cloud_ds, reference_cloud_ds):
            if (cloud_ds.shape != target_ds.shape or cloud_ds.transform != target_ds.transform or
                    cloud_ds.crs != target_ds.crs or cloud_ds.count != 1 or cloud_ds.dtypes[0] != "uint8"):
                raise ValueError("Cloud mask grid or dtype contract failed: " + sid)
        current = np.moveaxis(target_ds.read(list(range(1, 7))), 0, -1).astype(np.float32)
        previous = np.moveaxis(reference_ds.read(list(range(1, 7))), 0, -1).astype(np.float32)
        target_cloud = target_cloud_ds.read(1)
        reference_cloud = reference_cloud_ds.read(1)
        if not np.isin(target_cloud, [0, 1, 2, 3, 4]).all() or not np.isin(reference_cloud, [0, 1, 2, 3, 4]).all():
            raise ValueError("Unknown CloudSEN12 code: " + sid)
        native = ((target_cloud == 0) & (reference_cloud == 0) &
                  np.isfinite(current).all(-1) & np.isfinite(previous).all(-1) &
                  (current > 0).all(-1) & (previous > 0).all(-1))
        for ds, image in ((target_ds, current), (reference_ds, previous)):
            if ds.nodata is not None:
                native &= (image != ds.nodata).all(-1)
        valid = native.reshape(100, 2, 100, 2).all(axis=(1, 3))
        fraction = float(valid.mean())
        if fraction < .90:
            return None, sources, "less_than_90pct_bidate_clear_valid", fraction
        native_valid = np.repeat(np.repeat(valid, 2, axis=0), 2, axis=1)
        label = np.zeros((200, 200), dtype=np.uint8)
        if metadata[sid]["label_state"] == "PLUME":
            with rasterio.open(paths["plume_mask"]) as label_ds:
                if (label_ds.shape != target_ds.shape or label_ds.transform != target_ds.transform or
                        label_ds.crs != target_ds.crs or label_ds.count != 1 or label_ds.dtypes[0] != "uint8"):
                    raise ValueError("Plume annotation grid/dtype contract failed: " + sid)
                label = label_ds.read(1)
                if not np.isin(label, [0, 1]).all() or not np.any(label):
                    raise ValueError("Invalid producer positive annotation: " + sid)
        coarse = label.reshape(100, 2, 100, 2).max(axis=(1, 3)).astype(np.float32)
        if metadata[sid]["label_state"] == "PLUME" and not np.any((coarse == 1) & valid):
            return None, sources, "no_supported_positive_20m_cell", fraction
        arrays = {"target_image": current.reshape(100, 2, 100, 2, 6).mean(axis=(1, 3)),
                  "reference_image": previous.reshape(100, 2, 100, 2, 6).mean(axis=(1, 3)),
                  "valid": valid, "target": coarse, "label_native": label, "valid_native": native_valid}
        grids = (grid(valid.shape, target_ds.transform * Affine.scale(2), target_ds.crs),
                 grid(label.shape, target_ds.transform, target_ds.crs))
        return (arrays, grids), sources, None, fraction


def site_round_robin(sids, originals):
    by_site = collections.defaultdict(list)
    for sid in sids:
        by_site[originals[sid]["id_location"]].append(sid)
    for site in by_site:
        by_site[site].sort(key=stable)
    sites = sorted(by_site, key=stable)
    for round_index in range(SITE_CAP):
        for site in sites:
            if round_index < len(by_site[site]):
                yield by_site[site][round_index]


def component_site_round_robin(sids, originals, site_component):
    groups = collections.defaultdict(list)
    for sid in sids:
        groups[site_component[originals[sid]["id_location"]]].append(sid)
    ordered = {g: list(site_round_robin(items, originals)) for g, items in groups.items()}
    keys = sorted(ordered, key=stable)
    for round_index in range(max((len(v) for v in ordered.values()), default=0)):
        for g in keys:
            if round_index < len(ordered[g]):
                yield ordered[g][round_index]


def check_split(scenes):
    for key in ("id", "group_id", "physical_location_id", "target_scene_id", "reference_scene_id"):
        seen = {}
        for scene in scenes:
            prior = seen.setdefault(scene[key], scene["split"])
            if prior != scene["split"]:
                raise ValueError(f"Cross-partition {key}: {scene[key]}")
    seen_paths = {}
    seen_image_hashes = {}
    for scene in scenes:
        for item in scene["sources"]:
            prior_path = seen_paths.setdefault(item["path"], scene["split"])
            if prior_path != scene["split"]:
                raise ValueError("Cross-partition source path")
            if item["role"] in ("target_image", "reference_image"):
                prior_image = seen_image_hashes.setdefault(item["sha256"], scene["split"])
                if prior_image != scene["split"]:
                    raise ValueError("Cross-partition spectral image hash")
    coords = collections.defaultdict(list)
    for scene in scenes:
        coords[scene["physical_location_id"]].append((scene["latitude"], scene["longitude"], scene["split"]))
    sites = [values[0] for values in coords.values()]
    for i, a in enumerate(sites):
        for b in sites[i+1:]:
            if a[2] != b[2] and kilometers(a[:2], b[:2]) < 25:
                raise ValueError("Cross-partition sites within 25 km")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Repaired data directory must be new: " + str(dest))
    originals, metadata, csv_paths, mixed_path = load_release()
    site_component, components_by_id, graph = components(originals, metadata)
    pairs, assets, pairing_audit = reference_pairs(originals, metadata, site_component)
    assignment, component_capacity, singleton_group = allocate(originals, metadata, components_by_id, pairs)
    positives = sorted((sid for sid in pairs if metadata[sid]["label_state"] == "PLUME"), key=stable)
    negatives = {split: [sid for sid in pairs if metadata[sid]["label_state"] == "NO_PLUME" and
                         assignment[site_component[originals[sid]["id_location"]]] == split]
                 for split in NEGATIVE_LIMIT}
    dest.mkdir(parents=True)
    scenes, exclusions = [], []
    def add(sid):
        original, row = originals[sid], metadata[sid]
        reference_id = pairs[sid]["reference_id"]
        payload, sources, reason, fraction = raster_pair(sid, reference_id, metadata, originals, assets)
        if reason:
            exclusions.append({"sample_id": sid, "split": assignment[site_component[original["id_location"]]],
                               "label_state": row["label_state"], "reason": reason,
                               "bidate_valid_fraction": fraction})
            return False
        arrays, (coarse_grid, native_grid) = payload
        array_path = dest / f"MARS_{sid}.npz"
        np.savez_compressed(array_path, **arrays)
        split = assignment[site_component[original["id_location"]]]
        scene = {"id": f"MARS_{sid}", "name": f"{row['label_state']} | {row['target_datetime'][:10]} | {sid[:8]}",
                 "split": split, "group_id": site_component[original["id_location"]],
                 "physical_location_id": original["id_location"], "scene_kind": row["label_state"],
                 "source_level": "L1C", "latitude": float(original["lat"]), "longitude": float(original["lon"]),
                 "target_scene_id": row["target_scene_id"],
                 "reference_scene_id": metadata[reference_id]["target_scene_id"],
                 "reference_observation_id": reference_id,
                 "original_release_reference_scene_id": row["reference_scene_id"],
                 "satellite_id": row["target_scene_id"], "satellite_time": row["target_datetime"],
                 "reference_time": metadata[reference_id]["target_datetime"],
                 "target_minus_reference_hours": pairs[sid]["offset_hours"],
                 "input_grid": coarse_grid, "reference_grid": native_grid, "reference_input_grid": coarse_grid,
                 "clear_fraction": fraction, "array_file": array_path.name, "array_sha256": sha(array_path),
                 "sources": sources,
                 "source_row": {key: value for key, value in row.items() if key != "paper_baseline"},
                 "official_csv_row": original,
                 "reference_source_row": {key: value for key, value in metadata[reference_id].items()
                                          if key != "paper_baseline"},
                 "source_label": row["label_state"], "reference_mapping": "repeat_2",
                 "reference_kind": "reviewed_binary_annotation",
                 "notes": "Internal MARS repartition; official test rows may be used for fitting. Earlier reviewed no-plume reference and both observed cloud masks verified. Historical project test exposure applies to Kansas, Mississippi, and Oklahoma."}
        scenes.append(scene)
        return True
    for sid in positives:
        add(sid)
    for split in ("train", "dev", "test"):
        selected = 0
        for sid in component_site_round_robin(negatives[split], originals, site_component):
            if selected >= NEGATIVE_LIMIT[split]:
                break
            if add(sid):
                selected += 1
    check_split(scenes)
    scenes.sort(key=lambda scene: ({"train": 0, "dev": 1, "test": 2}[scene["split"]], scene["id"]))
    counts = {split: {label: sum(scene["split"] == split and scene["scene_kind"] == label for scene in scenes)
                      for label in ("PLUME", "NO_PLUME")} for split in NEGATIVE_LIMIT}
    if any(counts[split]["PLUME"] == 0 or counts[split]["NO_PLUME"] == 0 for split in NEGATIVE_LIMIT):
        raise ValueError("Every split must contain supported positive and negative scenes")
    pixels = {split: sum(int(np.load(dest / scene["array_file"])["label_native"].sum()) for scene in scenes
                         if scene["split"] == split and scene["scene_kind"] == "PLUME")
              for split in NEGATIVE_LIMIT}
    summary = {"schema": "GTM_Model7_temporal_cohort_v1", "synthetic": False,
               "role": "exploratory_repartition_of_official_mars_train_and_test",
               "historical_test_exposure": "Kansas and Mississippi test positives were directly consumed in October 23 confirmation and publication strict evaluation; held out from these new models only",
               "omitted_upstream_prediction_fields": ["paper_baseline"],
               "downloads": 0, "seed": SEED, "bands": BANDS,
               "input_channels": {"target_image": BANDS, "reference_image": BANDS},
               "source_csv": [file_source(path) for path in csv_paths],
               "source_manifest": file_source(mixed_path),
               "protocol": file_source(PROTOCOL), "preparation_source": file_source(Path(__file__)),
               "target_20m_rule": "binary max over each 2x2 native annotation block",
               "native_scoring_rule": "repeat each 20m probability into its 2x2 native 10m block; score only valid_native",
               "reference_selection": "closest observed reviewed NO_PLUME at least 24 hours earlier and on a different UTC calendar date, same official site and exact CSV grid, target metadata clear>=90%, <=60 days; first six bands and target cloud mask of that observation; both dates clear in all four native pixels",
               "quality_limitations": ["Two positive test sites only; exploratory regression, not national generalization or official benchmark",
                                       "Both positive test examples have prior project evaluation exposure",
                                       "Release original background six bands are replaced consistently by verified observed earlier reference assets"],
               "graph_summary": graph, "pairing_audit": pairing_audit,
               "positive_component_regions": {"main_permain": next(g for g, ids in components_by_id.items() if sum(metadata[s]["label_state"] == "PLUME" for s in ids) == 105),
                                              "west_permian": next(g for g, ids in components_by_id.items() if sum(metadata[s]["label_state"] == "PLUME" for s in ids) == 12),
                                              **singleton_group},
               "component_assignment": {g: assignment[g] for g in sorted(assignment)},
               "metadata_negative_capacity_after_site_cap": component_capacity,
               "split_counts": counts, "native_positive_pixels": pixels,
               "raster_quality_exclusions": exclusions, "scenes": scenes}
    save(dest / "manifest.json", summary)
    receipt = {"schema": "GTM_Model7_mars_repair_preparation_receipt_v1",
               "manifest": file_source(dest / "manifest.json"), "preparation_source": file_source(Path(__file__)),
               "protocol": file_source(PROTOCOL), "downloads": 0,
               "scene_count": len(scenes), "split_counts": counts,
               "split_components": {split: len({s["group_id"] for s in scenes if s["split"] == split})
                                    for split in NEGATIVE_LIMIT},
               "positive_sites": {split: len({s["physical_location_id"] for s in scenes if s["split"] == split and s["scene_kind"] == "PLUME"})
                                  for split in NEGATIVE_LIMIT},
               "native_positive_pixels": pixels,
               "reference_gap_hours_range": {split: [min(s["target_minus_reference_hours"] for s in scenes if s["split"] == split),
                                                    max(s["target_minus_reference_hours"] for s in scenes if s["split"] == split)]
                                             for split in NEGATIVE_LIMIT},
               "pairing_audit": pairing_audit, "raster_quality_exclusions": exclusions,
               "cross_partition_checks": ["sample_id", "25km_component", "physical_site", "target_acquisition",
                                          "reference_acquisition", "source_path", "spectral_image_sha256",
                                          "25km_site_distance"],
               "source_hash_note": "Cloud and binary label TIFFs can have byte-identical contents across unrelated sites; their paths and verified upstream identities are checked, while cross-split byte collision is enforced for spectral imagery."}
    save(dest / "receipt.json", receipt)
    print(json.dumps({"manifest": str(dest / "manifest.json"), "counts": counts,
                      "components": receipt["split_components"], "positive_sites": receipt["positive_sites"],
                      "raster_exclusions": len(exclusions), "downloads": 0}, indent=2))


if __name__ == "__main__":
    main()
