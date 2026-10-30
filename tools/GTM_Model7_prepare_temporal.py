"""Prepare a sealed, grouped US MARS temporal cohort from official train rows.

The failure protocol was written first in docs/GTM_Model7_temporal_cohort.md.
No network access, official test labels, or prior model predictions are used.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
MARS = ROOT.parent / ("Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/"
                     "Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L")
PROTOCOL = ROOT / "docs/GTM_Model7_temporal_cohort.md"
DEFAULT_OUTPUT = ROOT / "outputs/GTM_Model7_temporal_20261023/data"
SEED = "GTM_Model7_temporal_20261023_seed_1"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
FULL_BANDS = BANDS + [f"{b}_bg" for b in BANDS]
GROUP_RADIUS_KM = 25.0
NEGATIVE_LIMIT = {"train": 120, "dev": 24, "test": 24}
SITE_LIMIT = {"train": 12, "dev": 12, "test": 12}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    y1, y2 = math.radians(a[0]), math.radians(b[0])
    dy, dx = y2 - y1, math.radians(b[1] - a[1])
    z = math.sin(dy / 2)**2 + math.cos(y1) * math.cos(y2) * math.sin(dx / 2)**2
    return 12742 * math.asin(min(1, math.sqrt(z)))


class UnionFind:
    def __init__(self, names):
        self.parent = {name: name for name in names}

    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def official_rows():
    csv_path = MARS / "train.csv"
    manifest_path = MARS / "publication_v3_training_samples.jsonl"
    with csv_path.open(newline="", encoding="utf-8") as f:
        official = [r for r in csv.DictReader(f)
                    if r["country"] == "United States of America" and r["satellite"] in ("S2A", "S2B")]
    official_by_id = {r["id_loc_image"]: r for r in official}
    if len(official_by_id) != len(official):
        raise ValueError("Duplicate official train sample ID")
    metadata = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    eligible = [r for r in metadata if r["country"] == "United States of America"
                and r["sensor"] == "Sentinel-2 MSI" and r["product_level"] == "L1C"
                and r["split"] == "train" and r["research_role"] in ("internal_training", "internal_validation")]
    ids = set()
    for row in eligible:
        sid = row["sample_id"]
        if sid in ids or sid not in official_by_id:
            raise ValueError("Publication metadata duplicates or lacks official train row: " + sid)
        ids.add(sid)
        original = official_by_id[sid]
        if (row["physical_location_id"] != original["id_location"] or
                (row["label_state"] == "PLUME") != (original["isplume"] == "True") or
                row["label_state"] not in ("PLUME", "NO_PLUME") or
                row["observability"] != original["observability"] or
                abs(row["longitude"] - float(original["lon"])) > 1e-5 or
                abs(row["latitude"] - float(original["lat"])) > 1e-5):
            raise ValueError("Publication metadata disagrees with official train row: " + sid)
        by_role = {a["role"]: a for a in row["assets"]}
        if (by_role["image"]["path"] != original["s2path"] or
                by_role["cloud_mask"]["path"] != original["cloudmaskpath"] or
                (row["label_state"] == "PLUME" and by_role["plume_mask"]["path"] != original["plumepath"])):
            raise ValueError("Publication asset paths disagree with official train row: " + sid)
    missing_metadata = sorted(set(official_by_id) - ids)
    if any(official_by_id[sid]["isplume"] == "True" for sid in missing_metadata):
        raise ValueError("Positive official train row lacks provenance metadata")
    return eligible, original_rows_summary(official, eligible, missing_metadata), csv_path, manifest_path


def original_rows_summary(official, eligible, missing_metadata):
    return {"official_us_s2_l1c_train_csv": len(official),
            "with_publication_v3_provenance": len(eligible),
            "excluded_no_publication_v3_provenance": len(missing_metadata),
            "excluded_sample_ids": missing_metadata,
            "candidate_labels": dict(collections.Counter(r["label_state"] for r in eligible)),
            "candidate_prior_research_roles": dict(collections.Counter(r["research_role"] for r in eligible))}


def components(rows):
    uf = UnionFind({r["group_id"] for r in rows})
    sites = collections.defaultdict(set)
    acquisitions = collections.defaultdict(set)
    for r in rows:
        sites[r["physical_location_id"]].add(r["group_id"])
        for key in ("target_scene_id", "reference_scene_id"):
            acquisitions[r[key]].add(r["group_id"])
    for groups in (*sites.values(), *acquisitions.values()):
        groups = sorted(groups)
        for group in groups[1:]:
            uf.union(groups[0], group)
    unique_sites = {r["physical_location_id"]: (r["latitude"], r["longitude"], r["group_id"])
                    for r in rows}
    points = list(unique_sites.values())
    proximity_edges = 0
    for i, a in enumerate(points):
        for b in points[i+1:]:
            if a[2] != b[2] and haversine(a[:2], b[:2]) < GROUP_RADIUS_KM:
                uf.union(a[2], b[2])
                proximity_edges += 1
    members = collections.defaultdict(list)
    for r in rows:
        members[uf.find(r["group_id"])].append(r)
    return dict(members), {g: uf.find(g) for g in uf.parent}, {
        "raw_geo25_groups": len(uf.parent), "connected_components": len(members),
        "shared_acquisition_identities_across_geo25_groups": sum(len(v) > 1 for v in acquisitions.values()),
        "cross_geo25_proximity_edges": proximity_edges,
        "positive_components": {g: sum(r["label_state"] == "PLUME" for r in rs)
                                for g, rs in members.items() if any(r["label_state"] == "PLUME" for r in rs)}}


def partition_components(members):
    positives = sorted((g for g, rows in members.items() if any(r["label_state"] == "PLUME" for r in rows)),
                       key=lambda g: -sum(r["label_state"] == "PLUME" for r in members[g]))
    if len(positives) != 2 or [sum(r["label_state"] == "PLUME" for r in members[g]) for g in positives] != [33, 3]:
        raise ValueError("Positive component geometry changed; review split protocol")
    negative = sorted((g for g in members if g not in positives),
                      key=lambda g: (-sum(r["label_state"] == "NO_PLUME" for r in members[g]), stable_key(g)))
    # The largest negative-only component gives dev enough geographically distinct
    # observations. The next three add independent sites to the test partition.
    if len(negative) < 6:
        raise ValueError("Too few negative geographic components")
    assigned = {positives[0]: "train", positives[1]: "test", negative[0]: "dev"}
    for g in negative[1:4]:
        assigned[g] = "test"
    for g in negative[4:6]:
        assigned[g] = "dev"
    for g in negative[6:]:
        assigned[g] = "train"
    if len(set(assigned.values())) != 3:
        raise ValueError("Incomplete partition")
    return assigned


def verified_asset(row, role):
    item = next(a for a in row["assets"] if a["role"] == role)
    path = (MARS / item["path"]).resolve()
    if not path.is_relative_to(MARS.resolve()):
        raise ValueError("Source path escapes cached release")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != item["size"]:
        raise ValueError("Cached source size differs from pinned upstream: " + str(path))
    h = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
            blob.update(part)
    actual = h.hexdigest()
    remote = actual if item["remote_oid_type"] == "sha256_lfs" else blob.hexdigest() \
        if item["remote_oid_type"] == "git_blob_sha1" else None
    if remote != item["remote_oid"]:
        raise ValueError("Cached source differs from pinned upstream hash: " + str(path))
    return path, {"path": str(path), "sha256": actual, "bytes": item["size"],
                  "role": role, "source_url": item["source_url"], "upstream_identity_verified": True}


def grid(shape, transform, crs):
    return {"shape": list(shape), "affine": list(transform)[:6], "crs": str(crs)}


def source_arrays(row):
    if (row["height"] != 200 or row["width"] != 200 or row["band_order"] != FULL_BANDS or
            row["observability"] != "clear"):
        raise ValueError("Source metadata fails scene contract")
    image_path, image_source = verified_asset(row, "image")
    cloud_path, cloud_source = verified_asset(row, "cloud_mask")
    sources = [image_source, cloud_source]
    with rasterio.open(image_path) as image, rasterio.open(cloud_path) as cloud:
        if (image.shape != (200, 200) or image.count != 12 or image.res != (10, 10) or
                list(image.descriptions) != FULL_BANDS or len(set(image.dtypes)) != 1 or
                image.dtypes[0] != "uint16" or str(image.crs) != row["crs"] or
                cloud.shape != image.shape or cloud.transform != image.transform or
                cloud.crs != image.crs or cloud.count != 1 or cloud.dtypes[0] != "uint8"):
            raise ValueError("Changed Sentinel/cloud grid, dtype, or band contract")
        raw = np.moveaxis(image.read(), 0, -1)
        codes = cloud.read(1)
        if not np.isin(codes, [0, 1, 2, 3, 4]).all():
            raise ValueError("Unknown target CloudSEN12 code")
        radiometric_valid = np.isfinite(raw).all(-1)
        if image.nodata is not None:
            radiometric_valid &= (raw != image.nodata).all(-1)
        native_support = (codes == 0) & radiometric_valid
        valid = native_support.reshape(100, 2, 100, 2).all(axis=(1, 3))
        valid_native = np.repeat(np.repeat(valid, 2, axis=0), 2, axis=1)
        if float(valid.mean()) < .90:
            return None, sources, "less_than_90pct_clear_and_radiometrically_valid", float(valid.mean())
        target_image = raw[..., :6].astype(np.float32).reshape(100, 2, 100, 2, 6).mean(axis=(1, 3))
        reference_image = raw[..., 6:].astype(np.float32).reshape(100, 2, 100, 2, 6).mean(axis=(1, 3))
        native = np.zeros((200, 200), dtype=np.uint8)
        transform = image.transform
        crs = image.crs
        if row["label_state"] == "PLUME":
            label_path, label_source = verified_asset(row, "plume_mask")
            sources.append(label_source)
            with rasterio.open(label_path) as label:
                if (label.shape != image.shape or label.transform != transform or label.crs != crs or
                        label.count != 1 or label.dtypes[0] != "uint8"):
                    raise ValueError("Changed producer annotation grid or dtype")
                native = label.read(1)
                if not np.isin(native, [0, 1]).all() or not np.any(native):
                    raise ValueError("Producer positive annotation is empty or not binary")
        elif row["label_state"] != "NO_PLUME":
            raise ValueError("Unknown producer label state")
        target = native.reshape(100, 2, 100, 2).max(axis=(1, 3)).astype(np.float32)
        if row["label_state"] == "PLUME" and not np.any((target == 1) & valid):
            return None, sources, "no_supported_positive_20m_cell", float(valid.mean())
        arrays = {"target_image": target_image, "reference_image": reference_image,
                  "valid": valid, "target": target, "label_native": native,
                  "valid_native": valid_native}
        return (arrays, grid(valid.shape, transform * Affine.scale(2), crs),
                grid(native.shape, transform, crs)), sources, None, float(valid.mean())


def temporal_offset(row):
    target = datetime.fromisoformat(row["target_datetime"])
    reference = datetime.strptime(row["reference_scene_id"].split("_")[2], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    if target.tzinfo is None:
        raise ValueError("Target timestamp lacks timezone")
    target_from_id = datetime.strptime(row["target_scene_id"].split("_")[2], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    if abs((target - target_from_id).total_seconds()) > 1:
        raise ValueError("Target timestamp disagrees with acquisition ID")
    return reference.isoformat(), (target - reference).total_seconds() / 3600


def negative_order(rows, group_to_component):
    by_component = collections.defaultdict(list)
    for row in rows:
        by_component[group_to_component[row["group_id"]]].append(row)
    component_orders = {group: site_balanced_order(group_rows)
                        for group, group_rows in by_component.items()}
    groups = sorted(component_orders, key=stable_key)
    ordered = []
    for round_index in range(max((len(v) for v in component_orders.values()), default=0)):
        for group in groups:
            if round_index < len(component_orders[group]):
                ordered.append(component_orders[group][round_index])
    return ordered


def site_balanced_order(rows):
    by_site = collections.defaultdict(list)
    for row in rows:
        if row["label_state"] == "NO_PLUME" and row["observability"] == "clear" and row["cloud_fraction"] <= .1:
            by_site[row["physical_location_id"]].append(row)
    for site in by_site:
        by_site[site].sort(key=lambda r: stable_key(r["sample_id"]))
    sites = sorted(by_site, key=stable_key)
    ordered = []
    for round_index in range(max((len(v) for v in by_site.values()), default=0)):
        for site in sites:
            if round_index < len(by_site[site]):
                ordered.append(by_site[site][round_index])
    return ordered


def check_partition(scenes):
    for field in ("id", "group_id", "physical_location_id", "target_scene_id", "reference_scene_id"):
        seen = {}
        for scene in scenes:
            key = scene[field]
            prior = seen.setdefault(key, scene["split"])
            if prior != scene["split"]:
                raise ValueError(f"Cross-partition {field}: {key}")
    seen_hash = {}
    for scene in scenes:
        for item in scene["sources"]:
            prior = seen_hash.setdefault(item["sha256"], scene["split"])
            if prior != scene["split"]:
                raise ValueError("Cross-partition source hash")
    for i, a in enumerate(scenes):
        for b in scenes[i+1:]:
            if a["split"] != b["split"] and haversine((a["latitude"], a["longitude"]),
                                                        (b["latitude"], b["longitude"])) < GROUP_RADIUS_KM:
                raise ValueError("Cross-partition 25 km proximity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Sealed output directory must be new: " + str(dest))
    rows, source_summary, csv_path, v3_path = official_rows()
    members, group_to_component, graph_summary = components(rows)
    assignment = partition_components(members)
    positive_rows = sorted((r for r in rows if r["label_state"] == "PLUME"),
                           key=lambda r: r["sample_id"])
    # Fixed positive list and group allocation precede all raster access and inference.
    candidates = {split: [] for split in NEGATIVE_LIMIT}
    for group, rs in members.items():
        split = assignment[group]
        candidates[split].extend(r for r in rs if r["label_state"] == "NO_PLUME")
    ordered = {split: negative_order(candidates[split], group_to_component) for split in NEGATIVE_LIMIT}
    dest.mkdir(parents=True)
    scenes, exclusions, quality_exclusion_rows = [], collections.Counter(), []
    def add(row, split):
        payload, sources, reason, clear_fraction = source_arrays(row)
        if reason:
            exclusions[reason] += 1
            quality_exclusion_rows.append({"sample_id": row["sample_id"], "label_state": row["label_state"],
                                           "reason": reason, "valid_fraction": clear_fraction})
            return False
        arrays, input_grid, reference_grid = payload
        sid = "MARS_" + row["sample_id"]
        array_path = dest / f"{sid}.npz"
        np.savez_compressed(array_path, **arrays)
        reference_time, offset_hours = temporal_offset(row)
        scenes.append({"id": sid, "name": f"{row['label_state']} | {row['target_datetime'][:10]} | {row['sample_id'][:8]}",
                       "split": split, "group_id": group_to_component[row["group_id"]],
                       "source_group_id": row["group_id"], "physical_location_id": row["physical_location_id"],
                       "scene_kind": row["label_state"], "source_level": "L1C",
                       "latitude": row["latitude"], "longitude": row["longitude"],
                       "target_scene_id": row["target_scene_id"], "reference_scene_id": row["reference_scene_id"],
                       "satellite_id": row["target_scene_id"], "satellite_time": row["target_datetime"],
                       "reference_time": reference_time, "target_minus_reference_hours": offset_hours,
                       "input_grid": input_grid, "reference_grid": reference_grid,
                       "clear_fraction": clear_fraction, "array_file": array_path.name,
                       "array_sha256": digest(array_path), "sources": sources,
                       "source_row": row, "source_label": row["label_state"],
                       "reference_mapping": "repeat_2", "reference_kind": "reviewed_binary_annotation",
                       "notes": "Official train.csv scene; held out from these new models by 25 km/acquisition component. This is an exploratory internal split; prior project work has used MARS. Background cloud mask is unavailable."})
        return True
    for row in positive_rows:
        add(row, assignment[group_to_component[row["group_id"]]])
    for split in ("train", "dev", "test"):
        site_counts = collections.Counter()
        for row in ordered[split]:
            if sum(r["split"] == split and r["scene_kind"] == "NO_PLUME" for r in scenes) >= NEGATIVE_LIMIT[split]:
                break
            site = row["physical_location_id"]
            if site_counts[site] >= SITE_LIMIT[split]:
                continue
            if add(row, split):
                site_counts[site] += 1
        chosen = sum(r["split"] == split and r["scene_kind"] == "NO_PLUME" for r in scenes)
        if chosen != NEGATIVE_LIMIT[split]:
            raise ValueError(f"Only {chosen} quality-passing negative scenes for {split}")
    check_partition(scenes)
    scenes.sort(key=lambda s: ({"train": 0, "dev": 1, "test": 2}[s["split"]], s["id"]))
    counts = {split: {label: sum(s["split"] == split and s["scene_kind"] == label for s in scenes)
                      for label in ("PLUME", "NO_PLUME")} for split in NEGATIVE_LIMIT}
    if counts != {"train": {"PLUME": 32, "NO_PLUME": 120},
                  "dev": {"PLUME": 0, "NO_PLUME": 24},
                  "test": {"PLUME": 3, "NO_PLUME": 24}}:
        raise ValueError("Unexpected selected counts: " + repr(counts))
    native_positive = {split: sum(int(np.load(dest / s["array_file"])["label_native"].sum())
                                  for s in scenes if s["split"] == split and s["scene_kind"] == "PLUME")
                       for split in NEGATIVE_LIMIT}
    coarse_positive = {split: sum(int(np.load(dest / s["array_file"])["target"].sum())
                                  for s in scenes if s["split"] == split and s["scene_kind"] == "PLUME")
                       for split in NEGATIVE_LIMIT}
    manifest = {"schema": "GTM_Model7_temporal_cohort_v1", "synthetic": False,
                "role": "exploratory_official_train_internal_group_holdout",
                "downloads": 0, "seed": SEED, "bands": BANDS,
                "input_channels": {"target_image": BANDS, "reference_image": [f"{b}_bg" for b in BANDS]},
                "source_csv": source(csv_path), "source_manifest": source(v3_path),
                "dataset_readme": source(MARS / "README.md"),
                "selection_rule": "Official US S2 L1C train rows with publication-v3 provenance; connected 25 km, site, and target/reference-acquisition groups; fixed 33-positive train/3-positive test components; negative-only dev; deterministic site-balanced negative selection with <=12 per physical site, 120/24/24 negative limits; >=90% all-four target-cloud-clear and 12-band finite/non-nodata native support; no inference score selection",
                "quality_limitations": ["No reference-date cloud mask is supplied. Background radiometry is checked for finite/non-nodata only.",
                                        "This is an exploratory split of official train observations; MARS has appeared in prior project work.",
                                        "Positive scenes exist in only two connected geographic components, so dev contains negatives only.",
                                        "Train positives carried source research_role=internal_validation and test positives carried source research_role=internal_training; both are in the official train.csv."],
                "target_20m_rule": "binary max over each 2x2 native annotation block",
                "native_scoring_rule": "repeat each 20m predicted probability into its 2x2 native 10m block; score only valid_native",
                "split_counts": counts, "native_positive_pixels": native_positive,
                "coarse_positive_pixels": coarse_positive,
                "graph_summary": graph_summary, "source_summary": source_summary,
                "quality_exclusions": dict(exclusions), "quality_exclusion_rows": quality_exclusion_rows,
                "partition_components": {g: assignment[g] for g in sorted(assignment)},
                "scenes": scenes}
    save(dest / "manifest.json", manifest)
    receipt = {"schema": "GTM_Model7_temporal_preparation_receipt_v1",
               "manifest": source(dest / "manifest.json"), "preparation_source": source(Path(__file__)),
               "protocol": source(PROTOCOL), "scene_count": len(scenes), "split_counts": counts,
               "split_groups": {split: len({s["group_id"] for s in scenes if s["split"] == split})
                                for split in NEGATIVE_LIMIT},
               "positive_sites": {split: len({s["physical_location_id"] for s in scenes
                                              if s["split"] == split and s["scene_kind"] == "PLUME"})
                                  for split in NEGATIVE_LIMIT},
               "native_positive_pixels": native_positive, "coarse_positive_pixels": coarse_positive,
               "offset_hours": {split: sorted({s["target_minus_reference_hours"] for s in scenes
                                               if s["split"] == split}) for split in NEGATIVE_LIMIT},
               "source_summary": source_summary, "graph_summary": graph_summary,
               "quality_exclusions": dict(exclusions), "quality_exclusion_rows": quality_exclusion_rows,
               "downloads": 0,
               "cross_partition_checks": ["sample_id", "component_group_id", "physical_location_id",
                                          "target_acquisition", "reference_acquisition", "asset_sha256",
                                          "25km_haversine_site_distance"]}
    save(dest / "receipt.json", receipt)
    print(json.dumps({"data": str(dest), "scene_count": len(scenes), "counts": counts,
                      "groups": receipt["split_groups"], "positive_sites": receipt["positive_sites"],
                      "positive_pixels_native": native_positive, "quality_exclusions": dict(exclusions),
                      "downloads": 0}, indent=2))


if __name__ == "__main__":
    main()
