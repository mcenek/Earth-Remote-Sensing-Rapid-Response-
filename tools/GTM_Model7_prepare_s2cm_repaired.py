"""Seal a US MethaneS2CM official-train crop cohort for two temporal models.

The fixed failure list predates this exporter: docs/GTM_Model7_s2cm_repair_protocol.md.
This reads no official location-test imagery and makes no network request.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "Earth-Remote-Sensing-Rapid-Response-"
SOURCE = PARENT / ("EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
                   "publication-v1/external/MethaneS2CM/l2a_location_split_32x32")
H5 = SOURCE / "v5_train_packed.h5"
ROWS = SOURCE / "v5_train_development_manifest.jsonl"
MARS = ROOT / "outputs/GTM_Model7_temporal_20261023/data/manifest.json"
STATES = ROOT / "GTM_Viewer/assets/us-states-10m.json"
PROTOCOL = ROOT / "docs/GTM_Model7_s2cm_repair_protocol.md"
DEFAULT = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/s2cm_data"
SEED = "GTM_Model7_s2cm_repair_20261024_seed_1"
EXPECTED_MARS = "93ebc93f87183e63e09e6611fdec1b3b8a9be0acaa2fe4f2b24bfe67e1611bd8"
EXPECTED_ROWS = "6f2137a0cf33bcae80a9ecee57d62cd9b37d4121903808bcdc7e7951a82558e3"
EXPECTED_H5 = "f75ea0c35bba240d56e8deae8828722253c5b82fb59b515d8627fb4106951ca4"
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
INDICES = (1, 2, 3, 7, 10, 11)
MIN_VALID = .90
MAX_PER_SITE_CLASS = 4


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def stable(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def states_polygon():
    raw = json.loads(STATES.read_text(encoding="utf-8"))
    if raw.get("type") != "Topology" or "states" not in raw.get("objects", {}):
        raise ValueError("Local US state boundary has changed")
    scale, shift = raw["transform"]["scale"], raw["transform"]["translate"]
    arcs = []
    for delta in raw["arcs"]:
        x = y = 0
        points = []
        for dx, dy in delta:
            x += dx
            y += dy
            points.append((x * scale[0] + shift[0], y * scale[1] + shift[1]))
        arcs.append(points)

    def ring(indices):
        points = []
        for index in indices:
            arc = arcs[index] if index >= 0 else arcs[~index][::-1]
            points.extend(arc[1:] if points else arc)
        return points

    polygons = []
    for geometry in raw["objects"]["states"]["geometries"]:
        for rings in ([geometry["arcs"]] if geometry["type"] == "Polygon" else geometry["arcs"]):
            exterior = ring(rings[0])
            holes = [ring(part) for part in rings[1:]]
            box = (min(p[0] for p in exterior), min(p[1] for p in exterior),
                   max(p[0] for p in exterior), max(p[1] for p in exterior))
            polygons.append((geometry["properties"]["name"], exterior, holes, box))
    return polygons


def in_ring(lon: float, lat: float, ring: list) -> bool:
    inside = False
    for index in range(len(ring)):
        a, b = ring[index - 1], ring[index]
        if (a[1] > lat) != (b[1] > lat) and lon < (b[0] - a[0]) * (lat - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def state_for(lat: float, lon: float, polygons: list) -> str | None:
    for name, exterior, holes, (left, bottom, right, top) in polygons:
        if left <= lon <= right and bottom <= lat <= top and in_ring(lon, lat, exterior):
            if not any(in_ring(lon, lat, hole) for hole in holes):
                return name
    return None


def distances_km(lat: float, lon: float, radians: np.ndarray) -> np.ndarray:
    y, x = np.radians((lat, lon))
    value = np.sin((radians[:, 0] - y) / 2) ** 2 + np.cos(y) * np.cos(radians[:, 0]) * np.sin((radians[:, 1] - x) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def verify_h5(source: h5py.File, rows: list[dict]) -> None:
    if len(rows) != 80217 or source.attrs.get("source_split") != "l2a_location_split_32x32/train.csv":
        raise ValueError("Official TRAIN source universe changed")
    if source.attrs.get("source_revision") != "ee9a96d4994ca6bc45725c1e92d7a06258131eaf":
        raise ValueError("Packed source revision changed")
    expected = {"sample_id": ((80217,), "int64"), "label": ((80217,), "uint8"),
                "latitude": ((80217,), "float64"), "longitude": ((80217,), "float64"),
                "target": ((80217, 12, 32, 32), "uint16"),
                "reference90": ((80217, 12, 32, 32), "uint16"),
                "reference365": ((80217, 12, 32, 32), "uint16"),
                "mask": ((80217, 32, 32), "uint8")}
    for key, (shape, dtype) in expected.items():
        if key not in source or source[key].shape != shape or str(source[key].dtype) != dtype:
            raise ValueError(f"Packed TRAIN schema changed: {key}")
    ids = source["sample_id"][:]
    labels = source["label"][:]
    latitudes = source["latitude"][:]
    longitudes = source["longitude"][:]
    for index, row in enumerate(rows):
        if (str(ids[index]) != row["id"] or int(labels[index]) != int(row["label"]) or
                not math.isclose(float(latitudes[index]), float(row["latitude"]), abs_tol=1e-8) or
                not math.isclose(float(longitudes[index]), float(row["longitude"]), abs_tol=1e-8)):
            raise ValueError(f"Packed/manifest row disagreement: {index}")


def source_arrays(source: h5py.File, index: int, expected_label: int) -> tuple[dict, str, float, int]:
    target_raw = source["target"][index, INDICES, :, :]
    reference_raw = source["reference90"][index, INDICES, :, :]
    native = source["mask"][index]
    if not np.isin(native, [0, 1]).all() or bool(native.any()) != bool(expected_label):
        raise ValueError(f"Nonbinary mask or released label mismatch: packed row {index}")
    target = np.moveaxis(target_raw, 0, -1).astype(np.float32)
    reference = np.moveaxis(reference_raw, 0, -1).astype(np.float32)
    valid = np.isfinite(target).all(-1) & np.isfinite(reference).all(-1) & (target != 0).all(-1) & (reference != 0).all(-1)
    fraction = float(valid.mean())
    arrays = {"target_image": target, "reference_image": reference, "valid": valid,
              "target": native.astype(np.float32), "label_native": native,
              "valid_native": valid.copy()}
    fingerprint = hashlib.sha256(target_raw.tobytes() + reference_raw.tobytes()).hexdigest()
    return arrays, fingerprint, fraction, int(native.sum())


def check_cross_split_sites(selected: list[dict]) -> dict:
    by_site = {}
    by_group = {}
    for row in selected:
        site, group, split = row["exact_location_id"], row["group_id"], row["split"]
        for table, key in ((by_site, site), (by_group, group)):
            if key in table and table[key] != split:
                raise ValueError(f"Source site/group crosses partitions: {key}")
            table[key] = split
    sites = {row["exact_location_id"]: (float(row["latitude"]), float(row["longitude"]), row["split"])
             for row in selected}
    points = list(sites.values())
    for i, (lat, lon, split) in enumerate(points):
        others = [(a, b) for a, b, other in points[i + 1:] if other != split]
        if others and float(np.min(distances_km(lat, lon, np.radians(np.asarray(others))))) < 25:
            raise ValueError("Different partitions have sites closer than 25 km")
    return {split: {"sites": len({r["exact_location_id"] for r in selected if r["split"] == split}),
                    "groups": len({r["group_id"] for r in selected if r["split"] == split}),
                    "states": len({r["state"] for r in selected if r["split"] == split})}
            for split in ("train", "dev", "test")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError(f"Immutable output directory exists: {dest}")
    if sha(MARS) != EXPECTED_MARS or sha(ROWS) != EXPECTED_ROWS or sha(H5) != EXPECTED_H5:
        raise ValueError("Pinned MARS or MethaneS2CM source identity changed")
    mars = json.loads(MARS.read_text(encoding="utf-8"))
    old_points = np.radians(np.asarray([(s["latitude"], s["longitude"]) for s in mars["scenes"]]))
    polygons = states_polygon()
    rows = [json.loads(line) for line in ROWS.open(encoding="utf-8")]
    if any(row.get("source_split") != "l2a_location_split_32x32/train.csv" for row in rows):
        raise ValueError("Non-TRAIN MethaneS2CM row found")
    site_location = {row["exact_location_id"]: (float(row["latitude"]), float(row["longitude"])) for row in rows}
    if len(site_location) != 3460:
        raise ValueError("Exact-location universe changed")
    state = {site: state_for(*point, polygons) for site, point in site_location.items()}
    eligible_sites = {site for site, point in site_location.items()
                      if state[site] and float(np.min(distances_km(*point, old_points))) > 25}
    if len(eligible_sites) != 1292:
        raise ValueError(f"US state/MARS spatial gate changed: {len(eligible_sites)} sites")
    candidates = [(index, row) for index, row in enumerate(rows) if row["exact_location_id"] in eligible_sites]
    if len(candidates) != 24420:
        raise ValueError(f"US state/MARS candidate count changed: {len(candidates)}")
    by_site_class = collections.defaultdict(list)
    for index, row in candidates:
        by_site_class[(row["exact_location_id"], int(row["label"]))].append((index, row))
    capped = []
    for (site, label), observations in sorted(by_site_class.items()):
        observations.sort(key=lambda item: stable(item[1]["id"]))
        capped.extend(observations[:MAX_PER_SITE_CLASS])
    internal_development = collections.defaultdict(list)
    for _, row in capped:
        if row["research_role"] == "internal_development":
            internal_development[row["group_id"]].append(row)
        elif row["research_role"] != "internal_fitting":
            raise ValueError("Unexpected source research role")
    if len(internal_development) < 4:
        raise ValueError("Too few held source groups")
    dev_groups, test_groups = set(), set()
    load = {"dev": 0, "test": 0}
    for group in sorted(internal_development, key=stable):
        split = "dev" if load["dev"] <= load["test"] else "test"
        (dev_groups if split == "dev" else test_groups).add(group)
        load[split] += sum(int(row["label"]) for row in internal_development[group])
    chosen, exclusions = [], collections.Counter()
    wide_mask = collections.Counter()
    with h5py.File(H5, "r") as source:
        verify_h5(source, rows)
        for index, row in sorted(capped, key=lambda item: int(item[1]["id"])):
            arrays, pair_hash, valid_fraction, mask_pixels = source_arrays(source, index, int(row["label"]))
            if valid_fraction < MIN_VALID:
                exclusions["less_than_90pct_nonzero_both_dates"] += 1
                continue
            split = "train" if row["research_role"] == "internal_fitting" else "dev" if row["group_id"] in dev_groups else "test"
            chosen.append({"index": index, "row": row, "split": split, "state": state[row["exact_location_id"]],
                           "pair_hash": pair_hash, "valid_fraction": valid_fraction, "mask_pixels": mask_pixels})
        duplicates = collections.defaultdict(list)
        for item in chosen:
            duplicates[item["pair_hash"]].append(item)
        duplicate_exclusions = set()
        for items in duplicates.values():
            if len(items) > 1:
                if len({item["split"] for item in items}) > 1:
                    duplicate_exclusions.update(item["row"]["id"] for item in items)
                    exclusions["cross_split_identical_input_pair_quarantined"] += len(items)
                else:
                    keep = min(items, key=lambda item: stable(item["row"]["id"]))
                    discarded = [item for item in items if item is not keep]
                    duplicate_exclusions.update(item["row"]["id"] for item in discarded)
                    exclusions["same_split_identical_input_pair_deduplicated"] += len(discarded)
        chosen = [item for item in chosen if item["row"]["id"] not in duplicate_exclusions]
        selected = [{**item["row"], "split": item["split"], "state": item["state"]} for item in chosen]
        split_summary = check_cross_split_sites(selected)
        for split in ("train", "dev", "test"):
            subset = [item for item in chosen if item["split"] == split]
            if split_summary[split]["groups"] < (1 if split == "train" else 2) or not any(int(item["row"]["label"]) for item in subset) or not any(not int(item["row"]["label"]) for item in subset):
                raise ValueError(f"Class/geography-deficient {split} partition")
        dest.mkdir(parents=True)
        scenes = []
        source_info = {"path": str(H5.resolve()), "sha256": EXPECTED_H5, "bytes": H5.stat().st_size,
                       "role": "official_train_packed_h5", "source_revision": source.attrs["source_revision"]}
        for item in chosen:
            row = item["row"]
            arrays, pair_hash, valid_fraction, mask_pixels = source_arrays(source, item["index"], int(row["label"]))
            if pair_hash != item["pair_hash"] or valid_fraction != item["valid_fraction"] or mask_pixels != item["mask_pixels"]:
                raise ValueError("Source imagery changed during preparation")
            sid = "S2CM_" + row["id"]
            file = dest / f"{sid}.npz"
            np.savez_compressed(file, **arrays)
            if mask_pixels == 1024:
                wide_mask["full_box_positive"] += 1
            if mask_pixels >= 973:
                wide_mask["at_least_95pct_positive"] += 1
            grid = {"shape": [32, 32], "coordinate_system": "pixel", "nominal_resolution_m": 20,
                    "affine": None, "crs": None, "georeferencing": "unavailable_in_released_crop"}
            scenes.append({"id": sid, "name": f"{('PLUME' if int(row['label']) else 'NO_PLUME')} | MethaneS2CM {row['id']}",
                           "split": item["split"], "group_id": row["group_id"], "source_group_id": row["group_id"],
                           "physical_location_id": row["exact_location_id"], "scene_kind": "PLUME" if int(row["label"]) else "NO_PLUME",
                           "source_level": "L2A", "state": item["state"], "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
                           "target_scene_id": f"S2CM_{row['id']}_target_acquisition_unknown",
                           "reference_scene_id": f"S2CM_{row['id']}_reference90_acquisition_unknown",
                           "acquisition_ids_verified": False, "reference_time": None, "target_time": None,
                           "reference_quality": "unknown; no per-pixel cloud mask or exact acquisition timestamp",
                           "input_grid": grid, "reference_grid": grid.copy(), "valid_fraction": valid_fraction,
                           "native_positive_pixels": mask_pixels, "array_file": file.name, "array_sha256": sha(file),
                           "input_pair_sha256": pair_hash, "sources": [source_info], "source_row": row,
                           "source_label": row["label"], "reference_mapping": "identity", "native_repeat_factor": 1,
                           "reference_kind": "released_binary_source_mask",
                           "notes": "Official MethaneS2CM TRAIN crop, US state polygon and >25 km from October 23 MARS cohort. Source 25 km groups; exact acquisition IDs and cloud masks unavailable."})
    scenes.sort(key=lambda s: ({"train": 0, "dev": 1, "test": 2}[s["split"]], int(s["source_row"]["id"])))
    counts = {split: {kind: sum(s["split"] == split and s["scene_kind"] == kind for s in scenes)
                      for kind in ("PLUME", "NO_PLUME")} for split in ("train", "dev", "test")}
    manifest = {"schema": "GTM_Model7_temporal_cohort_v1", "cohort_name": "GTM_Model7_s2cm_repair_20261024",
                "synthetic": False, "role": "exploratory_official_train_internal_group_holdout",
                "source_dataset": "H1deaki/MethaneS2CM", "source_product": "Sentinel-2 L2A surface reflectance",
                "source_revision": "ee9a96d4994ca6bc45725c1e92d7a06258131eaf", "seed": SEED,
                "bands": list(BANDS), "input_channels": {"target_image": list(BANDS), "reference_image": [f"{b}_reference90" for b in BANDS]},
                "source_manifest": {"path": str(ROWS.resolve()), "sha256": EXPECTED_ROWS, "bytes": ROWS.stat().st_size},
                "source_h5": source_info, "source_mars_exclusion_manifest": {"path": str(MARS.resolve()), "sha256": EXPECTED_MARS},
                "us_boundary": {"path": str(STATES.resolve()), "sha256": sha(STATES)},
                "selection_rule": "Official source TRAIN only; exact US state polygon; >25 km from all October 23 MARS cohort point coordinates; existing source internal-fitting groups train; deterministic split of source internal-development 25 km groups into dev/test; <=4 positive and <=4 negative observations per exact site by seeded metadata hash; >=90% nonzero six-band target/reference90 support; cross-split identical input pairs quarantined",
                "quality_limitations": ["L2A 32x32 crops are a separate product/resolution domain from MARS L1C.",
                                        "No exact acquisition IDs, timestamps, affine/CRS, or per-pixel cloud masks in source.",
                                        "Source masks are released binary labels; prior MethaneS2CM work in this project means this is not project-untouched test data.",
                                        "The official MethaneS2CM location test is reserved and is not used here."],
                "target_20m_rule": "identity; released source mask is already 32x32 nominal 20m",
                "native_scoring_rule": "identity; do not upsample or invent georeferencing",
                "native_repeat_factor": 1, "split_counts": counts, "split_groups": split_summary,
                "exclusions": dict(exclusions), "wide_mask_counts": dict(wide_mask), "scenes": scenes}
    save(dest / "manifest.json", manifest)
    replay = {}
    with h5py.File(H5, "r") as source:
        for split in ("train", "dev", "test"):
            subset = sorted((s for s in scenes if s["split"] == split), key=lambda s: stable(s["id"]))[:2]
            for scene in subset:
                index = next(i for i, row in enumerate(rows) if row["id"] == scene["source_row"]["id"])
                expected, _, _, _ = source_arrays(source, index, int(scene["source_label"]))
                with np.load(dest / scene["array_file"], allow_pickle=False) as actual:
                    if any(not np.array_equal(actual[key], value) for key, value in expected.items()):
                        raise ValueError(f"Source replay mismatch: {scene['id']}")
                replay[scene["id"]] = "all seven arrays match original TRAIN HDF5"
    receipt = {"schema": "GTM_Model7_s2cm_repair_receipt_v1", "manifest": str(dest / "manifest.json"),
               "manifest_sha256": sha(dest / "manifest.json"), "exporter_sha256": sha(Path(__file__)),
               "protocol_sha256": sha(PROTOCOL), "source_h5_sha256": EXPECTED_H5,
               "source_rows_sha256": EXPECTED_ROWS, "source_mars_exclusion_sha256": EXPECTED_MARS,
               "state_boundary_sha256": sha(STATES), "candidate_rows": len(candidates),
               "eligible_us_sites": len(eligible_sites), "site_class_cap": MAX_PER_SITE_CLASS,
               "scene_count": len(scenes), "split_counts": counts, "split_geography": split_summary,
               "exclusions": dict(exclusions), "wide_mask_counts": dict(wide_mask),
               "source_replay": replay, "official_location_test_opened": False,
               "new_downloads": 0, "acquisition_identity_limitation": "Source lacks exact acquisition IDs; cannot prove cross-source acquisition disjointness"}
    save(dest / "receipt.json", receipt)
    print(json.dumps({"directory": str(dest), "scenes": len(scenes), "split_counts": counts,
                      "split_geography": split_summary, "exclusions": dict(exclusions),
                      "replay_scenes": len(replay)}, indent=2))


if __name__ == "__main__":
    main()
