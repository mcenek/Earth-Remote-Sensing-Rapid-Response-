"""Freeze a metadata-only v12 split, then audit FIT arrays only.

Failure modes checked before any array is read: changed source manifest; a site,
nearby site, observation, acquisition, or spectral image crossing partitions;
missing classes or an under-supported holdout; an unexpected assignment count;
and source array drift. FIT audit rejects malformed grids, empty/unequal support,
nonbinary labels, nonfinite imagery, or a target that disagrees with native labels.
Historical exposure remains explicit; this split cannot certify new-site accuracy.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json"
OUTPUT = ROOT / "outputs/GTM_Model7_validation_v12_20261026"
SOURCE_SHA = "1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a"
SEED = 20261026


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def distance_km(a: dict, b: dict) -> float:
    p, q = math.radians(a["latitude"]), math.radians(b["latitude"])
    dp, dl = q - p, math.radians(a["longitude"] - b["longitude"])
    h = math.sin(dp / 2) ** 2 + math.cos(p) * math.cos(q) * math.sin(dl / 2) ** 2
    return 12742 * math.asin(min(1, math.sqrt(h)))


def identity_keys(scene: dict) -> set[tuple[str, str]]:
    keys = set()
    for field in ("target_scene_id", "reference_scene_id"):
        if scene.get(field):
            keys.add(("acquisition", scene[field]))
    if scene.get("reference_observation_id"):
        keys.add(("observation", scene["reference_observation_id"]))
    keys.add(("observation", scene["id"].removeprefix("MARS_")))
    for source in scene["sources"]:
        if source["role"] in ("target_image", "reference_image"):
            keys.add(("spectral_image_sha256", source["sha256"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--metadata-only", action="store_true", help="Freeze split and receipt without opening arrays")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if sha(source) != SOURCE_SHA:
        raise ValueError("Source manifest hash drift")
    source_data = json.loads(source.read_text(encoding="utf-8"))
    scenes = source_data["scenes"]
    train = [s for s in scenes if s["split"] == "train"]
    regression = [s for s in scenes if s["split"] == "dev"]
    if len(train) != 958 or len(regression) != 198 or len(scenes) != 1156:
        raise ValueError("Source inventory drift")
    by_site = collections.defaultdict(list)
    for scene in scenes:
        by_site[scene["physical_location_id"]].append(scene)
    sites = sorted(by_site)
    if len({s["physical_location_id"] for s in train}) != 224:
        raise ValueError("TRAIN site count drift")
    parent = {site: site for site in sites}

    def root(site: str) -> str:
        while parent[site] != site:
            parent[site] = parent[parent[site]]
            site = parent[site]
        return site

    def union(a: str, b: str) -> None:
        parent[root(a)] = root(b)

    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            if distance_km(by_site[a][0], by_site[b][0]) < 25:
                union(a, b)
    key_sites = collections.defaultdict(set)
    for site, group in by_site.items():
        for scene in group:
            for key in identity_keys(scene):
                key_sites[key].add(site)
    for members in key_sites.values():
        members = sorted(members)
        for site in members[1:]:
            union(members[0], site)
    components = collections.defaultdict(list)
    for site in sites:
        components[root(site)].append(site)
    train_components = []
    regression_components = []
    for members in components.values():
        group = [s for site in members for s in by_site[site]]
        splits = {s["split"] for s in group}
        if len(splits) != 1:
            raise ValueError("TRAIN / original DEV identity or proximity connection")
        (train_components if "train" in splits else regression_components).append(members)
    if len(train_components) != 55:
        raise ValueError("TRAIN component count drift")

    rows = []
    for members in train_components:
        group = [s for site in members for s in by_site[site]]
        positives = {s["physical_location_id"] for s in group if s["scene_kind"] == "PLUME"}
        rows.append({"id": min(members), "sites": sorted(members),
                     "observation_ids": sorted(s["id"] for s in group),
                     "n": len(group), "p": sum(s["scene_kind"] == "PLUME" for s in group),
                     "clear_only": len(members) - len(positives),
                     "countries": sorted({s["source_row"]["country"] for s in group})})
    rows.sort(key=lambda row: (-row["n"], hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()))
    rng = random.Random(SEED)
    best = None
    for iteration in range(30000):
        groups = {"fit": [rows[0]], "select": [], "assess": []}
        for row in rows[1:]:
            draw = rng.random()
            role = "select" if draw < .23 else "assess" if draw < .46 else "fit"
            groups[role].append(row)
        score = 0.0
        for role in ("select", "assess"):
            group = groups[role]
            n = sum(r["n"] for r in group)
            p = sum(r["p"] for r in group)
            clear_only = sum(r["clear_only"] for r in group)
            country_count = len(set().union(*(set(r["countries"]) for r in group)))
            if len(group) < 5 or p < 20 or n - p < 40 or clear_only < 1 or not 100 <= n <= 210:
                break
            score += 2 * abs(n - 144) + 1.5 * abs(p - 40) + 4 * max(0, 5 - country_count) + 5 * max(0, 2 - clear_only)
        else:
            if best is None or score < best[0]:
                best = (score, iteration, groups)
    if best is None:
        raise ValueError("No feasible metadata split")
    score, iteration, groups = best
    role_by_id = {sid: role for role, group in groups.items() for row in group for sid in row["observation_ids"]}
    counts = {role: {"observations": sum(r["n"] for r in group),
                     "plume": sum(r["p"] for r in group),
                     "clear": sum(r["n"] - r["p"] for r in group),
                     "sites": sum(len(r["sites"]) for r in group),
                     "components": len(group),
                     "countries": sorted(set().union(*(set(r["countries"]) for r in group)))}
              for role, group in groups.items()}
    if [(counts[r]["observations"], counts[r]["plume"], counts[r]["clear"]) for r in ("fit", "select", "assess")] != [(670, 188, 482), (144, 39, 105), (144, 41, 103)]:
        raise ValueError("Pinned split counts not reproduced")
    role_by_site = {site: role_by_id[by_site[site][0]["id"]] for site in sites if by_site[site][0]["split"] == "train"}
    for site, group in by_site.items():
        assigned = {role_by_id.get(s["id"], "regression") for s in group}
        if len(assigned) != 1:
            raise ValueError("Physical site crosses split")
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            ra, rb = role_by_site.get(a, "regression"), role_by_site.get(b, "regression")
            if ra != rb and distance_km(by_site[a][0], by_site[b][0]) < 25:
                raise ValueError("Near sites cross split")
    for key, members in key_sites.items():
        if len({role_by_site.get(site, "regression") for site in members}) > 1:
            raise ValueError(f"Shared identity crosses split: {key}")

    source_split = {"source_manifest": str(source), "source_manifest_sha256": SOURCE_SHA,
                    "algorithm": "v12 metadata component split, seed 20261026, 30000 trials",
                    "selected_trial_zero_based": iteration, "selected_score": score,
                    "historical_exposure": "All rows were previously available; assessment is new-run held-out, not pristine external evidence.",
                    "counts": counts,
                    "components": {role: group for role, group in groups.items()},
                    "members": {role: sorted(sid for row in group for sid in row["observation_ids"])
                                for role, group in groups.items()},
                    "regression_members": sorted(s["id"] for s in regression),
                    "regression_components": len(regression_components)}
    split_path = output / "data/frozen_split.json"
    save(split_path, source_split)
    new_scenes = []
    for s in scenes:
        row = dict(s)
        old = row["split"]
        role = role_by_id.get(s["id"], "regression")
        row["original_split"] = old
        row["validation_role"] = role
        row["country"] = s["source_row"]["country"]
        row["split"] = {"fit": "train", "select": "dev", "assess": "assessment", "regression": "regression"}[role]
        row["source_array_file"] = str((source.parent / s["array_file"]).resolve())
        row["array_file"] = row["source_array_file"]
        new_scenes.append(row)
    manifest = {k: v for k, v in source_data.items() if k != "scenes"}
    manifest.update({"schema": "GTM_Model7_validation_v12_manifest_v1", "source_manifest_sha256": SOURCE_SHA,
                     "frozen_split_sha256": sha(split_path), "historical_exposure": source_split["historical_exposure"],
                     "scenes": new_scenes})
    manifest_path = output / "data/combined_manifest.json"
    save(manifest_path, manifest)
    receipt_path = output / "data/split_receipt.json"
    save(receipt_path, {"split": str(split_path), "split_sha256": sha(split_path),
                        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                        "source_sha256": SOURCE_SHA, "counts": counts,
                        "boundary_checks": "physical, 25km, acquisition, observation, spectral image hash clear"})

    if args.metadata_only:
        print(json.dumps({"split": str(split_path), "split_sha256": sha(split_path),
                          "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                          "receipt": str(receipt_path), "counts": counts, "trial": iteration}))
        return

    # The split and its content hashes are now pinned. Only FIT pixels may be opened below.
    totals = collections.Counter()
    sizes = []
    for row in new_scenes:
        if row["validation_role"] != "fit":
            continue
        path = Path(row["source_array_file"])
        if sha(path) != row["array_sha256"]:
            raise ValueError("FIT array hash mismatch: " + row["id"])
        with np.load(path, allow_pickle=False) as z:
            required = ("target_image", "reference_image", "valid", "target", "label_native", "valid_native")
            if any(key not in z for key in required):
                raise ValueError("FIT array key missing: " + row["id"])
            data = {key: z[key] for key in required}
        if (data["target_image"].shape != (100, 100, 6) or data["reference_image"].shape != (100, 100, 6)
            or data["valid"].shape != (100, 100) or data["target"].shape != (100, 100)
            or data["label_native"].shape != (200, 200) or data["valid_native"].shape != (200, 200)):
            raise ValueError("Malformed FIT grid: " + row["id"])
        if not all(np.isfinite(data[k]).all() for k in required):
            raise ValueError("Nonfinite FIT array: " + row["id"])
        valid = data["valid"].astype(bool)
        native_valid = data["valid_native"].astype(bool)
        label = data["label_native"]
        target = data["target"]
        if not valid.any() or not np.isin(label, [0, 1]).all() or not np.isin(target, [0, 1]).all():
            raise ValueError("Invalid FIT support/label: " + row["id"])
        if not np.array_equal(native_valid, np.repeat(np.repeat(valid, 2, 0), 2, 1)):
            raise ValueError("FIT native support mismatch: " + row["id"])
        block = label.reshape(100, 2, 100, 2).sum(axis=(1, 3))
        if not np.array_equal(target, block > 0):
            raise ValueError("FIT any-positive target mismatch: " + row["id"])
        native_pos = int((label & native_valid).sum())
        if (row["scene_kind"] == "PLUME") != (native_pos > 0):
            raise ValueError("FIT class flag mismatch: " + row["id"])
        expanded = np.repeat(np.repeat(target, 2, 0), 2, 1)
        totals["native_positive_pixels"] += native_pos
        totals["expanded_any_positive_pixels"] += int((expanded * native_valid).sum())
        totals["supported_native_pixels"] += int(native_valid.sum())
        totals["plume_observations"] += int(native_pos > 0)
        sizes.append({"id": row["id"], "native_positive_pixels": native_pos,
                      "any_positive_20m_cells": int((target * valid).sum()),
                      "expanded_any_positive_pixels": int((expanded * native_valid).sum()),
                      "native_block_occupancy": [int(((block == n) & valid).sum()) for n in range(5)]})
        for threshold in range(1, 5):
            pred = np.repeat(np.repeat(block >= threshold, 2, 0), 2, 1) & native_valid
            truth = label.astype(bool) & native_valid
            totals[f"oracle_{threshold}_tp"] += int((pred & truth).sum())
            totals[f"oracle_{threshold}_fp"] += int((pred & ~truth).sum())
            totals[f"oracle_{threshold}_fn"] += int((~pred & truth).sum())
    ceilings = {}
    for threshold in range(1, 5):
        tp, fp, fn = (totals[f"oracle_{threshold}_{k}"] for k in ("tp", "fp", "fn"))
        ceilings[str(threshold)] = {"tp": tp, "fp": fp, "fn": fn,
                                    "f1": 2 * tp / (2 * tp + fp + fn)}
    audit = {"source_split_sha256": sha(split_path), "manifest_sha256": sha(manifest_path),
             "fit_only": True, "fit_arrays_verified": len(sizes), "totals": dict(totals),
             "expanded_to_native_area_ratio": totals["expanded_any_positive_pixels"] / totals["native_positive_pixels"],
             "ideal_block_constant_thresholds": ceilings,
             "ideal_block_constant_best_f1": max(v["f1"] for v in ceilings.values()),
             "scenes": sizes}
    audit_path = output / "audit/fit_label_audit.json"
    save(audit_path, audit)
    print(json.dumps({"split": str(split_path), "split_sha256": sha(split_path),
                      "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
                      "receipt": str(receipt_path), "audit": str(audit_path),
                      "counts": counts, "trial": iteration,
                      "area_ratio": audit["expanded_to_native_area_ratio"],
                      "ideal_f1": audit["ideal_block_constant_best_f1"]}))


if __name__ == "__main__":
    main()
