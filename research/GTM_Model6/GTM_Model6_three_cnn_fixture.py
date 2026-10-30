"""Deterministic local engineering fixture for the three-CNN workflow.

All values, sources, coordinates, and labels are synthetic. The LOCAL_METRE_GRID
coordinates are separated for split checks and have no geographic meaning.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from GTM_Model6_three_cnn_data import (BANDS, SCHEMA, SCENES_SCHEMA,
                                       load_arrays, load_manifest, load_scenes, sha256)


def _scene(rng, positive, scene_index):
    height = width = 48
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x = (xx - 23.5) / 24.0
    y = (yy - 23.5) / 24.0
    texture = (0.035 * np.sin(5.4 * x + scene_index * 0.3)
               + 0.025 * np.cos(4.1 * y - 2.2 * x)
               + rng.normal(0, 0.012, (height, width))).astype(np.float32)
    image = np.empty((height, width, 5), dtype=np.float32)
    for band, base in enumerate((0.19, 0.24, 0.29, 0.34, 0.37)):
        image[:, :, band] = base + texture * (0.6 + band * 0.1)

    # A winding, tapering plume whose geometry cannot be learned as a rectangle.
    center_line = 23.0 + 5.0 * np.sin((xx - 8.0) / 10.0 + scene_index * 0.13)
    width_at_x = 2.0 + np.clip((xx - 9.0) / 38.0, 0, 1) * 5.8
    tail = np.exp(-((yy - center_line) / width_at_x) ** 2)
    taper = np.clip((xx - 7.0) / 10.0, 0, 1) * np.clip((49.0 - xx) / 9.0, 0, 1)
    plume = (tail * taper).astype(np.float32) if positive else np.zeros((height, width), np.float32)
    image[:, :, 3] += plume * 0.12
    image[:, :, 4] -= plume * 0.105
    image[:, :, 2] += plume * 0.018
    image = np.clip(image, 0, 1).astype(np.float32)

    input_valid = np.ones((height, width), dtype=np.uint8)
    input_valid[:3, :3] = 0
    image[:3, :3, :] = np.nan
    target_valid = np.ones((16, 16), dtype=np.uint8)
    target_valid[0, 0] = 0
    # Mean of a 3x3 native footprint. Narrow curved margins remain unknown only
    # where the input is invalid; all valid negatives are reviewed background.
    native_signal = plume.reshape(16, 3, 16, 3).mean(axis=(1, 3))
    target = (native_signal > 0.27).astype(np.float32)
    target[0, 0] = np.nan
    return image, input_valid, target, target_valid


def write_fixture(destination, seed=17):
    """Write a fresh synthetic dataset and return its validated manifest path."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"fixture destination already exists: {destination}")
    destination.mkdir(parents=True)
    arrays_dir = destination / "arrays"
    arrays_dir.mkdir()
    rng = np.random.default_rng(seed)
    rows = []
    cases = []
    for kind in ("airborne", "emit"):
        for role in ("branch_fit", "validation", "test"):
            for positive in (True, False):
                cases.append((kind, role, positive))
    for positive in (True, False):
        cases.append(("emit", "fusion_fit", positive))

    for index, (kind, role, positive) in enumerate(cases):
        polarity = "plume" if positive else "background"
        row_id = f"synthetic-{kind}-{role}-{polarity}"
        array_path = arrays_dir / f"{row_id}.npz"
        image, input_valid, target, target_valid = _scene(rng, positive, index)
        np.savez_compressed(array_path, image=image, input_valid=input_valid,
                            target=target, target_valid=target_valid)
        # Every role has a disjoint engineering footprint with a large gap.
        x_origin = index * 2000
        y_origin = 10000
        input_grid = {"shape": [48, 48], "crs": "LOCAL_METRE_GRID",
                      "affine": [20, 0, x_origin, 0, -20, y_origin]}
        target_grid = {"shape": [16, 16], "crs": "LOCAL_METRE_GRID",
                       "affine": [60, 0, x_origin, 0, -60, y_origin]}
        rows.append({
            "id": row_id,
            "parent_id": f"synthetic-parent-{index}",
            "site": f"synthetic-site-{index}",
            "group": f"synthetic-group-{index}",
            "source_ids": [f"synthetic-sentinel-{index}", f"synthetic-{kind}-source-{index}"],
            "role": role,
            "target_kind": kind,
            "input": {
                "path": f"arrays/{array_path.name}", "sha256": sha256(array_path),
                "sensor": "Sentinel-2", "product_level": "L2A", "units": "reflectance",
                "radiometry_verified": True, "acquired_at": "2026-01-01T12:00:00Z",
                "grid": input_grid,
            },
            "target": {
                "kind": "plume_mask", "units": "binary",
                "acquired_at": "2026-01-01T12:03:00Z", "support_reviewed": True,
                "reviewer": "synthetic-fixture-generator", "source": f"synthetic-{kind}-labels",
                "grid": target_grid,
            },
            "review": {
                "status": "approved", "alignment_verified": True,
                "timing_verified": True, "independence_verified": True,
                "reviewer": "synthetic-fixture-generator",
                "notes": "Algorithmically generated engineering scene; no geographic or measured methane claim.",
            },
        })
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": SCHEMA, "data_kind": "synthetic",
                                         "bands": BANDS, "observations": rows}, indent=2) + "\n",
                             encoding="utf-8")
    load_manifest(manifest_path)
    return manifest_path


def write_scenes(training_manifest, destination):
    """Copy four synthetic test inputs into a fresh target-free scene fixture."""
    training = load_manifest(training_manifest)
    if training["data_kind"] != "synthetic":
        raise ValueError("scene fixture requires a synthetic paired manifest")
    tests = [row for row in training["observations"] if row["role"] == "test"]
    if len(tests) != 4:
        raise ValueError(f"scene fixture requires exactly four test observations; got {len(tests)}")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"scene fixture destination already exists: {destination}")
    destination.mkdir(parents=True)
    arrays_dir = destination / "arrays"
    arrays_dir.mkdir()
    rows = []
    for original in tests:
        sentinel_sources = [source for source in original["source_ids"]
                            if "sentinel" in source.casefold()]
        if not sentinel_sources:
            raise ValueError(f"test observation {original['id']} lacks an explicit Sentinel source ID")
        row_id = original["id"]
        array_path = arrays_dir / f"{row_id}.npz"
        arrays = load_arrays(original)
        np.savez_compressed(array_path, image=arrays["image"],
                            input_valid=arrays["input_valid"])
        inp = copy.deepcopy(original["input"])
        inp["path"] = f"arrays/{array_path.name}"
        inp["sha256"] = sha256(array_path)
        rows.append({
            "id": row_id,
            "source_ids": sentinel_sources,
            "input": inp,
            "review": {"status": "approved", "reviewer": "synthetic-fixture-generator",
                       "notes": "Synthetic Sentinel-only engineering scene; no geographic claim."},
        })
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": SCENES_SCHEMA, "data_kind": "synthetic",
                                         "bands": BANDS, "observations": rows}, indent=2) + "\n",
                             encoding="utf-8")
    load_scenes(manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Write a fresh synthetic three-CNN engineering fixture")
    parser.add_argument("--output", type=Path, required=True, help="new destination directory")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(write_fixture(args.output, seed=args.seed))


if __name__ == "__main__":
    main()
