"""Generate an explicitly synthetic, negative-rich three-CNN fixture.

These artificial labels and metre-grid locations are for an optimization
rehearsal. They make no physical methane or geographic validity claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1] / "research" / "GTM_Model6"
sys.path.insert(0, str(RESEARCH))
from GTM_Model6_three_cnn_data import BANDS, SCHEMA, load_arrays, load_manifest, sha256  # noqa: E402
from GTM_Model6_three_cnn_fixture import _scene  # noqa: E402


COUNTS = {
    "branch_fit": (4, 16),
    "validation": (2, 8),
    "test": (16, 64),
    "fusion_fit": (4, 16),
}


def _orient(image, input_valid, target, target_valid, turns, flip):
    arrays = [image, input_valid, target, target_valid]
    arrays = [np.rot90(a, turns, axes=(0, 1)) for a in arrays]
    if flip:
        arrays = [np.flip(a, axis=1) for a in arrays]
    return [np.ascontiguousarray(a) for a in arrays]


def write_fixture(destination: Path, seed: int) -> Path:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    arrays_dir = destination / "arrays"
    arrays_dir.mkdir()
    rng = np.random.default_rng(seed)
    rows = []
    index = 0
    for kind in ("airborne", "emit"):
        roles = ("branch_fit", "validation", "test")
        if kind == "emit":
            roles += ("fusion_fit",)
        for role in roles:
            for positive, count in ((True, COUNTS[role][0]), (False, COUNTS[role][1])):
                for _ in range(count):
                    index += 1
                    polarity = "positive" if positive else "negative"
                    row_id = f"synthetic-{index:04d}-{kind}-{role}-{polarity}"
                    path = arrays_dir / f"{row_id}.npz"
                    # Continuous phase and independently drawn pixel noise have the
                    # same distribution regardless of class or source role.
                    phase = float(rng.uniform(-np.pi, np.pi))
                    arrays = _scene(rng, positive, phase)
                    turns = int(rng.integers(0, 4))
                    flip = bool(rng.integers(0, 2))
                    image, input_valid, target, target_valid = _orient(*arrays, turns, flip)
                    np.savez_compressed(path, image=image, input_valid=input_valid,
                                        target=target, target_valid=target_valid)
                    x_origin = index * 2000
                    y_origin = 10000
                    rows.append({
                        "id": row_id,
                        "parent_id": f"synthetic-parent-{index:04d}",
                        "site": f"synthetic-site-{index:04d}",
                        "group": f"synthetic-group-{index:04d}",
                        "source_ids": [f"synthetic-sentinel-{index:04d}",
                                       f"synthetic-{kind}-source-{index:04d}"],
                        "role": role,
                        "target_kind": kind,
                        "input": {
                            "path": f"arrays/{path.name}", "sha256": sha256(path),
                            "sensor": "Sentinel-2", "product_level": "L2A",
                            "units": "reflectance", "radiometry_verified": True,
                            "acquired_at": "2026-01-01T12:00:00Z",
                            "grid": {"shape": [48, 48], "crs": "LOCAL_METRE_GRID",
                                     "affine": [20, 0, x_origin, 0, -20, y_origin]},
                        },
                        "target": {
                            "kind": "plume_mask", "units": "binary",
                            "acquired_at": "2026-01-01T12:03:00Z",
                            "support_reviewed": True,
                            "reviewer": "synthetic-fixture-generator",
                            "source": f"synthetic-{kind}-labels",
                            "grid": {"shape": [16, 16], "crs": "LOCAL_METRE_GRID",
                                     "affine": [60, 0, x_origin, 0, -60, y_origin]},
                        },
                        "review": {
                            "status": "approved", "alignment_verified": True,
                            "timing_verified": True, "independence_verified": True,
                            "reviewer": "synthetic-fixture-generator",
                            "notes": ("Artificial synthetic scene and supervision role only; "
                                      "no measured methane, physical validity, or independent real geography claim."),
                        },
                    })

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": SCHEMA, "data_kind": "synthetic",
                                         "bands": BANDS, "observations": rows}, indent=2) + "\n",
                             encoding="utf-8")
    loaded = load_manifest(manifest_path)
    counts = Counter()
    seen = {key: set() for key in ("id", "parent_id", "site", "group")}
    sources = set()
    for row in loaded["observations"]:
        for key in seen:
            assert row[key] not in seen[key], (key, row[key])
            seen[key].add(row[key])
        assert not sources.intersection(row["source_ids"])
        sources.update(row["source_ids"])
        arr = load_arrays(row)
        image, iv, target, tv = (arr[name] for name in
                                 ("image", "input_valid", "target", "target_valid"))
        assert image.shape == (48, 48, 5) and iv.shape == (48, 48)
        assert target.shape == (16, 16) and tv.shape == (16, 16)
        assert np.count_nonzero(iv == 0) == 9 and np.count_nonzero(tv == 0) == 1
        assert np.isnan(image[iv == 0]).all() and np.isnan(target[tv == 0]).all()
        positive = bool(np.any(target[tv == 1] == 1))
        assert positive == row["id"].endswith("-positive")
        counts[f"{row['target_kind']}/{row['role']}/{('positive' if positive else 'negative')}"] += 1
    expected = {f"{kind}/{role}/{polarity}": n
                for kind in ("airborne", "emit")
                for role in (("branch_fit", "validation", "test", "fusion_fit")
                             if kind == "emit" else ("branch_fit", "validation", "test"))
                for polarity, n in zip(("positive", "negative"), COUNTS[role])}
    assert dict(counts) == expected, (dict(counts), expected)
    receipt = {
        "description": "Synthetic optimization fixture; no measured methane or real geography",
        "invocation": f"python tools/GTM_Model6_negative_rich_fixture.py --output {destination} --seed {seed}",
        "seed": seed,
        "manifest": str(manifest_path),
        "manifest_sha256": loaded["manifest_sha256"],
        "original_scenes": len(rows),
        "positive": sum(n for key, n in counts.items() if key.endswith("/positive")),
        "negative": sum(n for key, n in counts.items() if key.endswith("/negative")),
        "counts": dict(sorted(counts.items())),
        "verification": "Existing manifest/array loader plus every array shape, support, label, and identity checked",
    }
    (destination / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(write_fixture(args.output, args.seed))


if __name__ == "__main__":
    main()
