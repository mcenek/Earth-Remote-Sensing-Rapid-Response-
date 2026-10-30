"""One bounded full-scene median-centering pilot for repaired MARS inputs.

Frozen protocol: docs/GTM_Model7_scene_centering_pilot_20261024.md.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_domain_v4 as v4

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL = v4.ORIGINAL
STABLE = v4.STABLE
V3 = v4.V3
V4 = v4.HERE
PROTOCOL = ROOT / "docs/GTM_Model7_scene_centering_pilot_20261024.md"
BASE_MANIFEST = v4.BASE_MANIFEST
COMBINED_MANIFEST = v4.COMBINED_MANIFEST
BASE_DRAWS = v4.BASE_DRAWS
SEED = v4.SEED
ACTIVE_BRANCH = v4.ACTIVE_BRANCH
THRESHOLD = 0.5
REPLAY_ATOL = v4.REPLAY_ATOL
ARM = "balanced_full_scene_centered"

sha = v4.sha
write_json = v4.write_json
load_scene = v4.load_scene
new_model = v4.new_model
predict = v4.predict
native_factor = v4.native_factor
side = v4.side


def set_branch(name: str):
    global ACTIVE_BRANCH
    v4.set_branch(name)
    ACTIVE_BRANCH = v4.ACTIVE_BRANCH


def _centered_date(data: dict, key: str):
    valid = data["valid"].astype(bool)
    pixels = v4.v3.radiometry(data[key][valid])
    if not np.isfinite(pixels).all() or not len(pixels):
        raise ValueError("Invalid centered-date support")
    median = np.median(pixels, axis=0)
    return pixels - median


def normalization(manifest_path: Path, scenes: list[dict]):
    count = 0
    sums = np.zeros(6, dtype=np.float64)
    squares = np.zeros(6, dtype=np.float64)
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = load_scene(manifest_path, scene)
        for key in ("target_image", "reference_image"):
            pixels = _centered_date(data, key)
            count += len(pixels)
            sums += pixels.sum(axis=0)
            squares += np.square(pixels).sum(axis=0)
    if not count:
        raise ValueError("No centered TRAIN support")
    center = sums / count
    scale = np.maximum(np.sqrt(np.maximum(squares / count - center * center, 0)), 1e-6)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite centered train normalizer")
    return {"center": center.tolist(), "scale": scale.tolist(), "train_date_pixels": count,
            "clip_standardized": 8.0, "invalid_fill": 0.0,
            "radiometry": "log1p_dn_date_band_median_centered",
            "date_median_rule": "per-date per-band common valid scene pixels"}


def prepare(data: dict, norm: dict):
    if norm.get("radiometry") != "log1p_dn_date_band_median_centered":
        raise ValueError("Centered checkpoint requires centered normalization")
    valid = data["valid"].astype(bool)
    dates = []
    for key in ("target_image", "reference_image"):
        x = np.zeros((side(), side(), 6), dtype=np.float32)
        centered = _centered_date(data, key)
        x[valid] = np.clip((centered - norm["center"]) / norm["scale"], -8, 8).astype(np.float32)
        dates.append(x)
    result = np.concatenate(dates, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite centered scene")
    return result


# V4's full-scene batch and CPU evaluator call the V3 module globals. Rebind only
# this process so saved V4 arms and their source hashes remain unchanged.
v4.v3.prepare = prepare


def confusion(prob, target, valid):
    v4.v3.THRESHOLD = THRESHOLD
    return v4.v3.confusion(prob, target, valid)


def evaluate(model, manifest_path, scenes, norm, split, folder):
    v4.v3.THRESHOLD = THRESHOLD
    return v4.v3.evaluate(model, manifest_path, scenes, norm, split, folder)


def run(output: Path):
    if output.exists():
        raise FileExistsError(output)
    device = v4.configure_gpu()
    set_branch("mars_repaired")
    manifest_path = COMBINED_MANIFEST
    _, scenes, counts, positives = v4.v3.checked_scenes(manifest_path)
    norm = normalization(manifest_path, scenes)
    draws = v4.draw_schedule("balanced_full_scene", manifest_path, scenes)
    control = output.parent / "balanced_full_scene" / "draw_schedule.json"
    if not control.exists() or draws != json.loads(control.read_text(encoding="utf-8")):
        raise ValueError("Full-scene control schedule differs")
    domain_counts = v4.draw_counts(draws, scenes)
    if domain_counts != {"US_positive": 9600, "US_negative": 9600,
                         "aux_positive": 9600, "aux_negative": 9600}:
        raise ValueError("Unbalanced centered pilot schedule")
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    meta = {"schema": "GTM_Model7_repaired_centered_v5", "arm": ARM,
            "manifest": str(manifest_path.resolve()), "trainer_path": str(HERE),
            "manifest_sha256": sha(manifest_path), "trainer_sha256": sha(HERE),
            "v3_source_sha256": sha(V3), "v4_source_sha256": sha(V4),
            "original_sha256": sha(ORIGINAL), "stable_source_sha256": sha(STABLE),
            "protocol_sha256": sha(PROTOCOL), "normalization": norm,
            "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
            "control_draw_schedule_sha256": sha(control),
            "baseline_schedule_file_sha256": sha(BASE_DRAWS),
            "baseline_manifest_sha256": sha(BASE_MANIFEST),
            "split_counts": counts, "positive_counts": positives,
            "draw_domain_class_counts": domain_counts,
            "config": {"seed": SEED, "branch": ACTIVE_BRANCH, "side": 100,
                       "crop": 100, "batch": 8, "steps": 4800,
                       "milestones": (1200, 2400, 3600, 4800),
                       "learning_rate": v4.v3.LR, "weight_decay": v4.v3.WEIGHT_DECAY,
                       "positive_weight": v4.v3.POS_WEIGHT, "dice_weight": v4.v3.DICE_WEIGHT,
                       "gradient_cap": v4.v3.GRAD_CAP, "threshold": THRESHOLD,
                       "device": str(device), "dtype": "float32", "amp": False, "tf32": False,
                       "max_wall_seconds_per_model": v4.MAX_WALL_SECONDS_PER_MODEL},
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "test_array_access": "none"}
    write_json(output / "metadata.json", meta)
    models = {}
    for name in ("stable_residual", "siamese"):
        models[name] = v4.fit_arm_model(name, "balanced_full_scene", manifest_path,
                                        scenes, norm, draws, output / name, device)
    receipt = {"schema": "GTM_Model7_repaired_centered_v5_receipt", "arm": ARM,
               "metadata_sha256": sha(output / "metadata.json"), "models": models,
               "draw_domain_class_counts": domain_counts, "test_array_access": "none"}
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"training": "complete", "arm": ARM, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
