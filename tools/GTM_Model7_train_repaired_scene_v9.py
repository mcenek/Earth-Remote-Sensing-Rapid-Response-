"""One per-positive-scene Dice fit on the frozen v8 schedule and v7 U-Net."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import GTM_Model7_train_repaired_broader_v7 as v7
import GTM_Model7_train_repaired_site_v8 as v8

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
V7 = v7.HERE
V8 = v8.HERE
V5, V4, V3, ORIGINAL, STABLE = v7.V5, v7.V4, v7.V3, v7.ORIGINAL, v7.STABLE
PROTOCOL = ROOT / "docs/GTM_Model7_scene_dice_v9_protocol_20261025.md"
V8_SHA256 = "aab52ac10522fb653275d6bdb9f789b3d17e0fcfe92294f5afd46bab47a78abc"
V8_FIT = ROOT / "outputs/GTM_Model7_site_balanced_v8_20261024/fit"
V7_SHA256 = "822d5b542f1dabbd28d8a7c3bba6d13685507c3a8768d4214e8cd4c06722bec4"
V7_METADATA = ROOT / "outputs/GTM_Model7_broader_unet_20261024/early_fusion/metadata.json"
MILESTONES = (1200, 2400, 3600, 4800)
MAX_WALL_SECONDS_PER_MODEL = 1800
REPLAY_ATOL = v7.REPLAY_ATOL
THRESHOLD = .5
ACTIVE_BRANCH = "mars_repaired"
ARM = None

sha, write_json, load_scene = v7.sha, v7.write_json, v7.load_scene
side, native_factor = v7.side, v7.native_factor
v5 = v7.v5


def check_sources(meta=None):
    if sha(V8) != V8_SHA256:
        raise ValueError("Frozen v8 implementation changed")
    if sha(V7) != V7_SHA256:
        raise ValueError("Frozen v7 implementation changed")
    if meta:
        for field, source in (("trainer_sha256", HERE), ("v8_source_sha256", V8), ("v7_source_sha256", V7),
                              ("v5_source_sha256", V5), ("v4_source_sha256", V4),
                              ("v3_source_sha256", V3), ("original_sha256", ORIGINAL),
                              ("stable_source_sha256", STABLE), ("protocol_sha256", PROTOCOL)):
            if meta.get(field) != sha(source):
                raise ValueError("Frozen dependency drift: " + field)


def configure(meta):
    global ARM
    check_sources(meta if "trainer_sha256" in meta else None)
    arm = meta.get("config", {}).get("arm", meta.get("arm"))
    if arm != "scene_dice":
        raise ValueError("v9 requires scene_dice arm")
    v7.configure({"arm": "early_fusion"})
    ARM = arm


def set_branch(name):
    global ACTIVE_BRANCH
    v7.set_branch(name)
    ACTIVE_BRANCH = name


def new_model(name):
    if ARM != "scene_dice":
        raise ValueError("Configure v9 before model construction")
    return v7.new_model(name)


def prepare(data, norm):
    if ARM != "scene_dice":
        raise ValueError("Configure v9 before feature construction")
    return v7.prepare(data, norm)


def predict(model, x):
    return v7.predict(model, x)


def confusion(prob, target, valid):
    hit, truth = prob[valid.astype(bool)] >= THRESHOLD, target[valid.astype(bool)] == 1
    return {"tp": int(np.count_nonzero(hit & truth)), "fp": int(np.count_nonzero(hit & ~truth)),
            "fn": int(np.count_nonzero(~hit & truth)), "tn": int(np.count_nonzero(~hit & ~truth))}


def evaluate(model, manifest_path, scenes, norm, split, folder):
    v7.THRESHOLD = THRESHOLD
    return v7.evaluate(model, manifest_path, scenes, norm, split, folder)


make_batch = v7.make_batch


def loss_parts(logits, target, valid, positive):
    if logits.ndim != 4 or logits.shape[1:] != (1, 100, 100) or logits.shape != target.shape or valid.shape != target.shape or positive.shape != (logits.shape[0],):
        raise ValueError("Loss grid/flag shape mismatch")
    if logits.shape[0] != 8:
        raise ValueError("Fixed batch size changed")
    if not all(torch.isfinite(v).all() for v in (logits, target, valid, positive)):
        raise FloatingPointError("Nonfinite loss input")
    if not torch.all((target == 0) | (target == 1)) or not torch.all((valid == 0) | (valid == 1)) or not torch.all((positive == 0) | (positive == 1)):
        raise ValueError("Nonbinary label, support or class flag")
    support = valid.sum(dim=(1, 2, 3))
    if not torch.all(support > 0):
        raise ValueError("Empty per-scene support")
    if positive.sum() != 4 or (positive == 0).sum() != 4:
        raise ValueError("Fixed four/four batch changed")
    if not torch.equal(positive > .5, (target * valid).sum(dim=(1, 2, 3)) > 0):
        raise ValueError("Draw class and actual scene label differ")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=logits.new_tensor(2.0))
    bce = (raw * valid).sum() / valid.sum()
    p = torch.sigmoid(logits)
    intersection = (p * target * valid).sum(dim=(1, 2, 3))
    prediction = (p * valid).sum(dim=(1, 2, 3))
    truth = (target * valid).sum(dim=(1, 2, 3))
    dice_each = 1 - (2 * intersection + 1) / (prediction + truth + 1)
    dice = dice_each[positive.bool()].mean()
    loss = bce + .1 * dice
    if not all(torch.isfinite(v) for v in (loss, bce, dice)):
        raise FloatingPointError("Nonfinite loss")
    return loss, bce, dice


def fit_step(model, optimizer, sample):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, bce, dice = loss_parts(model(sample[0]), *sample[1:])
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return {"loss": float(loss.item()), "bce": float(bce.item()),
            "mean_positive_scene_dice_loss": float(dice.item()), "gradient_norm": float(gradient.item())}


def frozen_inputs(manifest_path):
    check_sources()
    v8_meta = json.loads((V8_FIT / "metadata.json").read_text(encoding="utf-8"))
    if v8_meta["trainer_sha256"] != sha(V8) or v8_meta["manifest_sha256"] != sha(manifest_path):
        raise ValueError("Frozen v8 source or manifest drift")
    old = json.loads(V7_METADATA.read_text(encoding="utf-8"))
    if sha(manifest_path) != old["manifest_sha256"]:
        raise ValueError("Combined TRAIN/DEV manifest drift")
    base, scenes, _ = v7.frozen_inputs(manifest_path, Path(old["reservation_path"]))
    if base["normalization"] != old["normalization"]:
        raise ValueError("Frozen TRAIN normalizer changed")
    if base["normalization"] != v8_meta["normalization"]:
        raise ValueError("v8 TRAIN normalizer changed")
    return base, scenes, old


def site_schedule(scenes):
    by_class = {True: collections.defaultdict(list), False: collections.defaultdict(list)}
    by_id = {}
    for row in scenes:
        if row["split"] != "train":
            continue
        positive = row["scene_kind"] == "PLUME"
        by_class[positive][row["physical_location_id"]].append(row["id"])
        by_id[row["id"]] = row
    sites = {class_: sorted(groups) for class_, groups in by_class.items()}
    for groups in by_class.values():
        for ids in groups.values():
            ids.sort()
    if len(by_id) != 958 or not all(sites.values()):
        raise ValueError("TRAIN site inventory changed")
    rng = random.Random(20261024)
    draws = []
    for _ in range(4800):
        batch = []
        for positive in (True, False):
            for _ in range(4):
                site = rng.choice(sites[positive])
                sid = rng.choice(by_class[positive][site])
                batch.append([sid, 0, 0, rng.randrange(8), positive])
        rng.shuffle(batch)
        draws.extend(batch)
    if len(draws) != 38400 or any(sum(bool(d[4]) for d in draws[i:i+8]) != 4 for i in range(0, 38400, 8)):
        raise ValueError("Site schedule class balance changed")
    return draws, by_id, sites


def schedule_counts(draws, by_id, sites):
    country = collections.Counter()
    site = collections.Counter()
    observation = collections.Counter()
    for sid, _, _, _, positive in draws:
        row = by_id[sid]
        kind = "PLUME" if positive else "NO_PLUME"
        if row["scene_kind"] != kind:
            raise ValueError("Draw class drift")
        country[(kind, row["source_row"]["country"])] += 1
        site[(kind, row["physical_location_id"])] += 1
        observation[sid] += 1
    return {"class_site_inventory": {"PLUME": len(sites[True]), "NO_PLUME": len(sites[False])},
            "unique_sampled_observations": len(observation),
            "unique_sampled_sites_by_class": {"PLUME": len({s for k, s in site if k == "PLUME"}),
                                              "NO_PLUME": len({s for k, s in site if k == "NO_PLUME"})},
            "country_draws": {f"{k}/{c}": v for (k, c), v in sorted(country.items())},
            "site_draws": {f"{k}/{s}": v for (k, s), v in sorted(site.items())},
            "observation_draws": dict(sorted(observation.items()))}


def train(manifest_path, preflight_path, output):
    if output.exists():
        raise FileExistsError(output)
    configure({"arm": "scene_dice"})
    set_branch("mars_repaired")
    base, scenes, old = frozen_inputs(manifest_path)
    draws, by_id, sites = site_schedule(scenes)
    v8_schedule = V8_FIT / "draw_schedule.json"
    if sha(v8_schedule) != json.loads((V8_FIT / "metadata.json").read_text(encoding="utf-8"))["draw_schedule_file_sha256"] or draws != json.loads(v8_schedule.read_text(encoding="utf-8")):
        raise ValueError("Frozen v8 draw schedule changed")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("passed") or preflight["trainer_sha256"] != sha(HERE) or preflight["manifest_sha256"] != sha(manifest_path) or preflight["schedule_sha256"] != hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest() or preflight["v8_schedule_file_sha256"] != sha(v8_schedule):
        raise ValueError("Matching successful real-data preflight required")
    device = v5.v4.configure_gpu()
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    write_json(output / "schedule_counts.json", schedule_counts(draws, by_id, sites))
    dependencies = {"trainer_sha256": sha(HERE), "v8_source_sha256": sha(V8), "v7_source_sha256": sha(V7),
                    "v5_source_sha256": sha(V5), "v4_source_sha256": sha(V4),
                    "v3_source_sha256": sha(V3), "original_sha256": sha(ORIGINAL),
                    "stable_source_sha256": sha(STABLE), "protocol_sha256": sha(PROTOCOL)}
    meta = {"schema": "GTM_Model7_scene_dice_v9", "arm": "scene_dice",
            "trainer_path": str(HERE), "manifest": str(manifest_path),
            "manifest_sha256": sha(manifest_path), "normalization": base["normalization"],
            "baseline_metadata_sha256": sha(v7.BASELINE / "metadata.json"),
            "v7_metadata_sha256": sha(V7_METADATA), "v8_metadata_sha256": sha(V8_FIT / "metadata.json"),
            "v8_schedule_file_sha256": sha(v8_schedule), **dependencies,
            "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
            "schedule_counts_sha256": sha(output / "schedule_counts.json"),
            "preflight_sha256": sha(preflight_path),
            "split_counts": {part: sum(s["split"] == part for s in scenes) for part in ("train", "dev")},
            "config": {"arm": "scene_dice", "architecture": "v7_early_fusion", "seed": 20261024,
                       "branch": "mars_repaired", "side": 100, "crop": 100, "batch": 8,
                       "steps": 4800, "milestones": MILESTONES, "learning_rate": .0005,
                       "weight_decay": .0001, "positive_weight": 2., "dice_weight": .1,
                       "dice_reduction": "mean_positive_scene",
                       "gradient_cap": 1., "threshold": .5,
                       "max_wall_seconds_per_model": MAX_WALL_SECONDS_PER_MODEL,
                       "device": str(device), "dtype": "float32", "amp": False, "tf32": False},
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "test_array_access": "none"}
    write_json(output / "metadata.json", meta)
    model_dir = output / "siamese"
    model_dir.mkdir()
    torch.manual_seed(20261024)
    torch.cuda.manual_seed_all(20261024)
    model = new_model("siamese").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    original_initial = torch.load(V8_FIT / "siamese" / "initial.pt", map_location="cpu", weights_only=False)
    if original_initial["normalization"] != base["normalization"] or any(not torch.equal(value.cpu(), original_initial["model"][key]) for key, value in model.state_dict().items()):
        raise ValueError("v9 initialization differs from v8")
    v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", 0, model_dir / "initial.pt")
    cache, curve, milestones = {}, [], {}
    started = time.monotonic()
    for step in range(1, 4801):
        if time.monotonic() - started > MAX_WALL_SECONDS_PER_MODEL:
            raise TimeoutError(f"v9 exceeded 30-minute fit cap at update {step}")
        sample = make_batch(manifest_path, by_id, draws[(step - 1)*8:step*8], base["normalization"], cache)
        values = fit_step(model, optimizer, tuple(value.to(device) for value in sample))
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, **values})
        if step % 200 == 0:
            print(json.dumps({"arm": "scene_dice", "step": step,
                              "seconds": round(time.monotonic() - started, 1), "loss": values["loss"]}), flush=True)
        if step in MILESTONES:
            checkpoint = model_dir / f"step_{step}.pt"
            state = v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", step, checkpoint)
            cpu_model = new_model("siamese")
            cpu_model.load_state_dict(state)
            train_score = evaluate(cpu_model, manifest_path, scenes, base["normalization"], "train", model_dir / f"train_step_{step}")
            dev_score = evaluate(cpu_model, manifest_path, scenes, base["normalization"], "dev", model_dir / f"dev_step_{step}")
            milestones[str(step)] = {"checkpoint_sha256": sha(checkpoint), "train": train_score,
                                     "dev": dev_score, "elapsed_seconds": time.monotonic()-started}
            write_json(model_dir / "progress.json", {"milestones": milestones, "curve": curve})
            print(json.dumps({"arm": "scene_dice", "milestone": step,
                              "dev_native_iou": dev_score["native_pooled"]["iou"]}), flush=True)
    report = {"model": "siamese", "steps": 4800,
              "initial_checkpoint_sha256": sha(model_dir / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "elapsed_seconds": time.monotonic()-started}
    write_json(model_dir / "receipt.json", report)
    write_json(output / "receipt.json", {"schema": "GTM_Model7_scene_dice_v9_receipt",
                                       "metadata_sha256": sha(output / "metadata.json"),
                                       "models": {"siamese": report}, "test_array_access": "none"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    train(a.manifest.resolve(), a.preflight.resolve(), a.output.resolve())


if __name__ == "__main__":
    main()
