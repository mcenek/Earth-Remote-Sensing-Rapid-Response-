"""One fixed TRAIN-only hard-negative replay of the v8 early-fusion fit."""
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

import GTM_Model7_train_repaired_broader_v7 as v7
import GTM_Model7_train_repaired_site_v8 as v8

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
V7 = v7.HERE
V5, V4, V3, ORIGINAL, STABLE = v7.V5, v7.V4, v7.V3, v7.ORIGINAL, v7.STABLE
PROTOCOL = ROOT / "docs/GTM_Model7_hard_negative_v11_protocol_20261025.md"
V7_SHA256 = "822d5b542f1dabbd28d8a7c3bba6d13685507c3a8768d4214e8cd4c06722bec4"
V8_SHA256 = "aab52ac10522fb653275d6bdb9f789b3d17e0fcfe92294f5afd46bab47a78abc"
V8_FIT = ROOT / "outputs/GTM_Model7_site_balanced_v8_20261024/fit"
V8_METADATA_SHA256 = "7300cb3c602e9ffb29db1e7b88216c2f2eb23c86c2ab5955ae73ddf2b835c241"
V8_RECEIPT_SHA256 = "546fb226bbd4b6f603c39ee94d70eb92a531b4066b041932e729875c4c68032f"
V8_SCHEDULE_SHA256 = "8140900c25048e7d55e242271441e37e0ac8dc1d88b5a982dedb272ce2819852"
V8_INITIAL_SHA256 = "364e9e7b2f99ec7006119e033cd7ec1002c0454528827e35d0019d686d2170d9"
V8_STEP2400_SHA256 = "d592e371babcb5e788c69caa4b9c9853d4c04c2338155a203bf75640ad117e21"
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
    fixed = ((V7, V7_SHA256), (v8.HERE, V8_SHA256),
             (V8_FIT / "metadata.json", V8_METADATA_SHA256),
             (V8_FIT / "receipt.json", V8_RECEIPT_SHA256),
             (V8_FIT / "draw_schedule.json", V8_SCHEDULE_SHA256),
             (V8_FIT / "siamese/initial.pt", V8_INITIAL_SHA256),
             (V8_FIT / "siamese/step_2400.pt", V8_STEP2400_SHA256))
    if any(sha(path) != digest for path, digest in fixed):
        raise ValueError("Frozen v7/v8 dependency changed")
    if meta:
        for field, source in (("trainer_sha256", HERE), ("v7_source_sha256", V7),
                              ("v8_source_sha256", v8.HERE),
                              ("v8_fit_metadata_sha256", V8_FIT / "metadata.json"),
                              ("v8_fit_receipt_sha256", V8_FIT / "receipt.json"),
                              ("v8_draw_schedule_file_sha256", V8_FIT / "draw_schedule.json"),
                              ("v8_initial_checkpoint_sha256", V8_FIT / "siamese/initial.pt"),
                              ("v8_step2400_checkpoint_sha256", V8_FIT / "siamese/step_2400.pt"),
                              ("v5_source_sha256", V5), ("v4_source_sha256", V4),
                              ("v3_source_sha256", V3), ("original_sha256", ORIGINAL),
                              ("stable_source_sha256", STABLE), ("protocol_sha256", PROTOCOL)):
            if meta.get(field) != sha(source):
                raise ValueError("Frozen dependency drift: " + field)


def configure(meta):
    global ARM
    check_sources(meta if "trainer_sha256" in meta else None)
    arm = meta.get("config", {}).get("arm", meta.get("arm"))
    if arm != "hard_negative_replay":
        raise ValueError("v11 requires hard_negative_replay arm")
    v7.configure({"arm": "early_fusion"})
    v8.configure({"arm": "site_balanced"})
    ARM = arm


def set_branch(name):
    global ACTIVE_BRANCH
    v7.set_branch(name)
    ACTIVE_BRANCH = name


def new_model(name):
    if ARM != "hard_negative_replay" or name != "siamese":
        raise ValueError("Configure v11 before siamese model construction")
    return v8.new_model(name)


def prepare(data, norm):
    if ARM != "hard_negative_replay":
        raise ValueError("Configure v11 before feature construction")
    return v8.prepare(data, norm)


def predict(model, x):
    return v7.predict(model, x)


def confusion(prob, target, valid):
    hit, truth = prob[valid.astype(bool)] >= THRESHOLD, target[valid.astype(bool)] == 1
    return {"tp": int(np.count_nonzero(hit & truth)), "fp": int(np.count_nonzero(hit & ~truth)),
            "fn": int(np.count_nonzero(~hit & truth)), "tn": int(np.count_nonzero(~hit & ~truth))}


def evaluate(model, manifest_path, scenes, norm, split, folder):
    v8.THRESHOLD = THRESHOLD
    return v8.evaluate(model, manifest_path, scenes, norm, split, folder)


make_batch = v8.make_batch


def loss_parts(logits, target, valid, positive):
    if logits.shape != target.shape or valid.shape != target.shape or positive.shape != (target.shape[0],):
        raise ValueError("Loss grid/flag shape mismatch")
    if not all(torch.isfinite(value).all() for value in (logits, target, valid, positive)):
        raise FloatingPointError("Nonfinite loss tensor")
    if not torch.all((valid == 0) | (valid == 1)) or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Nonbinary support or labels")
    if not torch.all((positive == 0) | (positive == 1)) or not torch.all(valid.sum((1, 2, 3)) > 0):
        raise ValueError("Invalid class flags or empty per-scene support")
    return v8.loss_parts(logits, target, valid, positive)


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
            "pooled_positive_dice_loss": float(dice.item()), "gradient_norm": float(gradient.item())}


def frozen_inputs(manifest_path):
    check_sources()
    old = json.loads(V7_METADATA.read_text(encoding="utf-8"))
    if sha(manifest_path) != old["manifest_sha256"]:
        raise ValueError("Combined TRAIN/DEV manifest drift")
    base, scenes, _ = v7.frozen_inputs(manifest_path, Path(old["reservation_path"]))
    if base["normalization"] != old["normalization"]:
        raise ValueError("Frozen TRAIN normalizer changed")
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
    if draws != json.loads((V8_FIT / "draw_schedule.json").read_text(encoding="utf-8")):
        raise ValueError("Baseline v8 schedule changed")
    return draws, by_id, sites


def hard_pool(scenes):
    """Freeze recurring clear-site mistakes from v8 TRAIN update2400 only."""
    report = json.loads((V8_FIT / "receipt.json").read_text(encoding="utf-8"))
    milestone = report["models"]["siamese"]["milestones"]["2400"]
    if milestone["checkpoint_sha256"] != V8_STEP2400_SHA256:
        raise ValueError("v8 hard-mining checkpoint drift")
    records = milestone["train"]["scenes"]
    train = {row["id"]: row for row in scenes if row["split"] == "train"}
    if len(train) != 958 or set(records) != set(train):
        raise ValueError("Hard-pool TRAIN identity drift")
    grouped = collections.defaultdict(list)
    for sid, row in train.items():
        record = records[sid]
        positive = row["scene_kind"] == "PLUME"
        if bool(record["positive"]) != positive:
            raise ValueError("v8 TRAIN scene class mismatch")
        if not positive:
            native = record["native_confusion"]
            if native["tp"] or native["fn"] or native["fp"] < 0:
                raise ValueError("Malformed clear TRAIN native counts")
            grouped[row["physical_location_id"]].append((sid, native["fp"]))
    if sum(map(len, grouped.values())) != 690 or len(grouped) != 223:
        raise ValueError("Clear TRAIN inventory changed")
    hard = {}
    for site, pairs in sorted(grouped.items()):
        pairs.sort()
        active = sum(fp > 0 for _, fp in pairs)
        if active >= 2:
            hard[site] = {"members": [sid for sid, _ in pairs],
                          "native_fp_by_member": {sid: fp for sid, fp in pairs},
                          "activated_clear_observations": active,
                          "native_fp_total": sum(fp for _, fp in pairs)}
    if len(hard) != 49 or sum(v["native_fp_total"] for v in hard.values()) != 41768:
        raise ValueError("TRAIN-only hard-site criterion drift")
    return {"criterion": "at least two distinct clear TRAIN observations with native FP > 0 at v8 update2400/cutoff0.50",
            "v8_fit_receipt_sha256": sha(V8_FIT / "receipt.json"),
            "v8_step2400_checkpoint_sha256": V8_STEP2400_SHA256,
            "sites": hard}


def hard_schedule(original_draws, by_id, pool):
    eligible = sorted(pool["sites"])
    if len(eligible) != 49 or len(original_draws) != 38400:
        raise ValueError("Hard-site pool or v8 schedule changed")
    for site in eligible:
        members = pool["sites"][site]["members"]
        if not members or len(members) != len(set(members)) or members != sorted(members):
            raise ValueError("Duplicate, unsorted or empty hard-site members")
        if set(members) != set(pool["sites"][site]["native_fp_by_member"]):
            raise ValueError("Hard-site native counts/member mismatch")
        if any(sid not in by_id or by_id[sid]["split"] != "train" or
               by_id[sid]["scene_kind"] != "NO_PLUME" or
               by_id[sid]["physical_location_id"] != site for sid in members):
            raise ValueError("Hard-site contains non-clear, non-TRAIN or wrong-site member")
    rng = random.Random(20261025)
    result = []
    replaced = retained_clear = 0
    for start in range(0, 38400, 8):
        original = original_draws[start:start+8]
        if len(original) != 8 or sum(bool(draw[4]) for draw in original) != 4:
            raise ValueError("Original four/four batch changed")
        clear_index = 0
        for draw in original:
            fresh = list(draw)
            if not bool(draw[4]):
                clear_index += 1
                if clear_index > 2:
                    site = rng.choice(eligible)
                    sid = rng.choice(pool["sites"][site]["members"])
                    if sid not in by_id or by_id[sid]["split"] != "train" or by_id[sid]["scene_kind"] != "NO_PLUME" or by_id[sid]["physical_location_id"] != site:
                        raise ValueError("Hard replacement is not a member clear TRAIN scene")
                    fresh[0] = sid
                    replaced += 1
                else:
                    retained_clear += 1
            if fresh[1:] != draw[1:] or (bool(draw[4]) and fresh[0] != draw[0]):
                raise ValueError("Positive/transform/flag/position drift")
            result.append(fresh)
        if clear_index != 4:
            raise ValueError("Clear batch count changed")
    if replaced != 9600 or retained_clear != 9600:
        raise ValueError("Hard replay slot count changed")
    represented = {by_id[d[0]]["physical_location_id"] for d in result if not d[4]}
    all_clear = {r["physical_location_id"] for r in by_id.values() if r["scene_kind"] == "NO_PLUME"}
    if represented != all_clear:
        raise ValueError("Hard replay lost clear TRAIN site coverage")
    return result


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
    configure({"arm": "hard_negative_replay"})
    set_branch("mars_repaired")
    base, scenes, old = frozen_inputs(manifest_path)
    original_draws, by_id, sites = site_schedule(scenes)
    pool = hard_pool(scenes)
    draws = hard_schedule(original_draws, by_id, pool)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    pool_file = preflight_path.parent / "hard_pool.json"
    preflight_schedule = preflight_path.parent / "draw_schedule.json"
    if (not preflight.get("passed") or preflight["trainer_sha256"] != sha(HERE) or
        preflight["manifest_sha256"] != sha(manifest_path) or
        preflight["protocol_sha256"] != sha(PROTOCOL) or
        preflight["v8_fit_receipt_sha256"] != sha(V8_FIT / "receipt.json") or
        preflight["hard_pool_sha256"] != sha(pool_file) or
        preflight["schedule_file_sha256"] != sha(preflight_schedule) or
        json.loads(pool_file.read_text(encoding="utf-8")) != pool or
        json.loads(preflight_schedule.read_text(encoding="utf-8")) != draws or
        preflight["schedule_sha256"] != hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest()):
        raise ValueError("Matching successful real-data preflight required")
    device = v5.v4.configure_gpu()
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    write_json(output / "hard_pool.json", pool)
    write_json(output / "schedule_counts.json", schedule_counts(draws, by_id, sites))
    dependencies = {"trainer_sha256": sha(HERE), "v7_source_sha256": sha(V7),
                    "v8_source_sha256": sha(v8.HERE),
                    "v8_fit_metadata_sha256": sha(V8_FIT / "metadata.json"),
                    "v8_fit_receipt_sha256": sha(V8_FIT / "receipt.json"),
                    "v8_draw_schedule_file_sha256": sha(V8_FIT / "draw_schedule.json"),
                    "v8_initial_checkpoint_sha256": sha(V8_FIT / "siamese/initial.pt"),
                    "v8_step2400_checkpoint_sha256": sha(V8_FIT / "siamese/step_2400.pt"),
                    "v5_source_sha256": sha(V5), "v4_source_sha256": sha(V4),
                    "v3_source_sha256": sha(V3), "original_sha256": sha(ORIGINAL),
                    "stable_source_sha256": sha(STABLE), "protocol_sha256": sha(PROTOCOL)}
    meta = {"schema": "GTM_Model7_hard_negative_v11", "arm": "hard_negative_replay",
            "trainer_path": str(HERE), "manifest": str(manifest_path),
            "manifest_sha256": sha(manifest_path), "normalization": base["normalization"],
            "baseline_metadata_sha256": sha(v7.BASELINE / "metadata.json"),
            "v7_metadata_sha256": sha(V7_METADATA), **dependencies,
            "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
            "hard_pool_file_sha256": sha(output / "hard_pool.json"),
            "schedule_counts_sha256": sha(output / "schedule_counts.json"),
            "preflight_sha256": sha(preflight_path),
            "split_counts": {part: sum(s["split"] == part for s in scenes) for part in ("train", "dev")},
            "config": {"arm": "hard_negative_replay", "architecture": "v7_early_fusion", "seed": 20261024,
                       "hard_draw_seed": 20261025, "hard_sites": 49,
                       "retained_clear_draws_per_batch": 2, "hard_clear_draws_per_batch": 2,
                       "branch": "mars_repaired", "side": 100, "crop": 100, "batch": 8,
                       "steps": 4800, "milestones": MILESTONES, "learning_rate": .0005,
                       "weight_decay": .0001, "positive_weight": 2., "dice_weight": .1,
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
    if sum(p.numel() for p in model.parameters()) != 121937:
        raise ValueError("v8 architecture parameter count drift")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", 0, model_dir / "initial.pt")
    if sha(model_dir / "initial.pt") != V8_INITIAL_SHA256:
        raise ValueError("v8 deterministic initial checkpoint drift")
    cache, curve, milestones = {}, [], {}
    started = time.monotonic()
    for step in range(1, 4801):
        if time.monotonic() - started > MAX_WALL_SECONDS_PER_MODEL:
            raise TimeoutError(f"v11 exceeded 30-minute fit cap at update {step}")
        sample = make_batch(manifest_path, by_id, draws[(step - 1)*8:step*8], base["normalization"], cache)
        values = fit_step(model, optimizer, tuple(value.to(device) for value in sample))
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, **values})
        if step % 200 == 0:
            print(json.dumps({"arm": "hard_negative_replay", "step": step,
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
            print(json.dumps({"arm": "hard_negative_replay", "milestone": step,
                              "dev_native_iou": dev_score["native_pooled"]["iou"]}), flush=True)
    report = {"model": "siamese", "steps": 4800,
              "initial_checkpoint_sha256": sha(model_dir / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "elapsed_seconds": time.monotonic()-started}
    write_json(model_dir / "receipt.json", report)
    write_json(output / "receipt.json", {"schema": "GTM_Model7_hard_negative_v11_receipt",
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
