"""Controlled domain/exposure diagnosis for the repaired MARS trainer.

Protocol: docs/GTM_Model7_pipeline_diagnosis_20261024.md.
Only TRAIN arrays enter optimization; DEV is used for milestone evaluation.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_mars_log_v3 as v3

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL = v3.ORIGINAL
STABLE = v3.STABLE
PROTOCOL = ROOT / "docs/GTM_Model7_pipeline_diagnosis_20261024.md"
V3 = v3.HERE
SEED = v3.SEED
THRESHOLD = v3.THRESHOLD
REPLAY_ATOL = v3.REPLAY_ATOL
ACTIVE_BRANCH = v3.ACTIVE_BRANCH
BASE_MANIFEST = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json"
COMBINED_MANIFEST = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json"
BASE_DRAWS = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_training_log_v3/draw_schedule.json"
ARMS = {"us_original_norm": 2400, "us_combined_norm": 2400, "expanded_long": 4800,
        "balanced_domains": 4800, "balanced_full_scene": 4800}
MILESTONES = {"us_original_norm": (1200, 2400), "us_combined_norm": (1200, 2400),
              "expanded_long": (1200, 2400, 3600, 4800),
              "balanced_domains": (1200, 2400, 3600, 4800),
              "balanced_full_scene": (1200, 2400, 3600, 4800)}
MAX_WALL_SECONDS_PER_MODEL = 1200

sha = v3.sha
write_json = v3.write_json
load_scene = v3.load_scene
normalization = v3.normalization
prepare = v3.prepare
new_model = v3.new_model
metrics = v3.metrics
native_factor = v3.native_factor
side = v3.side
predict = v3.predict
choose_pareto = v3.choose_pareto


def confusion(prob, target, valid):
    v3.THRESHOLD = THRESHOLD
    return v3.confusion(prob, target, valid)


def evaluate(model, manifest_path, scenes, norm, split, folder):
    v3.THRESHOLD = THRESHOLD
    return v3.evaluate(model, manifest_path, scenes, norm, split, folder)


def set_branch(name: str):
    global ACTIVE_BRANCH
    v3.set_branch(name)
    ACTIVE_BRANCH = v3.ACTIVE_BRANCH


def configure_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for controlled fits")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda:0")


def _baseline_draws():
    rows = json.loads(BASE_DRAWS.read_text(encoding="utf-8"))
    if len(rows) != 19200 or any(bool(r[4]) != (i % 2 == 0) for i, r in enumerate(rows)):
        raise ValueError("Frozen baseline schedule shape or class sequence changed")
    return rows


def _aux_draws(manifest_path: Path, scenes: list[dict], count: int):
    base_ids = {s["id"] for s in json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))["scenes"]}
    aux = [s for s in scenes if s["split"] == "train" and s["id"] not in base_ids]
    sites, points = v3.train_catalog(manifest_path, aux)
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for i in range(count):
        kind = i % 2 == 0
        site_ids = sorted(sites[kind])
        site = site_ids[int(rng.integers(len(site_ids)))]
        ids = sorted(sites[kind][site])
        sid = ids[int(rng.integers(len(ids)))]
        coords = points[sid]
        cy, cx = coords[int(rng.integers(len(coords)))]
        top = int(np.clip(cy - 24, 0, 52))
        left = int(np.clip(cx - 24, 0, 52))
        rows.append([sid, top, left, int(rng.integers(8)), kind])
    return rows


def draw_schedule(arm: str, manifest_path: Path, scenes: list[dict]):
    if arm in ("us_original_norm", "us_combined_norm"):
        return _baseline_draws()
    if arm == "expanded_long":
        return v3.schedule(manifest_path, scenes, 4800)
    us = _baseline_draws()
    aux = _aux_draws(manifest_path, scenes, len(us))
    result = []
    for i in range(4800):
        result.extend(us[4*i:4*i+4])
        result.extend(aux[4*i:4*i+4])
    assert [r for i, r in enumerate(result) if i % 8 < 4] == us
    return result


def draw_counts(draws: list, scenes: list[dict]):
    base_ids = {s["id"] for s in json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))["scenes"]}
    counts = {"US_positive": 0, "US_negative": 0, "aux_positive": 0, "aux_negative": 0}
    for sid, _, _, _, positive in draws:
        key = ("US" if sid in base_ids else "aux") + ("_positive" if positive else "_negative")
        counts[key] += 1
    return counts


def make_batch(manifest_path: Path, by_id: dict, draws: list, norm: dict, cache: dict, *, full_scene: bool):
    if not full_scene:
        return v3.batch(manifest_path, by_id, draws, norm, cache)
    xs, ys, masks, positives = [], [], [], []
    for sid, _, _, transform, positive in draws:
        x, y, valid = v3.prepared(manifest_path, by_id[sid], norm, cache)
        xs.append(v3.old.d4(x, transform))
        ys.append(v3.old.d4(y, transform))
        masks.append(v3.old.d4(valid, transform))
        positives.append(positive)
    x = v3.tensor_chw(np.stack(xs))
    y = torch.from_numpy(np.stack(ys)[:, None].copy()).float()
    valid = torch.from_numpy(np.stack(masks)[:, None].copy()).float()
    flag = torch.tensor(positives, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not valid.any():
        raise ValueError("Invalid full-scene batch")
    return x, y, valid, flag


def cpu_checkpoint(model, norm: dict, name: str, step: int, path: Path):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({"model": state, "normalization": norm, "model_name": name, "steps": step}, path)
    return state


def fit_arm_model(name: str, arm: str, manifest_path: Path, scenes: list, norm: dict,
                  draws: list, output: Path, device):
    output.mkdir()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = new_model(name).to(device)
    optimizer = v3.optimizer(model)
    cpu_checkpoint(model, norm, name, 0, output / "initial.pt")
    by_id = {s["id"]: s for s in scenes}
    cache = {}
    curve, milestones = [], {}
    started = time.monotonic()
    full_scene = arm == "balanced_full_scene"
    for step in range(1, ARMS[arm] + 1):
        if time.monotonic() - started > MAX_WALL_SECONDS_PER_MODEL:
            raise TimeoutError(f"{arm}/{name} wall-time cap exceeded at step {step}")
        sample = make_batch(manifest_path, by_id, draws[(step-1)*8:step*8], norm, cache,
                            full_scene=full_scene)
        sample = tuple(t.to(device, non_blocking=False) for t in sample)
        loss, grad = v3.fit_step(model, optimizer, sample, ordinary=False)
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, "loss": loss, "gradient_norm": grad})
        if step in MILESTONES[arm]:
            checkpoint = output / f"step_{step}.pt"
            state = cpu_checkpoint(model, norm, name, step, checkpoint)
            cpu_model = new_model(name)
            cpu_model.load_state_dict(state)
            train = evaluate(cpu_model, manifest_path, scenes, norm, "train", output / f"train_step_{step}")
            dev = evaluate(cpu_model, manifest_path, scenes, norm, "dev", output / f"dev_step_{step}")
            milestones[str(step)] = {"checkpoint_sha256": sha(checkpoint), "train": train,
                                     "dev": dev, "elapsed_seconds": time.monotonic()-started}
            write_json(output / "progress.json", {"milestones": milestones, "curve": curve})
            print(json.dumps({"arm": arm, "model": name, "step": step,
                              "train_iou": train["pooled"]["iou"],
                              "dev_positive_iou": dev["positive"]["iou"],
                              "dev_negative_fpr": dev["negative"]["false_positive_area_rate"],
                              "seconds": round(time.monotonic()-started, 1)}), flush=True)
    report = {"model": name, "steps": ARMS[arm], "initial_checkpoint_sha256": sha(output / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "selection": choose_pareto(milestones),
              "elapsed_seconds": time.monotonic()-started}
    write_json(output / "receipt.json", report)
    return report


def benchmark(output: Path):
    device = configure_gpu()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = COMBINED_MANIFEST
    _, scenes, _, _ = v3.checked_scenes(manifest_path)
    norm = normalization(manifest_path, scenes)
    draws = draw_schedule("balanced_domains", manifest_path, scenes)
    by_id = {s["id"]: s for s in scenes}
    results = {}
    for name in ("stable_residual", "siamese"):
        for full in (False, True):
            torch.manual_seed(SEED)
            model = new_model(name).to(device)
            opt = v3.optimizer(model)
            cache = {}
            sample = make_batch(manifest_path, by_id, draws[:8], norm, cache, full_scene=full)
            sample = tuple(x.to(device) for x in sample)
            times = []
            for repeat in range(3):
                torch.cuda.synchronize(); begin = time.perf_counter_ns()
                for _ in range(100):
                    v3.fit_step(model, opt, sample, ordinary=False)
                torch.cuda.synchronize(); times.append((time.perf_counter_ns()-begin)/1e11)
            results[f"{name}_{'full' if full else 'crop'}"] = {"seconds_per_update_median_after_warmup":
                  float(np.median(times[1:])), "projected_4800_update_seconds": float(np.median(times[1:])*4800)}
    receipt = {"device": str(device), "device_name": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "deterministic": True, "tf32": False, "amp": False,
               "manifest_sha256": sha(manifest_path), "trainer_sha256": sha(HERE),
               "results": results}
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def run(arm: str, output: Path):
    if output.exists():
        raise FileExistsError(output)
    device = configure_gpu()
    set_branch("mars_repaired")
    manifest_path = COMBINED_MANIFEST
    _, scenes, counts, positives = v3.checked_scenes(manifest_path)
    if arm == "us_original_norm":
        _, old_scenes, _, _ = v3.checked_scenes(BASE_MANIFEST)
        norm = normalization(BASE_MANIFEST, old_scenes)
    else:
        norm = normalization(manifest_path, scenes)
    draws = draw_schedule(arm, manifest_path, scenes)
    dc = draw_counts(draws, scenes)
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    meta = {"schema": "GTM_Model7_repaired_domain_v4", "arm": arm,
            "manifest": str(manifest_path.resolve()), "trainer_path": str(HERE),
            "manifest_sha256": sha(manifest_path), "trainer_sha256": sha(HERE),
            "v3_source_sha256": sha(V3), "original_sha256": sha(ORIGINAL),
            "stable_source_sha256": sha(STABLE), "protocol_sha256": sha(PROTOCOL),
            "normalization": norm, "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
            "baseline_schedule_file_sha256": sha(BASE_DRAWS), "baseline_manifest_sha256": sha(BASE_MANIFEST),
            "split_counts": counts, "positive_counts": positives, "draw_domain_class_counts": dc,
            "config": {"seed": SEED, "branch": ACTIVE_BRANCH, "side": 100,
                       "crop": 100 if arm == "balanced_full_scene" else 48,
                       "batch": 8, "steps": ARMS[arm], "milestones": MILESTONES[arm],
                       "learning_rate": v3.LR, "weight_decay": v3.WEIGHT_DECAY,
                       "positive_weight": v3.POS_WEIGHT, "dice_weight": v3.DICE_WEIGHT,
                       "gradient_cap": v3.GRAD_CAP, "threshold": THRESHOLD,
                       "device": str(device), "dtype": "float32", "amp": False, "tf32": False,
                       "max_wall_seconds_per_model": MAX_WALL_SECONDS_PER_MODEL},
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "test_array_access": "none"}
    write_json(output / "metadata.json", meta)
    models = {}
    for name in ("stable_residual", "siamese"):
        models[name] = fit_arm_model(name, arm, manifest_path, scenes, norm, draws, output / name, device)
    receipt = {"schema": "GTM_Model7_repaired_domain_v4_receipt", "arm": arm,
               "metadata_sha256": sha(output / "metadata.json"), "models": models,
               "draw_domain_class_counts": dc, "test_array_access": "none"}
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"training": "complete", "arm": arm, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("benchmark"); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("train"); p.add_argument("--arm", choices=tuple(ARMS), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "benchmark":
        benchmark(args.output.resolve())
    else:
        run(args.arm, args.output.resolve())


if __name__ == "__main__":
    main()
