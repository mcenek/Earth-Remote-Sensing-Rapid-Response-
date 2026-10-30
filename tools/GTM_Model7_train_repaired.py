"""Shared float32 trainer for a sealed repaired real temporal cohort.

The failure protocol was fixed first in docs/GTM_Model7_repaired_training_protocol.md.
Train and development arrays only; test array files are never opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL = ROOT / "tools/GTM_Model7_train_temporal.py"
STABLE = ROOT / "tools/GTM_Model7_stable_residual.py"
PROTOCOL = ROOT / "docs/GTM_Model7_repaired_training_protocol.md"
old_spec = importlib.util.spec_from_file_location("temporal_original", ORIGINAL)
old = importlib.util.module_from_spec(old_spec)
assert old_spec.loader is not None
old_spec.loader.exec_module(old)
stable_spec = importlib.util.spec_from_file_location("temporal_stable", STABLE)
stable = importlib.util.module_from_spec(stable_spec)
assert stable_spec.loader is not None
stable_spec.loader.exec_module(stable)

torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

SEED = 20261024
BRANCHES = {
    "mars_repaired": {"side": 100, "native_factor": 2, "crop": 48, "radiometry": "raw"},
    "methanes2cm": {"side": 32, "native_factor": 1, "crop": 32, "radiometry": "log1p_dn"},
}
ACTIVE_BRANCH = "mars_repaired"


def set_branch(name: str):
    global ACTIVE_BRANCH
    if name not in BRANCHES:
        raise ValueError(f"Unknown repaired branch: {name}")
    ACTIVE_BRANCH = name


def side() -> int:
    return BRANCHES[ACTIVE_BRANCH]["side"]


def crop_size() -> int:
    return BRANCHES[ACTIVE_BRANCH]["crop"]


def native_factor() -> int:
    return BRANCHES[ACTIVE_BRANCH]["native_factor"]


def radiometry(x: np.ndarray) -> np.ndarray:
    value = x.astype(np.float64)
    if ACTIVE_BRANCH == "methanes2cm":
        if (value < 0).any():
            raise ValueError("Negative supported DN under log1p")
        value = np.log1p(value)
    return value


BATCH = 8
DRAWS_PER_EPOCH = 48
STEPS = 2400
MILESTONES = (600, 1200, 1800, 2400)
PROBE_STEPS = 500
LR = 0.0005
WEIGHT_DECAY = 0.0001
POS_WEIGHT = 2.0
DICE_WEIGHT = 0.1
GRAD_CAP = 1.0
THRESHOLD = 0.5
REPLAY_ATOL = 1e-6
MAX_SECONDS_PER_MODEL = 900


def new_model(name: str):
    if name == "stable_residual":
        return stable.StableResidual().float()
    if name == "siamese":
        return old.SiameseUNet().float()
    raise ValueError(f"Unknown model: {name}")


def sha(path: Path) -> str:
    return old.sha(path)


def write_json(path: Path, value):
    old.write_json(path, value)


def load_scene(manifest_path: Path, scene: dict) -> dict:
    n, factor = side(), native_factor()
    with np.load(old.array_path(manifest_path, scene["array_file"]), allow_pickle=False) as z:
        required = ("target_image", "reference_image", "valid", "target", "label_native", "valid_native")
        if any(key not in z for key in required):
            raise ValueError(f"Missing array key: {scene['id']}")
        data = {key: z[key] for key in required}
    if (data["target_image"].shape != (n, n, 6) or
        data["reference_image"].shape != (n, n, 6) or
        data["valid"].shape != (n, n) or data["target"].shape != (n, n) or
        data["label_native"].shape != (n * factor, n * factor) or
        data["valid_native"].shape != (n * factor, n * factor)):
        raise ValueError(f"Malformed grids: {scene['id']}")
    valid = data["valid"].astype(bool)
    native_valid = data["valid_native"].astype(bool)
    if not valid.any() or not np.array_equal(native_valid,
                                             np.repeat(np.repeat(valid, factor, 0), factor, 1)):
        raise ValueError(f"Invalid native support: {scene['id']}")
    if not np.isin(data["target"], (0, 1)).all() or not np.isin(data["label_native"], (0, 1)).all():
        raise ValueError(f"Nonbinary target: {scene['id']}")
    projected = (data["label_native"].reshape(n, factor, n, factor) == 1).any(axis=(1, 3))
    if not np.array_equal(data["target"][valid].astype(bool), projected[valid]):
        raise ValueError(f"Native/coarse label mismatch: {scene['id']}")
    for key in ("target_image", "reference_image"):
        pixels = data[key][valid]
        if not np.isfinite(pixels).all() or (ACTIVE_BRANCH == "methanes2cm" and (pixels < 0).any()):
            raise ValueError(f"Invalid supported radiometry: {scene['id']}")
    return data


def normalization(manifest_path: Path, scenes: list[dict]) -> dict:
    count = 0
    sums = np.zeros(6, np.float64)
    squares = np.zeros(6, np.float64)
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = load_scene(manifest_path, scene)
        valid = data["valid"].astype(bool)
        for key in ("target_image", "reference_image"):
            pixels = radiometry(data[key][valid])
            count += len(pixels)
            sums += pixels.sum(axis=0)
            squares += np.square(pixels).sum(axis=0)
    if count == 0:
        raise ValueError("Empty train normalization support")
    center = sums / count
    scale = np.maximum(np.sqrt(np.maximum(squares / count - center * center, 0)), 1e-6)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite train normalizer")
    return {"center": center.tolist(), "scale": scale.tolist(), "train_date_pixels": count,
            "clip_standardized": 8.0, "invalid_fill": 0.0,
            "radiometry": BRANCHES[ACTIVE_BRANCH]["radiometry"]}


def prepare(data: dict, norm: dict) -> np.ndarray:
    valid = data["valid"].astype(bool)
    dates = []
    for key in ("target_image", "reference_image"):
        x = np.zeros((side(), side(), 6), np.float32)
        x[valid] = np.clip((radiometry(data[key][valid]) - norm["center"]) / norm["scale"],
                           -8, 8).astype(np.float32)
        dates.append(x)
    return np.concatenate(dates, axis=-1)


def checked_scenes(manifest_path: Path, *, require_dev_positive: bool = True):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_schemas = {"GTM_Model7_temporal_cohort_v1", "GTM_Model7_repaired_cohort_v1",
                       "GTM_Model7_methanes2cm_cohort_v1"}
    if manifest.get("schema") not in allowed_schemas:
        raise ValueError("Wrong repaired cohort schema")
    if ACTIVE_BRANCH == "methanes2cm":
        if (manifest.get("source_dataset") != "H1deaki/MethaneS2CM" or
                manifest.get("native_repeat_factor") != 1 or
                "L2A" not in manifest.get("source_product", "")):
            raise ValueError("MethaneS2CM branch/product/grid mismatch")
    elif manifest.get("source_dataset") == "H1deaki/MethaneS2CM" or manifest.get("native_repeat_factor", 2) != 2:
        raise ValueError("MARS branch/product/grid mismatch")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Missing scenes")
    ids = set()
    ownership = {}
    source_hashes = {}
    counts = defaultdict(int)
    positive_counts = defaultdict(int)
    for scene in scenes:
        sid, split = scene.get("id"), scene.get("split")
        if not isinstance(sid, str) or not sid or sid in ids or "/" in sid or "\\" in sid:
            raise ValueError(f"Bad/duplicate scene id: {sid}")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Bad split: {sid}")
        ids.add(sid)
        counts[split] += 1
        identity_keys = ["group_id", "physical_location_id"]
        if ACTIVE_BRANCH == "mars_repaired":
            identity_keys += ["target_scene_id", "reference_scene_id"]
        for key in identity_keys:
            value = scene.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing {key}: {sid}")
            identity = (("acquisition" if key in ("target_scene_id", "reference_scene_id") else key), value)
            if identity in ownership and ownership[identity] != split:
                raise ValueError(f"{key} crosses splits: {value}")
            ownership[identity] = split
        old.array_path(manifest_path, scene.get("array_file"))
        if not isinstance(scene.get("array_sha256"), str) or len(scene["array_sha256"]) != 64:
            raise ValueError(f"Missing array hash: {sid}")
        sources = scene.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Missing sources: {sid}")
        for source in sources:
            raw = source.get("path")
            digest = source.get("sha256")
            if not isinstance(raw, str) or not isinstance(digest, str):
                raise ValueError(f"Bad source metadata: {sid}")
            if ACTIVE_BRANCH == "mars_repaired":
                identity = ("source", raw)
                if identity in ownership and ownership[identity] != split:
                    raise ValueError(f"Source asset crosses splits: {raw}")
                ownership[identity] = split
            if split != "test":
                path = old.source_path(manifest_path, raw)
                if path not in source_hashes:
                    source_hashes[path] = sha(path)
                actual = source_hashes[path]
                if actual != digest:
                    raise ValueError(f"Source changed: {path}")
        if split == "test":
            # Test metadata checks end here. The test array is not opened or hashed.
            continue
        path = old.array_path(manifest_path, scene.get("array_file"))
        if sha(path) != scene.get("array_sha256"):
            raise ValueError(f"Array changed: {sid}")
        data = load_scene(manifest_path, scene)
        if np.any((data["target"] == 1) & data["valid"].astype(bool)):
            positive_counts[split] += 1
    if counts["train"] == 0 or counts["dev"] == 0 or counts["test"] == 0:
        raise ValueError("All three splits required")
    if positive_counts["train"] == 0 or (require_dev_positive and positive_counts["dev"] == 0):
        raise ValueError("Positive train and development scenes required")
    if counts["dev"] - positive_counts["dev"] == 0:
        raise ValueError("Negative development scenes required")
    return manifest, scenes, dict(counts), dict(positive_counts)


def tensor_chw(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.moveaxis(np.ascontiguousarray(x), -1, -3).copy()).float()


def train_catalog(manifest_path: Path, scenes: list[dict]):
    sites = {True: defaultdict(list), False: defaultdict(list)}
    labels = {}
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = load_scene(manifest_path, scene)
        valid = data["valid"].astype(bool)
        positives = np.argwhere((data["target"] == 1) & valid)
        kind = len(positives) > 0
        sites[kind][scene["physical_location_id"]].append(scene["id"])
        labels[scene["id"]] = positives if kind else np.argwhere(valid)
    if not sites[True] or not sites[False]:
        raise ValueError("Both train classes required")
    return sites, labels


def schedule(manifest_path: Path, scenes: list[dict], steps: int):
    if steps <= 0 or steps * BATCH % DRAWS_PER_EPOCH:
        raise ValueError("Steps must divide the fixed 48-draw schedule")
    sites, points = train_catalog(manifest_path, scenes)
    rng = np.random.default_rng(SEED)
    draws = []
    for i in range(steps * BATCH):
        kind = (i % 2 == 0)
        site_ids = sorted(sites[kind])
        site = site_ids[int(rng.integers(len(site_ids)))]
        scene_ids = sorted(sites[kind][site])
        sid = scene_ids[int(rng.integers(len(scene_ids)))]
        choices = points[sid]
        cy, cx = choices[int(rng.integers(len(choices)))]
        top = int(np.clip(cy - crop_size() // 2, 0, side() - crop_size()))
        left = int(np.clip(cx - crop_size() // 2, 0, side() - crop_size()))
        draws.append((sid, top, left, int(rng.integers(8)), kind))
    return draws


def prepared(manifest_path: Path, scene: dict, norm: dict, cache: dict):
    sid = scene["id"]
    if sid not in cache:
        data = load_scene(manifest_path, scene)
        cache[sid] = (prepare(data, norm), data["target"].astype(np.float32),
                      data["valid"].astype(bool))
    return cache[sid]


def batch(manifest_path: Path, by_id: dict, draws: list[tuple], norm: dict, cache: dict):
    xs, ys, masks, positives = [], [], [], []
    for sid, top, left, transform, positive in draws:
        x, y, valid = prepared(manifest_path, by_id[sid], norm, cache)
        sl = np.s_[top:top + crop_size(), left:left + crop_size()]
        xs.append(old.d4(x[sl], transform))
        ys.append(old.d4(y[sl], transform))
        masks.append(old.d4(valid[sl], transform))
        positives.append(positive)
    if not all(mask.any() for mask in masks):
        raise ValueError("Empty sampled support")
    x = tensor_chw(np.stack(xs))
    y = torch.from_numpy(np.stack(ys)[:, None].copy()).float()
    valid = torch.from_numpy(np.stack(masks)[:, None].copy()).float()
    flag = torch.tensor(positives, dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite batch")
    return x, y, valid, flag


def fit_step(model, optim, sample, *, ordinary: bool):
    x, y, valid, positive = sample
    model.train()
    optim.zero_grad(set_to_none=True)
    logits = model(x)
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                              pos_weight=torch.tensor(1.0 if ordinary else POS_WEIGHT))
    bce = (raw * valid).sum() / valid.sum()
    if ordinary:
        loss = bce
    else:
        on_positive = positive[:, None, None, None] * valid
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * y * on_positive).sum() + 1) / (
            (p * on_positive).sum() + (y * on_positive).sum() + 1)
        loss = bce + DICE_WEIGHT * dice
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CAP, error_if_nonfinite=True)
    if not torch.isfinite(grad):
        raise FloatingPointError("Nonfinite gradient")
    optim.step()
    return float(loss.item()), float(grad.item())


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def predict(model, x: np.ndarray):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(tensor_chw(x[None])))[0, 0].detach().cpu().numpy().astype(np.float32)
    if p.shape != (side(), side()) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise FloatingPointError("Invalid prediction")
    return p


def confusion(prob, target, valid):
    hit = prob[valid] >= THRESHOLD
    truth = target[valid] == 1
    return {"tp": int((hit & truth).sum()), "fp": int((hit & ~truth).sum()),
            "fn": int((~hit & truth).sum()), "tn": int((~hit & ~truth).sum())}


def metrics(c):
    return old.score(c)


def evaluate(model, manifest_path: Path, scenes: list[dict], norm: dict,
             split: str, folder: Path):
    folder.mkdir(parents=True)
    pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    positive = {key: 0 for key in pooled}
    negative = {key: 0 for key in pooled}
    native_pooled = {key: 0 for key in pooled}
    native_positive = {key: 0 for key in pooled}
    native_negative = {key: 0 for key in pooled}
    records = {}
    for scene in scenes:
        if scene["split"] != split:
            continue
        data = load_scene(manifest_path, scene)
        valid = data["valid"].astype(bool)
        p = predict(model, prepare(data, norm))
        c = confusion(p, data["target"], valid)
        factor = native_factor()
        native = np.repeat(np.repeat(p, factor, axis=0), factor, axis=1)
        native_c = confusion(native, data["label_native"], data["valid_native"].astype(bool))
        kind = bool(np.any((data["target"] == 1) & valid))
        for key in pooled:
            pooled[key] += c[key]
            (positive if kind else negative)[key] += c[key]
            native_pooled[key] += native_c[key]
            (native_positive if kind else native_negative)[key] += native_c[key]
        file = folder / f"{scene['id']}.npz"
        np.savez_compressed(file, probability20=p, valid20=valid,
                            probability_native=native, valid_native=data["valid_native"].astype(bool))
        records[scene["id"]] = {"positive": kind, "confusion": c,
                                "native_confusion": native_c,
                                "probability_sha256": sha(file),
                                "activated": bool(c["tp"] + c["fp"])}
    return {"pooled": metrics(pooled), "positive": metrics(positive),
            "negative": metrics(negative), "scene_count": len(records),
            "native_pooled": metrics(native_pooled), "native_positive": metrics(native_positive),
            "native_negative": metrics(native_negative),
            "positive_scenes": sum(v["positive"] for v in records.values()),
            "negative_scenes": sum(not v["positive"] for v in records.values()),
            "activated_scenes": sum(v["activated"] for v in records.values()),
            "scenes": records}


def probe_draws(manifest_path: Path, scenes: list[dict]):
    sites, points = train_catalog(manifest_path, scenes)
    rng = np.random.default_rng(SEED + 1)
    draws = []
    for index in range(4):
        kind = index % 2 == 0
        site_ids = sorted(sites[kind])
        site = site_ids[(index // 2) % len(site_ids)]
        sid = sorted(sites[kind][site])[0]
        coords = points[sid]
        cy, cx = coords[int(rng.integers(len(coords)))]
        draws.append((sid, int(np.clip(cy - crop_size() // 2, 0, side() - crop_size())),
                      int(np.clip(cx - crop_size() // 2, 0, side() - crop_size())), 0, kind))
    return draws


def run_probe(name: str, manifest_path: Path, scenes: list[dict], norm: dict,
              folder: Path):
    folder.mkdir()
    draws = probe_draws(manifest_path, scenes)
    sample = batch(manifest_path, {s["id"]: s for s in scenes}, draws, norm, {})
    x, y, valid, _ = sample
    torch.manual_seed(SEED)
    model = new_model(name)
    start = {k: v.clone() for k, v in model.state_dict().items()}
    optim = optimizer(model)
    model.eval()
    with torch.no_grad():
        initial_p = torch.sigmoid(model(x))[:, 0].detach().cpu().numpy().astype(np.float32)
        initial_loss = float(((F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * valid).sum() / valid.sum()).item())
    torch.save({"model": model.state_dict(), "normalization": norm, "model_name": name,
                "steps": 0}, folder / "initial.pt")
    curve = []
    started = time.monotonic()
    cpu_started = time.process_time()
    for step in range(1, PROBE_STEPS + 1):
        if time.process_time() - cpu_started > MAX_SECONDS_PER_MODEL:
            raise TimeoutError("Probe CPU-time cap exceeded")
        loss, grad = fit_step(model, optim, sample, ordinary=True)
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, "loss": loss, "gradient_norm": grad})
    torch.save({"model": model.state_dict(), "normalization": norm, "model_name": name,
                "steps": PROBE_STEPS}, folder / "final.pt")
    model.eval()
    with torch.no_grad():
        final_p = torch.sigmoid(model(x))[:, 0].detach().cpu().numpy().astype(np.float32)
    target = y[:, 0].numpy()
    support = valid[:, 0].numpy().astype(bool)
    positive_score = metrics(confusion(final_p[[0, 2]], target[[0, 2]], support[[0, 2]]))
    negative_score = metrics(confusion(final_p[[1, 3]], target[[1, 3]], support[[1, 3]]))
    movement = sum(float((v - start[k]).abs().sum().item()) for k, v in model.state_dict().items())
    qualified = (movement > 0 and loss <= initial_loss * 0.7 and
                 (positive_score["recall"] or 0) >= 0.8 and
                 (negative_score["false_positive_area_rate"] or 0) <= 0.05)
    file = folder / "probe_arrays.npz"
    np.savez_compressed(file, image12=x.numpy(), target=target, valid=support,
                        probability_initial=initial_p, probability_final=final_p)
    report = {"model": name, "draws": draws, "initial_loss": initial_loss,
              "final_loss": loss, "positive": positive_score, "negative": negative_score,
              "parameter_l1_change": movement, "qualified": qualified, "curve": curve,
              "checkpoint_sha256": {p.stem: sha(p) for p in folder.glob("*.pt")},
              "arrays_sha256": sha(file), "elapsed_seconds": time.monotonic() - started,
              "cpu_seconds": time.process_time() - cpu_started}
    write_json(folder / "receipt.json", report)
    return report


def choose_pareto(rows: dict):
    frontier = []
    for step, row in rows.items():
        iou = row["dev"]["positive"]["iou"] or 0.0
        fpr = row["dev"]["negative"]["false_positive_area_rate"] or 0.0
        dominated = any((other["dev"]["positive"]["iou"] or 0.0) >= iou and
                        (other["dev"]["negative"]["false_positive_area_rate"] or 0.0) <= fpr and
                        ((other["dev"]["positive"]["iou"] or 0.0) > iou or
                         (other["dev"]["negative"]["false_positive_area_rate"] or 0.0) < fpr)
                        for other in rows.values())
        if not dominated:
            frontier.append(int(step))
    eligible = [step for step in frontier if
                (rows[str(step)]["dev"]["positive"]["recall"] or 0) >= 0.3 and
                (rows[str(step)]["dev"]["negative"]["false_positive_area_rate"] or 0) <= 0.05]
    selected = max(eligible, key=lambda step: (
        rows[str(step)]["dev"]["positive"]["iou"] or 0,
        -(rows[str(step)]["dev"]["negative"]["false_positive_area_rate"] or 0), -step)) if eligible else None
    return {"pareto_steps": sorted(frontier), "qualified_steps": sorted(eligible),
            "selected_step": selected}


def train_model(name: str, manifest_path: Path, scenes: list[dict], norm: dict,
                draws: list[tuple], folder: Path):
    folder.mkdir()
    torch.manual_seed(SEED)
    model = new_model(name)
    optim = optimizer(model)
    torch.save({"model": model.state_dict(), "normalization": norm, "model_name": name,
                "steps": 0}, folder / "initial.pt")
    by_id = {s["id"]: s for s in scenes}
    cache = {}
    curve = []
    milestones = {}
    started = time.monotonic()
    cpu_started = time.process_time()
    for step in range(1, STEPS + 1):
        if time.process_time() - cpu_started > MAX_SECONDS_PER_MODEL:
            raise TimeoutError(f"{name} CPU-time cap exceeded before fixed schedule")
        sample = batch(manifest_path, by_id, draws[(step - 1) * BATCH:step * BATCH], norm, cache)
        loss, grad = fit_step(model, optim, sample, ordinary=False)
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, "loss": loss, "gradient_norm": grad})
        if step in MILESTONES:
            path = folder / f"step_{step}.pt"
            torch.save({"model": model.state_dict(), "normalization": norm,
                        "model_name": name, "steps": step}, path)
            train = evaluate(model, manifest_path, scenes, norm, "train", folder / f"train_step_{step}")
            dev = evaluate(model, manifest_path, scenes, norm, "dev", folder / f"dev_step_{step}")
            milestones[str(step)] = {"checkpoint_sha256": sha(path), "train": train,
                                     "dev": dev, "elapsed_seconds": time.monotonic() - started,
                                     "cpu_seconds": time.process_time() - cpu_started}
            write_json(folder / "progress.json", {"milestones": milestones, "curve": curve})
            print(json.dumps({"model": name, "step": step,
                              "dev_positive_iou": dev["positive"]["iou"],
                              "dev_positive_recall": dev["positive"]["recall"],
                              "dev_negative_fpr": dev["negative"]["false_positive_area_rate"]}), flush=True)
    selection = choose_pareto(milestones)
    report = {"model": name, "steps": STEPS, "initial_checkpoint_sha256": sha(folder / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "selection": selection,
              "elapsed_seconds": time.monotonic() - started,
              "cpu_seconds": time.process_time() - cpu_started}
    write_json(folder / "receipt.json", report)
    return report


def run(manifest_path: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing populated output: {output}")
    _, scenes, counts, positives = checked_scenes(manifest_path)
    norm = normalization(manifest_path, scenes)
    draws = schedule(manifest_path, scenes, STEPS)
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    config = {"seed": SEED, "branch": ACTIVE_BRANCH,
              "side": side(), "native_factor": native_factor(), "crop": crop_size(), "batch": BATCH,
              "draws_per_epoch": DRAWS_PER_EPOCH, "steps": STEPS,
              "milestones": MILESTONES, "probe_steps": PROBE_STEPS,
              "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
              "positive_weight": POS_WEIGHT, "dice_weight": DICE_WEIGHT,
              "gradient_cap": GRAD_CAP, "threshold": THRESHOLD,
              "dtype": "float32", "device": "cpu", "max_cpu_seconds_per_model": MAX_SECONDS_PER_MODEL}
    metadata = {"schema": "GTM_Model7_repaired_training_v1", "manifest": str(manifest_path),
                "manifest_sha256": sha(manifest_path), "trainer_sha256": sha(HERE),
                "original_sha256": sha(ORIGINAL), "stable_source_sha256": sha(STABLE),
                "protocol_sha256": sha(PROTOCOL), "normalization": norm,
                "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
                "split_counts": counts, "positive_counts": positives, "config": config,
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "test_array_access": "none"}
    write_json(output / "metadata.json", metadata)
    probes = {}
    for name in ("stable_residual", "siamese"):
        probes[name] = run_probe(name, manifest_path, scenes, norm, output / f"probe_{name}")
        print(json.dumps({"model": name, "probe_qualified": probes[name]["qualified"],
                          "probe_positive_recall": probes[name]["positive"]["recall"],
                          "probe_negative_fpr": probes[name]["negative"]["false_positive_area_rate"]}), flush=True)
    write_json(output / "probe_results.json", probes)
    if not all(p["qualified"] for p in probes.values()):
        raise RuntimeError("A real-crop control failed; long fit prohibited")
    models = {}
    for name in ("stable_residual", "siamese"):
        models[name] = train_model(name, manifest_path, scenes, norm, draws, output / name)
    receipt = {"schema": "GTM_Model7_repaired_training_receipt_v1",
               "metadata_sha256": sha(output / "metadata.json"), "probes": probes,
               "models": models, "test_array_access": "none",
               "checkpoint_selection": "DEV Pareto rule in fixed protocol"}
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"training": "complete", "receipt": str(output / "receipt.json")}), flush=True)


def benchmark(manifest_path: Path):
    _, scenes, counts, positives = checked_scenes(manifest_path, require_dev_positive=False)
    norm = normalization(manifest_path, scenes)
    draws = schedule(manifest_path, scenes, 6)
    by_id = {s["id"]: s for s in scenes}
    sample = batch(manifest_path, by_id, draws[:BATCH], norm, {})
    numbers = {}
    for name in ("stable_residual", "siamese"):
        torch.manual_seed(SEED)
        model = new_model(name)
        optim = optimizer(model)
        times = []
        for _ in range(4):
            started = time.monotonic()
            fit_step(model, optim, sample, ordinary=False)
            times.append(time.monotonic() - started)
        numbers[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                         "median_seconds_per_update_after_warmup": float(np.median(times[1:])),
                         "projected_2400_update_seconds_excluding_data_and_evaluation":
                         float(np.median(times[1:]) * STEPS)}
    return {"manifest_sha256": sha(manifest_path), "trainer_sha256": sha(HERE),
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__},
            "split_counts": counts, "positive_counts": positives,
            "models": numbers, "test_array_access": "none"}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "benchmark", "train"):
        entry = sub.add_parser(name)
        entry.add_argument("--branch", choices=tuple(BRANCHES), required=True)
        entry.add_argument("--manifest", type=Path, required=True)
        if name == "train":
            entry.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_branch(args.branch)
    if args.command == "validate":
        _, _, counts, positives = checked_scenes(args.manifest)
        print(json.dumps({"valid": True, "splits": counts, "positive_scenes": positives}))
    elif args.command == "benchmark":
        print(json.dumps(benchmark(args.manifest)))
    else:
        run(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
