"""Frozen three-arm, FIT-only Model 7 validation experiment."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import GTM_Model7_train_repaired_broader_v7 as v7

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
PROTOCOL = ROOT / "docs/GTM_Model7_validation_v12_protocol_20261026.md"
ORIGINAL, STABLE, V3, V4, V5 = v7.ORIGINAL, v7.STABLE, v7.V3, v7.V4, v7.V5
ADAPTER = ROOT / "EarthRemoteSensingRapidResponse/mars_s2l_adapter.py"
DATA = ROOT / "outputs/GTM_Model7_validation_v12_20261026/data"
ARMS = ("control", "fraction", "ratio")
MILESTONES = (1200, 2400, 3600, 4800)
MAX_WALL_SECONDS_PER_MODEL = 1800
REPLAY_ATOL = 1e-6
ARM = None
THRESHOLD = .5
sha, write_json = v7.sha, v7.write_json
side, native_factor = v7.side, v7.native_factor

_spec = importlib.util.spec_from_file_location("mars_s2l_adapter_v12", ADAPTER)
_adapter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _adapter
_spec.loader.exec_module(_adapter)


def configure(meta: dict):
    global ARM
    arm = meta.get("config", {}).get("arm", meta.get("arm"))
    if arm not in ARMS:
        raise ValueError(f"Unknown validation arm: {arm}")
    if meta.get("schema") == "GTM_Model7_validation_v12":
        check_sources(meta)
    ARM = arm
    v7.set_branch("mars_repaired")


def set_branch(name: str):
    v7.set_branch(name)


def source_hashes():
    return {"trainer_sha256": sha(HERE), "protocol_sha256": sha(PROTOCOL),
            "v7_source_sha256": sha(v7.HERE), "v5_source_sha256": sha(V5),
            "v4_source_sha256": sha(V4), "v3_source_sha256": sha(V3),
            "original_sha256": sha(ORIGINAL), "stable_source_sha256": sha(STABLE),
            "adapter_sha256": sha(ADAPTER)}


def check_sources(meta: dict):
    if any(meta.get(k) != v for k, v in source_hashes().items()):
        raise ValueError("Frozen validation source changed")
    for key, pathkey in (("manifest_sha256", "manifest"),
                         ("split_sha256", "split_path"),
                         ("split_receipt_sha256", "split_receipt_path")):
        if sha(Path(meta[pathkey])) != meta[key]:
            raise ValueError(f"Frozen validation {key} changed")


def frozen_inputs(manifest_path: Path):
    receipt_path = DATA / "split_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split_path = Path(receipt["split"])
    if (sha(manifest_path) != receipt["manifest_sha256"] or
        sha(split_path) != receipt["split_sha256"] or
        receipt["source_sha256"] != "1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a"):
        raise ValueError("Split provenance changed")
    if sha(Path(json.loads(split_path.read_text(encoding="utf-8"))["source_manifest"])) != receipt["source_sha256"]:
        raise ValueError("Source manifest changed")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (document["frozen_split_sha256"] != receipt["split_sha256"] or
        document["source_manifest_sha256"] != receipt["source_sha256"]):
        raise ValueError("Manifest split provenance drift")
    rows = document["scenes"]
    expected = {"train": (670, 188, 482), "dev": (144, 39, 105),
                "assessment": (144, 41, 103), "regression": (198, 46, 152)}
    for role, (n, p, q) in expected.items():
        group = [s for s in rows if s["split"] == role]
        if (len(group), sum(s["scene_kind"] == "PLUME" for s in group),
            sum(s["scene_kind"] == "NO_PLUME" for s in group)) != (n, p, q):
            raise ValueError(f"Frozen {role} membership changed")
    if len({s["id"] for s in rows}) != len(rows):
        raise ValueError("Duplicate observation identity")
    for row in rows:
        if (not Path(row["array_file"]).is_absolute() or
            row["array_file"] != row["source_array_file"]):
            raise ValueError(f"Row provenance drift: {row['id']}")
    return rows, receipt, split_path, receipt_path


def load_scene(manifest_path: Path, scene: dict) -> dict:
    path = Path(scene["array_file"])
    if not path.is_absolute() or path != Path(scene["source_array_file"]) or sha(path) != scene["array_sha256"]:
        raise ValueError(f"Array provenance changed: {scene['id']}")
    with np.load(path, allow_pickle=False) as z:
        keys = ("target_image", "reference_image", "valid", "target", "label_native", "valid_native")
        if any(k not in z for k in keys):
            raise ValueError(f"Missing array key: {scene['id']}")
        data = {k: z[k] for k in keys}
    if (data["target_image"].shape != (100, 100, 6) or data["reference_image"].shape != (100, 100, 6)
        or data["valid"].shape != (100, 100) or data["target"].shape != (100, 100)
        or data["label_native"].shape != (200, 200) or data["valid_native"].shape != (200, 200)):
        raise ValueError(f"Malformed grids: {scene['id']}")
    valid = data["valid"].astype(bool)
    support = data["valid_native"].astype(bool)
    if (not valid.any() or not np.array_equal(support, np.repeat(np.repeat(valid, 2, 0), 2, 1))
        or not np.isin(data["target"], (0, 1)).all()
        or not np.isin(data["label_native"], (0, 1)).all()):
        raise ValueError(f"Invalid support or labels: {scene['id']}")
    native = data["label_native"].reshape(100, 2, 100, 2)
    if not np.array_equal(data["target"].astype(bool), native.any(axis=(1, 3))):
        raise ValueError(f"Native/coarse label mismatch: {scene['id']}")
    for k in ("target_image", "reference_image"):
        if not np.isfinite(data[k][valid]).all() or np.any(data[k][valid] < 0):
            raise ValueError(f"Invalid radiometry: {scene['id']}")
    if (scene["scene_kind"] == "PLUME") != bool(data["label_native"][support].any()):
        raise ValueError(f"Class flag mismatch: {scene['id']}")
    return data


def normalization(manifest_path: Path, rows: list[dict]):
    fit = [s for s in rows if s["split"] == "train"]
    if len(fit) != 670:
        raise ValueError("Normalizer requires all 670 FIT scenes")
    # v5's original loader only accepts relative paths; rebind it in this process
    # while preserving its exact normalization algorithm and immutable source.
    previous = v7.v5.load_scene
    try:
        v7.v5.load_scene = load_scene
        return v7.v5.normalization(manifest_path, fit)
    finally:
        v7.v5.load_scene = previous


def ratio_log(data: dict):
    valid = data["valid"].astype(bool)
    # Preserve the raw six bands. The adapter normalizes B12/B11 by scene
    # medians and accepts the native data scale directly.
    a = data["target_image"].transpose(2, 0, 1).astype(np.float32)
    b = data["reference_image"].transpose(2, 0, 1).astype(np.float32)
    ratio = _adapter.compute_mbmp(a, b, valid_mask=valid)
    out = np.zeros((100, 100), dtype=np.float32)
    out[valid] = np.log(np.maximum(ratio[valid], 1e-4))
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite MBMP feature")
    return out


def ratio_normalization(manifest_path: Path, rows: list[dict]):
    total = 0
    summation = squared = 0.
    for s in rows:
        if s["split"] != "train":
            continue
        data = load_scene(manifest_path, s)
        values = ratio_log(data)[data["valid"].astype(bool)].astype(np.float64)
        total += values.size
        summation += float(values.sum())
        squared += float(np.square(values).sum())
    mean = summation / total
    scale = max(float(np.sqrt(max(squared / total - mean * mean, 0.))), 1e-6)
    if not np.isfinite([mean, scale]).all():
        raise FloatingPointError("Nonfinite FIT MBMP normalization")
    return {"mean": mean, "scale": scale, "train_pixels": total,
            "method": "validity_aware_mbmp_log_clamp_1e-4_fit_standardize_clip8"}


def new_model(name: str):
    if name != "siamese" or ARM not in ARMS:
        raise ValueError("configure(metadata) required before constructing validation model")
    base = v7.FusionUNet(24).float()
    if ARM != "ratio":
        return base
    extended = v7.FusionUNet(25).float()
    old = base.state_dict()
    state = extended.state_dict()
    for k in state:
        if k == "enc1.layers.0.weight":
            state[k] = torch.zeros_like(state[k])
            state[k][:, :24] = old[k]
        else:
            state[k] = old[k].clone()
    extended.load_state_dict(state)
    return extended


def prepare(data: dict, norm: dict) -> np.ndarray:
    if ARM not in ARMS:
        raise ValueError("Arm not configured")
    dates = v7.v5.prepare(data, norm)
    a, b = dates[..., :6], dates[..., 6:]
    features = [a, b, a - b, np.abs(a - b)]
    valid = data["valid"].astype(bool)
    if ARM == "ratio":
        rn = norm["mbmp"]
        feature = np.zeros((100, 100), np.float32)
        feature[valid] = np.clip((ratio_log(data)[valid] - rn["mean"]) / rn["scale"], -8, 8)
        features.append(feature[..., None])
    x = np.concatenate(features, axis=-1).astype(np.float32)
    x[~valid] = 0
    if x.shape != (100, 100, 25 if ARM == "ratio" else 24) or not np.isfinite(x).all():
        raise ValueError("Malformed validation features")
    return x


def target20(data: dict):
    if ARM == "control":
        return data["target"].astype(np.float32)
    return data["label_native"].reshape(100, 2, 100, 2).mean(axis=(1, 3)).astype(np.float32)


def loss_parts(logits, target, valid, positive):
    if (logits.shape != target.shape or target.shape != valid.shape or
        positive.shape != (target.shape[0],)):
        raise ValueError("Loss grid/flag shape mismatch")
    if (positive.sum() != 4 or (positive == 0).sum() != 4 or
        not torch.equal(positive > .5, (target * valid).sum(dim=(1, 2, 3)) > 0)):
        raise ValueError("Balanced class flag mismatch")
    if (not all(torch.isfinite(x).all() for x in (logits, target, valid)) or
        not torch.all((target >= 0) & (target <= 1)) or valid.sum() <= 0):
        raise FloatingPointError("Invalid supported loss")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                             pos_weight=logits.new_tensor(2.))
    bce = (raw * valid).sum() / valid.sum()
    selected = positive[:, None, None, None] * valid
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target * selected).sum() + 1) / ((p * selected).sum() +
                                                      (target * selected).sum() + 1)
    loss = bce + .1 * dice
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite validation loss")
    return loss, bce, dice


def draw_schedule(rows: list[dict]):
    train = [s for s in rows if s["split"] == "train"]
    pools = {}
    for positive in (True, False):
        by_site = {}
        for s in train:
            if (s["scene_kind"] == "PLUME") == positive:
                by_site.setdefault(s["physical_location_id"], []).append(s["id"])
        pools[positive] = [sorted(by_site[k]) for k in sorted(by_site)]
    rng = random.Random(20261026)
    draws = []
    for _ in range(4800):
        for positive in (True, False):
            for _ in range(4):
                site = rng.choice(pools[positive])
                draws.append([rng.choice(site), rng.randrange(8), int(positive)])
    if len(draws) != 38400:
        raise AssertionError("Schedule length")
    return draws


def make_batch(manifest_path, by_id, draws, norm, cache):
    xs, ys, masks, flags = [], [], [], []
    for sid, transform, positive in draws:
        if sid not in by_id or by_id[sid]["split"] != "train":
            raise ValueError("Non-FIT draw")
        if sid not in cache:
            data = load_scene(manifest_path, by_id[sid])
            cache[sid] = (prepare(data, norm), target20(data), data["valid"].astype(bool))
        x, y, valid = cache[sid]
        d4 = v7.v5.v4.v3.old.d4
        xs.append(d4(x, transform))
        ys.append(d4(y, transform))
        masks.append(d4(valid, transform))
        flags.append(positive)
    sample = (v7.v5.v4.v3.tensor_chw(np.stack(xs)),
              torch.from_numpy(np.stack(ys)[:, None].copy()).float(),
              torch.from_numpy(np.stack(masks)[:, None].copy()).float(),
              torch.tensor(flags, dtype=torch.float32))
    if not torch.isfinite(sample[0]).all() or not torch.all(sample[2].sum(dim=(1, 2, 3)) > 0):
        raise ValueError("Malformed real batch")
    return sample


def fit_step(model, optimizer, sample):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, bce, dice = loss_parts(model(sample[0]), *sample[1:])
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return {"loss": float(loss.item()), "bce": float(bce.item()),
            "pooled_positive_dice_loss": float(dice.item()), "gradient_norm": float(gradient.item())}


def predict(model, x):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(v7.v5.v4.v3.tensor_chw(x[None])))[0, 0].cpu().numpy().astype(np.float32)
    if p.shape != (100, 100) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise FloatingPointError("Invalid prediction")
    return p


def confusion(prob, target, valid, cutoff=.5):
    hit, truth = prob[valid] >= cutoff, target[valid] == 1
    return {"tp": int((hit & truth).sum()), "fp": int((hit & ~truth).sum()),
            "fn": int((~hit & truth).sum()), "tn": int((~hit & ~truth).sum())}


def evaluate(model, manifest_path, scenes, norm, split, folder):
    folder.mkdir(parents=True, exist_ok=False)
    keys = ("tp", "fp", "fn", "tn")
    totals = {kind: {k: 0 for k in keys} for kind in
              ("pooled", "positive", "negative", "native_pooled", "native_positive", "native_negative")}
    records = {}
    for s in scenes:
        if s["split"] != split:
            continue
        data = load_scene(manifest_path, s)
        p = predict(model, prepare(data, norm))
        native = np.repeat(np.repeat(p, 2, 0), 2, 1)
        valid = data["valid"].astype(bool)
        nc = confusion(native, data["label_native"], data["valid_native"].astype(bool), THRESHOLD)
        c = confusion(p, data["target"], valid, THRESHOLD)
        positive = s["scene_kind"] == "PLUME"
        for k in keys:
            totals["pooled"][k] += c[k]
            totals["positive" if positive else "negative"][k] += c[k]
            totals["native_pooled"][k] += nc[k]
            totals["native_positive" if positive else "native_negative"][k] += nc[k]
        path = folder / f"{s['id']}.npz"
        np.savez_compressed(path, probability20=p, valid20=valid,
                            probability_native=native, valid_native=data["valid_native"].astype(bool))
        records[s["id"]] = {"positive": positive, "confusion": c, "native_confusion": nc,
                            "probability_sha256": sha(path), "activated": bool(c["tp"] + c["fp"])}
    return {**{kind: v7.v5.v4.v3.metrics(v) for kind, v in totals.items()},
            "scene_count": len(records), "positive_scenes": sum(r["positive"] for r in records.values()),
            "negative_scenes": sum(not r["positive"] for r in records.values()),
            "activated_scenes": sum(r["activated"] for r in records.values()), "scenes": records}


def train(arm: str, manifest: Path, preflight_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rows, receipt, split_path, receipt_path = frozen_inputs(manifest)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (not preflight.get("passed") or preflight.get("manifest_sha256") != sha(manifest)
        or preflight.get("split_sha256") != sha(split_path)
        or preflight.get("trainer_sha256") != sha(HERE)
        or preflight.get("protocol_sha256") != sha(PROTOCOL)
        or arm not in preflight.get("arms", {})):
        raise ValueError("Matching passed v12 E2E preflight required")
    configure({"arm": arm})
    norm = json.loads((preflight_path.parent / "normalization.json").read_text(encoding="utf-8"))
    draws = json.loads((preflight_path.parent / "draw_schedule.json").read_text(encoding="utf-8"))
    if sha(preflight_path.parent / "normalization.json") != preflight["normalization_sha256"] or sha(preflight_path.parent / "draw_schedule.json") != preflight["draw_schedule_sha256"]:
        raise ValueError("Preflight normalizer or schedule changed")
    device = v7.v5.v4.configure_gpu()
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    metadata = {"schema": "GTM_Model7_validation_v12", "arm": arm, "trainer_path": str(HERE),
                "manifest": str(manifest), "manifest_sha256": sha(manifest),
                "split_path": str(split_path), "split_sha256": sha(split_path),
                "split_receipt_path": str(receipt_path), "split_receipt_sha256": sha(receipt_path),
                **source_hashes(), "normalization": norm,
                "normalization_member_ids": preflight["normalization_member_ids"],
                "normalization_member_sha256": preflight["normalization_member_sha256"],
                "normalization_file_sha256": preflight["normalization_sha256"],
                "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
                "preflight_sha256": sha(preflight_path),
                "config": {"arm": arm, "seed": 20261026, "branch": "mars_repaired", "side": 100,
                           "batch": 8, "steps": 4800, "milestones": MILESTONES, "learning_rate": .0005,
                           "weight_decay": .0001, "positive_weight": 2., "dice_weight": .1,
                           "gradient_cap": 1., "threshold": .5, "device": str(device),
                           "dtype": "float32", "amp": False, "tf32": False,
                           "max_wall_seconds_per_model": MAX_WALL_SECONDS_PER_MODEL},
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "assessment_array_access": "none", "regression_array_access": "none"}
    write_json(output / "metadata.json", metadata)
    model_dir = output / "siamese"
    model_dir.mkdir()
    torch.manual_seed(20261026)
    torch.cuda.manual_seed_all(20261026)
    model = new_model("siamese").to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    v7.v5.v4.cpu_checkpoint(model, norm, "siamese", 0, model_dir / "initial.pt")
    by_id, cache, curve, milestones = {s["id"]: s for s in rows}, {}, [], {}
    started = time.monotonic()
    for step in range(1, 4801):
        if time.monotonic() - started > MAX_WALL_SECONDS_PER_MODEL:
            raise TimeoutError(f"{arm} exceeded 30-minute fit cap at update {step}")
        sample = make_batch(manifest, by_id, draws[(step-1)*8:step*8], norm, cache)
        values = fit_step(model, optim, tuple(t.to(device) for t in sample))
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, **values})
        if step % 200 == 0:
            print(json.dumps({"arm": arm, "step": step, "seconds": round(time.monotonic()-started, 1), "loss": values["loss"]}), flush=True)
        if step in MILESTONES:
            path = model_dir / f"step_{step}.pt"
            state = v7.v5.v4.cpu_checkpoint(model, norm, "siamese", step, path)
            cpu_model = new_model("siamese")
            cpu_model.load_state_dict(state)
            fit = evaluate(cpu_model, manifest, rows, norm, "train", model_dir / f"train_step_{step}")
            select = evaluate(cpu_model, manifest, rows, norm, "dev", model_dir / f"dev_step_{step}")
            milestones[str(step)] = {"checkpoint_sha256": sha(path), "train": fit, "dev": select,
                                     "elapsed_seconds": time.monotonic()-started}
            write_json(model_dir / "progress.json", {"milestones": milestones, "curve": curve})
            print(json.dumps({"arm": arm, "milestone": step, "select_native_f1": select["native_pooled"].get("f1")}), flush=True)
    report = {"model": "siamese", "steps": 4800,
              "initial_checkpoint_sha256": sha(model_dir / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "elapsed_seconds": time.monotonic()-started}
    write_json(model_dir / "receipt.json", report)
    write_json(output / "receipt.json", {"schema": "GTM_Model7_validation_v12_receipt", "arm": arm,
                                       "metadata_sha256": sha(output / "metadata.json"),
                                       "models": {"siamese": report},
                                       "assessment_array_access": "none", "regression_array_access": "none"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--manifest", type=Path, default=DATA / "combined_manifest.json")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.arm, args.manifest.resolve(), args.preflight.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
