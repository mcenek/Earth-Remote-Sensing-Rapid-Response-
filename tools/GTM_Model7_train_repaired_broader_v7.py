"""Two bounded broader-DEV U-Net fits under the frozen 2026-10-24 protocol.

Only TRAIN arrays enter fitting. TEST is metadata-only until a later freeze.
Run the real-data E2E preflight before invoking ``train``.
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
from torch import nn
from torch.nn import functional as F

import GTM_Model7_train_repaired_centered_v5 as v5

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL, STABLE, V3, V4, V5 = v5.ORIGINAL, v5.STABLE, v5.V3, v5.V4, v5.HERE
PROTOCOL = ROOT / "docs/GTM_Model7_broader_dev_model_protocol_20261024.md"
BASELINE = ROOT / "outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered"
ARMS = ("early_fusion", "spectral_fusion")
MILESTONES = (1200, 2400, 3600, 4800)
MAX_WALL_SECONDS_PER_MODEL = 1800
REPLAY_ATOL = v5.REPLAY_ATOL
ACTIVE_BRANCH = "mars_repaired"
THRESHOLD = .5
ARM: str | None = None

sha, write_json, load_scene = v5.sha, v5.write_json, v5.load_scene
side, native_factor = v5.side, v5.native_factor


def configure(meta: dict):
    global ARM
    arm = meta.get("config", {}).get("arm", meta.get("arm"))
    if arm not in ARMS:
        raise ValueError(f"Unknown broader v7 arm: {arm}")
    ARM = arm


def set_branch(name: str):
    global ACTIVE_BRANCH
    if name != "mars_repaired":
        raise ValueError("Broader v7 supports repaired MARS only")
    v5.set_branch(name)
    ACTIVE_BRANCH = name


class ConvBlock(nn.Module):
    def __init__(self, input_width: int, width: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class FusionUNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.enc1 = ConvBlock(channels, 16)
        self.enc2 = ConvBlock(16, 32)
        self.bridge = ConvBlock(32, 64)
        self.dec2 = ConvBlock(96, 32)
        self.dec1 = ConvBlock(48, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.bridge(F.avg_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), dim=1))
        e = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a), dim=1))
        return self.head(e)


class RetainedResidual(v5.v4.v3.stable.StableResidual):
    """Keep v5 weights and its original twelve-channel input under v7 prep."""

    def forward(self, x):
        return super().forward(x[:, :12])


def new_model(name: str):
    if name == "stable_residual":
        return RetainedResidual().float()
    if name != "siamese" or ARM not in ARMS:
        raise ValueError("configure(metadata) must set a v7 arm before model construction")
    return FusionUNet(24 if ARM == "early_fusion" else 26).float()


def prepare(data: dict, norm: dict) -> np.ndarray:
    if ARM not in ARMS:
        raise ValueError("Arm not configured")
    dates = v5.prepare(data, norm)
    current, reference = dates[..., :6], dates[..., 6:]
    channels = [current, reference, current - reference, np.abs(current - reference)]
    if ARM == "spectral_fusion":
        valid = data["valid"].astype(bool)
        scale = np.asarray(norm["scale"], dtype=np.float64)
        for second in (4, 3):  # B12-B11, B12-B08 in frozen band order.
            contrast = np.zeros(valid.shape, dtype=np.float32)
            # Recover centered-log units from the frozen TRAIN-standardized
            # channels, retaining v5's clipping and scene median convention.
            raw = ((current[valid, 5] - reference[valid, 5]) * scale[5] -
                   (current[valid, second] - reference[valid, second]) * scale[second])
            contrast[valid] = (raw / np.hypot(scale[5], scale[second])).astype(np.float32)
            channels.append(contrast[..., None])
    result = np.concatenate(channels, axis=-1).astype(np.float32)
    result[~data["valid"].astype(bool)] = 0
    if result.shape != (100, 100, 24 if ARM == "early_fusion" else 26) or not np.isfinite(result).all():
        raise ValueError("Malformed fusion features")
    return result


def predict(model, x: np.ndarray):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(v5.v4.v3.tensor_chw(x[None])))[0, 0].cpu().numpy().astype(np.float32)
    if p.shape != (100, 100) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise FloatingPointError("Invalid v7 prediction")
    return p


def confusion(prob, target, valid):
    hit, truth = prob[valid] >= THRESHOLD, target[valid] == 1
    return {"tp": int((hit & truth).sum()), "fp": int((hit & ~truth).sum()),
            "fn": int((~hit & truth).sum()), "tn": int((~hit & ~truth).sum())}


def evaluate(model, manifest_path, scenes, norm, split, folder):
    folder.mkdir(parents=True)
    keys = ("tp", "fp", "fn", "tn")
    totals = {part: {key: 0 for key in keys} for part in
              ("pooled", "positive", "negative", "native_pooled", "native_positive", "native_negative")}
    records = {}
    for scene in scenes:
        if scene["split"] != split:
            continue
        data = load_scene(manifest_path, scene)
        valid = data["valid"].astype(bool)
        p = predict(model, prepare(data, norm))
        native = np.repeat(np.repeat(p, native_factor(), axis=0), native_factor(), axis=1)
        c = confusion(p, data["target"], valid)
        nc = confusion(native, data["label_native"], data["valid_native"].astype(bool))
        positive = bool((data["target"][valid] == 1).any())
        for key in keys:
            totals["pooled"][key] += c[key]
            totals["positive" if positive else "negative"][key] += c[key]
            totals["native_pooled"][key] += nc[key]
            totals["native_positive" if positive else "native_negative"][key] += nc[key]
        file = folder / f"{scene['id']}.npz"
        np.savez_compressed(file, probability20=p, valid20=valid,
                            probability_native=native, valid_native=data["valid_native"].astype(bool))
        records[scene["id"]] = {"positive": positive, "confusion": c, "native_confusion": nc,
                                "probability_sha256": sha(file), "activated": bool(c["tp"] + c["fp"])}
    return {**{key: v5.v4.v3.metrics(value) for key, value in totals.items()},
            "scene_count": len(records), "positive_scenes": sum(r["positive"] for r in records.values()),
            "negative_scenes": sum(not r["positive"] for r in records.values()),
            "activated_scenes": sum(r["activated"] for r in records.values()), "scenes": records}


def loss_parts(logits, target, valid, positive):
    if logits.shape != target.shape or valid.shape != target.shape or positive.shape != (target.shape[0],):
        raise ValueError("Loss grid/flag shape mismatch")
    if not all(torch.isfinite(v).all() for v in (logits, target, valid)) or valid.sum() <= 0:
        raise FloatingPointError("Nonfinite or empty loss support")
    if positive.sum() != 4 or (positive == 0).sum() != 4:
        raise ValueError("Fixed four/four batch changed")
    if not torch.equal(positive > .5, (target * valid).sum(dim=(1, 2, 3)) > 0):
        raise ValueError("Draw class and actual scene label differ")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=logits.new_tensor(2.0))
    bce = (raw * valid).sum() / valid.sum()
    selected = positive[:, None, None, None] * valid
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target * selected).sum() + 1) / ((p * selected).sum() + (target * selected).sum() + 1)
    loss = bce + .1 * dice
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    return loss, bce, dice


def make_batch(manifest_path, by_id, draws, norm, cache):
    xs, ys, masks, flags = [], [], [], []
    for sid, _, _, transform, positive in draws:
        if sid not in cache:
            data = load_scene(manifest_path, by_id[sid])
            cache[sid] = (prepare(data, norm), data["target"].astype(np.float32), data["valid"].astype(bool))
        x, y, valid = cache[sid]
        xs.append(v5.v4.v3.old.d4(x, transform))
        ys.append(v5.v4.v3.old.d4(y, transform))
        masks.append(v5.v4.v3.old.d4(valid, transform))
        flags.append(positive)
    sample = (v5.v4.v3.tensor_chw(np.stack(xs)),
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
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return {"loss": float(loss.item()), "bce": float(bce.item()),
            "pooled_positive_dice_loss": float(dice.item()), "gradient_norm": float(gradient.item())}


def frozen_inputs(manifest_path: Path, reservation_path: Path):
    base = json.loads((BASELINE / "metadata.json").read_text(encoding="utf-8"))
    base_manifest = Path(base["manifest"])
    if sha(base_manifest) != base["manifest_sha256"] or sha(BASELINE / "draw_schedule.json") != base["draw_schedule_file_sha256"]:
        raise ValueError("Original v5 manifest or schedule drift")
    if base["normalization"]["radiometry"] != "log1p_dn_date_band_median_centered":
        raise ValueError("Unexpected v5 normalizer")
    if not reservation_path.is_file():
        raise ValueError("Frozen TEST reservation metadata required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["scenes"]
    original = json.loads(base_manifest.read_text(encoding="utf-8"))["scenes"]
    old_train = {s["id"]: s for s in original if s["split"] == "train"}
    new_train = {s["id"]: s for s in rows if s["split"] == "train"}
    if len(old_train) != 958 or old_train.keys() != new_train.keys():
        raise ValueError("TRAIN observation identity changed")
    for sid, old in old_train.items():
        new = new_train[sid]
        for key in ("array_sha256", "group_id", "physical_location_id", "target_scene_id", "reference_scene_id"):
            if old[key] != new[key]:
                raise ValueError(f"TRAIN identity drift: {sid}/{key}")
    dev = [s for s in rows if s["split"] == "dev"]
    if not dev or not any(s["scene_kind"] == "PLUME" for s in dev) or not any(s["scene_kind"] == "NO_PLUME" for s in dev):
        raise ValueError("Broader DEV must contain both classes")
    if any(s["split"] not in ("train", "dev") for s in rows):
        raise ValueError("Training manifest must contain only TRAIN and DEV arrays")
    train_sites = {s["physical_location_id"] for s in new_train.values()}
    train_groups = {s["group_id"] for s in new_train.values()}
    train_acquisitions = {s[key] for s in new_train.values() for key in ("target_scene_id", "reference_scene_id")}
    train_spectral_hashes = {a["sha256"] for s in new_train.values() for a in s["sources"]
                             if a["role"] in ("target_image", "reference_image")}
    if any(s["physical_location_id"] in train_sites or s["group_id"] in train_groups or
           s["target_scene_id"] in train_acquisitions or s["reference_scene_id"] in train_acquisitions for s in dev):
        raise ValueError("TRAIN/DEV source or site overlap")
    dev_spectral_hashes = {a["sha256"] for s in dev for a in s["sources"]
                           if a["role"] in ("target_image", "reference_image")}
    if train_spectral_hashes & dev_spectral_hashes:
        raise ValueError("TRAIN/DEV spectral content overlap")
    for scene in rows:
        if sha(v5.v4.v3.old.array_path(manifest_path, scene["array_file"])) != scene["array_sha256"]:
            raise ValueError(f"Array drift: {scene['id']}")
    draws = json.loads((BASELINE / "draw_schedule.json").read_text(encoding="utf-8"))
    if len(draws) != 38400 or any(sid not in new_train for sid, *_ in draws):
        raise ValueError("Original draw schedule incompatible")
    return base, rows, draws


def train(arm: str, manifest_path: Path, reservation_path: Path, output: Path, preflight_path: Path):
    if output.exists():
        raise FileExistsError(output)
    configure({"arm": arm})
    set_branch("mars_repaired")
    base, scenes, draws = frozen_inputs(manifest_path, reservation_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (not preflight.get("passed") or preflight.get("trainer_sha256") != sha(HERE) or
        preflight.get("manifest_sha256") != sha(manifest_path) or
        preflight.get("reservation_sha256") != sha(reservation_path) or
        arm not in preflight.get("arms", {})):
        raise ValueError("Matching passed real-data E2E preflight required")
    device = v5.v4.configure_gpu()
    output.mkdir(parents=True)
    write_json(output / "draw_schedule.json", draws)
    dependencies = {"trainer_sha256": sha(HERE), "v5_source_sha256": sha(V5),
                    "v4_source_sha256": sha(V4), "v3_source_sha256": sha(V3),
                    "original_sha256": sha(ORIGINAL), "stable_source_sha256": sha(STABLE),
                    "protocol_sha256": sha(PROTOCOL)}
    meta = {"schema": "GTM_Model7_broader_v7", "arm": arm, "trainer_path": str(HERE),
            "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
            "reservation_path": str(reservation_path), "reservation_sha256": sha(reservation_path),
            **dependencies, "normalization": base["normalization"],
            "baseline_metadata_sha256": sha(BASELINE / "metadata.json"),
            "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
            "preflight_sha256": sha(preflight_path),
            "split_counts": {kind: sum(s["split"] == kind for s in scenes) for kind in ("train", "dev")},
            "config": {"arm": arm, "seed": 20261024, "branch": "mars_repaired", "side": 100,
                       "crop": 100, "batch": 8, "steps": 4800, "milestones": MILESTONES,
                       "learning_rate": .0005, "weight_decay": .0001, "positive_weight": 2.,
                       "dice_weight": .1, "gradient_cap": 1., "threshold": .5,
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
    v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", 0, model_dir / "initial.pt")
    by_id, cache, curve, milestones = {s["id"]: s for s in scenes}, {}, [], {}
    started = time.monotonic()
    for step in range(1, 4801):
        if time.monotonic() - started > MAX_WALL_SECONDS_PER_MODEL:
            raise TimeoutError(f"{arm} exceeded 30-minute fit cap at update {step}")
        sample = make_batch(manifest_path, by_id, draws[(step - 1) * 8:step * 8], base["normalization"], cache)
        values = fit_step(model, optimizer, tuple(value.to(device) for value in sample))
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, **values})
        if step % 200 == 0:
            print(json.dumps({"arm": arm, "step": step, "seconds": round(time.monotonic() - started, 1), "loss": values["loss"]}), flush=True)
        if step in MILESTONES:
            checkpoint = model_dir / f"step_{step}.pt"
            state = v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", step, checkpoint)
            cpu_model = new_model("siamese")
            cpu_model.load_state_dict(state)
            train_score = evaluate(cpu_model, manifest_path, scenes, base["normalization"], "train", model_dir / f"train_step_{step}")
            dev_score = evaluate(cpu_model, manifest_path, scenes, base["normalization"], "dev", model_dir / f"dev_step_{step}")
            milestones[str(step)] = {"checkpoint_sha256": sha(checkpoint), "train": train_score,
                                     "dev": dev_score, "elapsed_seconds": time.monotonic() - started}
            write_json(model_dir / "progress.json", {"milestones": milestones, "curve": curve})
            print(json.dumps({"arm": arm, "milestone": step, "dev_native_iou": dev_score["native_pooled"]["iou"]}), flush=True)
    report = {"model": "siamese", "steps": 4800, "initial_checkpoint_sha256": sha(model_dir / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "elapsed_seconds": time.monotonic() - started}
    write_json(model_dir / "receipt.json", report)
    write_json(output / "receipt.json", {"schema": "GTM_Model7_broader_v7_receipt", "arm": arm,
                                       "metadata_sha256": sha(output / "metadata.json"),
                                       "models": {"siamese": report}, "test_array_access": "none"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.arm, args.manifest.resolve(), args.reservation.resolve(), args.output.resolve(), args.preflight.resolve())


if __name__ == "__main__":
    main()
