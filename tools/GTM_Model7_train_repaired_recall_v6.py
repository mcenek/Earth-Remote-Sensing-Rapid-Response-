"""Two bounded scene-recall U-Net fits, with the v5 residual retained as control.

Failure cases and selection rules were frozen first in
docs/GTM_Model7_unet_recall_protocol_20261024.md.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import GTM_Model7_train_repaired_centered_v5 as v5

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL = v5.ORIGINAL
STABLE = v5.STABLE
V3 = v5.V3
V4 = v5.V4
V5 = v5.HERE
PROTOCOL = ROOT / "docs/GTM_Model7_unet_recall_protocol_20261024.md"
BASELINE = ROOT / "outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered"
BASELINE_FREEZE = ROOT / "outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration/freeze.json"
ARMS = ("scene_dice", "delta_scene_dice")
MILESTONES = (1200, 2400, 3600, 4800)
MAX_WALL_SECONDS_PER_MODEL = 1200
THRESHOLD = 0.5
REPLAY_ATOL = v5.REPLAY_ATOL
ACTIVE_BRANCH = v5.ACTIVE_BRANCH
ARM = None

sha = v5.sha
write_json = v5.write_json
load_scene = v5.load_scene
prepare = v5.prepare
predict = v5.predict
native_factor = v5.native_factor
side = v5.side


def confusion(prob, target, valid):
    v5.THRESHOLD = THRESHOLD
    return v5.confusion(prob, target, valid)


def configure(meta: dict):
    """Set the frozen arm before new_model is called by inference."""
    global ARM
    value = meta.get("config", {}).get("arm", meta.get("arm"))
    if value not in ARMS:
        raise ValueError(f"Unknown v6 arm: {value}")
    ARM = value


def set_branch(name: str):
    global ACTIVE_BRANCH
    v5.set_branch(name)
    ACTIVE_BRANCH = v5.ACTIVE_BRANCH


class DeltaSiameseUNet(v5.v4.v3.old.SiameseUNet):
    def __init__(self):
        super().__init__()
        self.delta_skip = nn.Conv2d(6, 1, 1, bias=False)
        nn.init.zeros_(self.delta_skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.delta_skip(x[:, :6] - x[:, 6:])


def new_model(name: str):
    if name == "stable_residual":
        return v5.new_model(name)
    if name != "siamese":
        raise ValueError(f"Unknown model: {name}")
    if ARM == "scene_dice":
        return v5.new_model(name)
    if ARM == "delta_scene_dice":
        return DeltaSiameseUNet().float()
    raise ValueError("configure(metadata) must set the v6 arm before model construction")


def evaluate(model, manifest_path, scenes, norm, split, folder):
    v5.THRESHOLD = THRESHOLD
    return v5.evaluate(model, manifest_path, scenes, norm, split, folder)


def loss_parts(logits, target, valid, positive):
    if logits.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Loss grid shape mismatch")
    if positive.shape != (target.shape[0],):
        raise ValueError("Loss scene flag shape mismatch")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite loss input")
    if not torch.all(valid.sum(dim=(1, 2, 3)) > 0):
        raise ValueError("Empty valid scene support")
    if not torch.all((positive > 0.5) == ((target * valid).sum(dim=(1, 2, 3)) > 0)):
        raise ValueError("Positive scene flag disagrees with target")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                              pos_weight=logits.new_tensor(2.0))
    bce = (raw * valid).sum() / valid.sum()
    p = torch.sigmoid(logits)
    intersection = (p * target * valid).sum(dim=(1, 2, 3))
    support = (p * valid).sum(dim=(1, 2, 3)) + (target * valid).sum(dim=(1, 2, 3))
    per_scene_dice_loss = 1 - (2 * intersection + 1) / (support + 1)
    selected = positive > 0.5
    if not selected.any():
        raise ValueError("No positive scenes in fixed balanced batch")
    dice = per_scene_dice_loss[selected].mean()
    loss = bce + 0.5 * dice
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    return loss, bce, dice


def fit_step(model, optimizer, sample):
    x, y, valid, positive = sample
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, bce, dice = loss_parts(model(x), y, valid, positive)
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), v5.v4.v3.GRAD_CAP,
                                              error_if_nonfinite=True)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return {"loss": float(loss.item()), "bce": float(bce.item()),
            "mean_positive_scene_dice_loss": float(dice.item()),
            "gradient_norm": float(gradient.item())}


def _retained_control(output: Path):
    source = BASELINE / "stable_residual"
    dest = output / "stable_residual"
    dest.mkdir()
    copied = {}
    for path in sorted(source.iterdir()):
        if path.name == "initial.pt" or path.name.startswith("step_") and path.suffix == ".pt" or path.name in ("progress.json", "receipt.json"):
            shutil.copy2(path, dest / path.name)
            if sha(path) != sha(dest / path.name):
                raise ValueError(f"Retained control copy mismatch: {path.name}")
            copied[path.name] = sha(dest / path.name)
        elif path.name.startswith("dev_step_") and path.is_dir():
            shutil.copytree(path, dest / path.name)
            for original in path.glob("*.npz"):
                if sha(original) != sha(dest / path.name / original.name):
                    raise ValueError(f"Retained DEV probability mismatch: {original}")
    original_receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
    return {**original_receipt, "retained_control": {
            "status": "not_retrained", "source": str(source.resolve()),
            "source_receipt_sha256": sha(source / "receipt.json"), "files": copied}}


def _frozen_inputs():
    meta = json.loads((BASELINE / "metadata.json").read_text(encoding="utf-8"))
    freeze = json.loads(BASELINE_FREEZE.read_text(encoding="utf-8"))
    if meta != freeze["metadata"] or sha(BASELINE / "receipt.json") != freeze["training_receipt_sha256"]:
        raise ValueError("Frozen v5 source or receipt changed")
    if sha(v5.HERE) != meta["trainer_sha256"] or sha(V4) != meta["v4_source_sha256"] or sha(V3) != meta["v3_source_sha256"]:
        raise ValueError("Frozen trainer dependency changed")
    manifest_path = Path(meta["manifest"])
    if sha(manifest_path) != meta["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    draws_path = BASELINE / "draw_schedule.json"
    if sha(draws_path) != meta["draw_schedule_file_sha256"]:
        raise ValueError("Frozen draw schedule changed")
    draws = json.loads(draws_path.read_text(encoding="utf-8"))
    if len(draws) != 38400:
        raise ValueError("Frozen draw count changed")
    set_branch(meta["config"]["branch"])
    _, scenes, counts, positives = v5.v4.v3.checked_scenes(manifest_path)
    if counts != meta["split_counts"] or positives != meta["positive_counts"]:
        raise ValueError("Frozen split counts changed")
    return meta, manifest_path, scenes, draws


def _initial_model_check():
    torch.manual_seed(v5.SEED)
    original = v5.new_model("siamese")
    torch.manual_seed(v5.SEED)
    candidate = new_model("siamese")
    base_state = original.state_dict()
    current = candidate.state_dict()
    if any(not torch.equal(value, current[key]) for key, value in base_state.items()):
        raise ValueError("Candidate backbone initialization differs from v5")
    if ARM == "delta_scene_dice" and not torch.count_nonzero(candidate.delta_skip.weight) == 0:
        raise ValueError("Delta skip is not zero initialized")
    return hashlib.sha256(b"".join(v.numpy().tobytes() for v in base_state.values())).hexdigest()


def train(arm: str, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if arm not in ARMS:
        raise ValueError(arm)
    configure({"config": {"arm": arm}})
    device = v5.v4.configure_gpu()
    baseline_meta, manifest_path, scenes, draws = _frozen_inputs()
    initial_backbone_hash = _initial_model_check()
    norm = baseline_meta["normalization"]
    output.mkdir(parents=True)
    shutil.copy2(BASELINE / "draw_schedule.json", output / "draw_schedule.json")
    control = _retained_control(output)
    meta = {"schema": "GTM_Model7_repaired_recall_v6", "arm": arm,
            "manifest": str(manifest_path.resolve()), "trainer_path": str(HERE),
            "manifest_sha256": sha(manifest_path), "trainer_sha256": sha(HERE),
            "v3_source_sha256": sha(V3), "v4_source_sha256": sha(V4), "v5_source_sha256": sha(V5),
            "original_sha256": sha(ORIGINAL), "stable_source_sha256": sha(STABLE),
            "protocol_sha256": sha(PROTOCOL), "normalization": norm,
            "draw_schedule_file_sha256": sha(output / "draw_schedule.json"),
            "baseline_metadata_sha256": sha(BASELINE / "metadata.json"),
            "baseline_receipt_sha256": sha(BASELINE / "receipt.json"),
            "baseline_freeze_sha256": sha(BASELINE_FREEZE),
            "split_counts": baseline_meta["split_counts"], "positive_counts": baseline_meta["positive_counts"],
            "draw_domain_class_counts": baseline_meta["draw_domain_class_counts"],
            "initial_backbone_hash": initial_backbone_hash,
            "config": {**baseline_meta["config"], "arm": arm, "dice_weight": 0.5,
                       "dice_reduction": "mean_per_positive_scene", "delta_skip": arm == "delta_scene_dice"},
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "test_array_access": "none"}
    write_json(output / "metadata.json", meta)
    model_dir = output / "siamese"
    model_dir.mkdir()
    torch.manual_seed(v5.SEED)
    torch.cuda.manual_seed_all(v5.SEED)
    model = new_model("siamese").to(device)
    optimizer = v5.v4.v3.optimizer(model)
    v5.v4.cpu_checkpoint(model, norm, "siamese", 0, model_dir / "initial.pt")
    by_id = {scene["id"]: scene for scene in scenes}
    cache = {}
    milestones, curve = {}, []
    start = time.monotonic()
    for step in range(1, 4801):
        if time.monotonic() - start > MAX_WALL_SECONDS_PER_MODEL:
            raise TimeoutError(f"{arm} wall-time cap at update {step}")
        sample = v5.v4.make_batch(manifest_path, by_id, draws[(step-1)*8:step*8], norm,
                                  cache, full_scene=True)
        sample = tuple(value.to(device, non_blocking=False) for value in sample)
        result = fit_step(model, optimizer, sample)
        if step == 1 or step % 25 == 0:
            curve.append({"step": step, **result})
        if step % 200 == 0:
            print(json.dumps({"arm": arm, "step": step, "seconds": round(time.monotonic()-start, 1),
                              "loss": result["loss"]}), flush=True)
        if step in MILESTONES:
            checkpoint = model_dir / f"step_{step}.pt"
            state = v5.v4.cpu_checkpoint(model, norm, "siamese", step, checkpoint)
            cpu_model = new_model("siamese")
            cpu_model.load_state_dict(state)
            train_score = evaluate(cpu_model, manifest_path, scenes, norm, "train", model_dir / f"train_step_{step}")
            dev_score = evaluate(cpu_model, manifest_path, scenes, norm, "dev", model_dir / f"dev_step_{step}")
            milestones[str(step)] = {"checkpoint_sha256": sha(checkpoint), "train": train_score,
                                     "dev": dev_score, "elapsed_seconds": time.monotonic()-start}
            write_json(model_dir / "progress.json", {"milestones": milestones, "curve": curve})
            print(json.dumps({"arm": arm, "milestone": step, "dev_native_iou": dev_score["native_pooled"]["iou"],
                              "dev_native_recall": dev_score["native_pooled"]["recall"],
                              "seconds": round(time.monotonic()-start, 1)}), flush=True)
    report = {"model": "siamese", "steps": 4800, "initial_checkpoint_sha256": sha(model_dir / "initial.pt"),
              "draw_schedule_sha256": hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest(),
              "curve": curve, "milestones": milestones, "elapsed_seconds": time.monotonic()-start}
    write_json(model_dir / "receipt.json", report)
    write_json(output / "receipt.json", {"schema": "GTM_Model7_repaired_recall_v6_receipt", "arm": arm,
                                       "metadata_sha256": sha(output / "metadata.json"),
                                       "models": {"siamese": report, "stable_residual": control},
                                       "test_array_access": "none"})
    print(json.dumps({"training": "complete", "arm": arm, "output": str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.arm, args.output.resolve())


if __name__ == "__main__":
    main()
