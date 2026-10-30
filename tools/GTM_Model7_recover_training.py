"""Train-only recovery of the sealed Model 7 temporal comparison.

See docs/GTM_Model7_recovery_protocol.md, written before this implementation.
Never reads test label keys or prior test predictions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL = ROOT / "tools/GTM_Model7_train_temporal.py"
spec = importlib.util.spec_from_file_location("sealed_temporal_architecture", ORIGINAL)
old = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(old)
MANIFEST = ROOT / "outputs/GTM_Model7_temporal_20261023/data/manifest.json"
DEFAULT_OUT = ROOT / "outputs/GTM_Model7_recovery_20261023/train_diagnostics"
STEPS_PER_EPOCH = old.BASE["draws_per_epoch"] // old.BASE["batch_size"]


def setup(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing populated recovery directory: {output}")
    _, scenes = old.load_manifest(MANIFEST)  # test is loaded with labels=False
    norm = old.normalization(MANIFEST, scenes)
    output.mkdir(parents=True)
    return scenes, norm


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=old.BASE["learning_rate"],
                             weight_decay=old.BASE["weight_decay"])


def update(model, opt, sample, pos_weight: float, ordinary: bool):
    x, y, valid, positive = sample
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    if ordinary:
        raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (raw * valid).sum() / valid.sum()
    else:
        loss = old.fit_loss(logits, y, valid, positive, pos_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), old.BASE["gradient_norm_cap"],
                                               error_if_nonfinite=True)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Nonfinite gradient")
    opt.step()
    return float(loss.item()), float(gradient.item())


def evaluate(model, scenes, norm, split: str, *, save_to: Path | None = None):
    pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    per_scene = {}
    activated = 0
    if save_to:
        save_to.mkdir(parents=True)
    for scene in scenes:
        if scene["split"] != split:
            continue
        data = old.load_scene(MANIFEST, scene, labels=True)
        valid = data["valid"].astype(bool)
        p = old.infer(model, old.prepare(data, norm), valid)
        c = old.confusion(p, data["target"], valid)
        for key in pooled:
            pooled[key] += c[key]
        activated += bool(c["fp"])
        per_scene[scene["id"]] = {**c, "probability_min": float(p[valid].min()),
                                   "probability_max": float(p[valid].max())}
        if save_to:
            np.savez_compressed(save_to / f"{scene['id']}.npz", probability20=p,
                                valid20=valid)
            per_scene[scene["id"]]["prediction_sha256"] = old.sha(save_to / f"{scene['id']}.npz")
    return {**old.score(pooled), "activated_scenes": activated, "scene_count": len(per_scene),
            "scenes": per_scene}


def probe(output: Path, scenes, norm):
    draws = old.schedule(MANIFEST, scenes, 1)[0][:4]
    by_id = {s["id"]: s for s in scenes}
    sample = old.batch(MANIFEST, by_id, draws, norm, {})
    x, y, valid, _ = sample
    result = {"draws": draws, "steps": 300, "loss": "ordinary masked BCE", "models": {}}
    for name in ("residual", "siamese"):
        torch.manual_seed(old.BASE["seed"])
        model = old.new_model(name)
        opt = optimizer(model)
        start = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            initial = float(((F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * valid).sum() / valid.sum()).item())
        curve = []
        for step in range(300):
            loss, grad = update(model, opt, sample, 1, True)
            if (step + 1) % 25 == 0:
                curve.append({"step": step + 1, "loss": loss, "gradient_norm": grad})
        model.eval()
        with torch.no_grad():
            p = old.array(torch.sigmoid(model(x))[:, 0])
        target = old.array(y[:, 0])
        support = old.array(valid[:, 0]).astype(bool)
        positive = old.confusion(p[[0, 2]], target[[0, 2]], support[[0, 2]])
        negative = old.confusion(p[[1, 3]], target[[1, 3]], support[[1, 3]])
        changed = sum(float((value - start[key]).abs().sum().item()) for key, value in model.state_dict().items())
        result["models"][name] = {"initial_loss": initial, "final_loss": loss,
                                   "parameter_l1_change": changed,
                                   "positive_probe": old.score(positive),
                                   "negative_probe": old.score(negative), "curve": curve}
    old.write_json(output / "probe.json", result)
    return result


def train_arm(output: Path, scenes, norm, name: str, weight: float, steps: int, *, checkpoints=(360, 1800)):
    arm = output / f"{name}_w{int(weight)}"
    arm.mkdir()
    torch.manual_seed(old.BASE["seed"])
    model = old.new_model(name)
    opt = optimizer(model)
    draws = old.schedule(MANIFEST, scenes, steps // STEPS_PER_EPOCH)
    by_id = {s["id"]: s for s in scenes}
    cache = {}
    curve = []
    results = {}
    initial = {key: val.clone() for key, val in model.state_dict().items()}
    start_time = time.monotonic()
    n = 0
    for epoch_draws in draws:
        for offset in range(0, len(epoch_draws), old.BASE["batch_size"]):
            if time.monotonic() - start_time > 900:
                raise TimeoutError(f"{name} weight {weight} exceeded 15-minute cap")
            sample = old.batch(MANIFEST, by_id, epoch_draws[offset:offset + old.BASE["batch_size"]], norm, cache)
            loss, grad = update(model, opt, sample, weight, False)
            n += 1
            if n % 60 == 0:
                curve.append({"step": n, "loss": loss, "gradient_norm": grad})
            if n in checkpoints:
                checkpoint = arm / f"step_{n}.pt"
                torch.save({"model": model.state_dict(), "normalization": norm,
                            "model_name": name, "positive_weight": weight, "steps": n,
                            "original_source_sha256": old.sha(ORIGINAL),
                            "recovery_source_sha256": old.sha(HERE)}, checkpoint)
                train = evaluate(model, scenes, norm, "train", save_to=arm / f"train_predictions_step_{n}")
                dev = evaluate(model, scenes, norm, "dev")
                results[str(n)] = {"train": train, "dev": dev,
                                   "checkpoint_sha256": old.sha(checkpoint),
                                   "elapsed_seconds": time.monotonic() - start_time}
                print(json.dumps({"arm": arm.name, "steps": n,
                                  "train_iou": train["iou"], "train_recall": train["recall"],
                                  "dev_fpr": dev["false_positive_area_rate"]}), flush=True)
    changed = sum(float((val - initial[key]).abs().sum().item()) for key, val in model.state_dict().items())
    report = {"model": name, "positive_weight": weight, "steps": steps,
              "draw_schedule_sha256": old.schedule_hash(draws),
              "parameter_l1_change": changed, "curve": curve, "checkpoints": results}
    old.write_json(arm / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    scenes, norm = setup(output)
    metadata = {"schema": "GTM_Model7_training_recovery_v1",
                "manifest": str(MANIFEST), "manifest_sha256": old.sha(MANIFEST),
                "original_source_sha256": old.sha(ORIGINAL),
                "recovery_source_sha256": old.sha(HERE),
                "protocol_sha256": old.sha(ROOT / "docs/GTM_Model7_recovery_protocol.md"),
                "normalization": norm,
                "splits": {split: sum(s["split"] == split for s in scenes) for split in ("train", "dev", "test")},
                "test_access": "manifest validator input-only; no test label keys or predictions"}
    old.write_json(output / "metadata.json", metadata)
    probe_result = probe(output, scenes, norm)
    arms = {}
    for weight in (1.0, 10.0):
        arms[str(int(weight))] = train_arm(output, scenes, norm, "residual", weight, 1800)
    # Selection rule was fixed before inspecting results: retain positive train recall
    # >= 0.5, then maximize train IoU with dev FPR as a tie-breaker. If no arm
    # qualifies, mark recovery failed and still run the least collapsed arm.
    choices = []
    for weight, report in arms.items():
        for step, value in report["checkpoints"].items():
            train, dev = value["train"], value["dev"]
            choices.append((float(train["recall"] or 0) >= 0.5,
                            float(train["iou"] or 0), -float(dev["false_positive_area_rate"] or 0),
                            int(weight), int(step)))
    selected = max(choices)
    weight, steps = selected[3:]
    siamese = train_arm(output, scenes, norm, "siamese", float(weight), steps,
                        checkpoints=(steps,))
    final = {**metadata, "selection_rule": "require train recall >=0.5, then maximize train IoU, dev FPR tie-breaker",
             "selection_has_recall_floor": bool(selected[0]),
             "selected_positive_weight": weight, "selected_steps": steps,
             "probe": probe_result, "residual_arms": arms, "siamese": siamese,
             "test_selected": False}
    old.write_json(output / "recovery_receipt.json", final)
    print(json.dumps({"recovery": "complete", "selected_weight": weight, "selected_steps": steps,
                      "receipt": str(output / "recovery_receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
