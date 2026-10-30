"""Train-only stability and moderate-weight follow-up. See the prewritten protocol."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SOURCE = ROOT / "tools/GTM_Model7_recover_training.py"
spec = importlib.util.spec_from_file_location("temporal_recovery", SOURCE)
r = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r)
old = r.old
OUT = ROOT / "outputs/GTM_Model7_recovery_20261023/stability_followup"


def stable_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.0005,
                             weight_decay=old.BASE["weight_decay"])


def tiny_probe(scenes, norm):
    draws = old.schedule(r.MANIFEST, scenes, 1)[0][:4]
    sample = old.batch(r.MANIFEST, {s["id"]: s for s in scenes}, draws, norm, {})
    x, y, valid, _ = sample
    values = {}
    for lr in (0.0001, 0.0005):
        torch.manual_seed(old.BASE["seed"])
        model = old.new_model("residual")
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=old.BASE["weight_decay"])
        with torch.no_grad():
            initial = float(((F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * valid).sum() / valid.sum()).item())
        start = {key: val.clone() for key, val in model.state_dict().items()}
        curve = []
        for step in range(300):
            loss, grad = r.update(model, opt, sample, 1, True)
            if (step + 1) % 25 == 0:
                curve.append({"step": step + 1, "loss": loss, "gradient_norm": grad})
        with torch.no_grad():
            p = old.array(torch.sigmoid(model(x))[:, 0])
        target = old.array(y[:, 0])
        mask = old.array(valid[:, 0]).astype(bool)
        positive = old.score(old.confusion(p[[0, 2]], target[[0, 2]], mask[[0, 2]]))
        negative = old.score(old.confusion(p[[1, 3]], target[[1, 3]], mask[[1, 3]]))
        changed = sum(float((val - start[key]).abs().sum().item()) for key, val in model.state_dict().items())
        values[str(lr)] = {"initial_loss": initial, "final_loss": loss,
                            "parameter_l1_change": changed,
                            "positive_probe": positive, "negative_probe": negative,
                            "curve": curve}
    return {"draws": draws, "steps": 300, "values": values}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing populated follow-up: {OUT}")
    _, scenes = old.load_manifest(r.MANIFEST)
    norm = old.normalization(r.MANIFEST, scenes)
    OUT.mkdir(parents=True)
    hashes = {"manifest": old.sha(r.MANIFEST), "original_trainer": old.sha(r.ORIGINAL),
              "first_recovery": old.sha(SOURCE), "followup": old.sha(HERE),
              "protocol": old.sha(ROOT / "docs/GTM_Model7_recovery_followup.md")}
    probe = tiny_probe(scenes, norm)
    old.write_json(OUT / "residual_stability_probe.json", probe)
    r.optimizer = stable_optimizer
    arms = {}
    for weight in (2.0, 3.0):
        arms[str(int(weight))] = r.train_arm(OUT, scenes, norm, "siamese", weight, 1800)
    choices = []
    for weight, arm in arms.items():
        for step, checkpoint in arm["checkpoints"].items():
            train, dev = checkpoint["train"], checkpoint["dev"]
            choices.append((float(train["recall"] or 0) >= 0.5,
                            float(train["iou"] or 0), -float(dev["false_positive_area_rate"] or 0),
                            int(weight), int(step)))
    selected = max(choices)
    report = {"schema": "GTM_Model7_recovery_followup_v1", "hashes": hashes,
              "config": {"lr": 0.0005, "weights": [2, 3], "steps": [360, 1800],
                         "dice_weight": 0.1, "threshold": 0.5,
                         "same_seed_and_schedule_as_original": True},
              "normalization": norm, "residual_stability_probe": probe,
              "siamese_arms": arms,
              "selection_rule": "require train recall >=0.5, then maximize train IoU, dev FPR tie-breaker",
              "selected": {"weight": selected[3], "step": selected[4],
                           "recall_floor_met": bool(selected[0])},
              "test_labels_read": False, "test_predictions_generated": False}
    old.write_json(OUT / "receipt.json", report)
    print(json.dumps({"followup": "complete", "selected": report["selected"],
                      "receipt": str(OUT / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
