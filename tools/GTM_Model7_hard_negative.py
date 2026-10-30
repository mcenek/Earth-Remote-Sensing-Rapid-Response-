"""Controlled same-site negative resampling, restricted to TRAIN and DEV."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SOURCE = ROOT / "tools/GTM_Model7_recover_training.py"
spec = importlib.util.spec_from_file_location("recovery", SOURCE)
r = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r)
old = r.old
OUT = ROOT / "outputs/GTM_Model7_recovery_20261023/hard_negative"
ORIGINAL_SCHEDULE = old.schedule


def hard_schedule(path, scenes, epochs):
    baseline = ORIGINAL_SCHEDULE(path, scenes, epochs)
    train = [s for s in scenes if s["split"] == "train"]
    positives = {s["physical_location_id"] for s in train
                 if np.any((old.load_scene(path, s, labels=True)["target"] == 1) &
                           old.load_scene(path, s, labels=True)["valid"].astype(bool))}
    same, other, support = [], [], {}
    for scene in train:
        data = old.load_scene(path, scene, labels=True)
        if np.any((data["target"] == 1) & data["valid"].astype(bool)):
            continue
        sid = scene["id"]
        support[sid] = np.argwhere(data["valid"].astype(bool))
        (same if scene["physical_location_id"] in positives else other).append(sid)
    if not same or not other:
        raise ValueError("Both same-site and other-site train negatives required")
    rng = np.random.default_rng(20261024)
    result = []
    for epoch in baseline:
        updated = []
        for i, draw in enumerate(epoch):
            if draw[4]:
                updated.append(draw)
                continue
            pool = same if i % 4 == 1 else other
            sid = pool[int(rng.integers(len(pool)))]
            points = support[sid]
            cy, cx = points[int(rng.integers(len(points)))]
            top = int(np.clip(cy - old.BASE["crop"] // 2, 0, 100 - old.BASE["crop"]))
            left = int(np.clip(cx - old.BASE["crop"] // 2, 0, 100 - old.BASE["crop"]))
            updated.append((sid, top, left, draw[3], False))
        result.append(updated)
    if any(a != b for old_epoch, new_epoch in zip(baseline, result)
           for a, b in zip(old_epoch, new_epoch) if a[4]):
        raise AssertionError("A positive draw changed")
    return result


def lower_lr(model):
    return torch.optim.AdamW(model.parameters(), lr=0.0005,
                             weight_decay=old.BASE["weight_decay"])


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing populated output: {OUT}")
    _, scenes = old.load_manifest(r.MANIFEST)
    norm = old.normalization(r.MANIFEST, scenes)
    generated = hard_schedule(r.MANIFEST, scenes, 300)
    OUT.mkdir(parents=True)
    r.optimizer = lower_lr
    old.schedule = hard_schedule
    arms = {}
    for weight in (2.0, 3.0):
        arms[str(int(weight))] = r.train_arm(OUT, scenes, norm, "siamese", weight, 1800,
                                            checkpoints=(360, 1800))
        path = OUT / f"siamese_w{int(weight)}" / "step_1800.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = old.new_model("siamese")
        model.load_state_dict(state["model"])
        arms[str(int(weight))]["dev_predictions"] = r.evaluate(
            model, scenes, norm, "dev",
            save_to=OUT / f"siamese_w{int(weight)}" / "dev_predictions_step_1800")
    choices = []
    for weight, arm in arms.items():
        row = arm["checkpoints"]["1800"]
        train, dev = row["train"], row["dev"]
        choices.append((float(train["recall"] or 0) >= 0.5,
                        float(train["iou"] or 0), -float(dev["false_positive_area_rate"] or 0),
                        int(weight)))
    selected = max(choices)
    report = {"schema": "GTM_Model7_hard_negative_v1", "manifest_sha256": old.sha(r.MANIFEST),
              "original_trainer_sha256": old.sha(r.ORIGINAL), "recovery_source_sha256": old.sha(SOURCE),
              "source_sha256": old.sha(HERE),
              "protocol_sha256": old.sha(ROOT / "docs/GTM_Model7_hard_negative_protocol.md"),
              "normalization": norm, "schedule_sha256": old.schedule_hash(generated),
              "baseline_schedule_sha256": old.schedule_hash(ORIGINAL_SCHEDULE(r.MANIFEST, scenes, 300)),
              "config": {"lr": 0.0005, "weights": [2, 3], "steps": 1800,
                         "threshold": 0.5, "negative_same_site_fraction": 0.5},
              "arms": arms, "selection_rule": "require train recall >=0.5; maximize train IoU; dev FPR tie-breaker",
              "selected_weight": selected[3], "selection_has_recall_floor": bool(selected[0]),
              "test_labels_read": False, "test_predictions_generated": False}
    old.write_json(OUT / "receipt.json", report)
    print(json.dumps({"hard_negative": "complete", "selected_weight": selected[3],
                      "receipt": str(OUT / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
