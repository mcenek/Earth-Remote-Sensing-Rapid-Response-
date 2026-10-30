"""Real-scene preflight and checkpoint replay for the two frozen recall fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_recall_v6 as v6


def independent_loss(logits, target, valid, positive):
    z = logits.astype(np.float64)
    y = target.astype(np.float64)
    m = valid.astype(np.float64)
    pos = positive.astype(bool)
    raw = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    raw[y == 1] *= 2
    bce = float((raw * m).sum() / m.sum())
    p = 1 / (1 + np.exp(-z))
    intersections = (p * y * m).sum(axis=(1, 2, 3))
    support = ((p + y) * m).sum(axis=(1, 2, 3))
    dice = float((1 - (2 * intersections[pos] + 1) / (support[pos] + 1)).mean())
    return bce + 0.5 * dice, bce, dice


def preflight(output: Path):
    if output.exists():
        raise FileExistsError(output)
    meta, manifest, scenes, draws = v6._frozen_inputs()
    by_id = {row["id"]: row for row in scenes}
    sample = v6.v5.v4.make_batch(manifest, by_id, draws[:8], meta["normalization"], {}, full_scene=True)
    x, y, valid, positive = sample
    if x.shape != (8, 12, 100, 100) or not torch.all(valid.sum(dim=(1, 2, 3)) > 0):
        raise ValueError("Real full-scene batch is malformed")
    if not (positive.sum().item() == 4 and (positive == 0).sum().item() == 4):
        raise ValueError("Balanced positive/negative batch changed")
    results = {}
    for arm in v6.ARMS:
        v6.configure({"config": {"arm": arm}})
        backbone_hash = v6._initial_model_check()
        torch.manual_seed(v6.v5.SEED)
        model = v6.new_model("siamese")
        with torch.no_grad():
            logits = model(x)
        loss, bce, dice = v6.loss_parts(logits, y, valid, positive)
        independent = independent_loss(logits.numpy(), y.numpy(), valid.numpy(), positive.numpy())
        if not np.allclose([loss.item(), bce.item(), dice.item()], independent, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Independent loss reduction mismatch: {arm}")
        # Remove invalid support and explicit negatives in the independent formula.
        poisoned = logits.clone()
        poisoned[valid == 0] = 30
        isolated = v6.loss_parts(poisoned, y, valid, positive)
        if not torch.allclose(loss, isolated[0], atol=1e-6):
            raise ValueError("Invalid pixels changed loss")
        shifted = logits.clone()
        shifted[positive == 0] += 1
        changed = v6.loss_parts(shifted, y, valid, positive)
        if not torch.allclose(dice, changed[2], atol=1e-6) or torch.allclose(bce, changed[1]):
            raise ValueError("Negative scenes changed Dice or failed to change BCE")
        opt = v6.v5.v4.v3.optimizer(model)
        trained = v6.fit_step(model, opt, sample)
        if not all(np.isfinite(value) for value in trained.values()):
            raise FloatingPointError("Nonfinite real-scene gradient")
        if arm == "delta_scene_dice":
            torch.manual_seed(v6.v5.SEED)
            baseline = v6.v5.new_model("siamese")
            torch.manual_seed(v6.v5.SEED)
            delta = v6.new_model("siamese")
            with torch.no_grad():
                if not torch.equal(baseline(x), delta(x)):
                    raise ValueError("Zero skip changed initial predictions")
        results[arm] = {"backbone_initial_sha256": backbone_hash,
                        "loss": float(loss.item()), "bce": float(bce.item()),
                        "positive_scene_dice_loss": float(dice.item()), "step": trained}
    if results[v6.ARMS[0]]["backbone_initial_sha256"] != results[v6.ARMS[1]]["backbone_initial_sha256"]:
        raise ValueError("Initial backbones differ across arms")
    output.mkdir(parents=True)
    v6.write_json(output / "receipt.json", {"schema": "GTM_Model7_recall_v6_preflight_e2e",
                                             "source_sha256": v6.sha(v6.HERE),
                                             "baseline_draw_sha256": v6.sha(v6.BASELINE / "draw_schedule.json"),
                                             "manifest_sha256": v6.sha(manifest),
                                             "batch_draws": draws[:8], "results": results,
                                             "test_array_access": "none"})
    print(json.dumps({"preflight": "passed", "receipt": str(output / "receipt.json")}), flush=True)


def replay(run: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    v6.configure(meta)
    v6.set_branch(meta["config"]["branch"])
    if v6.sha(v6.HERE) != meta["trainer_sha256"] or v6.sha(v6.V5) != meta["v5_source_sha256"]:
        raise ValueError("Replay source drift")
    if v6.sha(run / "draw_schedule.json") != meta["draw_schedule_file_sha256"]:
        raise ValueError("Replay draw drift")
    _, manifest, scenes, _ = v6._frozen_inputs()
    rows = {}
    for step in v6.MILESTONES:
        checkpoint = run / "siamese" / f"step_{step}.pt"
        progress = json.loads((run / "siamese" / "progress.json").read_text(encoding="utf-8"))
        if v6.sha(checkpoint) != progress["milestones"][str(step)]["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash drift")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = v6.new_model("siamese")
        model.load_state_dict(bundle["model"])
        if bundle["normalization"] != meta["normalization"] or bundle["steps"] != step:
            raise ValueError("Checkpoint metadata mismatch")
        original = run / "siamese" / f"dev_step_{step}"
        count = 0
        max_error = 0.0
        native_pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
        for scene in scenes:
            if scene["split"] != "dev":
                continue
            data = v6.load_scene(manifest, scene)
            x = v6.prepare(data, meta["normalization"])
            probability = v6.predict(model, x)
            with np.load(original / f"{scene['id']}.npz", allow_pickle=False) as saved:
                error = float(np.max(np.abs(probability - saved["probability20"])))
                max_error = max(error, max_error)
                if error > v6.REPLAY_ATOL:
                    raise ValueError(f"CPU replay mismatch: {scene['id']} {step}")
            native = np.repeat(np.repeat(probability, 2, 0), 2, 1)
            confusion = v6.v5.v4.v3.confusion(native, data["label_native"],
                                                 data["valid_native"].astype(bool))
            for key in native_pooled:
                native_pooled[key] += confusion[key]
            count += 1
        expected = progress["milestones"][str(step)]["dev"]["native_pooled"]["confusion"]
        if any(native_pooled[key] != expected[key] for key in native_pooled):
            raise ValueError("Native DEV count mismatch")
        rows[str(step)] = {"checkpoint_sha256": v6.sha(checkpoint), "dev_count": count,
                           "max_probability_error": max_error, "native_pooled": native_pooled}
    output.mkdir(parents=True)
    report = {"schema": "GTM_Model7_recall_v6_replay_e2e", "passed": True,
              "run": str(run), "manifest_sha256": meta["manifest_sha256"],
              "trainer_sha256": meta["trainer_sha256"], "rows": rows,
              "threshold": v6.THRESHOLD, "test_array_access": "none"}
    v6.write_json(output / "receipt.json", report)
    v6.write_json(run / "e2e_receipt.json", report)
    print(json.dumps({"replay": "passed", "run": str(run), "receipt": str(output / "receipt.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("preflight")
    first.add_argument("--output", type=Path, required=True)
    second = sub.add_parser("replay")
    second.add_argument("--run", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.output.resolve())
    else:
        replay(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
