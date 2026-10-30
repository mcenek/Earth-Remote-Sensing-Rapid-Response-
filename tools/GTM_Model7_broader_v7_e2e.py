"""Real-array preflight and independent CPU replay for broader v7 fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_broader_v7 as t


def independent_loss(logits, target, valid, positive):
    z, y, m = (v.astype(np.float64) for v in (logits, target, valid))
    raw = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    raw[y == 1] *= 2
    bce = float((raw * m).sum() / m.sum())
    p = 1 / (1 + np.exp(-z))
    selected = positive[:, None, None, None] * m
    dice = 1 - (2 * (p * y * selected).sum() + 1) / ((p * selected).sum() + (y * selected).sum() + 1)
    return bce + .1 * dice, bce, float(dice)


def feature_check(data, norm, arm):
    t.configure({"arm": arm})
    features = t.prepare(data, norm)
    base = t.v5.prepare(data, norm)
    curr, ref = base[..., :6], base[..., 6:]
    expected = np.concatenate((curr, ref, curr - ref, np.abs(curr - ref)), axis=-1)
    valid = data["valid"].astype(bool)
    if not np.allclose(features[..., :24], expected, atol=1e-6) or not np.all(features[~valid] == 0):
        raise ValueError("Independent 24-channel construction differs")
    if arm == "spectral_fusion":
        scale = np.asarray(norm["scale"])
        for j, other in enumerate((4, 3)):
            independent = ((curr[..., 5] - ref[..., 5]) * scale[5] -
                           (curr[..., other] - ref[..., other]) * scale[other]) / np.hypot(scale[5], scale[other])
            if not np.allclose(features[valid, 24 + j], independent[valid], atol=1e-6):
                raise ValueError("Independent centered-log spectral channel differs")
    return {"shape": list(features.shape), "valid_pixels": int(valid.sum()),
            "spectral_nonzero_pixels": int(np.count_nonzero(features[..., 24:])) if arm == "spectral_fusion" else 0}


def preflight(manifest: Path, reservation: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    t.set_branch("mars_repaired")
    base, scenes, draws = t.frozen_inputs(manifest, reservation)
    if t.sha(t.BASELINE / "draw_schedule.json") != base["draw_schedule_file_sha256"]:
        raise ValueError("Draw schedule changed")
    by_id = {scene["id"]: scene for scene in scenes}
    if len(draws[:8]) != 8 or sum(bool(d[4]) for d in draws[:8]) != 4:
        raise ValueError("Unbalanced first real batch")
    result = {}
    for arm in t.ARMS:
        t.configure({"arm": arm})
        sample = t.make_batch(manifest, by_id, draws[:8], base["normalization"], {})
        x, y, valid, positive = sample
        channels = 24 if arm == "early_fusion" else 26
        if x.shape != (8, channels, 100, 100) or y.shape != (8, 1, 100, 100):
            raise ValueError("Full-scene batch shape changed")
        if int(positive.sum()) != 4 or not (valid == 0).any():
            raise ValueError("Positive/negative or invalid support missing")
        first = t.load_scene(manifest, by_id[draws[0][0]])
        features = feature_check(first, base["normalization"], arm)
        control_file = t.BASELINE / "stable_residual" / "step_4800.pt"
        retained = torch.load(control_file, map_location="cpu", weights_only=False)
        original_control = t.v5.new_model("stable_residual")
        adapted_control = t.new_model("stable_residual")
        original_control.load_state_dict(retained["model"])
        adapted_control.load_state_dict(retained["model"])
        original_control.eval(); adapted_control.eval()
        with torch.no_grad():
            control_error = float((original_control(x[:, :12]) - adapted_control(x)).abs().max())
        if control_error != 0:
            raise ValueError("Retained residual prediction changed under v7 features")
        torch.manual_seed(20261024)
        model = t.new_model("siamese")
        with torch.no_grad():
            logits = model(x)
        if logits.shape != (8, 1, 100, 100):
            raise ValueError("Model output shape changed")
        actual = t.loss_parts(logits, y, valid, positive)
        independent = independent_loss(logits.numpy(), y.numpy(), valid.numpy(), positive.numpy())
        if not np.allclose([v.item() for v in actual], independent, atol=1e-5, rtol=1e-5):
            raise ValueError("Pooled v5 loss parity failed")
        poisoned = logits.clone()
        poisoned[valid == 0] = 30
        if not torch.allclose(t.loss_parts(poisoned, y, valid, positive)[0], actual[0], atol=1e-6):
            raise ValueError("Invalid pixels contribute to loss")
        negative_shift = logits.clone()
        negative_shift[positive == 0] += 1
        shifted = t.loss_parts(negative_shift, y, valid, positive)
        if torch.allclose(shifted[1], actual[1]) or not torch.allclose(shifted[2], actual[2], atol=1e-6):
            raise ValueError("Negative scenes do not contribute correctly")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
        trained = t.fit_step(model, optimizer, sample)
        if not all(np.isfinite(v) for v in trained.values()):
            raise FloatingPointError("Nonfinite real-batch gradient")
        # A checkpoint must reconstruct the source-bound architecture and predictions.
        output.mkdir(parents=True, exist_ok=True)
        file = output / f"{arm}_roundtrip.pt"
        t.v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", 1, file)
        saved = torch.load(file, map_location="cpu", weights_only=False)
        rebuilt = t.new_model("siamese")
        rebuilt.load_state_dict(saved["model"])
        model.eval(); rebuilt.eval()
        with torch.no_grad():
            error = float((model(x) - rebuilt(x)).abs().max())
        if error != 0 or saved["normalization"] != base["normalization"]:
            raise ValueError("Checkpoint round trip failed")
        result[arm] = {"features": features, "loss": [float(v.item()) for v in actual],
                       "independent_loss": list(independent), "gradient": trained,
                       "checkpoint_sha256": t.sha(file), "roundtrip_max_error": error,
                       "retained_control_checkpoint_sha256": t.sha(control_file),
                       "retained_control_max_error": control_error}
    receipt = {"schema": "GTM_Model7_broader_v7_preflight_e2e", "passed": True,
               "trainer_sha256": t.sha(t.HERE), "protocol_sha256": t.sha(t.PROTOCOL),
               "manifest_sha256": t.sha(manifest), "reservation_sha256": t.sha(reservation),
               "baseline_draw_sha256": t.sha(t.BASELINE / "draw_schedule.json"),
               "batch_draws": draws[:8], "arms": result, "test_array_access": "none"}
    t.write_json(output / "receipt.json", receipt)
    print(json.dumps({"preflight": "passed", "receipt": str(output / "receipt.json")}), flush=True)


def replay(run: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    t.configure(meta)
    t.set_branch(meta["config"]["branch"])
    for field, source in (("trainer_sha256", t.HERE), ("v5_source_sha256", t.V5),
                          ("v4_source_sha256", t.V4), ("v3_source_sha256", t.V3),
                          ("original_sha256", t.ORIGINAL), ("stable_source_sha256", t.STABLE),
                          ("protocol_sha256", t.PROTOCOL)):
        if t.sha(source) != meta[field]:
            raise ValueError(f"Replay source drift: {field}")
    manifest = Path(meta["manifest"])
    reservation = Path(meta["reservation_path"])
    if t.sha(manifest) != meta["manifest_sha256"] or t.sha(reservation) != meta["reservation_sha256"]:
        raise ValueError("Frozen data metadata drift")
    if t.sha(run / "draw_schedule.json") != meta["draw_schedule_file_sha256"]:
        raise ValueError("Frozen draws changed")
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    if t.sha(run / "metadata.json") != receipt["metadata_sha256"]:
        raise ValueError("Training metadata changed")
    _, scenes, _ = t.frozen_inputs(manifest, reservation)
    progress = json.loads((run / "siamese" / "progress.json").read_text(encoding="utf-8"))
    rows = {}
    for step in t.MILESTONES:
        milestone = progress["milestones"][str(step)]
        checkpoint = run / "siamese" / f"step_{step}.pt"
        if t.sha(checkpoint) != milestone["checkpoint_sha256"]:
            raise ValueError("Milestone checkpoint changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["normalization"] != meta["normalization"] or saved["steps"] != step:
            raise ValueError("Checkpoint preprocessing differs")
        model = t.new_model("siamese")
        model.load_state_dict(saved["model"])
        total = {key: 0 for key in ("tp", "fp", "fn", "tn")}
        maximum = 0.
        count = 0
        for scene in scenes:
            if scene["split"] != "dev":
                continue
            data = t.load_scene(manifest, scene)
            p = t.predict(model, t.prepare(data, meta["normalization"]))
            file = run / "siamese" / f"dev_step_{step}" / f"{scene['id']}.npz"
            original = milestone["dev"]["scenes"][scene["id"]]
            if t.sha(file) != original["probability_sha256"]:
                raise ValueError("Saved DEV scores changed")
            with np.load(file, allow_pickle=False) as z:
                error = float(np.max(np.abs(p - z["probability20"])))
                if not np.array_equal(z["valid20"], data["valid"].astype(bool)):
                    raise ValueError("DEV support changed")
            maximum = max(maximum, error)
            if error > t.REPLAY_ATOL:
                raise ValueError(f"CPU prediction replay mismatch: {scene['id']}/{step}")
            native = np.repeat(np.repeat(p, t.native_factor(), axis=0), t.native_factor(), axis=1)
            mask = data["valid_native"].astype(bool)
            prediction = native[mask] >= .5
            truth = data["label_native"][mask] == 1
            independent = {"tp": int(np.count_nonzero(prediction & truth)),
                           "fp": int(np.count_nonzero(prediction & ~truth)),
                           "fn": int(np.count_nonzero(~prediction & truth)),
                           "tn": int(np.count_nonzero(~prediction & ~truth))}
            if independent != original["native_confusion"]:
                raise ValueError("DEV scene native count mismatch")
            for key in total:
                total[key] += independent[key]
            count += 1
        expected = milestone["dev"]["native_pooled"]["confusion"]
        if total != expected:
            raise ValueError("Pooled native DEV count mismatch")
        rows[str(step)] = {"checkpoint_sha256": t.sha(checkpoint), "dev_count": count,
                           "max_probability_error": maximum, "native_pooled": total}
    output.mkdir(parents=True)
    report = {"schema": "GTM_Model7_broader_v7_replay_e2e", "passed": True,
              "run": str(run), "trainer_sha256": meta["trainer_sha256"],
              "manifest_sha256": meta["manifest_sha256"], "reservation_sha256": meta["reservation_sha256"],
              "rows": rows, "threshold": t.THRESHOLD, "fresh_cpu_process": True, "test_array_access": "none"}
    t.write_json(output / "receipt.json", report)
    t.write_json(run / "e2e_receipt.json", report)
    print(json.dumps({"replay": "passed", "receipt": str(output / "receipt.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--manifest", type=Path, required=True)
    pre.add_argument("--reservation", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    rep = commands.add_parser("replay")
    rep.add_argument("--run", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.manifest.resolve(), args.reservation.resolve(), args.output.resolve())
    else:
        replay(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
