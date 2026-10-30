"""Real TRAIN preflight and fresh CPU DEV replay for one site-balanced v8 fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_site_v8 as t


def expect_failure(name, action):
    try:
        action()
    except (ValueError, FloatingPointError):
        return name
    raise AssertionError("Failure control unexpectedly passed: " + name)


def independent_loss(logits, target, valid, positive):
    z, y, m = (a.astype(np.float64) for a in (logits, target, valid))
    raw = np.maximum(z, 0) - z*y + np.log1p(np.exp(-np.abs(z)))
    raw[y == 1] *= 2
    bce = float((raw*m).sum()/m.sum())
    p = 1/(1+np.exp(-z))
    selected = positive[:, None, None, None]*m
    dice = 1-(2*(p*y*selected).sum()+1)/((p*selected).sum()+(y*selected).sum()+1)
    return [bce+.1*dice, bce, float(dice)]


def preflight(manifest, output):
    if output.exists():
        raise FileExistsError(output)
    t.configure({"arm": "site_balanced"})
    t.set_branch("mars_repaired")
    base, scenes, old = t.frozen_inputs(manifest)
    draws, by_id, sites = t.site_schedule(scenes)
    schedule_digest = hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest()
    controls = []
    controls.append(expect_failure("wrong_arm", lambda: t.configure({"arm": "early_fusion"})))
    controls.append(expect_failure("changed_manifest", lambda: t.frozen_inputs(t.PROTOCOL)))
    original_v7_hash = t.V7_SHA256
    try:
        t.V7_SHA256 = "0"*64
        controls.append(expect_failure("changed_v7_source", lambda: t.configure({"arm": "site_balanced"})))
    finally:
        t.V7_SHA256 = original_v7_hash
        t.configure({"arm": "site_balanced"})
    if len(draws) != 38400 or {row[0] for row in draws} - by_id.keys():
        raise ValueError("Draw schedule includes non-TRAIN identity")
    if any(sum(bool(d[4]) for d in draws[i:i+8]) != 4 for i in range(0, len(draws), 8)):
        raise ValueError("Draw batch class imbalance")
    counts = t.schedule_counts(draws, by_id, sites)
    if counts["unique_sampled_sites_by_class"] != counts["class_site_inventory"]:
        raise ValueError("Site sampling did not reach every site")
    first = t.load_scene(manifest, by_id[draws[0][0]])
    features = t.prepare(first, base["normalization"])
    prepared = t.v5.prepare(first, base["normalization"])
    curr, ref = prepared[..., :6], prepared[..., 6:]
    expected = np.concatenate((curr, ref, curr-ref, np.abs(curr-ref)), axis=-1)
    valid = first["valid"].astype(bool)
    if features.shape != (100, 100, 24) or not np.array_equal(features, expected) or np.any(features[~valid]) or not np.isfinite(features).all():
        raise ValueError("Frozen 24-channel feature construction changed")
    sample = t.make_batch(manifest, by_id, draws[:8], base["normalization"], {})
    x, y, mask, positive = sample
    if x.shape != (8, 24, 100, 100) or y.shape != (8, 1, 100, 100) or int(positive.sum()) != 4 or not (mask == 0).any():
        raise ValueError("Real full-scene batch malformed")
    controls.append(expect_failure("class_flag_mismatch", lambda: t.loss_parts(torch.zeros_like(y), y, mask, 1-positive)))
    controls.append(expect_failure("empty_support", lambda: t.loss_parts(torch.zeros_like(y), y, torch.zeros_like(mask), positive)))
    d4 = draws[0][3]
    independently_rotated = t.v5.v4.v3.old.d4(features, d4)
    if not np.array_equal(x[0].permute(1, 2, 0).numpy(), independently_rotated):
        raise ValueError("D4 feature alignment failed")
    for key, index in (("target", 1), ("valid", 2)):
        direct = t.v5.v4.v3.old.d4(first[key], d4)
        if not np.array_equal(sample[index][0, 0].numpy(), direct):
            raise ValueError("D4 label/support alignment failed: " + key)
    torch.manual_seed(20261024)
    model = t.new_model("siamese")
    with torch.no_grad():
        logits = model(x)
    actual = t.loss_parts(logits, y, mask, positive)
    independent = independent_loss(logits.numpy(), y.numpy(), mask.numpy(), positive.numpy())
    if not np.allclose([v.item() for v in actual], independent, atol=1e-5, rtol=1e-5):
        raise ValueError("Frozen v7 loss parity failed")
    poison = logits.clone()
    poison[mask == 0] = 30
    if not torch.allclose(t.loss_parts(poison, y, mask, positive)[0], actual[0], atol=1e-6):
        raise ValueError("Invalid support affects loss")
    negative_shift = logits.clone()
    negative_shift[positive == 0] += 1
    shifted = t.loss_parts(negative_shift, y, mask, positive)
    if torch.allclose(shifted[1], actual[1]) or not torch.allclose(shifted[2], actual[2], atol=1e-6):
        raise ValueError("Class contributions to loss changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    gradient = t.fit_step(model, optimizer, sample)
    if not all(np.isfinite(v) for v in gradient.values()):
        raise FloatingPointError("Nonfinite real gradient")
    output.mkdir(parents=True)
    checkpoint = output / "cpu_roundtrip.pt"
    t.v5.v4.cpu_checkpoint(model, base["normalization"], "siamese", 1, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = t.new_model("siamese")
    restored.load_state_dict(saved["model"])
    model.eval(); restored.eval()
    with torch.no_grad():
        error = float((model(x)-restored(x)).abs().max())
    if error != 0 or saved["normalization"] != base["normalization"]:
        raise ValueError("CPU checkpoint roundtrip changed")
    t.THRESHOLD = .7
    dummy = np.array([.6, .8], dtype=np.float32)
    if t.confusion(dummy, np.array([0, 1]), np.array([True, True])) != {"tp": 1, "fp": 0, "fn": 0, "tn": 1}:
        raise ValueError("Own threshold not used in confusion")
    t.THRESHOLD = .5
    receipt = {"schema": "GTM_Model7_site_v8_preflight_e2e", "passed": True,
               "trainer_sha256": t.sha(t.HERE), "v7_source_sha256": t.sha(t.V7),
               "v5_source_sha256": t.sha(t.V5), "v4_source_sha256": t.sha(t.V4),
               "v3_source_sha256": t.sha(t.V3), "original_sha256": t.sha(t.ORIGINAL),
               "stable_source_sha256": t.sha(t.STABLE), "protocol_sha256": t.sha(t.PROTOCOL),
               "manifest_sha256": t.sha(manifest), "v7_metadata_sha256": t.sha(t.V7_METADATA),
               "schedule_sha256": schedule_digest, "schedule_counts": counts,
               "failure_controls": controls, "features": {"shape": list(features.shape),
               "valid_pixels": int(valid.sum())}, "loss": [float(v.item()) for v in actual],
               "independent_loss": independent, "gradient": gradient,
               "checkpoint_sha256": t.sha(checkpoint), "checkpoint_roundtrip_error": error,
               "test_array_access": "none"}
    t.write_json(output / "receipt.json", receipt)
    print(json.dumps({"preflight": "passed", "receipt": str(output / "receipt.json"),
                      "failure_controls": controls}), flush=True)


def replay(run, output):
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    t.configure(meta)
    t.set_branch(meta["config"]["branch"])
    t.THRESHOLD = meta["config"]["threshold"]
    t.check_sources(meta)
    manifest = Path(meta["manifest"])
    if t.sha(manifest) != meta["manifest_sha256"] or t.sha(run / "draw_schedule.json") != meta["draw_schedule_file_sha256"] or t.sha(run / "schedule_counts.json") != meta["schedule_counts_sha256"]:
        raise ValueError("Frozen data/schedule drift")
    if t.sha(t.V7_METADATA) != meta["v7_metadata_sha256"]:
        raise ValueError("Frozen v7 metadata drift")
    fit_receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    if t.sha(run / "metadata.json") != fit_receipt["metadata_sha256"]:
        raise ValueError("Training metadata drift")
    base, scenes, _ = t.frozen_inputs(manifest)
    if base["normalization"] != meta["normalization"]:
        raise ValueError("Frozen normalizer drift")
    progress = json.loads((run / "siamese" / "progress.json").read_text(encoding="utf-8"))
    results = {}
    for step in t.MILESTONES:
        prior = progress["milestones"][str(step)]
        checkpoint = run / "siamese" / f"step_{step}.pt"
        if t.sha(checkpoint) != prior["checkpoint_sha256"]:
            raise ValueError("Milestone checkpoint drift")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["normalization"] != meta["normalization"] or saved["steps"] != step:
            raise ValueError("Checkpoint normalization or step drift")
        model = t.new_model("siamese")
        model.load_state_dict(saved["model"])
        native_total = {k: 0 for k in ("tp", "fp", "fn", "tn")}
        max_error = 0.
        count = 0
        for scene in scenes:
            if scene["split"] != "dev":
                continue
            data = t.load_scene(manifest, scene)
            probability = t.predict(model, t.prepare(data, meta["normalization"]))
            file = run / "siamese" / f"dev_step_{step}" / f"{scene['id']}.npz"
            previous = prior["dev"]["scenes"][scene["id"]]
            if t.sha(file) != previous["probability_sha256"]:
                raise ValueError("Saved DEV score file changed")
            with np.load(file, allow_pickle=False) as archive:
                max_error = max(max_error, float(np.max(np.abs(probability-archive["probability20"]))))
                if not np.array_equal(archive["valid20"], data["valid"].astype(bool)):
                    raise ValueError("DEV support drift")
            if max_error > t.REPLAY_ATOL:
                raise ValueError("CPU prediction replay differs")
            native = np.repeat(np.repeat(probability, t.native_factor(), axis=0), t.native_factor(), axis=1)
            confusion = t.confusion(native, data["label_native"], data["valid_native"])
            if confusion != previous["native_confusion"]:
                raise ValueError("Native confusion drift")
            for key in native_total:
                native_total[key] += confusion[key]
            count += 1
        if count != 198 or native_total != prior["dev"]["native_pooled"]["confusion"]:
            raise ValueError("DEV scene count or pooled native totals changed")
        results[str(step)] = {"dev_scenes": count, "max_probability_error": max_error,
                              "checkpoint_sha256": t.sha(checkpoint), "native_pooled": native_total}
    output.mkdir(parents=True)
    report = {"schema": "GTM_Model7_site_v8_replay_e2e", "passed": True,
              "trainer_sha256": meta["trainer_sha256"], "manifest_sha256": meta["manifest_sha256"],
              "threshold": t.THRESHOLD, "fresh_cpu_process": True, "test_array_access": "none",
              "rows": results}
    t.write_json(output / "receipt.json", report)
    t.write_json(run / "e2e_receipt.json", report)
    print(json.dumps({"replay": "passed", "receipt": str(output / "receipt.json")}), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--manifest", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    rep = sub.add_parser("replay")
    rep.add_argument("--run", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "preflight":
        preflight(args.manifest.resolve(), args.output.resolve())
    else:
        replay(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
