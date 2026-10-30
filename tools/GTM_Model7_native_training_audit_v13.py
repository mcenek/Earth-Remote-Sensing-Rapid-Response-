"""Read-only TRAIN audit of the frozen v13 native transfer experiment."""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_native_transfer_v13 as native

ROOT = Path(__file__).resolve().parents[1]
OUT = native.OUT / "training_audit"
SEED = 20261026
FEATURE_SCENES = 64
BINS = np.linspace(0, 10, 1001)
NEGATIVE_ACCEPTANCE = .0067


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def constant_terms(p: float, valid: int, pos: int, positive_valid: int) -> dict:
    neg = valid - pos
    bce_pos = -2 * pos * math.log(p) / valid
    bce_neg = -neg * math.log1p(-p) / valid
    dice = 1 - (2 * p * pos + 1) / (p * positive_valid + pos + 1)
    # Derivative of the actual scalar objective with respect to one shared logit.
    bce_grad_pos = 2 * pos * (p - 1) / valid
    bce_grad_neg = neg * p / valid
    dice_dp = -((2 * pos) * (p * positive_valid + pos + 1) -
                (2 * p * pos + 1) * positive_valid) / (p * positive_valid + pos + 1) ** 2
    dice_grad = .1 * dice_dp * p * (1 - p)
    return {"p": p, "bce_positive": bce_pos, "bce_negative": bce_neg,
            "bce": bce_pos + bce_neg, "dice": dice, "weighted_dice": .1 * dice,
            "total": bce_pos + bce_neg + .1 * dice,
            "shared_logit_grad_positive_bce": bce_grad_pos,
            "shared_logit_grad_negative_bce": bce_grad_neg,
            "shared_logit_grad_dice": dice_grad,
            "shared_logit_grad_total": bce_grad_pos + bce_grad_neg + dice_grad}


def exact_sample_auc(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float, float]:
    """All positive pixels versus a Bernoulli sample of negative pixels."""
    ordered = np.sort(positive)
    left = np.searchsorted(ordered, negative, side="left")
    right = np.searchsorted(ordered, negative, side="right")
    per_negative = (len(ordered) - right + .5 * (right - left)) / len(ordered)
    return (float(per_negative.mean()), float(per_negative.std(ddof=1) / np.sqrt(len(negative))),
            float(((right - left) / len(ordered)).mean()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = native.index()
    train = [r for r in items if r["split"] == "train"]
    assert len(train) == 958 and collections.Counter(r["scene_kind"] for r in train) == {"PLUME": 268, "NO_PLUME": 690}
    for row in train:
        sources = {s["role"]: s["path"] for s in row["sources"]}
        assert sources["target_image"] != sources["reference_image"]
        assert sources["target_cloud"] != sources["reference_cloud"] or sources["target_image"] != sources["reference_image"]
    source_manifest = native.SOURCE
    assert sha(source_manifest) == native.SOURCE_SHA
    metadata = {r["id"]: r for r in json.loads(source_manifest.read_text(encoding="utf-8"))["scenes"] if r["split"] == "train"}
    date_checks = collections.Counter()
    for row in train:
        meta = metadata[row["id"]]
        target_time = datetime.datetime.fromisoformat(meta["satellite_time"])
        reference_time = datetime.datetime.fromisoformat(meta["reference_time"])
        hours = (target_time - reference_time).total_seconds() / 3600
        assert abs(hours - float(meta["target_minus_reference_hours"])) < .01
        assert meta["reference_observation_id"] in next(s["path"] for s in row["sources"] if s["role"] == "reference_image")
        assert row["id"].removeprefix("MARS_") in next(s["path"] for s in row["sources"] if s["role"] == "target_image")
        date_checks["reference_before_target" if hours > 0 else "reference_after_target" if hours < 0 else "same_time"] += 1
    schedule_path = native.OUT / "preflight/draw_schedule.json"
    draws = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert len(draws) == 4800 and all(len(d) == 8 for d in draws)
    assert all(all(items[a["index"]]["split"] == "train" for a in d) for d in draws)
    counts = collections.Counter(a["index"] for draw in draws for a in draw)
    assert all(sum(items[a["index"]]["scene_kind"] == "PLUME" for a in d) == 4 for d in draws)
    pooled_valid = sum(r["valid_pixels"] for r in train)
    pooled_pos = sum(r["label_pixels"] for r in train)
    scheduled_valid = sum(counts[i] * r["valid_pixels"] for i, r in enumerate(items) if r["split"] == "train")
    scheduled_pos = sum(counts[i] * r["label_pixels"] for i, r in enumerate(items) if r["split"] == "train")
    scheduled_positive_valid = sum(counts[i] * r["valid_pixels"] for i, r in enumerate(items) if r["split"] == "train" and r["scene_kind"] == "PLUME")

    # Every TRAIN scene contributes its supported pixels to the fixed 0.01-wide histogram.
    positive_hist = np.zeros(1000, np.int64)
    background_hist = np.zeros(1000, np.int64)
    rng = np.random.default_rng(SEED)
    positive_values = []
    sampled_background_values = []
    feature_ids = set(rng.choice([r["id"] for r in train], FEATURE_SCENES, replace=False).tolist())
    channel_samples = [[] for _ in range(16)]
    invalid_nonzero = np.zeros(16, np.int64)
    target_ref_checks = []
    for row in train:
        data = native.scene(row)
        x, y, valid = data["x"], data["y"].astype(bool), data["valid"].astype(bool)
        assert x.shape == (16, 200, 200) and np.isfinite(x).all()
        assert int(valid.sum()) == row["valid_pixels"] and int((y & valid).sum()) == row["label_pixels"]
        assert (not y.any()) if row["scene_kind"] == "NO_PLUME" else bool((y & valid).any())
        positive_hist += np.histogram(x[0, y & valid], BINS)[0]
        background_hist += np.histogram(x[0, ~y & valid], BINS)[0]
        positive_values.append(x[0, y & valid])
        background = x[0, ~y & valid]
        sampled_background_values.append(background[rng.random(len(background)) < NEGATIVE_ACCEPTANCE])
        if row["id"] in feature_ids:
            # Fixed 1,024 pixel positions per sampled scene keep memory bounded.
            pixel_rng = np.random.default_rng(SEED ^ int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16))
            sample = pixel_rng.choice(np.flatnonzero(valid), min(1024, int(valid.sum())), replace=False)
            for c in range(16):
                channel_samples[c].append(x[c].reshape(-1)[sample])
                invalid_nonzero[c] += int(np.count_nonzero(x[c, ~valid]))
            target_ref_checks.append({"id": row["id"], "target_minus_reference_B12_mean": float((x[6, valid] - x[12, valid]).mean()),
                                      "wind_raw": row["wind_raw"], "wind_cache": data["wind"].tolist(),
                                      "wind_channels_equal_cache": bool(np.all(x[13] == data["wind"][0]) and np.all(x[14] == data["wind"][1]))})

    all_positive = np.concatenate(positive_values)
    sampled_background = np.concatenate(sampled_background_values)
    assert len(all_positive) == pooled_pos
    auc, auc_se, tie_fraction = exact_sample_auc(all_positive, sampled_background)
    scales = []
    names = ["MBMP", *["target_" + b for b in native.BANDS], *["reference_" + b for b in native.BANDS], "wind_u", "wind_v", "cloud"]
    for name, chunks in zip(names, channel_samples):
        a = np.concatenate(chunks)
        scales.append({"channel": name, "min": float(a.min()), "p01": float(np.quantile(a, .01)),
                       "median": float(np.median(a)), "p99": float(np.quantile(a, .99)), "max": float(a.max())})

    # Actual loss is measured only on the first frozen TRAIN draw and an existing checkpoint.
    first = draws[0]
    fv = sum(items[d["index"]]["valid_pixels"] for d in first)
    fp = sum(items[d["index"]]["label_pixels"] for d in first)
    fvp = sum(items[d["index"]]["valid_pixels"] for d in first if items[d["index"]]["scene_kind"] == "PLUME")
    grid = [constant_terms(float(p), fv, fp, fvp) for p in np.linspace(.001, .999, 999)]
    baseline = min(grid, key=lambda r: r["total"])
    checkpoint = native.OUT / "fit_native_zero_wind/step_1200.pt"
    actual = None
    constant_torch_verification = None
    if checkpoint.is_file():
        torch.set_num_threads(4)
        xx, yy, vv, pp = native.batch(items, first, "native_zero_wind", torch.device("cpu"))
        fixed_logits = torch.zeros_like(yy, requires_grad=True)
        fixed_loss = native.loss(fixed_logits, yy, vv, pp)
        fixed_loss.backward()
        half_terms = constant_terms(.5, fv, fp, fvp)
        fixed_grad = float(fixed_logits.grad.sum())
        assert abs(float(fixed_loss.detach()) - half_terms["total"]) < 1e-6
        assert abs(fixed_grad - half_terms["shared_logit_grad_total"]) < 1e-6
        constant_torch_verification = {"loss": float(fixed_loss.detach()), "shared_logit_gradient": fixed_grad}
        model = native.NativeUNet().float()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["step"] == 1200 and payload["arm"] == "native_zero_wind"
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(xx)
            actual = {"loss": float(native.loss(logits, yy, vv, pp)), "prob_mean_valid": float((torch.sigmoid(logits) * vv).sum() / vv.sum()),
                      "prob_mean_positive": float((torch.sigmoid(logits) * yy * vv).sum() / (yy * vv).sum())}

    v11_sources = [ROOT / "tools/GTM_Model7_train_repaired_mars_log_v3.py", ROOT / "tools/GTM_Model7_train_repaired_centered_v5.py",
                   ROOT / "tools/GTM_Model7_train_repaired_broader_v7.py", ROOT / "tools/GTM_Model7_train_repaired_site_v8.py",
                   ROOT / "tools/GTM_Model7_train_repaired_hard_v11.py"]
    receipt_files = [Path(__file__), Path(native.__file__), ROOT / "EarthRemoteSensingRapidResponse/mars_s2l_adapter.py",
                     native.OUT / "data/index.json", native.OUT / "data/receipt.json", source_manifest, schedule_path, *v11_sources]
    if checkpoint.is_file():
        receipt_files.append(checkpoint)
    result = {"scope": "TRAIN native caches only; checkpoint inference on first TRAIN batch; no optimizer update",
              "inventory": {"scenes": len(train), "positive_scenes": 268, "clear_scenes": 690,
                            "date_relation_counts": dict(date_checks),
                            "pooled_valid": pooled_valid, "pooled_positive": pooled_pos, "pooled_positive_fraction": pooled_pos / pooled_valid,
                            "scheduled_valid": scheduled_valid, "scheduled_positive": scheduled_pos,
                            "scheduled_positive_fraction": scheduled_pos / scheduled_valid,
                            "scheduled_positive_scene_valid": scheduled_positive_valid,
                            "scheduled_positive_fraction_within_positive_scenes": scheduled_pos / scheduled_positive_valid},
              "first_batch": {"valid": fv, "positive": fp, "positive_fraction": fp / fv,
                              "positive_scene_valid": fvp, "at_p_half": constant_terms(.5, fv, fp, fvp),
                              "at_p_half_torch_verification": constant_torch_verification,
                              "best_constant_grid_0.001": baseline, "actual_step1200_eval_mode": actual},
              "mbmp_train_histogram": {"edges": [0, 10, .01], "positive_counts": positive_hist.tolist(),
                                       "background_counts": background_hist.tolist(),
                                       "auc_higher_is_positive_exact_sample": auc,
                                       "auc_lower_is_positive_exact_sample": 1 - auc,
                                       "auc_standard_error_background_sampling_approx": auc_se,
                                       "auc_tie_pair_fraction_exact_sample": tie_fraction,
                                       "background_acceptance_probability": NEGATIVE_ACCEPTANCE,
                                       "background_sample_seed": SEED,
                                       "background_sample_size": len(sampled_background),
                                       "all_positive_size": len(all_positive),
                                       "positive_mean_bin_center": float((positive_hist * (BINS[:-1] + .005)).sum() / positive_hist.sum()),
                                       "background_mean_bin_center": float((background_hist * (BINS[:-1] + .005)).sum() / background_hist.sum())},
              "feature_sample": {"seed": SEED, "scenes_without_replacement": FEATURE_SCENES,
                                 "pixels_per_scene": 1024, "channels": scales, "invalid_nonzero_in_sampled_scenes": invalid_nonzero.tolist(),
                                 "target_reference_checks": target_ref_checks},
              "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in receipt_files}}
    native.save_json(OUT / "audit.json", result)
    at_half = constant_terms(.5, fv, fp, fvp)
    md = f"""# Native v13 TRAIN signal audit

Read-only audit of {len(train)} TRAIN caches and the first fixed TRAIN batch. Source hashes and full histograms are in `audit.json`.

- Native TRAIN plume support: {pooled_pos:,}/{pooled_valid:,} valid pixels = {pooled_pos/pooled_valid:.4%}. Across all 4,800 frozen balanced draws: {scheduled_pos:,}/{scheduled_valid:,} = {scheduled_pos/scheduled_valid:.4%}. Four plume scenes per batch does not imply balanced plume pixels.
- First batch: {fp:,}/{fv:,} = {fp/fv:.4%}. At constant p=0.5, positive weighted BCE {at_half['bce_positive']:.5f}, background BCE {at_half['bce_negative']:.5f}; their shared-logit gradient contributions are {at_half['shared_logit_grad_positive_bce']:.5f} and {at_half['shared_logit_grad_negative_bce']:.5f}. Dice gradient contributes {at_half['shared_logit_grad_dice']:.5f}. The positive total gradient {at_half['shared_logit_grad_total']:.5f} pushes shared logits down under gradient descent.
- First-batch constant grid optimum: p={baseline['p']:.3f}, loss {baseline['total']:.5f} (BCE {baseline['bce']:.5f} + weighted Dice {baseline['weighted_dice']:.5f}). Existing zero-wind step-1200 checkpoint eval-mode loss: {actual['loss'] if actual else 'unavailable'}. This is a diagnostic comparison, not a train-mode replay.
- Native MBMP TRAIN supported-pixel AUC for **higher MBMP = plume** is {auc:.4f}; for **lower MBMP = plume** it is {1-auc:.4f}. This compares all {len(all_positive):,} plume pixels against {len(sampled_background):,} background pixels selected independently with probability {NEGATIVE_ACCEPTANCE} and seed {SEED}; exact ranks give half credit to ties. Approximate background-sampling standard error is {auc_se:.4f} (rough 95% margin ±{1.96*auc_se:.4f}); spatial dependence makes that an optimistic uncertainty measure. The inverted direction is expected for methane absorption reducing target B12/B11. The U-Net receives the raw channel and can learn either sign. This tests one-channel separability, not model capacity or generalization.
- Channel order in frozen source and sampled cache is MBMP, target B02/B03/B04/B08/B11/B12, reference same six, wind u/v, cloud. Source computes MBMP as target normalized B12/B11 ratio divided by reference ratio; lower values indicate a target-side B12/B11 decrease. Sampled wind channels match cached wind. The TRAIN manifest's {date_checks['reference_before_target']} earlier and {date_checks['reference_after_target']} later references agree with its signed time differences and source observation IDs. The frozen producer reconstruction checks date arrays against the source cache. This audit did not reopen external imagery.
- v11 used `log1p(DN)`, per-date valid-pixel median centering, then TRAIN center/scale normalization clipped to [-8,8] at 20 m. v13 uses raw `DN/5000` clipped to [0,2] at native 10 m, with MBMP [0,10] and zero-filled wind for the zero-wind arm. Across 64 fixed sampled TRAIN scenes (up to 1,024 valid pixels each), MBMP p01/median/p99 is {scales[0]['p01']:.3f}/{scales[0]['median']:.3f}/{scales[0]['p99']:.3f}, target B12 is {scales[6]['p01']:.3f}/{scales[6]['median']:.3f}/{scales[6]['p99']:.3f}, and reference B12 is {scales[12]['p01']:.3f}/{scales[12]['median']:.3f}/{scales[12]['p99']:.3f}. All 16 channel quantiles are in `audit.json`. Raw date bands and wind can remain nonzero outside valid support; the loss masks these pixels, while convolutions and BatchNorm still see them. v11 additionally provided current minus reference, absolute difference and two spectral contrasts in its spectral arm. These are preprocessing differences and possible optimization hypotheses, not proven code bugs.

The arithmetic demonstrates strong background pressure in the actual objective. It cannot establish that this alone caused missed plumes, or that a deeper model would fix them. Next controlled work would compare TRAIN loss/gradients over multiple fixed draws, inspect positive-logit distributions over TRAIN, and test a changed loss or normalized features in a separate protocol after the frozen fits finish.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps({"report": str(OUT / "REPORT.md"), "pooled_fraction": pooled_pos / pooled_valid,
                      "scheduled_fraction": scheduled_pos / scheduled_valid, "mbmp_auc": auc,
                      "constant_loss": baseline["total"], "actual_loss": actual["loss"] if actual else None}))


if __name__ == "__main__":
    main()
