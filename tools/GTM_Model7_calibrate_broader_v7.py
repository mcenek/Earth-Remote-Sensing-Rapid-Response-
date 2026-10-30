"""Select broader-DEV model and cutoff without opening reserved TEST pixels."""
from __future__ import annotations

import argparse
import json
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np

import GTM_Model7_train_repaired_broader_v7 as t

CUTOFFS = [round(i / 20, 3) for i in range(1, 20)] + [.975, .99, .995]
KEYS = ("tp", "fp", "fn", "tn")


def counts(prob, truth, valid, cutoff):
    hit, label = prob[valid] >= cutoff, truth[valid] == 1
    return {"tp": int((hit & label).sum()), "fp": int((hit & ~label).sum()),
            "fn": int((~hit & label).sum()), "tn": int((~hit & ~label).sum())}


def score(c):
    tp, fp, fn = (c[k] for k in ("tp", "fp", "fn"))
    return {**c, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1)}


def rank(row):
    return (row["mean_positive_site_recall"], row["mean_positive_scene_recall"],
            row["pooled"]["iou"], -row["clear_pixel_fpr"], -row["step"], row["cutoff"])


def measure(probabilities, samples, scenes, cutoff, step):
    pooled = {key: 0 for key in KEYS}
    site_recalls = defaultdict(list)
    scene_recalls = []
    recovered = clear_fp = clear_pixels = activated_clear = positive_scenes = clear_scenes = 0
    scene_counts = {}
    for scene in scenes:
        sid = scene["id"]
        data = samples[sid]
        valid = data["valid_native"].astype(bool)
        probability = probabilities[sid]
        if probability.shape != valid.shape or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise ValueError(f"Malformed DEV score: {sid}")
        c = counts(probability, data["label_native"], valid, cutoff)
        scene_counts[sid] = c
        for key in KEYS:
            pooled[key] += c[key]
        if scene["scene_kind"] == "PLUME":
            positive_scenes += 1
            recall = c["tp"] / max(c["tp"] + c["fn"], 1)
            scene_recalls.append(recall)
            site_recalls[scene["physical_location_id"]].append(recall)
            recovered += recall >= .1
        else:
            clear_scenes += 1
            clear_fp += c["fp"]
            clear_pixels += int(valid.sum())
            activated_clear += c["fp"] > 0
    pooled_score = score(pooled)
    fpr = clear_fp / max(clear_pixels, 1)
    qualified = (pooled_score["precision"] >= .5 and pooled_score["recall"] >= .3 and
                 pooled_score["iou"] >= .2 and fpr <= .001 and recovered >= positive_scenes / 2)
    return {"step": step, "cutoff": cutoff, "pooled": pooled_score,
            "clear_pixel_fpr": fpr, "recovered_positive_scenes": int(recovered),
            "positive_scenes": positive_scenes, "activated_clear_scenes": int(activated_clear),
            "clear_scenes": clear_scenes, "positive_sites": len(site_recalls),
            "mean_positive_site_recall": float(np.mean([np.mean(v) for v in site_recalls.values()])),
            "mean_positive_scene_recall": float(np.mean(scene_recalls)),
            "qualified": bool(qualified), "scene_native_confusion": scene_counts}


def load_probabilities(folder, scenes, samples, expected_records=None):
    scores = {}
    for scene in scenes:
        sid = scene["id"]
        file = folder / f"{sid}.npz"
        if expected_records is not None and t.sha(file) != expected_records[sid]["probability_sha256"]:
            raise ValueError(f"DEV prediction hash changed: {sid}")
        with np.load(file, allow_pickle=False) as z:
            native = z["probability_native"]
            coarse = z["probability20"]
            if (not np.array_equal(z["valid_native"], samples[sid]["valid_native"].astype(bool)) or
                not np.array_equal(native, np.repeat(np.repeat(coarse, t.native_factor(), 0), t.native_factor(), 1))):
                raise ValueError(f"DEV score/support mismatch: {sid}")
            scores[sid] = native.copy()
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reservation", type=Path, required=True)
    p.add_argument("--baseline-freeze", type=Path, required=True)
    p.add_argument("--baseline-inference", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest, reservation = args.manifest.resolve(), args.reservation.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    t.set_branch("mars_repaired")
    base, rows, _ = t.frozen_inputs(manifest, reservation)
    scenes = [scene for scene in rows if scene["split"] == "dev"]
    samples = {scene["id"]: t.load_scene(manifest, scene) for scene in scenes}
    freeze_path = args.baseline_freeze.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    baseline_choice = freeze["models"]["siamese"]
    if baseline_choice.get("threshold", freeze.get("threshold")) != .95:
        raise ValueError("Frozen v5 comparator cutoff must be 0.95")
    if freeze["metadata"]["normalization"] != base["normalization"]:
        raise ValueError("Frozen comparator normalizer changed")
    inference_dir = args.baseline_inference.resolve()
    inference_receipt = json.loads((inference_dir / "receipt.json").read_text(encoding="utf-8"))
    if inference_receipt.get("manifest_sha256") != t.sha(manifest) or inference_receipt.get("split") != "dev":
        raise ValueError("Baseline broader-DEV inference identity differs")
    baseline_record = inference_receipt["models"]["siamese"]
    if baseline_record.get("checkpoint_sha256") != baseline_choice["checkpoint_sha256"]:
        raise ValueError("Baseline DEV predictions were not frozen v5 weights")
    baseline_scores = load_probabilities(inference_dir / "siamese", scenes, samples, baseline_record["scenes"])
    residual_record = inference_receipt["models"]["stable_residual"]
    residual_choice = freeze["models"]["stable_residual"]
    if residual_record.get("checkpoint_sha256") != residual_choice["checkpoint_sha256"]:
        raise ValueError("Retained residual checkpoint changed")
    residual_scores = load_probabilities(inference_dir / "stable_residual", scenes, samples, residual_record["scenes"])
    residual_new_dev = measure(residual_scores, samples, scenes, .95, int(residual_choice["steps"]))
    baseline_step = int(baseline_choice["steps"])
    original = measure(baseline_scores, samples, scenes, .95, baseline_step)
    calibration_rows = [measure(baseline_scores, samples, scenes, cutoff, baseline_step) for cutoff in CUTOFFS]
    candidates = {"baseline_frozen": [original], "baseline_calibration_only": calibration_rows}
    choices = {"baseline_frozen": {"run": freeze.get("run"), "checkpoint": baseline_choice["checkpoint"],
                                   "checkpoint_sha256": baseline_choice["checkpoint_sha256"],
                                   "step": baseline_step, "cutoff": .95, "development": original},
               "baseline_calibration_only": {}}
    for run_arg in args.runs:
        run = run_arg.resolve()
        meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        arm = meta["arm"]
        if arm not in t.ARMS or arm in candidates:
            raise ValueError("Duplicate/unknown candidate arm")
        if (meta["manifest_sha256"] != t.sha(manifest) or meta["reservation_sha256"] != t.sha(reservation) or
            meta["normalization"] != base["normalization"] or meta["trainer_sha256"] != t.sha(t.HERE)):
            raise ValueError("Candidate data/source drift")
        replay = json.loads((run / "e2e_receipt.json").read_text(encoding="utf-8"))
        receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
        if (not replay["passed"] or replay["manifest_sha256"] != meta["manifest_sha256"] or
            t.sha(run / "metadata.json") != receipt["metadata_sha256"]):
            raise ValueError("Completed matching CPU replay required")
        candidate_rows = []
        for step in t.MILESTONES:
            milestone = receipt["models"]["siamese"]["milestones"][str(step)]
            checkpoint = run / "siamese" / f"step_{step}.pt"
            if t.sha(checkpoint) != milestone["checkpoint_sha256"]:
                raise ValueError("Candidate weights changed")
            scores = load_probabilities(run / "siamese" / f"dev_step_{step}", scenes, samples, milestone["dev"]["scenes"])
            candidate_rows.extend(measure(scores, samples, scenes, cutoff, step) for cutoff in CUTOFFS)
        candidates[arm] = candidate_rows
        qualified_rows = [row for row in candidate_rows if row["qualified"]]
        arm_choice = max(qualified_rows or candidate_rows, key=rank)
        selected_checkpoint = run / "siamese" / f"step_{arm_choice['step']}.pt"
        choices[arm] = {"run": str(run), "trainer_sha256": meta["trainer_sha256"],
                        "checkpoint": str(selected_checkpoint),
                        "checkpoint_sha256": t.sha(selected_checkpoint),
                        "step": arm_choice["step"], "cutoff": arm_choice["cutoff"],
                        "development": arm_choice}
    if set(candidates) != {"baseline_frozen", "baseline_calibration_only", *t.ARMS}:
        raise ValueError("Both bounded fits required")
    best_calibration = max((r for r in calibration_rows if r["qualified"]), key=rank, default=None)
    if best_calibration is None:
        best_calibration = max(calibration_rows, key=rank)
    choices["baseline_calibration_only"] = {**choices["baseline_frozen"], "cutoff": best_calibration["cutoff"],
                                            "development": best_calibration, "unchanged_weights": True}
    all_rows = [(name, row) for name, rows in candidates.items() for row in rows]
    eligible = [(name, row) for name, row in all_rows if row["qualified"]]
    selected_name, selected = max(eligible or all_rows, key=lambda item: rank(item[1]))
    promoted = bool(eligible and rank(selected) > rank(original))
    if selected_name in t.ARMS:
        run = Path(choices[selected_name]["run"])
        checkpoint = run / "siamese" / f"step_{selected['step']}.pt"
        choices[selected_name].update(checkpoint=str(checkpoint), checkpoint_sha256=t.sha(checkpoint),
                                      step=selected["step"], cutoff=selected["cutoff"], development=selected)
    output.mkdir(parents=True)
    frozen = {"schema": "GTM_Model7_broader_v7_selection", "manifest": str(manifest),
              "manifest_sha256": t.sha(manifest), "reservation_path": str(reservation),
              "reservation_sha256": t.sha(reservation), "protocol_sha256": t.sha(t.PROTOCOL),
              "calibrator_sha256": t.sha(Path(__file__).resolve()),
              "baseline_freeze_sha256": t.sha(freeze_path),
              "baseline_inference_receipt_sha256": t.sha(inference_dir / "receipt.json"),
              "original_frozen_comparator": original, "selected_name": selected_name,
              "selected": selected, "eligible": bool(eligible), "promoted_for_final_evaluation": promoted,
              "choices": choices, "test_predictions_used_for_selection": False}
    t.write_json(output / "freeze.json", frozen)
    t.write_json(output / "candidates.json", {"cutoffs": CUTOFFS, "models": candidates,
                                                "manifest_sha256": t.sha(manifest), "test_array_access": "none"})
    t.write_json(output / "receipt.json", {"passed": True, "freeze_sha256": t.sha(output / "freeze.json"),
                                             "candidate_rows": {key: len(value) for key, value in candidates.items()},
                                             "test_array_access": "none"})
    # Each visual option has an ordinary inference freeze with the actual
    # trainer that reconstructs its weights. The retained residual receives
    # NEW DEV metrics instead of its historical v5 selection metrics.
    selected_options = {}
    for name in ("baseline_frozen", "baseline_calibration_only", *t.ARMS):
        choice = choices[name]
        if name in t.ARMS:
            run_path = Path(choice["run"])
            variant_meta = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
            variant_receipt = run_path / "receipt.json"
            variant_replay = run_path / "e2e_receipt.json"
        else:
            variant_meta = freeze["metadata"]
            run_path = Path(freeze["run"])
            variant_receipt = run_path / "receipt.json"
            variant_replay = run_path / "e2e_receipt.json"
        siamese = {"checkpoint": choice["checkpoint"], "checkpoint_sha256": choice["checkpoint_sha256"],
                   "steps": choice["step"], "threshold": choice["cutoff"],
                   "qualified_development_gate": choice["development"]["qualified"],
                   "development": choice["development"]}
        residual = {"checkpoint": residual_choice["checkpoint"],
                    "checkpoint_sha256": residual_choice["checkpoint_sha256"],
                    "steps": residual_choice["steps"], "threshold": .95,
                    "qualified_development_gate": residual_new_dev["qualified"],
                    "development": residual_new_dev, "retained_control": True}
        variant = {"schema": "GTM_Model7_repaired_freeze_v1", "selection_kind": "broader_DEV_site_balanced_v7",
                   "run": str(run_path), "metadata": variant_meta,
                   "training_receipt_sha256": t.sha(variant_receipt),
                   "training_replay_sha256": t.sha(variant_replay),
                   "selection": "Broader DEV site-balanced ranking under frozen gates; residual is retained control",
                   "threshold": .5, "models": {"stable_residual": residual, "siamese": siamese},
                   "evaluation_manifest": str(manifest), "evaluation_manifest_sha256": t.sha(manifest),
                   "test_reservation_sha256": t.sha(reservation),
                   "calibration_source_sha256": t.sha(Path(__file__).resolve()),
                   "qualified_development_gate": siamese["qualified_development_gate"],
                   "test_predictions_used_for_selection": False}
        directory = output / "freezes" / name
        directory.mkdir(parents=True)
        t.write_json(directory / "freeze.json", variant)
        selected_options[name] = {"path": str(directory / "freeze.json"),
                                  "sha256": t.sha(directory / "freeze.json")}
    selected_dir = output / "freezes" / "selected"
    selected_dir.mkdir()
    source = Path(selected_options[selected_name]["path"])
    selected_freeze = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    selected_freeze["broader_selection_status"] = {"selected_name": selected_name,
                                                   "eligible": bool(eligible),
                                                   "promoted_for_final_evaluation": promoted}
    t.write_json(selected_dir / "freeze.json", selected_freeze)
    t.write_json(output / "freeze_index.json", {"variants": selected_options,
                                                "selected": {"path": str(selected_dir / "freeze.json"),
                                                             "sha256": t.sha(selected_dir / "freeze.json")}})
    print(json.dumps({"selected": selected_name, "eligible": bool(eligible), "promoted": promoted,
                      "site_recall": selected["mean_positive_site_recall"]}), flush=True)


if __name__ == "__main__":
    main()
