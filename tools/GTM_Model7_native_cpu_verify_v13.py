"""Read-only cross-device diagnostic for the frozen v13 native checkpoints.

This does not amend the trainer's failed 1e-6 replay gate or create an accepted
replay receipt. It measures every saved DEV probability and every fixed cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_native_transfer_v13 as study


def run(arm: str) -> dict:
    study.settings()
    meta = study.fit_metadata(arm)
    items = study.index()
    study.verify_all_cache(items)
    scenes = [item for item in items if item["split"] == "dev"]
    if len(scenes) != 198:
        raise ValueError("DEV inventory changed")
    result = {}
    for step in study.MILESTONES:
        model = study.loaded_model(arm, step, torch.device("cpu"), meta)
        maximum, absolute_sum, pixel_count = 0.0, 0.0, 0
        changed_probabilities, exact_support, exact_labels = 0, True, True
        mismatch = {str(cutoff): {"pixels": 0, "scenes": 0, "truth_tp_delta": 0,
                                  "false_positive_delta": 0} for cutoff in study.CUTOFFS}
        repeat_delta = 0.0
        with torch.inference_mode():
            for item in scenes:
                source = study.scene(item, verify=False)
                inputs = source["x"].copy()
                if arm == "native_zero_wind":
                    inputs[13:15] = 0
                tensor = torch.from_numpy(inputs[None])
                fresh = torch.sigmoid(model(tensor))[0, 0].numpy()
                if item is scenes[0]:
                    repeat_delta = max(repeat_delta, float(np.max(np.abs(fresh - torch.sigmoid(model(tensor))[0, 0].numpy()))))
                path = study.OUT / f"fit_{arm}" / f"dev_step_{step}" / f"{item['id']}.npz"
                expected = meta["milestones"][str(step)]["dev"]["scenes"][item["id"]]
                if study.sha(path) != expected["probability_sha256"]:
                    raise ValueError("GPU score artifact changed: " + item["id"])
                with np.load(path, allow_pickle=False) as z:
                    gpu = z["probability_native"]
                    saved_support = z["valid_native"]
                    saved_label = z["target_native"]
                if gpu.shape != (200, 200) or fresh.shape != (200, 200):
                    raise ValueError("Native probability shape changed")
                support = source["valid"].astype(bool)
                label = source["y"].astype(bool)
                exact_support &= np.array_equal(saved_support.astype(bool), support)
                exact_labels &= np.array_equal(saved_label.astype(bool), label)
                exact_support &= int(support.sum()) == expected["valid_native_pixels"]
                exact_labels &= int((support & label).sum()) == expected["positive_native_pixels"]
                delta = np.abs(fresh.astype(np.float64) - gpu.astype(np.float64))
                maximum = max(maximum, float(delta.max()))
                absolute_sum += float(delta.sum())
                pixel_count += delta.size
                changed_probabilities += int(np.any(delta > 0))
                for cutoff in study.CUTOFFS:
                    key = str(cutoff)
                    gpu_mask = (gpu >= cutoff) & support
                    cpu_mask = (fresh >= cutoff) & support
                    difference = gpu_mask != cpu_mask
                    n = int(difference.sum())
                    if n:
                        mismatch[key]["pixels"] += n
                        mismatch[key]["scenes"] += 1
                        mismatch[key]["truth_tp_delta"] += int((cpu_mask & label).sum()) - int((gpu_mask & label).sum())
                        mismatch[key]["false_positive_delta"] += int((cpu_mask & ~label).sum()) - int((gpu_mask & ~label).sum())
        result[str(step)] = {"scene_count": len(scenes), "max_abs_probability_error": maximum,
                             "mean_abs_probability_error": absolute_sum / pixel_count,
                             "changed_probability_scenes": changed_probabilities,
                             "exact_native_support": exact_support, "exact_native_labels": exact_labels,
                             "fresh_cpu_repeat_max_abs_delta": repeat_delta,
                             "cutoff_mask_mismatches": mismatch}
    return {"scope": "diagnostic only; original 1e-6 replay gate failed; not an accepted replay receipt",
            "arm": arm, "source_manifest_sha256": study.SOURCE_SHA,
            "protocol_sha256": study.sha(study.PROTOCOL),
            "trainer_sha256": study.sha(Path(study.__file__)),
            "fit_metadata_sha256": study.sha(study.OUT / f"fit_{arm}/metadata.json"),
            "cutoffs": list(study.CUTOFFS), "milestones": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=study.ARMS)
    args = parser.parse_args()
    output = study.OUT / f"replay_diagnostic_{args.arm}/receipt.json"
    if output.exists():
        raise FileExistsError("Diagnostic is immutable: " + str(output))
    report = run(args.arm)
    study.save_json(output, report)
    print(json.dumps({"arm": args.arm, "milestones": {step: {
        "max_abs_probability_error": entry["max_abs_probability_error"],
        "total_cutoff_mismatch_pixels": sum(c["pixels"] for c in entry["cutoff_mask_mismatches"].values())
    } for step, entry in report["milestones"].items()}}), flush=True)


if __name__ == "__main__":
    main()
