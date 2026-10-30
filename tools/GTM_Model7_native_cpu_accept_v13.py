"""Verify frozen v13 CPU/GPU decisions under the explicit numerical amendment.

This preserves the original trainer and its failed 1e-6 replay. It writes replay
receipts only after both arms pass all source, score, mask and choice checks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_native_transfer_v13 as study

AMENDMENT = study.ROOT / "docs/GTM_Model7_native_replay_amendment_v13_20261026.md"
TOLERANCE = 5e-5


def choice(rows: list[dict]) -> tuple[dict, int]:
    eligible = [row for row in rows if row["selection_eligible"]]
    selected = max(eligible, key=study.rank) if eligible else next(row for row in rows if row["step"] == 1200 and row["cutoff"] == .7)
    return selected, len(eligible)


def source_check(items: list[dict]) -> dict:
    rows = study.manifest()
    by_id = {row["id"]: row for row in rows}
    if set(by_id) != {item["id"] for item in items}:
        raise ValueError("Original source and native cache inventory differ")
    checked_paths = {}
    original_arrays = set()
    for item in items:
        row = by_id[item["id"]]
        if row["array_sha256"] != item["source_array_sha256"]:
            raise ValueError("Original source array identity differs")
        original_path = (study.SOURCE.parent / row["array_file"]).resolve()
        if original_path not in original_arrays:
            if study.sha(original_path) != row["array_sha256"]:
                raise ValueError("Original source array changed: " + row["id"])
            original_arrays.add(original_path)
        for source in item["sources"]:
            path = Path(source["path"])
            if path in checked_paths:
                if checked_paths[path] != source["sha256"]:
                    raise ValueError("Source path reused with different hashes")
            else:
                study.source_path(source)
                checked_paths[path] = source["sha256"]
    study.verify_all_cache(items)
    return {"original_manifest_sha256": study.sha(study.SOURCE), "source_tiffs_verified": len(checked_paths),
            "original_arrays_verified": len(original_arrays), "native_caches_verified": len(items),
            "index_sha256": study.sha(study.OUT / "data/index.json"),
            "data_receipt_sha256": study.sha(study.OUT / "data/receipt.json")}


def arm_check(arm: str, items: list[dict]) -> dict:
    meta = study.fit_metadata(arm)
    diagnostic_path = study.OUT / f"replay_diagnostic_{arm}/receipt.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (diagnostic["arm"] != arm or diagnostic["trainer_sha256"] != study.sha(Path(study.__file__)) or
        diagnostic["fit_metadata_sha256"] != study.sha(study.OUT / f"fit_{arm}/metadata.json") or
        set(diagnostic["milestones"]) != {str(step) for step in study.MILESTONES}):
        raise ValueError("Complete matching diagnostic required")
    dev = [item for item in items if item["split"] == "dev"]
    if len(dev) != 198:
        raise ValueError("Frozen DEV inventory changed")
    gpu_rows, cpu_rows, result = [], [], {}
    selected_masks = {}
    for step in study.MILESTONES:
        model = study.loaded_model(arm, step, torch.device("cpu"), meta)
        gpu, cpu = {}, {}
        max_error, error_sum, pixels = 0.0, 0.0, 0
        mismatch = {str(cutoff): {"pixels": 0, "scenes": 0, "truth_tp_delta": 0,
                                  "false_positive_delta": 0} for cutoff in study.CUTOFFS}
        repeat_error = 0.0
        with torch.inference_mode():
            for item in dev:
                data = study.scene(item, verify=False)
                x = data["x"].copy()
                if arm == "native_zero_wind":
                    x[13:15] = 0
                tensor = torch.from_numpy(x[None])
                p_cpu = torch.sigmoid(model(tensor))[0, 0].numpy()
                if item is dev[0]:
                    repeat_error = float(np.max(np.abs(p_cpu - torch.sigmoid(model(tensor))[0, 0].numpy())))
                score_path = study.OUT / f"fit_{arm}" / f"dev_step_{step}" / f"{item['id']}.npz"
                expected = meta["milestones"][str(step)]["dev"]["scenes"][item["id"]]
                if study.sha(score_path) != expected["probability_sha256"]:
                    raise ValueError("Saved GPU probability changed")
                with np.load(score_path, allow_pickle=False) as saved:
                    p_gpu, label, valid = saved["probability_native"], saved["target_native"], saved["valid_native"]
                if (p_cpu.shape != (200, 200) or p_gpu.shape != (200, 200) or
                    not np.isfinite(p_cpu).all() or not np.isfinite(p_gpu).all() or
                    np.any(p_cpu < 0) or np.any(p_cpu > 1) or np.any(p_gpu < 0) or np.any(p_gpu > 1)):
                    raise ValueError("Invalid native probabilities")
                if (not np.array_equal(label, data["y"]) or not np.array_equal(valid, data["valid"]) or
                    int(valid.sum()) != expected["valid_native_pixels"] or
                    int((label * valid).sum()) != expected["positive_native_pixels"]):
                    raise ValueError("Native truth/support changed")
                delta = np.abs(p_cpu.astype(np.float64) - p_gpu.astype(np.float64))
                max_error = max(max_error, float(delta.max()))
                error_sum += float(delta.sum())
                pixels += delta.size
                gpu[item["id"]] = {"probability_native": p_gpu, "target_native": label, "valid_native": valid}
                cpu[item["id"]] = {"probability_native": p_cpu, "target_native": label, "valid_native": valid}
                support = valid.astype(bool)
                truth = label.astype(bool)
                for cutoff in study.CUTOFFS:
                    key = str(cutoff)
                    a, b = (p_gpu >= cutoff) & support, (p_cpu >= cutoff) & support
                    flips = int(np.count_nonzero(a != b))
                    if flips:
                        mismatch[key]["pixels"] += flips
                        mismatch[key]["scenes"] += 1
                        mismatch[key]["truth_tp_delta"] += int((b & truth).sum()) - int((a & truth).sum())
                        mismatch[key]["false_positive_delta"] += int((b & ~truth).sum()) - int((a & ~truth).sum())
        if max_error > TOLERANCE or repeat_error != 0:
            raise ValueError(f"CPU numerical bound failed {arm} step {step}: {max_error}, repeat {repeat_error}")
        if mismatch != diagnostic["milestones"][str(step)]["cutoff_mask_mismatches"]:
            raise ValueError("Cross-device cutoff mismatches differ from first diagnostic")
        if abs(max_error - diagnostic["milestones"][str(step)]["max_abs_probability_error"]) > 1e-12:
            raise ValueError("Cross-device error differs from first diagnostic")
        gpu_rows.extend(study.measure(gpu, items, cutoff, step) for cutoff in study.CUTOFFS)
        cpu_rows.extend(study.measure(cpu, items, cutoff, step) for cutoff in study.CUTOFFS)
        selected_masks[step] = (gpu, cpu)
        result[str(step)] = {"scene_count": len(dev), "max_abs_probability_native_error": max_error,
                             "mean_abs_probability_native_error": error_sum / pixels,
                             "fresh_cpu_repeat_exact": True, "support_and_native_counts_exact": True,
                             "cutoff_mask_mismatches": mismatch}
    gpu_choice, gpu_eligible = choice(gpu_rows)
    cpu_choice, cpu_eligible = choice(cpu_rows)
    if gpu_eligible != cpu_eligible or (gpu_choice["step"], gpu_choice["cutoff"]) != (cpu_choice["step"], cpu_choice["cutoff"]):
        raise ValueError("GPU/CPU selection or eligibility count changed")
    if gpu_choice["selection_eligible"] != cpu_choice["selection_eligible"] or gpu_choice["useful_target_met"] != cpu_choice["useful_target_met"]:
        raise ValueError("GPU/CPU quality or usefulness decision changed")
    chosen_step, cutoff = gpu_choice["step"], gpu_choice["cutoff"]
    for item in dev:
        g, c = (selected_masks[chosen_step][j][item["id"]] for j in (0, 1))
        support = g["valid_native"].astype(bool)
        if not np.array_equal((g["probability_native"] >= cutoff) & support,
                              (c["probability_native"] >= cutoff) & support):
            raise ValueError("Selected native binary mask changed: " + item["id"])
    return {"arm": arm, "milestones": result,
            "diagnostic_exceeds_original_1e_6_tolerance": any(x["max_abs_probability_native_error"] > 1e-6 for x in result.values()),
            "eligible_rows_gpu": gpu_eligible,
            "eligible_rows_cpu": cpu_eligible, "gpu_selected": gpu_choice,
            "cpu_selected": cpu_choice, "selected_masks_identical_all_198": True,
            "fit_metadata_sha256": study.sha(study.OUT / f"fit_{arm}/metadata.json"),
            "diagnostic_sha256": study.sha(diagnostic_path),
            "checkpoint_sha256": {str(step): meta["milestones"][str(step)]["checkpoint_sha256"] for step in study.MILESTONES}}


def main() -> None:
    study.settings()
    if not AMENDMENT.is_file():
        raise FileNotFoundError(AMENDMENT)
    for arm in study.ARMS:
        if (study.OUT / f"replay_{arm}/receipt.json").exists():
            raise FileExistsError("Accepted replay already exists")
    items = study.index()
    sources = source_check(items)
    checked = {arm: arm_check(arm, items) for arm in study.ARMS}
    gpu_winner = max((arm for arm in study.ARMS if checked[arm]["gpu_selected"]["selection_eligible"]),
                     key=lambda arm: study.rank(checked[arm]["gpu_selected"]), default=None)
    cpu_winner = max((arm for arm in study.ARMS if checked[arm]["cpu_selected"]["selection_eligible"]),
                     key=lambda arm: study.rank(checked[arm]["cpu_selected"]), default=None)
    if gpu_winner != cpu_winner:
        raise ValueError("Cross-arm selected winner changed")
    if any(checked[arm]["gpu_selected"]["useful_target_met"] != checked[arm]["cpu_selected"]["useful_target_met"] for arm in study.ARMS):
        raise ValueError("Cross-arm usefulness decision changed")
    common = {"passed": True, "cross_device_rule": "finite native probabilities; abs CPU-GPU <= 5e-5; selected masks and decisions identical",
              "original_replay_failure_arm": "native_zero_wind", "original_1e_6_rule_superseded": True,
              "gpu_selected_arm": gpu_winner, "cpu_selected_arm": cpu_winner,
              "amendment_sha256": study.sha(AMENDMENT), "verifier_sha256": study.sha(Path(__file__)),
              "trainer_sha256": study.sha(Path(study.__file__)), "protocol_sha256": study.sha(study.PROTOCOL),
              "source_manifest_sha256": study.SOURCE_SHA, "sources": sources,
              "selected_masks_identical_all_198": True, "quality_gates_unchanged": True}
    for arm in study.ARMS:
        value = {**common, **checked[arm]}
        study.save_json(study.OUT / f"replay_{arm}/receipt.json", value)
    print(json.dumps({"passed": True, "winner": gpu_winner, "arms": {arm: {
        "selected": [checked[arm]["gpu_selected"]["step"], checked[arm]["gpu_selected"]["cutoff"]],
        "eligible_rows": checked[arm]["eligible_rows_gpu"],
        "maximum_abs_error": max(x["max_abs_probability_native_error"] for x in checked[arm]["milestones"].values())
    } for arm in study.ARMS}}), flush=True)


if __name__ == "__main__":
    main()
