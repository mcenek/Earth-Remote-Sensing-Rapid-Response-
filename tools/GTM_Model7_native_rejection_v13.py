"""Preserve complete rejected v13 choices and GPU fallback maps without acceptance."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_native_transfer_v13 as study
import GTM_Model7_native_cpu_accept_v13 as cross_device


def compact(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "scene_native_confusion"}


def choose(rows: list[dict]) -> tuple[dict, dict, int]:
    eligible = [row for row in rows if row["selection_eligible"]]
    selected = max(eligible, key=study.rank) if eligible else next(row for row in rows if row["step"] == 1200 and row["cutoff"] == .7)
    return selected, max(rows, key=study.rank), len(eligible)


def main() -> None:
    study.settings()
    selection_dir = study.OUT / "selection"
    evaluation_dir = study.OUT / "evaluation"
    if selection_dir.exists() or evaluation_dir.exists():
        raise FileExistsError("Rejected v13 output already exists")
    items = study.index()
    sources = cross_device.source_check(items)
    dev = [item for item in items if item["split"] == "dev"]
    if len(dev) != 198:
        raise ValueError("DEV inventory changed")
    arms = {}
    for arm in study.ARMS:
        meta = study.fit_metadata(arm)
        diag_path = study.OUT / f"replay_diagnostic_{arm}/receipt.json"
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        if (diag["arm"] != arm or diag["fit_metadata_sha256"] != study.sha(study.OUT / f"fit_{arm}/metadata.json") or
            set(diag["milestones"]) != {str(step) for step in study.MILESTONES}):
            raise ValueError("Complete bound CPU diagnostic required")
        gpu_rows, cpu_rows = [], []
        for step in study.MILESTONES:
            model = study.loaded_model(arm, step, torch.device("cpu"), meta)
            gpu = study.saved_predictions(arm, step, items, meta)
            cpu = {}
            max_delta = 0.0
            with torch.inference_mode():
                for item in dev:
                    source = study.scene(item, verify=False)
                    x = source["x"].copy()
                    if arm == "native_zero_wind":
                        x[13:15] = 0
                    p_cpu = torch.sigmoid(model(torch.from_numpy(x[None])))[0, 0].numpy()
                    p_gpu = gpu[item["id"]]["probability_native"]
                    if (p_cpu.shape != (200, 200) or not np.isfinite(p_cpu).all() or
                        np.any(p_cpu < 0) or np.any(p_cpu > 1)):
                        raise ValueError("Invalid CPU probability")
                    if not np.array_equal(gpu[item["id"]]["target_native"], source["y"]) or not np.array_equal(gpu[item["id"]]["valid_native"], source["valid"]):
                        raise ValueError("Native source truth/support changed")
                    max_delta = max(max_delta, float(np.max(np.abs(p_cpu - p_gpu))))
                    cpu[item["id"]] = {"probability_native": p_cpu,
                                       "target_native": source["y"], "valid_native": source["valid"]}
            if abs(max_delta - diag["milestones"][str(step)]["max_abs_probability_error"]) > 1e-12:
                raise ValueError("CPU/GPU diagnostic drift changed")
            gpu_rows.extend(compact(study.measure(gpu, items, cutoff, step)) for cutoff in study.CUTOFFS)
            cpu_rows.extend(compact(study.measure(cpu, items, cutoff, step)) for cutoff in study.CUTOFFS)
        if len(gpu_rows) != 88 or len(cpu_rows) != 88:
            raise ValueError("Incomplete fixed choice grid")
        selected_gpu, best_gpu, eligible_gpu = choose(gpu_rows)
        selected_cpu, best_cpu, eligible_cpu = choose(cpu_rows)
        if (eligible_gpu != 0 or eligible_cpu != 0 or
            (selected_gpu["step"], selected_gpu["cutoff"]) != (1200, .7) or
            (selected_cpu["step"], selected_cpu["cutoff"]) != (1200, .7)):
            raise ValueError("Unexpected scientific choice; rejected freeze needs review")
        gpu_fallback = study.saved_predictions(arm, 1200, items, meta)
        model = study.loaded_model(arm, 1200, torch.device("cpu"), meta)
        with torch.inference_mode():
            for item in dev:
                source = study.scene(item, verify=False)
                x = source["x"].copy()
                if arm == "native_zero_wind":
                    x[13:15] = 0
                p_cpu = torch.sigmoid(model(torch.from_numpy(x[None])))[0, 0].numpy()
                p_gpu = gpu_fallback[item["id"]]["probability_native"]
                support = source["valid"].astype(bool)
                if not np.array_equal((p_cpu >= .7) & support, (p_gpu >= .7) & support):
                    raise ValueError("Fallback binary masks differ")
        arms[arm] = {"gpu_rows": gpu_rows, "cpu_rows": cpu_rows,
                     "frozen_fallback_gpu": selected_gpu, "frozen_fallback_cpu": selected_cpu,
                     "unconstrained_best_gpu_rejected": best_gpu,
                     "unconstrained_best_cpu_rejected": best_cpu,
                     "eligible_rows_gpu": eligible_gpu, "eligible_rows_cpu": eligible_cpu,
                     "fallback_masks_identical_all_198": True,
                     "fit_metadata_sha256": study.sha(study.OUT / f"fit_{arm}/metadata.json"),
                     "diagnostic_sha256": study.sha(diag_path),
                     "checkpoint_sha256": {str(step): meta["milestones"][str(step)]["checkpoint_sha256"] for step in study.MILESTONES},
                     "diagnostic_max_abs_probability_error": max(value["max_abs_probability_error"] for value in diag["milestones"].values()),
                     "fixed_cutoff_mask_mismatches": {step: value["cutoff_mask_mismatches"] for step, value in diag["milestones"].items()}}
    rejection = {"schema_version": 1,
                 "status": "REJECTED: numerical replay FAILED and zero eligible scientific choices",
                 "diagnostic_only": True, "promotion_allowed": False,
                 "numerical_replay_status": "FAILED",
                 "numerical_failure": {"original_1e_6_gate": "native_zero_wind failed",
                     "amended_5e_5_gate": "native_wind step4800 failed (6.243586540222168e-5)",
                     "accepted_replay_receipts_written": False},
                 "selection_kind": "historically exposed original DEV 198; fixed 22 cutoffs x 4 checkpoints",
                 "selected_arm": None, "retain_historical_v11": True,
                 "choices": {arm: {"step": 1200, "cutoff": .7, "eligible": False,
                                    "fallback_failed_diagnostic": True,
                                    "checkpoint_sha256": arms[arm]["checkpoint_sha256"]["1200"]} for arm in study.ARMS},
                 "arms": arms, "source_verification": sources,
                 "hashes": {"source_manifest": study.SOURCE_SHA, "protocol": study.sha(study.PROTOCOL),
                            "amendment": study.sha(cross_device.AMENDMENT),
                            "trainer": study.sha(Path(study.__file__)),
                            "diagnostic_script": study.sha(study.ROOT / "tools/GTM_Model7_native_cpu_verify_v13.py"),
                            "rejection_script": study.sha(Path(__file__)),
                            "schedule": study.sha(study.OUT / "preflight/draw_schedule.json")}}
    selection_dir.mkdir(parents=True)
    rejection_path = selection_dir / "rejection_summary.json"
    study.save_json(rejection_path, rejection)
    evaluation_dir.mkdir(parents=True)
    models = {}
    for arm in study.ARMS:
        folder = evaluation_dir / arm
        folder.mkdir()
        meta = study.fit_metadata(arm)
        predictions = study.saved_predictions(arm, 1200, items, meta)
        scenes = {}
        for item in dev:
            sid = item["id"]
            source_path = study.OUT / f"fit_{arm}/dev_step_1200/{sid}.npz"
            target_path = folder / f"{sid}.npz"
            shutil.copyfile(source_path, target_path)
            data = predictions[sid]
            scenes[sid] = {"probability_sha256": study.sha(target_path),
                           "native_confusion": study.native_confusion(data["probability_native"], data["target_native"], data["valid_native"], .7),
                           "valid_native_pixels": int(data["valid_native"].sum()),
                           "positive_native_pixels": int((data["target_native"] * data["valid_native"]).sum())}
        models[arm] = {"step": 1200, "cutoff": .7, "eligible": False,
                       "diagnostic_only": True, "numerical_replay_status": "FAILED", "scenes": scenes}
    study.save_json(evaluation_dir / "summary.json", {"schema_version": 1, "models": models,
                    "measurements": {arm: arms[arm]["frozen_fallback_gpu"] for arm in study.ARMS},
                    "rejection_summary_sha256": study.sha(rejection_path),
                    "numerical_replay_status": "FAILED", "diagnostic_only": True,
                    "promotion_allowed": False, "retain_historical_v11": True,
                    "source_manifest_sha256": study.SOURCE_SHA,
                    "grid": "200x200 native annotation; Sentinel-2 SWIR physical resolution remains 20m"})
    print(json.dumps({"status": "REJECTED", "selection": str(rejection_path),
                      "evaluation": str(evaluation_dir / "summary.json"),
                      "eligible": {arm: arms[arm]["eligible_rows_gpu"] for arm in study.ARMS}}), flush=True)


if __name__ == "__main__":
    main()
