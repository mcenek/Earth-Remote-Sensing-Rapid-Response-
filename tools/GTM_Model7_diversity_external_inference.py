"""Run frozen DEV-selected models on a separate, independently replayed external manifest."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA = "b7d44b43e113baa98d61353750a8721d8297ab6a9d94fd6e74ca0ae50da75836"
INFERENCE_PATH = ROOT / "tools/GTM_Model7_repaired_inference.py"
DEFAULT_MANIFEST = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/external_data/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/external_inference"

spec = importlib.util.spec_from_file_location("frozen_repaired_inference", INFERENCE_PATH)
inference = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(inference)


def run(freeze_path: Path, eval_manifest_path: Path, output: Path, verify: bool) -> dict:
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    meta = frozen["metadata"]
    training_manifest = Path(meta["manifest"])
    inference.check_implementation(meta)
    t = inference.t
    if meta["manifest_sha256"] != TRAIN_SHA or t.sha(training_manifest) != TRAIN_SHA:
        raise ValueError("Original combined training manifest changed")
    t.ACTIVE_BRANCH = meta["config"]["branch"]
    evaluation = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    evaluation_sha = t.sha(eval_manifest_path)
    allowed_roles = {"metadata_frozen_external_evaluation", "metadata_reserved_fresh_site_final_evaluation"}
    if evaluation["training_manifest"]["sha256"] != TRAIN_SHA or evaluation["role"] not in allowed_roles:
        raise ValueError("Evaluation manifest does not belong to frozen training cohort")
    e2e_path = eval_manifest_path.parent / "e2e_receipt.json"
    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))
    if not e2e["passed"] or e2e["evaluation_manifest_sha256"] != evaluation_sha:
        raise ValueError("Matching independent source reconstruction is required")
    if evaluation['role'] == 'metadata_reserved_fresh_site_final_evaluation':
        pinned = evaluation['freeze']
        baseline = ROOT / 'outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration/freeze.json'
        if (t.sha(Path(pinned['path'])) != pinned['sha256'] or
                e2e['freeze_sha256'] != pinned['sha256'] or
                t.sha(freeze_path) not in {pinned['sha256'], t.sha(baseline)}):
            raise ValueError('Final cohort accepts only the pre-pixel selected freeze and frozen v5 baseline')
    scenes = evaluation["scenes"]
    if not scenes or any(s["split"] != "test" for s in scenes):
        raise ValueError("External evaluation test split empty or mixed")
    for scene in scenes:
        if t.sha(eval_manifest_path.parent / scene["array_file"]) != scene["array_sha256"]:
            raise ValueError("External array identity failed: " + scene["id"])
    freeze_sha = t.sha(freeze_path)
    if not verify:
        output.mkdir(parents=True, exist_ok=False)
        receipt = {"schema": "GTM_Model7_diversity_external_inference_v1",
                   "freeze_sha256": freeze_sha, "training_manifest_sha256": TRAIN_SHA,
                   "manifest_sha256": evaluation_sha, "evaluation_manifest_sha256": evaluation_sha,
                   "source_replay_sha256": t.sha(e2e_path), "split": "test", "models": {}}
    else:
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        if (receipt["freeze_sha256"] != freeze_sha or receipt["manifest_sha256"] != evaluation_sha or
                receipt["training_manifest_sha256"] != TRAIN_SHA or receipt["source_replay_sha256"] != t.sha(e2e_path)):
            raise ValueError("Saved external inference identity differs")
    maximum = 0.0
    norm = meta["normalization"]
    for key, choice in frozen["models"].items():
        t.THRESHOLD = choice.get("threshold", frozen["threshold"])
        checkpoint = Path(choice["checkpoint"])
        if t.sha(checkpoint) != choice["checkpoint_sha256"]:
            raise ValueError("DEV-selected checkpoint changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["normalization"] != norm or saved["steps"] != choice["steps"]:
            raise ValueError("Checkpoint training normalization or selected step changed")
        model = t.new_model(key)
        model.load_state_dict(saved["model"])
        folder = output / key
        if not verify:
            measured = t.evaluate(model, eval_manifest_path, scenes, norm, "test", folder)
            receipt["models"][key] = dict(**measured, checkpoint=str(checkpoint), threshold=t.THRESHOLD,
                                           checkpoint_sha256=choice["checkpoint_sha256"],
                                           qualified_development_gate=choice["qualified_development_gate"])
        else:
            for scene in scenes:
                data = t.load_scene(eval_manifest_path, scene)
                p = t.predict(model, t.prepare(data, norm))
                file = folder / (scene["id"] + ".npz")
                expected = receipt["models"][key]["scenes"][scene["id"]]
                if t.sha(file) != expected["probability_sha256"]:
                    raise ValueError("Saved external prediction changed")
                with np.load(file, allow_pickle=False) as z:
                    error = float(np.max(np.abs(p - z["probability20"])))
                    maximum = max(maximum, error)
                    if error > t.REPLAY_ATOL or not np.array_equal(z["valid20"], data["valid"]):
                        raise ValueError("External prediction replay differs")
                    native = np.repeat(np.repeat(p, t.native_factor(), 0), t.native_factor(), 1)
                    if t.confusion(native, data["label_native"], data["valid_native"]) != expected["native_confusion"]:
                        raise ValueError("External native confusion differs")
        print(json.dumps({"model": key, "scenes": len(scenes), "replay": verify}), flush=True)
    if verify:
        replay = {"schema": "GTM_Model7_diversity_external_prediction_replay_v1", "passed": True,
                  "fresh_process": True, "scenes_per_model": len(scenes),
                  "receipt_sha256": t.sha(output / "receipt.json"), "freeze_sha256": freeze_sha,
                  "evaluation_manifest_sha256": evaluation_sha, "training_manifest_sha256": TRAIN_SHA,
                  "maximum_probability_error": maximum}
        t.write_json(output / "replay.json", replay)
        return replay
    t.write_json(output / "receipt.json", receipt)
    return receipt


def summarize(eval_manifest_path: Path, output: Path) -> dict:
    evaluation = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    replay = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    if (receipt["manifest_sha256"] != inference.t.sha(eval_manifest_path) or not replay["passed"] or
            replay["receipt_sha256"] != inference.t.sha(output / "receipt.json")):
        raise ValueError("External summary requires matching verified predictions")
    result = {"schema": "GTM_Model7_diversity_external_summary_v1", "diagnostic_only": True,
              "evaluation_role": evaluation['role'],
              "fresh_project_evaluation_at_first_inference": evaluation['role'] == 'metadata_reserved_fresh_site_final_evaluation',
              "evaluation_manifest_sha256": receipt["manifest_sha256"],
              "training_manifest_sha256": receipt["training_manifest_sha256"],
              "freeze_sha256": receipt["freeze_sha256"], "models": {}}
    scene_by_id = {s["id"]: s for s in evaluation["scenes"]}
    for key, model in receipt["models"].items():
        sites = collections.defaultdict(list)
        negative_activated = 0
        for sid, item in model["scenes"].items():
            scene = scene_by_id[sid]
            sites[scene["physical_location_id"]].append((scene, item))
            negative_activated += scene["scene_kind"] == "NO_PLUME" and item["activated"]
        site_records = {}
        for site, items in sites.items():
            counts = {part: {c: 0 for c in ("tp", "fp", "fn", "tn")}
                      for part in ("all", "positive", "negative")}
            for scene, item in items:
                part = "positive" if scene["scene_kind"] == "PLUME" else "negative"
                for c, value in item["native_confusion"].items():
                    counts["all"][c] += value
                    counts[part][c] += value
            site_records[site] = {"country": items[0][0]["country"],
                                  "positive_scenes": sum(s["scene_kind"] == "PLUME" for s, _ in items),
                                  "negative_scenes": sum(s["scene_kind"] == "NO_PLUME" for s, _ in items),
                                  "native_all": inference.t.metrics(counts["all"]),
                                  "native_positive": inference.t.metrics(counts["positive"]),
                                  "native_negative": inference.t.metrics(counts["negative"])}
        result["models"][key] = {"threshold": model["threshold"],
                                 "qualified_development_gate": model["qualified_development_gate"],
                                 "native_pooled": model["native_pooled"],
                                 "native_positive": model["native_positive"],
                                 "native_negative": model["native_negative"],
                                 "clear_scenes_activated": negative_activated,
                                 "clear_scenes_total": model["negative_scenes"],
                                 "sites": site_records}
    inference.t.write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        result = summarize(args.evaluation_manifest.resolve(), args.output.resolve())
        print(json.dumps({k: {"clear_scenes_activated": v["clear_scenes_activated"],
                              "clear_scenes_total": v["clear_scenes_total"]}
                          for k, v in result["models"].items()}, indent=2))
    else:
        result = run(args.freeze.resolve(), args.evaluation_manifest.resolve(), args.output.resolve(), args.verify)
        print(json.dumps({k: v for k, v in result.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
