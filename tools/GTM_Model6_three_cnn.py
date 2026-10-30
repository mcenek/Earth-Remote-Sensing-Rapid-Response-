"""Prepare, train, predict, and publish the staged three-CNN methane experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/GTM_Model6"
sys.path.insert(0, str(RESEARCH))
DEFAULT_CONFIG = RESEARCH / "GTM_Model6_three_cnn_config.json"
STAGES = ("emit", "airborne", "fusion")


#### Runtime-free inspection
def read_config(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("seed", "epochs", "batch_size", "base_channels", "learning_rate", "max_minutes", "tile_size", "halo", "threshold"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return config


def preflight(path):
    from GTM_Model6_three_cnn_data import load_arrays, load_manifest
    manifest = load_manifest(path)
    rows = manifest["observations"]
    counts = {}
    for row in rows:
        key = row["target_kind"] + ":" + row["role"]
        item = counts.setdefault(key, {"observations": 0, "groups": set(), "positive_cells": 0, "negative_cells": 0})
        a = load_arrays(row)
        item["observations"] += 1
        item["groups"].add(row["group"])
        valid = a["target_valid"].astype(bool)
        item["positive_cells"] += int(((a["target"] == 1) & valid).sum())
        item["negative_cells"] += int(((a["target"] == 0) & valid).sum())
    for item in counts.values():
        item["groups"] = sorted(item["groups"])
    stages = {}
    for stage in STAGES:
        kind, role = ("airborne", "branch_fit") if stage == "airborne" else ("emit", "fusion_fit" if stage == "fusion" else "branch_fit")
        required = [kind + ":" + role, kind + ":validation", kind + ":test"]
        if stage == "fusion":
            required += ["airborne:branch_fit", "airborne:validation", "emit:branch_fit"]
        missing = [key for key in required if key not in counts]
        class_gaps = [key for key in required if key in counts and (
            counts[key]["positive_cells"] == 0 or counts[key]["negative_cells"] == 0)]
        stages[stage] = {"data_ready": not missing and not class_gaps, "missing": missing, "class_gaps": class_gaps,
                         "needs_parent_checkpoints": stage == "fusion"}
    return {"schema": "GTM_Model6_three_cnn_preflight_v1", "data_kind": manifest["data_kind"],
            "manifest_sha256": manifest["manifest_sha256"], "cohorts": counts, "stages": stages,
            "software_contract_passed": True,
            "claim": "Synthetic engineering check only" if manifest["data_kind"] == "synthetic" else
                     "Prepared-data checks passed; scientific sufficiency and visual review remain separate",
            "downloads": 0}


def stages(value):
    return list(STAGES) if value == "all" else [value]


def export_one(args, stage):
    from GTM_Model6_three_cnn_export import export_run
    target = args.export_dir / (args.run_dir.name + "_" + stage)
    return export_run(args.manifest, args.run_dir, stage, target, args.bundles_dir if args.publish else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="Write synthetic engineering data, never methane labels")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--seed", type=int, default=41)
    scene_fixture = sub.add_parser("scene-fixture", help="Strip all targets from a synthetic engineering fixture")
    scene_fixture.add_argument("--manifest", type=Path, required=True)
    scene_fixture.add_argument("--output", type=Path, required=True)
    inference = sub.add_parser("infer", help="Predict scenes using only Sentinel arrays and saved model artifacts")
    inference.add_argument("--scenes", type=Path, required=True)
    inference.add_argument("--run-dir", type=Path, required=True)
    inference.add_argument("--stage", choices=STAGES, default="fusion")
    inference.add_argument("--output-dir", type=Path, required=True)
    inference.add_argument("--export-dir", type=Path, default=ROOT / "outputs/GTM_Model6_three_cnn_exports")
    inference.add_argument("--bundles-dir", type=Path, default=ROOT / "outputs/GTM_viewer_bundles")
    inference.add_argument("--publish", action="store_true")
    doctor = sub.add_parser("doctor", help="Check the installed TensorFlow runtime without installing anything")
    doctor.add_argument("--require-gpu", action="store_true")
    check = sub.add_parser("preflight", help="Validate arrays, grids, provenance, and independent splits")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path)
    for name in ("train", "predict", "replay", "export", "pipeline"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--stage", choices=(*STAGES, "all"), default="all")
        if name in ("train", "pipeline"):
            p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
            for key in ("seed", "epochs", "batch_size", "base_channels", "tile_size", "halo"):
                p.add_argument("--" + key.replace("_", "-"), type=int)
            for key in ("learning_rate", "max_minutes", "threshold"):
                p.add_argument("--" + key.replace("_", "-"), type=float)
            p.add_argument("--require-gpu", action="store_true")
        if name == "predict":
            p.add_argument("--roles", nargs="+", choices=("test", "validation", "branch_fit", "fusion_fit"), default=["test"])
        if name in ("export", "pipeline"):
            p.add_argument("--export-dir", type=Path, default=ROOT / "outputs/GTM_Model6_three_cnn_exports")
            p.add_argument("--bundles-dir", type=Path, default=ROOT / "outputs/GTM_viewer_bundles")
            p.add_argument("--publish", action="store_true", help="Atomically publish complete saved outputs to the viewer")
    args = parser.parse_args()
    if args.command == "fixture":
        from GTM_Model6_three_cnn_fixture import write_fixture
        result = {"manifest": str(write_fixture(args.output, seed=args.seed)), "data_kind": "synthetic"}
    elif args.command == "scene-fixture":
        from GTM_Model6_three_cnn_fixture import write_scenes
        result = {"scene_manifest": str(write_scenes(args.manifest, args.output))}
    elif args.command == "infer":
        from GTM_Model6_three_cnn_runtime import infer_scenes
        from GTM_Model6_three_cnn_export import export_inference
        result = {"inference": infer_scenes(args.scenes, args.run_dir, args.stage, args.output_dir)}
        target = args.export_dir / (args.run_dir.name + "_" + args.stage + "_unlabeled")
        result["viewer"] = export_inference(args.scenes, args.run_dir, args.stage, args.output_dir, target,
                                            args.bundles_dir if args.publish else None)
    elif args.command == "preflight":
        result = preflight(args.manifest)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
    elif args.command == "export":
        result = [export_one(args, stage) for stage in stages(args.stage)]
    else:
        from GTM_Model6_three_cnn_runtime import predict_run, runtime_info, train_stage, verify_replay
        if args.command == "doctor":
            result = runtime_info()
            if not result.get("tensorflow"):
                raise ValueError("TensorFlow is unavailable in this Python environment: " + str(result.get("error", "")))
            if args.require_gpu and not result.get("gpus"):
                raise ValueError("No TensorFlow GPU is available. Activate Martin's tf-env before retrying.")
        elif args.command == "predict":
            result = [predict_run(args.manifest, args.run_dir, stage, args.roles) for stage in stages(args.stage)]
        elif args.command == "replay":
            result = [verify_replay(args.manifest, args.run_dir, stage) for stage in stages(args.stage)]
        else:
            preflight(args.manifest)
            runtime = runtime_info()
            if args.require_gpu and not runtime.get("gpus"):
                raise ValueError("No TensorFlow GPU is available. Training was not started.")
            result = []
            for stage in stages(args.stage):
                item = {"stage": stage, "training": train_stage(args.manifest, args.run_dir, stage, read_config(args))}
                if args.command == "pipeline":
                    item["prediction"] = predict_run(args.manifest, args.run_dir, stage, ["test"])
                    item["viewer"] = export_one(args, stage)
                result.append(item)
                print(json.dumps({"completed_stage": stage, "viewer": item.get("viewer")}, allow_nan=False), flush=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(json.dumps({"error": str(error), "command": sys.argv[1:2]}, allow_nan=False), file=sys.stderr)
        sys.exit(2)
