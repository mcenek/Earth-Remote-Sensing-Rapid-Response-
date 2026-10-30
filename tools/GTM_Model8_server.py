#!/usr/bin/env python3
"""Portable Linux entry point for the frozen MARS baseline and its reviews.

Doctor and commands use only the standard library and never import Torch.
The frozen trainer and reviewer remain the authorities for data and model checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


PLAN_SHA256 = "3b29c62bb250dd9155b1c2af90637b605f2709f7ec6c64b18662a0c9ed2a6345"
MANIFEST_SHA256 = "59e9843ac329e6da81c5dbbebeb22cde5eaf170ee4a8a5a9c002e3afbfdc68d9"
RUN_NAME = "GTM_Model8_mars_baseline_20261030"
MODULES = ("numpy", "torch", "torchvision", "sklearn", "rasterio", "pandas",
           "shapely", "segmentation_models_pytorch", "marshsi", "georeader",
           "cloudsen12_models", "ee", "huggingface_hub")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_constants(script: Path) -> dict:
    """Read source pins without importing the GPU trainer."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    wanted = {"SOURCE_HASHES", "SOURCE_REVISION", "METADATA_REVISION"}
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise ValueError("Frozen trainer source pins are missing")
    return values


def paths(args: argparse.Namespace) -> dict[str, Path]:
    code = args.code_root.resolve()
    return {
        "code": code,
        "data": args.data_root.resolve(),
        "source": args.source_root.resolve(),
        "outputs": args.output_root.resolve(),
        "expected_data": code / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L",
        "expected_source": code / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/marss2l-code",
        "expected_outputs": code / "outputs",
        "plan": code / "tool-diagnostics/mars-baseline-20261030",
    }


def doctor(args: argparse.Namespace) -> dict:
    p = paths(args)
    missing = []
    checks = {}

    def check(name: str, okay: bool, detail: str) -> None:
        checks[name] = {"ok": bool(okay), "detail": detail}
        if not okay:
            missing.append(name)

    check("linux", sys.platform.startswith("linux"), sys.platform)
    check("code_root", p["code"] == Path(__file__).resolve().parents[1], str(p["code"]))
    for name, expected, actual in (("data_link", p["expected_data"], p["data"]),
                                   ("source_link", p["expected_source"], p["source"]),
                                   ("outputs_link", p["expected_outputs"], p["outputs"])):
        check(name, expected.exists() and expected.resolve() == actual, str(expected))
    trainer = p["code"] / "tools/GTM_Model8_mars_baseline.py"
    for name in ("GTM_Model8_mars_baseline.py", "GTM_Model8_mars_review.py",
                 "GTM_Model0_local_viewer.py", "evaluate_released_marss2l.py"):
        check("script:" + name, (p["code"] / "tools" / name).is_file(), name)
    for name in ("plan.json", "train_manifest.json", "select_ids.json",
                 "final_candidate_ids.json"):
        check("frozen:" + name, (p["plan"] / name).is_file(), name)
    plan_file = p["plan"] / "plan.json"
    check("plan_sha256", plan_file.is_file() and sha256(plan_file) == PLAN_SHA256,
          PLAN_SHA256)
    manifest = p["plan"] / "train_manifest.json"
    check("manifest_sha256", manifest.is_file() and sha256(manifest) == MANIFEST_SHA256,
          MANIFEST_SHA256)
    if plan_file.is_file() and sha256(plan_file) == PLAN_SHA256:
        try:
            id_hashes = json.loads(plan_file.read_text(encoding="utf-8"))["id_lists"]
            for name in ("select_ids.json", "final_candidate_ids.json"):
                file = p["plan"] / name
                check("ids_sha256:" + name, file.is_file() and sha256(file) == id_hashes[name],
                      id_hashes[name])
        except (OSError, ValueError, KeyError) as error:
            check("id_list_pins", False, str(error))
    for name in ("validated_images_all.csv", "validated_images_plumes.csv"):
        check("data:" + name, (p["data"] / name).is_file(), name)
    check("runtime_lock", args.runtime_lock.is_file(), str(args.runtime_lock))
    check("python", sys.version_info >= (3, 12), sys.version.split()[0])
    for module in MODULES:
        check("module:" + module, importlib.util.find_spec(module) is not None, module)
    driver = shutil.which("nvidia-smi")
    try:
        gpu = subprocess.run([driver, "-L"], capture_output=True, text=True, timeout=10) if driver else None
    except (OSError, subprocess.SubprocessError):
        gpu = None
    check("nvidia_driver", gpu is not None and gpu.returncode == 0 and bool(gpu.stdout.strip()),
          "nvidia-smi -L visible" if gpu and gpu.returncode == 0 else "NVIDIA driver not visible")
    if trainer.is_file():
        try:
            pins = frozen_constants(trainer)
            # A copied source tree can omit .git. Executable bytes are the authority.
            if (p["source"] / ".git").exists() and shutil.which("git"):
                revision = subprocess.run(["git", "-C", str(p["source"]), "rev-parse", "HEAD"],
                                          capture_output=True, text=True, timeout=10).stdout.strip()
                check("source_revision", revision == pins["SOURCE_REVISION"], pins["SOURCE_REVISION"])
            for name, expected in pins["SOURCE_HASHES"].items():
                file = p["source"] / "marss2l" / name
                check("source_sha256:" + name, file.is_file() and sha256(file) == expected, expected)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            check("source_pins", False, str(error))
    # The reviewer always loads the released reference as well as the new model.
    release = p["data"] / "trained_models/MARSS2L_20250326"
    check("released_reference", (release / "best_epoch").is_file() and
          (release / "config_experiment.json").is_file(),
          "Pinned released reference checkpoint and config must pass reviewer hashes")
    return {"status": "missing_requirements" if missing else "preflight_required",
            "missing": missing, "checks": checks,
            "note": "Doctor checks layout and imports only. Run frozen trainer preflight before training; CUDA and raster content are not proved here."}


def command(args: argparse.Namespace) -> list[str]:
    p = paths(args)
    fit = p["outputs"] / RUN_NAME / "fit"
    common = [sys.executable, "-u", str(p["code"] / "tools/GTM_Model8_mars_baseline.py"),
              "--data-root", str(p["data"]), "--manifest", str(p["plan"] / "train_manifest.json"),
              "--output", str(fit), "--runtime-lock", str(args.runtime_lock)]
    if args.action in {"preflight", "train"}:
        return common + ["--mode", args.action, "--batch-size", str(args.batch_size),
                         "--validation-batch-size", str(args.validation_batch_size),
                         "--precision", args.precision, "--num-workers", str(args.num_workers),
                         "--draws", str(args.draws), "--epochs", str(args.epochs),
                         "--patience", str(args.patience)]
    if args.action == "review":
        checkpoint = fit / ("epoch_000.pt" if args.stage == "first_pass" else "best_epoch.pt")
        output = p["outputs"] / RUN_NAME / "review" / args.stage / args.role
        result = [sys.executable, "-u", str(p["code"] / "tools/GTM_Model8_mars_review.py"),
                  "--data-root", str(p["data"]), "--checkpoint", str(checkpoint),
                  "--output", str(output), "--role", args.role, "--publish"]
        if args.stage == "final":
            result.append("--completed")
        return result
    if args.action == "viewer":
        return [sys.executable, "-u", str(p["code"] / "tools/GTM_Model0_local_viewer.py"),
                "--host", args.host, "--port", str(args.port),
                "--bundles-dir", str(p["outputs"] / "GTM_viewer_bundles")]
    raise ValueError(args.action)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("doctor", "commands", "preflight", "train", "review", "viewer"))
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path, help="MARS-S2L dataset root")
    parser.add_argument("--source-root", required=True, type=Path, help="Pinned marss2l checkout")
    parser.add_argument("--output-root", required=True, type=Path, help="Root linked as code-root/outputs")
    parser.add_argument("--runtime-lock", required=True, type=Path, help="Linux requirements lock used in the venv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, choices=(0, 2), default=2)
    parser.add_argument("--draws", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--stage", choices=("first_pass", "final"), default="first_pass")
    parser.add_argument("--role", choices=("validation", "holdout"), default="holdout")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.validation_batch_size, args.draws, args.epochs, args.patience) < 1:
        parser.error("batch sizes, draws, epochs, and patience must be positive")
    if args.action == "doctor":
        result = doctor(args)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "preflight_required" else 1
    if args.action == "commands":
        print(json.dumps({name: command(argparse.Namespace(**{**vars(args), "action": name}))
                          for name in ("preflight", "train", "review", "viewer")}, indent=2))
        return 0
    result = doctor(args)
    if result["status"] != "preflight_required":
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1
    if args.action == "review":
        fit = paths(args)["outputs"] / RUN_NAME / "fit"
        checkpoint = fit / ("epoch_000.pt" if args.stage == "first_pass" else "best_epoch.pt")
        if not checkpoint.is_file():
            parser.error(f"Predeclared checkpoint is missing: {checkpoint}")
        if args.stage == "final":
            status_file = fit / "run_status.json"
            if not status_file.is_file() or json.loads(status_file.read_text(encoding="utf-8")).get("state") != "completed":
                parser.error("Final review requires a completed fit")
    return subprocess.run(command(args), cwd=args.code_root.resolve(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
