#!/usr/bin/env python3
"""Rebase one trusted Model8 checkpoint onto a passed destination preflight.

This is an explicit, offline migration. It never changes a source checkpoint or
the frozen trainer. Supply SHA256 identities through an independent transfer
channel before loading either pickle-based PyTorch checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

FROZEN_RUNNER_SHA256 = "0e0f6adae01f89c7107f89ddd3e16338ff60896ffc3a30115c78c701695efe4a"
FROZEN_PLAN_SHA256 = "3b29c62bb250dd9155b1c2af90637b605f2709f7ec6c64b18662a0c9ed2a6345"
FROZEN_SCIENCE = {
    "manifest_sha256": "59e9843ac329e6da81c5dbbebeb22cde5eaf170ee4a8a5a9c002e3afbfdc68d9",
    "metadata_sha256": "799fa3272be6c313534c5d974894883db9f97874adb617eeaace1c8a4f9dc9b2",
    "plumes_sha256": "30cb278b9a7fca62e030d6efb7d362c28192f84421672203902ccd87eaaa2b5b",
    "source_revision": "f7d264c2c845dfba1cb27f76ef6026275f8d8758",
    "source_file_sha256": {
        "dataframe_image_plumes.py": "17e6a7dd216509141cd401c4cc2dfbadce529e64f6ecd93cd900bb37e603af80",
        "loaders.py": "db760531edd9b70c6c5b2679fb5a99dc0e492a2c766b83bc8eca37b0d5952b51",
        "loss.py": "fb662921052d55ac5fee0b207304cc74e9ea78da72c7a9456b4095197d464433",
        "mars_sentinel2/plumesimulation.py": "cf42119999c7676fdf7756b6e6532cbff8208d85c96a0de3fad3422102a338ca",
        "models.py": "8845fdd1db083271e25c55ea7896355ca631e94e6898903bb6be0ddc4c2b7f8b",
        "trainer.py": "479e13c0ff76f25d52af1bda9757581c0f7cb23cc792c612a921432917a55718",
    },
    "runner_sha256": FROZEN_RUNNER_SHA256,
    "released_unet_script_sha256": "22363ab23e5b98bff3c294ded35952636318f2d67bdfe99673ed481081edb5d7",
    "batch_size": 32, "validation_batch_size": 8, "draws": 65536,
    "precision": "bf16", "num_workers": 2, "cuda_allocator_fraction": 0.72,
}
FINGERPRINT_FIELDS = frozenset({
    "manifest_sha256", "metadata_sha256", "plumes_sha256", "draws", "batch_size",
    "validation_batch_size", "source_revision", "source_file_sha256", "runner_sha256",
    "released_unet_script_sha256", "runtime_lock_sha256", "torch_version",
    "numpy_version", "precision", "num_workers", "cuda_allocator_fraction",
})
ENVIRONMENT_FIELDS = frozenset({"runtime_lock_sha256", "torch_version", "numpy_version"})
SETTINGS = {"batch_size": 32, "precision": "bf16", "num_workers": 2,
            "draws": 65536, "epochs": 170, "patience": 30,
            "validation_batch_size": 8}
FULL_FIELDS = frozenset({"epoch", "model_state_dict", "optimizer_state_dict",
                         "torch_rng_state", "numpy_rng_state", "python_rng_state",
                         "cuda_rng_state", "best_ap", "best_epoch", "fingerprint"})
BEST_FIELDS = frozenset({"epoch", "model_state_dict", "fingerprint", "validation"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"A lowercase SHA256 is required for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, found {actual}")
    return actual


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def settings(path: Path) -> dict:
    value = read_json(path)
    found = value.get("settings", value)
    if found != SETTINGS:
        raise ValueError(f"Scientific training settings differ from frozen run: {path}")
    return found


def fingerprint_diff(source: dict, target: dict) -> dict:
    if set(source) != FINGERPRINT_FIELDS or set(target) != FINGERPRINT_FIELDS:
        raise ValueError("Unknown or missing fingerprint field")
    for key, expected in FROZEN_SCIENCE.items():
        if source[key] != expected or target[key] != expected:
            raise ValueError(f"Frozen scientific/source field differs: {key}")
    differences = {key: {"source": source[key], "destination": target[key]}
                   for key in sorted(FINGERPRINT_FIELDS) if source[key] != target[key]}
    forbidden = set(differences) - ENVIRONMENT_FIELDS
    if forbidden:
        raise ValueError(f"Scientific/source fingerprint differences: {sorted(forbidden)}")
    return differences


def assert_finite(value, location: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
            raise ValueError(f"Nonfinite tensor at {location}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite array at {location}")
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Nonfinite float at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{location}[{index}]")


def equal_state(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and all(
            equal_state(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            equal_state(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def validate_payload(full: dict, best: dict, source: dict) -> None:
    if set(full) != FULL_FIELDS or set(best) != BEST_FIELDS:
        raise ValueError("Checkpoint payload schema differs from frozen trainer")
    if full["fingerprint"] != source or best["fingerprint"] != source:
        raise ValueError("Checkpoint fingerprint differs from source preflight")
    if type(full["epoch"]) is not int or type(full["best_epoch"]) is not int:
        raise ValueError("Checkpoint epochs must be integers")
    if not (0 <= full["best_epoch"] <= full["epoch"] < SETTINGS["epochs"]):
        raise ValueError("Checkpoint epoch/best selection is invalid")
    if best["epoch"] != full["best_epoch"]:
        raise ValueError("Validation-best checkpoint epoch differs from full-state best_epoch")
    if not isinstance(full["best_ap"], (float, int)) or not math.isclose(
            full["best_ap"], best["validation"]["average_precision"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Validation-best AP differs from full-state best_ap")
    assert_finite(full)
    assert_finite(best)


def exclusive_save(path: Path, payload: dict) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as destination, temporary.open("rb") as source:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate(args: argparse.Namespace) -> dict:
    # These checks precede torch.load, which can execute pickle content.
    source_full_hash = require_hash(args.source_full, args.source_full_sha256)
    source_best_hash = require_hash(args.source_best, args.source_best_sha256)
    if args.plan_sha256 != FROZEN_PLAN_SHA256:
        raise ValueError("Supplied plan hash differs from frozen data plan")
    source_plan_hash = require_hash(args.source_plan, args.plan_sha256)
    target_plan_hash = require_hash(args.target_plan, args.plan_sha256)
    settings(args.source_settings)
    settings(args.target_settings)
    source_report = read_json(args.source_preflight)
    target_report = read_json(args.target_preflight)
    if source_report.get("status") != "passed" or target_report.get("status") != "passed":
        raise ValueError("Both source and destination preflights must have passed")
    source = source_report["fingerprint"]
    target = target_report["fingerprint"]
    differences = fingerprint_diff(source, target)
    if source_plan_hash != target_plan_hash:
        raise ValueError("Data plans differ")
    if args.target_fit.resolve() != args.target_preflight.resolve().parent:
        raise ValueError("Destination preflight must be inside target fit directory")
    for name in ("last_epoch.pt", "best_epoch.pt", "migration_receipt.json", "run_status.json", "run.claim"):
        if (args.target_fit / name).exists():
            raise FileExistsError(f"Destination already has training or migration state: {name}")
    if any(args.target_fit.glob("epoch_*.json")):
        raise FileExistsError("Destination already has epoch reports")
    full = torch.load(args.source_full, map_location="cpu", weights_only=False)
    best = torch.load(args.source_best, map_location="cpu", weights_only=False)
    if not isinstance(full, dict) or not isinstance(best, dict):
        raise ValueError("Expected dictionary checkpoints")
    validate_payload(full, best, source)
    derived_full = {**full, "fingerprint": target}
    derived_best = {**best, "fingerprint": target}
    assert equal_state({k: v for k, v in full.items() if k != "fingerprint"},
                       {k: v for k, v in derived_full.items() if k != "fingerprint"})
    assert equal_state({k: v for k, v in best.items() if k != "fingerprint"},
                       {k: v for k, v in derived_best.items() if k != "fingerprint"})
    receipt = {"schema": 1,
               "source_full": str(args.source_full.resolve()),
               "source_best": str(args.source_best.resolve()),
               "derived_full": str((args.target_fit / "last_epoch.pt").resolve()),
               "derived_best": str((args.target_fit / "best_epoch.pt").resolve()),
               "source_full_sha256": source_full_hash,
               "source_best_sha256": source_best_hash, "data_plan_sha256": source_plan_hash,
               "source_preflight_sha256": sha256(args.source_preflight),
               "target_preflight_sha256": sha256(args.target_preflight),
               "source_settings_sha256": sha256(args.source_settings),
               "target_settings_sha256": sha256(args.target_settings),
               "source_fingerprint": source, "target_fingerprint": target,
               "permitted_environment_differences": differences,
               "epoch": full["epoch"], "best_epoch": full["best_epoch"],
               "best_ap": full["best_ap"],
               "state_equality": "All payload fields except fingerprint compare equal before serialization",
               "precision_caveat": "Cross-platform resumed training is not promised bitwise identical"}
    created = []
    receipt_path = args.target_fit / "migration_receipt.json"
    receipt_created = False
    try:
        for name, payload in (("last_epoch.pt", derived_full), ("best_epoch.pt", derived_best)):
            path = args.target_fit / name
            receipt["derived_" + name.replace(".", "_") + "_sha256"] = exclusive_save(path, payload)
            created.append(path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if not equal_state(payload, loaded):
                raise ValueError(f"Serialized state differs: {path}")
        if require_hash(args.source_full, source_full_hash) != source_full_hash or \
           require_hash(args.source_best, source_best_hash) != source_best_hash:
            raise ValueError("Source checkpoint changed during migration")
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        receipt_created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
        return receipt
    except BaseException:
        if receipt_created:
            receipt_path.unlink(missing_ok=True)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-full", "source-best", "source-preflight", "source-settings",
                 "source-plan", "target-preflight", "target-settings", "target-plan", "target-fit"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-full-sha256", "source-best-sha256", "plan-sha256"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(migrate(parser.parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
