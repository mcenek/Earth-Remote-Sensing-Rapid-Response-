#!/usr/bin/env python3
"""One-command, user-owned Linux bootstrap for the frozen Model8 run.

Dry by default. --execute prepares paths and a private Python 3.12 venv;
--install-runtime installs only the pinned source lock; --start-training also
requires a passed doctor, frozen preflight, and an idle GPU before tmux launch.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import GTM_Model8_server as server
import GTM_Model8_server_prepare as prepare
import GTM_Model8_fetch_assets as assets


EXPECTED_ASSET_OBJECTS = 67687


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(argv: list[str], *, cwd: Path | None = None, log: Path | None = None) -> dict:
    started = stamp()
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as output:
            output.write(f"\n[{started}] {shlex.join(argv)}\n")
            output.flush()
            try:
                result = subprocess.run(argv, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                        check=False)
            except OSError as error:
                output.write(repr(error) + "\n")
                return {"command": argv, "started_at": started, "ended_at": stamp(),
                        "exit_code": -1, "log": str(log), "error": repr(error)}
        return {"command": argv, "started_at": started, "ended_at": stamp(),
                "exit_code": result.returncode, "log": str(log)}
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError as error:
        return {"command": argv, "started_at": started, "ended_at": stamp(),
                "exit_code": -1, "stdout": "", "stderr": repr(error)}
    return {"command": argv, "started_at": started, "ended_at": stamp(),
            "exit_code": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def data_report(root: Path) -> dict:
    required = ["validated_images_all.csv", "validated_images_plumes.csv",
                "trained_models/MARSS2L_20250326/best_epoch",
                "trained_models/MARSS2L_20250326/config_experiment.json"]
    present = {name: (root / name).is_file() for name in required}
    return {"root": str(root), "exists": root.is_dir(), "required_files": present,
            "missing": [name for name, okay in present.items() if not okay],
            "note": "Frozen preflight verifies manifest raster paths and scene content."}


def venv_python(venv: Path) -> Path:
    return venv / "bin/python"


def check_venv(venv: Path) -> dict:
    python = venv_python(venv)
    if not (venv / "pyvenv.cfg").is_file() or not python.is_file():
        raise RuntimeError(f"Existing venv is incomplete: {venv}")
    result = run_command([str(python), "-c", "import sys; print(sys.version.split()[0]); print(sys.prefix)"])
    if result["exit_code"] != 0:
        raise RuntimeError(f"Existing venv cannot run: {python}: {result['stderr']}")
    lines = result["stdout"].splitlines()
    if len(lines) < 2 or not lines[0].startswith("3.12.") or Path(lines[1]).resolve() != venv.resolve():
        raise RuntimeError(f"Existing venv is not private Python 3.12: {venv}")
    return {"executable": str(python), "version": lines[0], "prefix": lines[1]}


def ensure_venv(venv: Path, log: Path, interpreter: Path | None = None) -> dict:
    if venv.is_symlink():
        raise RuntimeError(f"Private venv path must not be a symlink: {venv}")
    if venv.exists() or venv.is_symlink():
        return check_venv(venv)
    candidate = str(interpreter) if interpreter is not None else shutil.which("python3.12")
    if candidate is None:
        raise RuntimeError("python3.12 is unavailable; no system Python changes were made")
    version = run_command([candidate, "-c", "import sys; print(sys.version.split()[0])"])
    if version["exit_code"] or not version["stdout"].startswith("3.12."):
        raise RuntimeError(f"Selected interpreter is not Python 3.12: {candidate}")
    venv.parent.mkdir(parents=True, exist_ok=True)
    result = run_command([candidate, "-m", "venv", str(venv)], log=log)
    if result["exit_code"]:
        raise RuntimeError(f"Private venv creation failed; see {log}")
    return check_venv(venv)


def idle_gpu() -> dict:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi unavailable; cannot establish an idle GPU")
    result = run_command([executable, "--query-compute-apps=pid,used_gpu_memory",
                          "--format=csv,noheader,nounits"])
    if result["exit_code"]:
        raise RuntimeError(f"GPU compute-process query failed: {result['stderr']}")
    processes = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f"GPU has active compute processes; leaving them untouched: {processes}")
    return {"query": result["command"], "compute_processes": [], "checked_at": stamp()}


def common_args(args: argparse.Namespace) -> list[str]:
    return ["--code-root", str(args.code_root), "--data-root", str(args.data_root),
            "--source-root", str(args.source_root), "--output-root", str(args.output_root),
            "--runtime-lock", str(args.source_root / "environment/requirements-test.lock"),
            "--batch-size", str(args.batch_size),
            "--validation-batch-size", str(args.validation_batch_size),
            "--precision", args.precision, "--num-workers", str(args.num_workers),
            "--draws", str(args.draws), "--epochs", str(args.epochs),
            "--patience", str(args.patience)]


def fresh_fit(args: argparse.Namespace) -> None:
    fit = args.output_root / server.RUN_NAME / "fit"
    job = args.output_root / server.RUN_NAME / "linux_job"
    fit_items = {path.name for path in fit.iterdir()} if fit.exists() else set()
    if fit_items - {"preflight.json"} or (job.exists() and any(job.iterdir())):
        raise RuntimeError(f"Existing fit or job files; refusing duplicate training: {fit}, {job}")


def plan(args: argparse.Namespace) -> dict:
    data = data_report(args.data_root)
    git = run_command(["git", "-C", str(args.code_root), "rev-parse", "HEAD"])
    return {"action": "bootstrap", "mode": "execute" if args.execute else "dry_plan",
            "at": stamp(), "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "code_root": str(args.code_root), "git_sha": git.get("stdout") if git["exit_code"] == 0 else None,
            "data": data, "source_root": str(args.source_root), "source_zip": str(args.source_zip),
            "source_zip_exists": args.source_zip.is_file(),
            "output_root": str(args.output_root), "venv": str(args.venv),
            "python3_12": str(args.python3_12) if args.python3_12 else shutil.which("python3.12"),
            "install_runtime": args.install_runtime, "start_training": args.start_training,
            "wait_for_assets": args.wait_for_assets,
            "tmux_session": args.tmux_session,
            "steps": (["wait for pinned asset acquisition"] if args.wait_for_assets else []) +
                     ["validate data", "verify/extract pinned source", "prepare links",
                      "create or verify private Python 3.12 venv"] +
                     (["pip install pinned lock"] if args.install_runtime else []) +
                     ["doctor", "frozen preflight"] +
                     (["verify GPU idle", "launch detached tmux training"] if args.start_training else [])}


def execute(args: argparse.Namespace, receipt: dict) -> dict:
    if receipt["data"]["missing"]:
        raise RuntimeError("Data root is missing required files: " + ", ".join(receipt["data"]["missing"]))
    if not args.source_root.exists() and not args.source_zip.is_file():
        raise RuntimeError(f"Pinned source archive missing: {args.source_zip}")
    if args.start_training:
        fresh_fit(args)
        if shutil.which("tmux") is None:
            raise RuntimeError("tmux unavailable")
        if run_command(["tmux", "has-session", "-t", args.tmux_session])["exit_code"] == 0:
            raise RuntimeError(f"tmux session already exists: {args.tmux_session}")
    prep_args = argparse.Namespace(code_root=args.code_root, data_root=args.data_root,
        source_root=args.source_root, output_root=args.output_root,
        source_zip=args.source_zip if args.source_zip.is_file() else None,
        receipt=args.output_root / "GTM_Model8_server_prepare.json")
    receipt["prepare"] = prepare.prepare(prep_args)
    logs = args.output_root / server.RUN_NAME / "bootstrap_logs"
    receipt["venv"] = ensure_venv(args.venv, logs / "venv.log", args.python3_12)
    python = venv_python(args.venv)
    lock = args.source_root / "environment/requirements-test.lock"
    if not lock.is_file():
        raise RuntimeError(f"Pinned runtime lock missing: {lock}")
    receipt["runtime_lock_sha256"] = server.sha256(lock)
    if args.install_runtime:
        install = run_command([str(python), "-m", "pip", "install", "-r", str(lock)],
                              cwd=args.code_root, log=logs / "pip.log")
        receipt["runtime_install"] = install
        if install["exit_code"]:
            raise RuntimeError(f"Pinned runtime install failed; see {install['log']}")
    launcher = args.code_root / "tools/GTM_Model8_server.py"
    for action in ("doctor", "preflight"):
        if action == "preflight":
            receipt["gpu_before_preflight"] = idle_gpu()
        result = run_command([str(python), str(launcher), action] + common_args(args),
                             cwd=args.code_root, log=logs / (action + ".log"))
        receipt[action] = result
        if result["exit_code"]:
            raise RuntimeError(f"{action} failed; see {result['log']}")
    if args.start_training:
        fresh_fit(args)
        proof = args.output_root / server.RUN_NAME / "fit/preflight.json"
        if not proof.is_file() or json.loads(proof.read_text(encoding="utf-8")).get("status") != "passed":
            raise RuntimeError(f"Frozen preflight did not leave a passed proof: {proof}")
        retained = logs / "bootstrap_preflight.json"
        retained.parent.mkdir(parents=True, exist_ok=True)
        os.replace(proof, retained)
        receipt["retained_preflight"] = {"path": str(retained), "sha256": server.sha256(retained)}
        fresh_fit(args)
        receipt["gpu"] = idle_gpu()
        controller = [str(python), str(args.code_root / "tools/GTM_Model8_linux_job.py"),
                      "run"] + common_args(args)
        job_log = logs / "tmux_job.log"
        shell_command = "exec " + shlex.join(controller) + " >> " + shlex.quote(str(job_log)) + " 2>&1"
        launched = run_command(["tmux", "new-session", "-d", "-s", args.tmux_session,
                                shell_command], cwd=args.code_root)
        receipt["tmux"] = {"session": args.tmux_session, "command": controller,
                           "log": str(job_log), "launch": launched}
        if launched["exit_code"]:
            raise RuntimeError(f"tmux launch failed: {launched['stderr']}")
        if run_command(["tmux", "has-session", "-t", args.tmux_session])["exit_code"]:
            raise RuntimeError(f"tmux session exited immediately; see {job_log}")
    receipt["status"] = "training_started" if args.start_training else "ready"
    return receipt


def write_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def complete_assets(state: dict, expected: int) -> None:
    """A complete label is insufficient without matching every frozen object."""
    try:
        verified = state["verified_existing"]
        missing = state["missing"]
        downloaded = state.get("downloaded", 0)
        remaining = state.get("remaining", 0)
        if not all(type(value) is int and value >= 0 for value in
                   (verified, missing, downloaded, remaining)):
            raise ValueError("Asset counts must be nonnegative integers")
        if (verified + downloaded != expected or verified + missing != expected or
                downloaded != missing or remaining != 0 or
                state.get("failed") or state.get("corrupt") or state.get("corrupt_count", 0)):
            raise ValueError("Complete acquisition receipt has inconsistent counts or errors")
    except KeyError as error:
        raise ValueError(f"Complete acquisition receipt lacks {error.args[0]}") from error


def wait_for_assets(args: argparse.Namespace, receipt: dict, receipt_path: Path) -> dict:
    catalog = args.code_root / "handoff/catalogs/GTM_Model8_mars_required_20261030.jsonl.gz"
    objects, catalog_hash = assets.load_catalog(catalog)
    if catalog_hash != assets.DEFAULT_CATALOG_SHA256 or len(objects) != EXPECTED_ASSET_OBJECTS:
        raise RuntimeError("Bundled acquisition catalog hash or object count differs from the pinned handoff")
    identity = {"data_root": str(args.data_root.resolve()), "catalog_sha256": catalog_hash}
    source_receipt = args.data_root / ".gtm_model8_fetch_receipt.json"
    deadline = time.monotonic() + args.asset_wait_timeout
    receipt["status"] = "waiting_for_assets"
    receipt["asset_wait"] = {"catalog_sha256": catalog_hash,
                             "expected_objects": EXPECTED_ASSET_OBJECTS,
                             "acquisition_session": args.acquisition_tmux_session,
                             "receipt": str(source_receipt)}
    write_receipt(receipt_path, receipt)
    while True:
        try:
            state = json.loads(source_receipt.read_text(encoding="utf-8")) if source_receipt.is_file() else None
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Acquisition receipt is unreadable: {error}") from error
        if state is not None:
            if state.get("identity") != identity:
                raise RuntimeError("Acquisition receipt has a different data root or catalog SHA-256")
            asset_state = state.get("state")
            if asset_state == "complete":
                complete_assets(state, EXPECTED_ASSET_OBJECTS)
                receipt["asset_wait"].update(state="complete", verified_existing=state["verified_existing"],
                                             downloaded=state.get("downloaded", 0),
                                             network_response_bytes=state.get("network_response_bytes"))
                write_receipt(receipt_path, receipt)
                print(json.dumps({"phase": "assets_complete", **receipt["asset_wait"]}), flush=True)
                return receipt
            if asset_state not in {"inventory", "downloading"}:
                raise RuntimeError(f"Acquisition stopped in state {asset_state!r}")
        else:
            asset_state = "receipt_missing"
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for acquisition after {args.asset_wait_timeout} seconds")
        session = run_command(["tmux", "has-session", "-t", args.acquisition_tmux_session])
        if session["exit_code"] != 0:
            # The fetcher can finish between the receipt read and this query.
            # Re-enter the full identity/count checks before reporting a lost job.
            if source_receipt.is_file():
                latest = json.loads(source_receipt.read_text(encoding="utf-8"))
                if latest.get("state") == "complete":
                    continue
            raise RuntimeError(f"Acquisition tmux session disappeared before completion: {args.acquisition_tmux_session}")
        progress = {"phase": "waiting_for_assets", "acquisition_state": asset_state,
                    "verified_existing": state.get("verified_existing") if state else None,
                    "downloaded": state.get("downloaded") if state else None,
                    "missing": state.get("missing") if state else None,
                    "network_response_bytes": state.get("network_response_bytes") if state else None}
        receipt["asset_wait"].update(progress)
        write_receipt(receipt_path, receipt)
        print(json.dumps(progress), flush=True)
        time.sleep(min(args.asset_poll_seconds, max(0, deadline - time.monotonic())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    home = Path.home()
    code = Path(__file__).resolve().parents[1]
    parser.add_argument("--code-root", type=Path, default=code)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=home / "gtm-model8/source/marss2l-code")
    parser.add_argument("--source-zip", type=Path,
                        default=code / "handoff/vendor/marss2l-f7d264c2c845dfba.zip")
    parser.add_argument("--output-root", type=Path, default=home / "gtm-model8/outputs")
    parser.add_argument("--venv", type=Path, default=home / "gtm-model8/venv")
    parser.add_argument("--python3-12", type=Path,
                        help="Explicit Python 3.12 executable if it is not on PATH")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--install-runtime", action="store_true")
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--wait-for-assets", action="store_true",
                        help="Wait for the pinned acquisition receipt before setup; training still requires --start-training")
    parser.add_argument("--acquisition-tmux-session", default="gtm-model8-data")
    parser.add_argument("--asset-wait-timeout", type=int, default=21600)
    parser.add_argument("--asset-poll-seconds", type=int, default=30)
    parser.add_argument("--tmux-session", default="gtm-model8-linux")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", choices=(0, 2), type=int, default=2)
    parser.add_argument("--draws", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.validation_batch_size, args.draws, args.epochs, args.patience) < 1:
        parser.error("Batch sizes, draws, epochs, and patience must be positive")
    if args.asset_wait_timeout < 1 or args.asset_poll_seconds < 1:
        parser.error("Asset wait timeout and poll seconds must be positive")
    args.code_root = args.code_root.resolve()
    args.data_root = args.data_root.resolve()
    args.source_root = args.source_root.absolute()
    args.source_zip = args.source_zip.absolute()
    args.output_root = args.output_root.absolute()
    args.venv = args.venv.absolute()
    if args.venv.is_symlink():
        parser.error("The private venv path must not be a symlink")
    if args.venv.resolve() != (home / "gtm-model8/venv").resolve():
        parser.error("The private venv must be ~/gtm-model8/venv")
    if args.install_runtime or args.start_training or args.wait_for_assets:
        if not args.execute:
            parser.error("--install-runtime, --start-training, and --wait-for-assets require --execute")
    receipt = plan(args)
    if not args.execute:
        print(json.dumps(receipt, indent=2))
        return 0
    destination = args.receipt or args.output_root / "GTM_Model8_server_bootstrap.json"
    try:
        if args.wait_for_assets:
            wait_for_assets(args, receipt, destination)
            receipt["data"] = data_report(args.data_root)
            receipt["status"] = "preparing"
            write_receipt(destination, receipt)
        receipt = execute(args, receipt)
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        print(json.dumps(receipt, indent=2), file=sys.stderr)
        write_receipt(destination, receipt)
        return 1
    write_receipt(destination, receipt)
    receipt["receipt"] = str(destination)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
