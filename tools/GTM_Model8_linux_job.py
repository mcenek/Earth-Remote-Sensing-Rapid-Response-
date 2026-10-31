#!/usr/bin/env python3
"""Durable, standard-library controller for one fresh Linux Model8 fit.

Run this inside tmux using a new Python 3.12 PyTorch venv. ``publish`` may be
called from another shell once epoch_000.pt exists. No environment is installed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import GTM_Model8_server as server


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def claim(path: Path) -> None:
    """Exclusive ownership without attempting to kill or adopt another process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Existing controller claim; inspect before retry: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump({"pid": os.getpid(), "created_at": now()}, target)


def package_fingerprint(args: argparse.Namespace) -> dict:
    p = server.paths(args)
    inputs = {
        "plan": p["plan"] / "plan.json",
        "manifest": p["plan"] / "train_manifest.json",
        "trainer": p["code"] / "tools/GTM_Model8_mars_baseline.py",
        "reviewer": p["code"] / "tools/GTM_Model8_mars_review.py",
        "launcher": p["code"] / "tools/GTM_Model8_server.py",
        "runtime_lock": args.runtime_lock,
    }
    return {"files": {name: server.sha256(path) for name, path in inputs.items()},
            "roots": {name: str(p[name]) for name in ("code", "data", "source", "outputs")},
            "settings": {name: getattr(args, name) for name in
                         ("batch_size", "validation_batch_size", "precision", "num_workers",
                          "draws", "epochs", "patience")},
            "python": sys.version.split()[0]}


def run_child(command: list[str], cwd: Path, log: Path, receipt: Path,
              phase: str, fingerprint: dict) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    state = {"phase": phase, "state": "starting", "command": command,
             "started_at": now(), "package_fingerprint": fingerprint}
    atomic_json(receipt, state)
    try:
        with log.open("a", encoding="utf-8") as output:
            output.write(f"\n[{state['started_at']}] {phase}: {json.dumps(command)}\n")
            output.flush()
            with subprocess.Popen(command, cwd=cwd, stdout=output,
                                  stderr=subprocess.STDOUT) as child:
                state.update(state="running", pid=child.pid)
                atomic_json(receipt, state)
                code = child.wait()
    except BaseException as error:
        state.update(state="failed", ended_at=now(), error=repr(error))
        atomic_json(receipt, state)
        raise
    state.update(state="completed" if code == 0 else "failed", ended_at=now(), exit_code=code)
    atomic_json(receipt, state)
    return code


def fit_dir(args: argparse.Namespace) -> Path:
    return server.paths(args)["outputs"] / server.RUN_NAME / "fit"


def job_dir(args: argparse.Namespace) -> Path:
    return server.paths(args)["outputs"] / server.RUN_NAME / "linux_job"


def review(args: argparse.Namespace, stage: str, role: str, fingerprint: dict) -> bool:
    root = job_dir(args)
    key = stage + "_" + role
    receipt = root / "reviews" / (key + ".json")
    destination = server.paths(args)["outputs"] / server.RUN_NAME / "review" / stage / role
    if receipt.exists() and read_json(receipt).get("state") == "completed":
        publication = destination / "publication.json"
        if publication.is_file() and set(read_json(publication)) == {"baseline", "released_reference"}:
            return True
        raise RuntimeError(f"Completed review receipt has no complete publication: {receipt}")
    checkpoint = fit_dir(args) / ("epoch_000.pt" if stage == "first_pass" else "best_epoch.pt")
    if stage == "final":
        status = fit_dir(args) / "run_status.json"
        if not status.is_file() or read_json(status).get("state") != "completed":
            raise RuntimeError("Final publication requires a successfully completed fit")
    if not checkpoint.is_file():
        raise RuntimeError(f"Predeclared checkpoint is missing: {checkpoint}")
    if destination.exists():
        raise RuntimeError(f"Review output already exists; inspect its publication before retry: {destination}")
    lock = root / "reviews" / (key + ".claim")
    claim(lock)
    try:
        review_args = argparse.Namespace(**{**vars(args), "action": "review", "stage": stage,
                                          "role": role})
        code = run_child(server.command(review_args), server.paths(args)["code"],
                         root / "logs" / (key + ".log"), receipt, "review", fingerprint)
        if code:
            raise RuntimeError(f"{key} reviewer exited {code}; see {receipt}")
        publication = destination / "publication.json"
        if not publication.is_file() or set(read_json(publication)) != {"baseline", "released_reference"}:
            result = read_json(receipt)
            result.update(state="failed", error="Reviewer returned without both publication receipts")
            atomic_json(receipt, result)
            raise RuntimeError(f"{key} reviewer returned without both publication receipts")
        result = read_json(receipt)
        result["publication"] = read_json(publication)
        atomic_json(receipt, result)
        return True
    finally:
        lock.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> int:
    root = job_dir(args)
    claim(root / "run.claim")
    status_path = root / "status.json"
    fit = fit_dir(args)
    if status_path.exists() or (fit.exists() and any(fit.iterdir())):
        (root / "run.claim").unlink(missing_ok=True)
        print("Existing fit or controller state found; refusing duplicate or incompatible fit", file=sys.stderr)
        return 1
    state = {"state": "doctor", "started_at": now(), "fit": "not_started",
             "publication": "not_started"}
    try:
        atomic_json(status_path, state)
        diagnosis = server.doctor(args)
        atomic_json(root / "doctor.json", diagnosis)
        if diagnosis["status"] != "preflight_required":
            raise RuntimeError("Doctor failed; see doctor.json")
        fingerprint = package_fingerprint(args)
        state.update(state="preflight", package_fingerprint=fingerprint)
        atomic_json(status_path, state)
        for phase in ("preflight", "train"):
            phase_args = argparse.Namespace(**{**vars(args), "action": phase})
            if phase == "train":
                report = fit_dir(args) / "preflight.json"
                if not report.is_file() or read_json(report).get("status") != "passed":
                    raise RuntimeError("Frozen preflight did not report passed")
                state.update(state="training", fit="running")
                atomic_json(status_path, state)
            code = run_child(server.command(phase_args), server.paths(args)["code"],
                             root / "logs" / (phase + ".log"), root / (phase + ".json"),
                             phase, fingerprint)
            if code:
                raise RuntimeError(f"{phase} exited {code}; see logs/{phase}.log")
        trainer_status = fit_dir(args) / "run_status.json"
        if not trainer_status.is_file() or read_json(trainer_status).get("state") != "completed":
            raise RuntimeError("Trainer exited without a completed fit status")
        state.update(state="publishing", fit="completed", publication="running")
        atomic_json(status_path, state)
        errors = {}
        for stage in ("first_pass", "final"):
            for role in ("validation", "holdout"):
                try:
                    review(args, stage, role, fingerprint)
                except Exception as error:
                    errors[stage + "_" + role] = str(error)
                    state["review_errors"] = errors
                    atomic_json(status_path, state)
        state.update(state="completed" if not errors else "publication_failed",
                     publication="completed" if not errors else "failed", ended_at=now())
        atomic_json(status_path, state)
        return 0 if not errors else 1
    except BaseException as error:
        state.update(state="failed", ended_at=now(), error=repr(error))
        if state["fit"] == "running":
            state["fit"] = "failed"
        atomic_json(status_path, state)
        print(f"Model8 Linux job failed: {error}", file=sys.stderr)
        return 1
    finally:
        (root / "run.claim").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "publish", "status"))
    for name in ("code-root", "data-root", "source-root", "output-root", "runtime-lock"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, choices=(0, 2), default=2)
    parser.add_argument("--draws", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--stage", choices=("first_pass", "final"), default="first_pass")
    parser.add_argument("--role", choices=("validation", "holdout", "both"), default="both")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.validation_batch_size, args.draws, args.epochs, args.patience) < 1:
        parser.error("batch sizes, draws, epochs, and patience must be positive")
    if args.action == "status":
        root = job_dir(args)
        status = root / "status.json"
        print(json.dumps({"job": read_json(status) if status.is_file() else None,
                          "trainer": read_json(fit_dir(args) / "run_status.json") if
                          (fit_dir(args) / "run_status.json").is_file() else None,
                          "reviews": {p.stem: read_json(p) for p in (root / "reviews").glob("*.json")}}, indent=2))
        return 0
    if args.action == "run":
        return run(args)
    diagnosis = server.doctor(args)
    if diagnosis["status"] != "preflight_required":
        print(json.dumps(diagnosis, indent=2), file=sys.stderr)
        return 1
    try:
        fingerprint = package_fingerprint(args)
        for role in (("validation", "holdout") if args.role == "both" else (args.role,)):
            review(args, args.stage, role, fingerprint)
    except Exception as error:
        print(f"Publication failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
