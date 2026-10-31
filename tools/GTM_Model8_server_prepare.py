#!/usr/bin/env python3
"""Probe a Linux host and prepare pinned Model8 paths without installing packages.

The probe only reads the current user's home and explicitly supplied search roots.
Prepare never replaces an existing path, and verifies frozen bytes before linking.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import GTM_Model8_server as server


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(argv: list[str]) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": repr(error)}


def memory_linux() -> dict:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(value.strip().split()[0]) * 1024
    return values


def candidates(roots: list[Path], max_depth: int = 4, max_dirs: int = 2000) -> dict:
    """Bounded, read-only search; skip other users and large hidden caches."""
    seen = 0
    datasets, repos = [], []
    for root in roots:
        if not root.is_dir():
            continue
        for base, dirs, files in os.walk(root, followlinks=False):
            seen += 1
            path = Path(base)
            depth = len(path.relative_to(root).parts)
            dirs[:] = [name for name in dirs if not name.startswith(".") and
                       name not in {"node_modules", "site-packages", "__pycache__", "tf-env", ".venv"}]
            if depth >= max_depth or seen >= max_dirs:
                dirs[:] = []
            if "validated_images_all.csv" in files and "validated_images_plumes.csv" in files:
                datasets.append(str(path))
            if (path / ".git").exists() and (
                    (path / "tools/GTM_Model8_server.py").is_file() or
                    (path / "EarthRemoteSensingRapidResponse").is_dir()):
                repos.append(str(path))
            if seen >= max_dirs:
                break
        if seen >= max_dirs:
            break
    return {"searched_roots": [str(root) for root in roots], "directories_seen": seen,
            "max_directories": max_dirs, "max_depth": max_depth,
            "candidate_data_roots": datasets, "own_repositories": repos}


def probe(args: argparse.Namespace) -> dict:
    home = Path.home().resolve()
    roots = [path.resolve() for path in args.search_root] if args.search_root else [home]
    # Explicit roots may be elsewhere, but default discovery stays in this account.
    disk = {}
    for root in roots:
        try:
            usage = shutil.disk_usage(root)
            disk[str(root)] = {"total": usage.total, "used": usage.used, "free": usage.free}
        except OSError as error:
            disk[str(root)] = {"error": repr(error)}
    python = {}
    for name in ("python3.12", "python3", "python"):
        executable = shutil.which(name)
        python[name] = {"path": executable, "version": command([executable, "--version"])
                        if executable else None}
    return {"action": "probe", "at": utcnow(), "host": platform.node(),
            "platform": platform.platform(), "user_home": str(home),
            "python": python, "tmux": shutil.which("tmux"),
            "nvidia_smi": command([shutil.which("nvidia-smi"), "-L"])
            if shutil.which("nvidia-smi") else {"error": "nvidia-smi unavailable"},
            "memory_bytes": memory_linux(), "disk_bytes": disk,
            "discovery": candidates(roots, args.max_depth, args.max_directories)}


def verify_frozen(code: Path, source: Path) -> dict:
    plan = code / "tool-diagnostics/mars-baseline-20261030"
    checks = {"plan_sha256": (plan / "plan.json", server.PLAN_SHA256),
              "manifest_sha256": (plan / "train_manifest.json", server.MANIFEST_SHA256)}
    pins = server.frozen_constants(code / "tools/GTM_Model8_mars_baseline.py")
    for name, digest in pins["SOURCE_HASHES"].items():
        checks["source_sha256:" + name] = (source / "marss2l" / name, digest)
    result = {}
    for name, (path, expected) in checks.items():
        actual = server.sha256(path) if path.is_file() else None
        result[name] = {"path": str(path), "expected": expected, "actual": actual,
                        "ok": actual == expected}
    if not all(check["ok"] for check in result.values()):
        raise ValueError("Frozen plan, manifest, or pinned source bytes differ: " +
                         ", ".join(name for name, item in result.items() if not item["ok"]))
    return {"source_revision": pins["SOURCE_REVISION"], "checks": result}


def archive_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    names = set()
    total = 0
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (not name or "\\" in name or name.startswith("/") or
                any(part in {"", ".", ".."} for part in path.parts) or
                re.match(r"^[A-Za-z]:", name)):
            raise ValueError(f"Unsafe archive path: {name!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in (0, 0o100000, 0o040000):
            raise ValueError(f"Unsupported archive entry type: {name!r}")
        normalized = str(path).rstrip("/")
        if normalized in names:
            raise ValueError(f"Duplicate archive entry: {name!r}")
        names.add(normalized)
        total += info.file_size
        if info.file_size > 200_000_000 or total > 1_000_000_000:
            raise ValueError("Source archive exceeds safety size limit")
        members.append(info)
    if not any(str(PurePosixPath(info.filename)).startswith("marss2l/") for info in members):
        raise ValueError("Source archive has no marss2l/ tree")
    return members


def extract_source(zip_path: Path, destination: Path, code: Path | None = None) -> dict:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Source destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive_members(archive)
        stage = Path(tempfile.mkdtemp(prefix=destination.name + ".stage-", dir=destination.parent))
        try:
            for info in members:
                target = stage.joinpath(*PurePosixPath(info.filename).parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            if code is not None:
                verify_frozen(code, stage)
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {"archive": str(zip_path), "archive_sha256": server.sha256(zip_path),
            "entries": len(members), "destination": str(destination)}


def link_plan(code: Path, data: Path, source: Path, outputs: Path) -> list[tuple[Path, Path]]:
    external = code / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external"
    return [(external / "MARS-S2L", data), (external / "marss2l-code", source),
            (code / "outputs", outputs)]


def validate_links(links: list[tuple[Path, Path]]) -> list[dict]:
    result = []
    for link, target in links:
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise FileExistsError(f"Conflicting symlink: {link} -> {link.resolve()}")
            action = "existing_correct_link"
        elif link.exists():
            raise FileExistsError(f"Conflicting existing path: {link}")
        else:
            action = "create_link"
        if link == target or link in target.parents:
            raise ValueError(f"Link target creates a cycle: {link} -> {target}")
        result.append({"link": str(link), "target": str(target), "action": action})
    return result


def prepare(args: argparse.Namespace) -> dict:
    code, data = args.code_root.resolve(), args.data_root.resolve()
    source = args.source_root.absolute()
    outputs = args.output_root.absolute()
    if not code.is_dir() or not (code / "tools/GTM_Model8_mars_baseline.py").is_file():
        raise FileNotFoundError(f"Model8 code root missing: {code}")
    if not data.is_dir() or not (data / "validated_images_all.csv").is_file():
        raise FileNotFoundError(f"MARS-S2L data root missing: {data}")
    if source.is_symlink() or outputs.is_symlink():
        raise ValueError("Source and output roots must be direct user-owned paths")
    links = link_plan(code, data, source, outputs)
    link_actions = validate_links(links)
    extracted = None
    if not source.exists():
        if args.source_zip is None:
            raise FileNotFoundError(f"Pinned source missing; supply --source-zip: {source}")
        extracted = extract_source(args.source_zip, source, code)
    elif not source.is_dir():
        raise ValueError(f"Source root is not a directory: {source}")
    frozen = verify_frozen(code, source)
    outputs.mkdir(parents=True, exist_ok=True)
    for item, (link, target) in zip(link_actions, links):
        if item["action"] == "create_link":
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target, target_is_directory=True)
    result = {"action": "prepare", "status": "ready", "at": utcnow(),
              "code_root": str(code), "data_root": str(data), "source_root": str(source),
              "output_root": str(outputs), "source_extraction": extracted,
              "frozen": frozen, "links": link_actions,
              "note": "No packages installed. Run doctor and frozen preflight before fitting."}
    receipt = args.receipt or outputs / "GTM_Model8_server_prepare.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(receipt.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    result["receipt"] = str(receipt)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    check = subparsers.add_parser("probe", help="Read-only bounded host inventory")
    check.add_argument("--search-root", type=Path, action="append", default=[])
    check.add_argument("--max-depth", type=int, default=4)
    check.add_argument("--max-directories", type=int, default=2000)
    setup = subparsers.add_parser("prepare", help="Verify frozen inputs and create missing links")
    for name in ("code-root", "data-root", "source-root", "output-root"):
        setup.add_argument("--" + name, required=True, type=Path)
    setup.add_argument("--source-zip", type=Path)
    setup.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.action == "probe" and (args.max_depth < 0 or args.max_directories < 1):
        parser.error("Probe limits must be nonnegative depth and positive directory count")
    try:
        result = probe(args) if args.action == "probe" else prepare(args)
    except Exception as error:
        print(json.dumps({"action": args.action, "status": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
