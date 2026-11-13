"""Derive an import-graph overlay from the official Python 3.12 test lock.

This is a package-name closure over pip-compile's ``# via`` annotations. It is
not a Python 3.11 resolver; use the import probe after installation.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/marss2l-code"
LOCK = SOURCE / "environment/requirements-test.lock"
NAME = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")
PY311_OVERRIDES = {"marshsi": "0.1.4"}
WINDOWS_MARKER_REQUIREMENTS = ["win32-setctime==1.2.0"]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked():
    entries = {}
    key = None
    via = False
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        match = NAME.match(line)
        if match:
            key = norm(match.group(1))
            entries[key] = (match.group(1), match.group(2), set())
            via = False
        elif key and line.startswith("    # via"):
            via = True
            parent = line.removeprefix("    # via").strip().split(" ")[0]
            if parent:
                entries[key][2].add(norm(parent))
        elif key and via and line.startswith("    #   "):
            entries[key][2].add(norm(line.removeprefix("    #   ").strip().split(" ")[0]))
        elif not line.startswith("    #"):
            via = False
    return entries


def main():
    entries = locked()
    requested = {norm(m.group(1)) for line in (HERE / "requirements-direct.in").read_text().splitlines() if (m := NAME.match(line))}
    existing = {norm(d.metadata["Name"]): d.version for d in metadata.distributions()}
    excluded = {norm(m.group(1)) for line in (HERE / "base-constraints.txt").read_text().splitlines() if (m := NAME.match(line))}
    stop = set(existing) | excluded
    seen = set(requested)
    while True:
        added = {key for key, (_, _, parents) in entries.items() if parents & (seen - stop)} - seen
        if not added:
            break
        seen.update(added)
    needed = sorted(seen - stop)
    missing = sorted(k for k in needed if k not in entries)
    if missing:
        raise ValueError(f"Missing official lock entries: {missing}")
    lines = [
        "# Derived from UNEP-IMEO-MARS/marss2l f7d264c2c845dfba1cb27f76ef6026275f8d8758",
        "# and its Python 3.12 environment/requirements-test.lock, filtered for the",
        "# existing Python 3.11 CUDA venv. Check wheels and imports before training.",
        "# Install with --no-deps --target into an isolated overlay only.",
    ] + [f"{entries[k][0]}=={PY311_OVERRIDES.get(k, entries[k][1])}" for k in needed]
    lines += WINDOWS_MARKER_REQUIREMENTS
    (HERE / "requirements-minimal-locked.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"locked={len(entries)} requested={len(requested)} closure={len(seen)} overlay={len(needed)}")
    print("base versions outside official lock:", {k: (existing.get(k), entries[k][1]) for k in sorted(excluded & seen) if k in entries and existing.get(k) != entries[k][1]})


if __name__ == "__main__":
    main()
