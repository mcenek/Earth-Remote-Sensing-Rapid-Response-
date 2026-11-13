#!/usr/bin/env python3
"""Audit actual overlay metadata against the base interpreter on this platform."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay", type=Path)
    args = parser.parse_args()
    overlay = args.overlay.resolve()
    if not overlay.is_dir():
        parser.error(f"Overlay absent: {overlay}")
    sys.path.insert(0, str(overlay))
    problems = []
    packages = list(metadata.distributions(path=[str(overlay)]))
    for package in packages:
        parent = package.metadata["Name"]
        for value in package.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                actual = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                actual = None
            if actual is None or (requirement.specifier and actual not in requirement.specifier):
                problems.append({"parent": parent, "requirement": str(requirement), "actual": actual})
    print(json.dumps({"overlay_packages": len(packages), "problems": problems}, indent=2, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
