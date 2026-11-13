#!/usr/bin/env python3
"""Batch-check pinned PyPI release metadata and Windows cp311 wheels.

Only fetches JSON metadata. No wheel, package, or dataset bytes are downloaded.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.request import urlopen

from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

HERE = Path(__file__).resolve().parent
TAGS = set(sys_tags())


def check(spec: str) -> dict:
    name, version = spec.split("==", 1)
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urlopen(url, timeout=20) as response:
        data = json.load(response)
    requires = data["info"].get("requires_python") or ""
    wheels = []
    for item in data["urls"]:
        filename = item["filename"]
        if item["packagetype"] != "bdist_wheel":
            continue
        _, _, _, tags = parse_wheel_filename(filename)
        if tags & TAGS:
            wheels.append({"filename": filename, "bytes": item["size"]})
    return {
        "package": name, "version": version, "requires_python": requires,
        "python311_allowed": not requires or SpecifierSet(requires).contains("3.11.14"),
        "compatible_wheels": wheels,
    }


def main() -> int:
    specs = [line.strip() for line in (HERE / "requirements-minimal-locked.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    result = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(check, spec): spec for spec in specs}
        for job in as_completed(jobs):
            try:
                result.append(job.result())
            except Exception as error:
                result.append({"package": jobs[job], "error": f"{type(error).__name__}: {error}"})
    result.sort(key=lambda item: item["package"])
    (HERE / "pypi_compat.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    issues = [item for item in result if "error" in item or not item["python311_allowed"] or not item["compatible_wheels"]]
    print(json.dumps({"checked": len(result), "issues": issues,
                      "compressed_wheel_bytes": sum(min(w["bytes"] for w in item["compatible_wheels"])
                                                    for item in result if item.get("compatible_wheels"))}, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
