#!/usr/bin/env python3
"""Summarize member paths in one verified MethaneUnion archive without extraction."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


def summarize(archive_path: Path, candidate_path: Path) -> dict[str, object]:
    candidates = {
        str(value)
        for value in json.loads(candidate_path.read_text(encoding="utf-8"))["candidate_ids"]
    }
    first_members: list[str] = []
    first_480m_s2: list[str] = []
    first_candidate_id_paths: list[str] = []
    prefixes: Counter[str] = Counter()
    file_members = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            file_members += 1
            parts = PurePosixPath(member.name).parts
            if len(first_members) < 20:
                first_members.append(member.name)
            prefixes["/".join(parts[:4])] += 1
            if "480m_GSD" in parts and "s2" in parts and len(first_480m_s2) < 20:
                first_480m_s2.append(member.name)
            if candidates.intersection(parts) and len(first_candidate_id_paths) < 20:
                first_candidate_id_paths.append(member.name)
    return {
        "archive": archive_path.name,
        "file_members": file_members,
        "first_members": first_members,
        "top_prefixes": prefixes.most_common(20),
        "first_480m_s2": first_480m_s2,
        "first_candidate_id_paths": first_candidate_id_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("reports/acquisition/methaneunion_novel_s2_candidates.json"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize(args.archive, args.candidates), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
