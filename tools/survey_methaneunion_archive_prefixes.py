#!/usr/bin/env python3
"""Bounded HTTP-prefix survey of pinned MethaneUnion gzip/tar archives.

This reads only a capped compressed prefix and decodes the first TAR header.  It
is an ordering diagnostic, never a complete member inventory or absence proof.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/methaneunion_novel_s2_acquisition_protocol.json"
DEFAULT_REPORT = ROOT / "reports/acquisition/methaneunion_remaining_archive_prefix_survey.json"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def first_tar_member_from_gzip_prefix(payload: bytes) -> str:
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    expanded = decoder.decompress(payload, 16 * 1024 * 1024)
    if len(expanded) < 512:
        raise ValueError("Compressed prefix did not yield one TAR header")
    try:
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            for _ in range(16):
                member = archive.next()
                if member is None:
                    break
                if member.isfile():
                    return member.name
    except (EOFError, tarfile.ReadError):
        pass
    raise ValueError("Compressed prefix did not yield a regular-file TAR header")


def capped_prefix(
    opener: Any, url: str, maximum_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    request = Request(
        url,
        headers={"Range": f"bytes=0-{maximum_bytes - 1}", "Accept-Encoding": "identity"},
    )
    with opener.open(request, timeout=90) as response:
        payload = response.read(maximum_bytes)
        return payload, {
            "status_code": response.status,
            "content_range": response.headers.get("Content-Range"),
            "content_length": response.headers.get("Content-Length"),
            "final_url_scheme": response.geturl().split(":", 1)[0],
        }


def survey(
    archives: Iterable[dict[str, Any]],
    repo_id: str,
    revision: str,
    maximum_bytes: int,
    opener: Any | None = None,
) -> tuple[list[dict[str, Any]], int]:
    client = opener or build_opener()
    results: list[dict[str, Any]] = []
    network_bytes = 0
    for archive in archives:
        url = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
            f"{quote(archive['name'])}?download=true"
        )
        payload, transport = capped_prefix(client, url, maximum_bytes)
        network_bytes += len(payload)
        results.append(
            {
                "archive": archive["name"],
                "archive_bytes": int(archive["bytes"]),
                "prefix_bytes_read": len(payload),
                "first_member": first_tar_member_from_gzip_prefix(payload),
                "transport": transport,
            }
        )
        print(f"{archive['name']}: {results[-1]['first_member']}", flush=True)
    return results, network_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=103)
    parser.add_argument("--last", type=int, default=144)
    parser.add_argument("--maximum-prefix-bytes", type=int, default=262144)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    archives = protocol["source"]["archives"]
    selected = archives[args.first - 1 : args.last]
    if not selected or len(selected) != args.last - args.first + 1:
        raise ValueError("Requested archive interval is outside the frozen inventory")
    results, network_bytes = survey(
        selected,
        protocol["source"]["repo_id"],
        protocol["source"]["revision"],
        args.maximum_prefix_bytes,
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Bounded first-member ordering diagnostic; not a complete membership audit",
        "source": {
            "repo_id": protocol["source"]["repo_id"],
            "revision": protocol["source"]["revision"],
        },
        "bounds": {
            "first_archive": args.first,
            "last_archive": args.last,
            "maximum_prefix_bytes_per_archive": args.maximum_prefix_bytes,
            "maximum_network_bytes": len(selected) * args.maximum_prefix_bytes,
            "actual_network_bytes": network_bytes,
        },
        "limitations": [
            "Only the first TAR member is decoded from each compressed prefix.",
            "A prefix cannot prove that later members are present or absent.",
            "Results may prioritize full scans but cannot change the frozen candidate set.",
        ],
        "results": results,
    }
    atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
