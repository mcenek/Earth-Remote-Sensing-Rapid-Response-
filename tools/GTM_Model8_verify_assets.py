"""Verify cached MARS objects against a frozen catalog without interpreting data.

No network requests or asset mutations. A separate receipt records reuse and
zero transfer bytes; the older CAFO pilot ledger is never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path: Path, kind: str = "sha256_lfs", size: int = 0) -> str:
    if kind == "sha256_lfs":
        value = hashlib.sha256()
    elif kind == "git_blob_sha1":
        value = hashlib.sha1()
        value.update(f"blob {size}\0".encode("ascii"))
    else:
        raise ValueError(f"Unsupported hash type: {kind}")
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    root, output = args.data_root.resolve(), args.output.resolve()
    objects = {}
    identities = []
    for catalog in args.catalog:
        identities.append({"path": str(catalog.resolve()), "sha256": digest(catalog)})
        with catalog.open(encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                old = objects.setdefault(item["path"], item)
                if any(old[key] != item[key] for key in ("remote_oid", "remote_oid_type", "size")):
                    raise ValueError(f"Conflicting catalog object: {item['path']}")
                if "/resolve/c26b1d7e31a0c5241fa37c9140802622c215eb32/" not in item["source_url"]:
                    raise ValueError("Catalog is not pinned to the selected immutable revision")
    total_bytes = sum(item["size"] for item in objects.values())
    state = dict(state="running", catalogs=identities, data_root=str(root),
                 objects=len(objects), logical_asset_bytes=total_bytes,
                 verified_objects=0, verified_bytes=0, failed=[], network_download_bytes=0,
                 acquisition="Reuse pinned cached objects; no network permitted by this tool")
    output.mkdir(parents=True, exist_ok=True)
    save(output / "asset_verification.json", state)
    started = time.perf_counter()

    def verify(item):
        path = (root / item["path"]).resolve()
        result = {"path": item["path"], "size": item["size"], "remote_oid": item["remote_oid"]}
        try:
            if not path.is_relative_to(root):
                raise ValueError("Asset path leaves the MARS cache")
            if path.stat().st_size != item["size"]:
                raise ValueError("Cached file has the wrong size")
            actual = digest(path, item["remote_oid_type"], item["size"])
            if actual != item["remote_oid"]:
                raise ValueError(f"Content hash mismatch: {actual}")
            result["verified"] = True
        except (OSError, ValueError) as error:
            result.update(verified=False, error=str(error))
        return result

    with (output / "verified_objects.jsonl").open("w", encoding="utf-8", newline="\n") as receipt:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for number, result in enumerate(executor.map(verify, objects.values()), 1):
                receipt.write(json.dumps(result, separators=(",", ":")) + "\n")
                if result["verified"]:
                    state["verified_objects"] += 1
                    state["verified_bytes"] += result["size"]
                else:
                    state["failed"].append(result)
                if number % 5000 == 0:
                    receipt.flush()
                    state["seconds"] = time.perf_counter() - started
                    save(output / "asset_verification.json", state)
                    print(json.dumps({key: state[key] for key in ("verified_objects", "verified_bytes", "seconds")}), flush=True)
    state.update(state="passed" if not state["failed"] else "failed",
                 seconds=time.perf_counter() - started, object_receipt_sha256=digest(output / "verified_objects.jsonl"))
    save(output / "asset_verification.json", state)
    print(json.dumps({key: value for key, value in state.items() if key != "catalogs"}, indent=2))
    return 0 if state["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
