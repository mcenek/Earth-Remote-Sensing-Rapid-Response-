#!/usr/bin/env python3
"""Join the frozen Model 8 role manifests to pinned MARS remote object metadata.

This is metadata-only. It checks paths and sizes but never reads asset bytes.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tool-diagnostics/mars-baseline-20261030"
META = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
SOURCE = META / "paper_v3_mixed_remote_catalog.jsonl"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
    wanted = {}
    for role in ("fit", "select", "final_candidate"):
        path = OUT / f"{role}.jsonl"
        if digest(path) != plan["files"][path.name]["sha256"]:
            raise ValueError(f"Frozen manifest changed: {path.name}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for a in row["assets"]:
                    p = a["path"]
                    old = wanted.setdefault(p, {"path": p, "roles": set(), "local_bytes": a["local_bytes"], "local_exists": a["local_exists"]})
                    if old["local_bytes"] != a["local_bytes"] or old["local_exists"] != a["local_exists"]:
                        raise ValueError(f"Inconsistent local metadata: {p}")
                    old["roles"].add(role)
    found = {}
    with SOURCE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            p = row["path"]
            if p in wanted:
                if p in found:
                    raise ValueError(f"Duplicate remote catalog path: {p}")
                found[p] = row
    missing_remote = sorted(set(wanted) - set(found))
    output = OUT / "required_asset_catalog.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for p in sorted(found):
            remote = found[p]
            local = wanted[p]
            if remote["size"] != local["local_bytes"]:
                raise ValueError(f"Local/remote size mismatch: {p}")
            if remote["remote_oid_type"] not in {"sha256_lfs", "git_blob_sha1"}:
                raise ValueError(f"Unknown remote hash type: {p}")
            f.write(json.dumps({**remote, "roles": sorted(local["roles"]), "local_exists": local["local_exists"], "local_bytes": local["local_bytes"]}, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {"schema_version": 1, "plan_sha256": digest(OUT / "plan.json"), "source_catalog": str(SOURCE.relative_to(ROOT)), "source_catalog_sha256": digest(SOURCE), "required_asset_catalog_sha256": digest(output), "required_paths": len(wanted), "resolved_remote_paths": len(found), "missing_remote_paths": missing_remote, "local_paths_present_size_matched": len(found), "total_reused_bytes_pending_hash_verification": sum(found[p]["size"] for p in found), "download_bytes": 0, "bytes_content_verified": 0, "status": "paths_and_sizes_only; remote object hash verification pending"}
    (OUT / "asset_catalog_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
