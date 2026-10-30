#!/usr/bin/env python3
"""Inventory all official MARS train assets without reading raster content."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
OUT = ROOT / "tool-diagnostics/mars-baseline-20261030"
CATALOG = META / "paper_v3_mixed_remote_catalog.jsonl"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
    if digest(META / "train.csv") != plan["source"]["sha256"]["train.csv"]:
        raise ValueError("Official train CSV changed")
    wanted = {}
    missing_fields = Counter()
    with (META / "train.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in ("s2path", "cloudmaskpath"):
            if not row[field]:
                missing_fields[field] += 1
            else:
                wanted[row[field]] = field
        # The released image TIFF is a current/reference concatenation. A
        # background product ID is metadata, not a separate asset path.
        if row["plumepath"]:
            wanted[row["plumepath"]] = "plume_mask"
        elif row["isplume"] == "True":
            missing_fields["positive_plumepath_empty_vector_wkt_available"] += 1
        if not row["background_image_tile"]:
            missing_fields["background_product_id"] += 1
    found = {}
    with CATALOG.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["path"] in wanted:
                if item["path"] in found:
                    raise ValueError(f"Duplicate remote path: {item['path']}")
                found[item["path"]] = item
    missing_remote = sorted(set(wanted) - set(found))
    output = OUT / "full_official_train_asset_catalog.jsonl"
    missing_local = []
    mismatched_local = []
    total_bytes = 0
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for path in sorted(found):
            item = found[path]
            local = META / path
            observed = local.stat().st_size if local.is_file() else None
            if observed is None:
                missing_local.append(path)
            elif observed != item["size"]:
                mismatched_local.append({"path": path, "expected": item["size"], "observed": observed})
            total_bytes += item["size"]
            f.write(json.dumps({**item, "role": wanted[path], "local_exists": observed is not None, "local_bytes": observed}, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {"schema_version": 1, "plan_sha256": digest(OUT / "plan.json"), "train_csv_sha256": digest(META / "train.csv"), "source_catalog_sha256": digest(CATALOG), "full_catalog_sha256": digest(output), "official_train_rows": len(rows), "required_unique_asset_paths": len(wanted), "remote_catalog_matched_paths": len(found), "missing_remote_paths": missing_remote, "missing_local_paths": missing_local, "local_size_mismatches": mismatched_local, "expected_total_bytes_matched_remote_paths": total_bytes, "missing_metadata_fields": dict(missing_fields), "background_asset_convention": "s2path TIFF contains current and background/reference bands; background_image_tile is a product ID, not a separate raster path", "content_hash_status": "not_verified"}
    (OUT / "full_official_train_inventory.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
