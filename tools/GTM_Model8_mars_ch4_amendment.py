#!/usr/bin/env python3
"""Add pinned CH4 assets to the frozen Model 8 inventory without changing role IDs.

Metadata-only: reads CSV/JSONL and file sizes; does not open TIFF content.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
OUT = ROOT / "tool-diagnostics/mars-baseline-20261030"
REMOTE = META / "paper_v3_mixed_remote_catalog.jsonl"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(name):
    with (META / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
    for name in ("train.csv", "val.csv", "test.csv"):
        if digest(META / name) != plan["source"]["sha256"][name]:
            raise ValueError(f"Source changed: {name}")
    role_rows = {}
    for role in ("fit", "select", "final_candidate"):
        path = OUT / f"{role}.jsonl"
        if digest(path) != plan["files"][path.name]["sha256"]:
            raise ValueError(f"Role changed: {role}")
        role_rows[role] = read_jsonl(path)
    train, val, test = (read_csv(name) for name in ("train.csv", "val.csv", "test.csv"))
    by_id = {row["id_loc_image"]: row for rows in (train, val, test) for row in rows}
    role_ids = {role: {row["sample_id"] for row in rows} for role, rows in role_rows.items()}
    for role, ids in role_ids.items():
        if ids != set(json.loads((OUT / f"{role}_ids.json").read_text(encoding="utf-8"))):
            raise ValueError(f"Role IDs changed: {role}")
    positive_train = {r["id_loc_image"] for r in train if r["isplume"] == "True"}
    positive_roles = {role: {i for i in ids if by_id[i]["isplume"] == "True"} for role, ids in role_ids.items()}
    all_positive = positive_train | set().union(*positive_roles.values())
    # The publication plume metadata can hold multiple plume records per image.
    plume_csv = META / "validated_images_plumes.csv"
    plume_ids = Counter(row["id_loc_image"] for row in read_csv("validated_images_plumes.csv"))
    missing_plume_metadata = sorted(i for i in all_positive if plume_ids[i] == 0)
    missing_label_paths = sorted(i for i in all_positive if not by_id[i]["plumepath"])
    missing_ch4_paths = sorted(i for i in all_positive if not by_id[i]["ch4path"])
    if missing_label_paths or missing_ch4_paths:
        raise ValueError("Positive images missing label or CH4 asset path")
    for role, rows in role_rows.items():
        for row in rows:
            source = by_id[row["sample_id"]]
            if row["label"] == "PLUME" and source["plumepath"] not in {a["path"] for a in row["assets"]}:
                raise ValueError(f"Previously frozen role omitted label TIFF: {role} {row['sample_id']}")
    wanted = {by_id[i]["ch4path"] for i in all_positive}
    found = {}
    with REMOTE.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["path"] in wanted:
                if item["path"] in found:
                    raise ValueError(f"Duplicate remote CH4 path: {item['path']}")
                found[item["path"]] = item
    missing_remote = sorted(wanted - set(found))
    if missing_remote:
        raise ValueError(f"CH4 paths missing pinned remote catalog: {missing_remote[:5]}")

    def write_catalog(name, ids):
        paths = {by_id[i]["ch4path"] for i in ids}
        missing_local, mismatched = [], []
        output = OUT / name
        with output.open("w", encoding="utf-8", newline="\n") as f:
            for path in sorted(paths):
                item = found[path]
                local = META / path
                size = local.stat().st_size if local.is_file() else None
                if size is None:
                    missing_local.append(path)
                elif size != item["size"]:
                    mismatched.append(path)
                f.write(json.dumps({**item, "role": "methane_enhancement", "local_exists": size is not None, "local_bytes": size}, sort_keys=True, separators=(",", ":")) + "\n")
        return {"file": name, "sha256": digest(output), "paths": len(paths), "bytes": sum(found[p]["size"] for p in paths), "missing_local_paths": missing_local, "size_mismatch_paths": mismatched}

    full = write_catalog("full_official_train_ch4_asset_catalog.jsonl", positive_train)
    fit_select = write_catalog("model_fit_select_ch4_asset_catalog.jsonl", positive_roles["fit"] | positive_roles["select"])
    final = write_catalog("final_candidate_ch4_asset_catalog.jsonl", positive_roles["final_candidate"])
    receipt = {"schema_version": 1, "status": "metadata_and_size_verified; content_hash_pending", "plan_sha256": digest(OUT / "plan.json"), "remote_catalog_sha256": digest(REMOTE), "plume_csv_sha256": digest(plume_csv), "positive_images": {"official_train": len(positive_train), **{role: len(ids) for role, ids in positive_roles.items()}}, "plume_metadata_coverage": {"positive_images_checked": len(all_positive), "missing_images": len(missing_plume_metadata), "missing_by_role": {role: sum(i in missing_plume_metadata for i in ids) for role, ids in positive_roles.items()}, "missing_official_train": sum(i in missing_plume_metadata for i in positive_train), "multiple_records": sum(plume_ids[i] > 1 for i in all_positive), "note": "The image-level isplume label and label TIFF remain available when validated_images_plumes.csv lacks an image row; investigate before using per-plume flux metadata."}, "label_and_ch4_paths_present_for_all_positive_images": True, "catalogs": {"full_official_train": full, "model_fit_select": fit_select, "final_candidate": final}, "correction": "Earlier claim that positive plumepath was empty was incorrect. Positive rows have _label.tif in plumepath and _ch4.tif in ch4path. Frozen role manifests contain label TIFFs; their asset lists omit CH4 TIFFs, supplied by this explicit amendment. Role IDs and plan hash are unchanged."}
    (OUT / "ch4_asset_amendment.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
