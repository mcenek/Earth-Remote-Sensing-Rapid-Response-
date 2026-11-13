"""Fetch public file metadata only and freeze the complete pinned MARS release."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import GTM_Model8_fetch_assets as fetch
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

REPO = "UNEP-IMEO/MARS-S2L"
OUT = ROOT / "handoff/catalogs"
LOCAL = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"


def main():
    api = HfApi(token=False)
    info = api.dataset_info(REPO, revision=fetch.REVISION, expand=["sha"], timeout=120)
    if info.sha != fetch.REVISION:
        raise ValueError("Provider resolved a different revision")
    records = {}
    groups = Counter()
    for file in api.list_repo_tree(REPO, repo_type="dataset", revision=fetch.REVISION,
                                   recursive=True, expand=False):
        if not isinstance(file, RepoFile):
            continue
        name = file.path
        if file.lfs is not None:
            digest = file.lfs.sha256
            kind = "sha256_lfs"
        else:
            digest = file.blob_id
            kind = "git_blob_sha1"
        item = fetch.validate_item({"path": name, "size": file.size,
            "remote_oid": digest, "remote_oid_type": kind,
            "source_url": "https://huggingface.co" + fetch.URL_PREFIX + quote(name, safe="/") + "?download=true"})
        if name in records:
            raise ValueError(f"Duplicate provider file {name}")
        records[name] = item
        groups["/".join(name.split("/")[:2])] += item["size"]
        if len(records) % 10000 == 0:
            print(json.dumps({"phase": "provider_metadata", "files": len(records)}), flush=True)
    required, _ = fetch.load_catalog(fetch.DEFAULT_CATALOG)
    for item in required:
        full = records[item["path"]]
        if item["size"] != full["size"]:
            raise ValueError(f"Baseline size differs for {item['path']}")
        if item["remote_oid_type"] == full["remote_oid_type"] and item["remote_oid"] != full["remote_oid"]:
            raise ValueError(f"Baseline hash differs for {item['path']}")
    local_missing, local_mismatch = [], []
    local_present = local_bytes = 0
    for name, item in records.items():
        path = LOCAL.joinpath(*name.split("/"))
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            local_missing.append(name)
        else:
            local_present += 1
            local_bytes += size
            if size != item["size"]:
                local_mismatch.append(name)
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = OUT / "GTM_MARS_full_release_c26b1d7e.jsonl.gz"
    with catalog.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as output:
            for name in sorted(records):
                output.write((json.dumps(records[name], sort_keys=True, separators=(",", ":")) + "\n").encode())
    loaded, digest = fetch.load_catalog(catalog)
    if len(loaded) != len(records):
        raise ValueError("Catalog roundtrip count differs")
    total = sum(row["size"] for row in records.values())
    baseline_bytes = sum(row["size"] for row in required)
    report = {"repository": REPO, "revision": fetch.REVISION,
        "scope": "Every file returned by the complete paginated recursive provider tree at this immutable revision. No training split changes.",
        "source": f"https://huggingface.co/api/datasets/{REPO}/tree/{fetch.REVISION}?recursive=true",
        "objects": len(records), "logical_asset_bytes": total,
        "catalog": catalog.name, "catalog_bytes": catalog.stat().st_size,
        "catalog_sha256": digest, "bytes_by_directory": dict(groups),
        "baseline_objects_reused": len(required), "baseline_logical_bytes_reused": baseline_bytes,
        "additional_logical_bytes_after_baseline": total - baseline_bytes,
        "license": "Provider data and released weights: CC BY-NC-SA 4.0; retain provider README.md."}
    receipt = catalog.with_suffix("").with_suffix(".json")
    receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    local_report = {"root": str(LOCAL.resolve()), "provider_objects": len(records),
        "present": local_present, "present_bytes": local_bytes,
        "missing": len(local_missing), "missing_paths": local_missing,
        "size_mismatches": local_mismatch,
        "verification": "Presence and size only for all provider files; no new complete content-hash audit."}
    (Path(__file__).parent / "pc_full_release_inventory.json").write_text(
        json.dumps(local_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"provider": report, "pc": {key: value for key, value in local_report.items()
                                                if key != "missing_paths"}}, indent=2))


if __name__ == "__main__":
    main()
