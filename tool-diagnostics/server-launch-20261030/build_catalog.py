"""Freeze the existing baseline's required public assets, without fetching data."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tool-diagnostics/mars-baseline-20261030"
DATA = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
OUT = ROOT / "handoff/catalogs"
REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"
PREFIX = f"https://huggingface.co/datasets/UNEP-IMEO/MARS-S2L/resolve/{REVISION}/"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


objects = {}
inputs = []
for name in ("required_asset_catalog.jsonl", "model_fit_select_ch4_asset_catalog.jsonl"):
    path = INPUT / name
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    inputs.append({"name": name, "sha256": sha(path), "objects": len(rows),
                   "bytes": sum(row["size"] for row in rows)})
    for row in rows:
        item = {key: row[key] for key in ("path", "size", "source_url", "remote_oid", "remote_oid_type")}
        old = objects.setdefault(item["path"], item)
        assert old == item, item["path"]

metadata = json.loads((DATA / "metadata_manifest.json").read_text(encoding="utf-8"))
for item in metadata["files"]:
    path = DATA / item["path"]
    assert sha(path) == item["sha256"]
    objects[item["path"]] = {"path": item["path"], "size": path.stat().st_size,
        "source_url": item["source_url"], "remote_oid": item["sha256"],
        "remote_oid_type": "sha256_content"}

reference = {
    "trained_models/MARSS2L_20250326/best_epoch": "be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246",
    "trained_models/MARSS2L_20250326/config_experiment.json": "abeb92d01313fbb2939e6c5fc1c6281846b8102ea5edd7081668fe0db05bf79f",
}
for name, expected in reference.items():
    path = DATA / name
    assert sha(path) == expected
    objects[name] = {"path": name, "size": path.stat().st_size,
        "source_url": PREFIX + name + "?download=true", "remote_oid": expected,
        "remote_oid_type": "sha256_content"}

for name, item in objects.items():
    assert item["source_url"] == PREFIX + name + "?download=true", name

OUT.mkdir(parents=True, exist_ok=True)
archive = OUT / "GTM_Model8_mars_required_20261030.jsonl.gz"
with archive.open("wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        for name in sorted(objects):
            compressed.write((json.dumps(objects[name], sort_keys=True, separators=(",", ":")) + "\n").encode())

receipt = {"repository": "UNEP-IMEO/MARS-S2L", "revision": REVISION,
    "description": "Existing frozen FIT, validation and reserved evaluation raster objects, FIT simulation CH4 patterns, provider metadata and pinned released reference. No new scene selection.",
    "license": "Provider dataset and released weights: CC BY-NC-SA 4.0; see downloaded README.md",
    "catalog": archive.name, "catalog_sha256": sha(archive),
    "catalog_bytes": archive.stat().st_size, "objects": len(objects),
    "logical_asset_bytes": sum(row["size"] for row in objects.values()),
    "input_catalogs": inputs,
    "metadata_manifest_sha256": sha(DATA / "metadata_manifest.json"),
    "hash_provenance": "Raster hashes preserve prior immutable-release catalogs. Metadata content SHA-256 matches the verified provider metadata receipt. Reference hashes match evaluate_released_marss2l.RELEASE_SPECS.",
}
destination = archive.with_suffix("").with_suffix(".json")
destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
with gzip.open(archive, "rt", encoding="utf-8") as reader:
    reloaded = [json.loads(line) for line in reader]
assert len(reloaded) == len(objects)
assert sum(row["size"] for row in reloaded) == receipt["logical_asset_bytes"]
print(json.dumps(receipt, indent=2))
