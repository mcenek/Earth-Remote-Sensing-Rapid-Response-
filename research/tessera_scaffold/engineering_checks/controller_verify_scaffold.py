"""Repeatable controller check of the offline synthetic Tessera user flow."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent
OUT = ROOT / "local-run/controller-verification"
OUT.mkdir(parents=True, exist_ok=True)
checks = []

def run(name, *arguments, expected_success=True):
    p = subprocess.run([sys.executable, "-X", "utf8", *map(str, arguments)],
                       cwd=CODE, text=True, capture_output=True, timeout=60)
    transcript = p.stdout + p.stderr
    log = OUT / (name + ".log")
    log.write_text(transcript, encoding="utf-8")
    assert log.read_text(encoding="utf-8") == transcript
    assert (p.returncode == 0) == expected_success, transcript
    checks.append(name)
    return p

run("unit-tests", "-m", "unittest", "discover", "-s", "tests", "-v")
demo, reloaded = OUT / "demo", OUT / "reloaded"
run("demo", "cli.py", "demo", "--output", demo)
run("saved-file-cli", "cli.py", "run", "--manifest", demo / "manifest.json",
    "--region", demo / "synthetic_region.npz", "--output", reloaded, "--max-pixels", 6)
result = json.loads((demo / "result.json").read_text(encoding="utf-8"))
assert result == json.loads((reloaded / "result.json").read_text(encoding="utf-8"))
matches = result["matches"]
assert [(m["point_id"], m["pixel_id"]) for m in matches] == [
    ("source-A", "r0c1"), ("source-A", "r1c0"),
    ("interior-B", "r0c2"), ("interior-B", "r1c2")]
assert math.isclose(matches[0]["cosine_similarity"], 9 / math.sqrt(82), abs_tol=1e-7)
assert math.isclose(matches[1]["cosine_similarity"], 4 / math.sqrt(17), abs_tol=1e-7)
assert (matches[0]["lon"], matches[0]["lat"]) == (-92.985, 41.995)
assert result["grid"] == {"crs": "EPSG:4326", "transform": [.01, 0, -93, 0, -.01, 42], "shape": [2, 3, 128]}
with (reloaded / "matches.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
geo = json.loads((reloaded / "matches.geojson").read_text(encoding="utf-8"))
assert len(rows) == len(geo["features"]) == len(matches) == 4
for match, row, feature in zip(matches, rows, geo["features"]):
    assert row["pixel_id"] == feature["properties"]["pixel_id"] == match["pixel_id"]
    assert row["provenance_key"] == match["provenance_key"]
    assert feature["geometry"]["coordinates"] == [match["lon"], match["lat"]]
checks.append("independent-ranking-and-export-oracle")

with np.load(demo / "synthetic_region.npz", allow_pickle=False) as saved:
    payload = {key: saved[key] for key in saved.files}
missing_path = OUT / "all-missing.npz"
np.savez_compressed(missing_path, **{**payload, "coverage": np.full((2, 3), "missing")})
missing_out = OUT / "all-missing-output"
run("all-missing-cli", "cli.py", "run", "--manifest", demo / "manifest.json",
    "--region", missing_path, "--output", missing_out, "--max-pixels", 6)
missing = json.loads((missing_out / "result.json").read_text(encoding="utf-8"))
assert missing["matches"] == [] and all(p["coverage_status"] == "missing" for p in missing["seeds"])
with (missing_out / "matches.csv").open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    assert "cosine_similarity" in reader.fieldnames and list(reader) == []
assert json.loads((missing_out / "matches.geojson").read_text(encoding="utf-8"))["features"] == []
checks.append("explicit-abstention-and-empty-export-readback")

mixed_path = OUT / "mismatched-dataset.npz"
np.savez_compressed(mixed_path, **{**payload, "dataset_version": "wrong-revision"})
failed = run("mismatched-pin-cli-rejected", "cli.py", "run", "--manifest", demo / "manifest.json",
    "--region", mixed_path, "--output", OUT / "must-not-exist", expected_success=False)
assert "ValueError: Mixed region SDK or dataset version" in failed.stderr
assert not (OUT / "must-not-exist/result.json").exists()
files = sorted(p for p in CODE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and "local-run" not in p.parts and "demo_output" not in p.parts)
inventory = [{"path": p.relative_to(CODE).as_posix(), "bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
review = {"state": "PASS_OFFLINE_SYNTHETIC_ONLY", "checks": checks, "files": inventory,
          "total_artifact_bytes": sum(x["bytes"] for x in inventory),
          "live_API_or_real_embedding_verified": False,
          "new_model_forwards_fits_or_real_facility_joins": 0}
text = json.dumps(review, indent=2) + "\n"
path = OUT / "CONTROLLER_VERIFICATION.json"
path.write_text(text, encoding="utf-8")
assert path.read_text(encoding="utf-8") == text
print(json.dumps({"state": review["state"], "checks": len(checks),
                  "bytes": review["total_artifact_bytes"], "evidence": str(path)}))
