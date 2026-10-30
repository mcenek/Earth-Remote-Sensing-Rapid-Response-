"""Exercise the cached-readiness public CLI without loading raster/model libraries.

Acceptance and failure cases were written first in
reports/research/GTM_Model6_cached_readiness_acceptance_20261001.md.
Every fault is injected into an isolated copy, never the research cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "reports/research/assets/GTM_Model6_data_contract_20261001/input_pins.json"
COMMAND = ROOT / "tools/GTM_Model6_cached_readiness.py"
CACHE = ROOT / "outputs/GTM_Model0_hf_pilot_20260925"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    originals = [(ROOT if p["scope"] == "repo" else CACHE) / p["path"] for p in pins["files"]]
    before = {str(p): sha(p) for p in originals}
    evidence = []
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def run(name, repo=ROOT, cache=CACHE, expected=0, output=None):
        target = output or out / f"{name}.json"
        argv = [sys.executable, "-I", str(COMMAND), "--repo-root", str(repo),
                "--cache-root", str(cache), "--pins", str(PINS), "--output", str(target)]
        result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
        record = {"name": name, "argv": argv, "exit_code": result.returncode,
                  "expected_exit_code": expected, "stdout": result.stdout, "stderr": result.stderr,
                  "passed": result.returncode == expected}
        evidence.append(record)
        if not record["passed"]:
            raise AssertionError(record)
        return json.loads(target.read_text(encoding="utf-8")) if expected != 3 else None

    try:
        baseline = run("baseline")
        assert baseline["status"] == "verified_cache_scientifically_blocked"
        assert baseline["training_ready"] is False
        assert baseline["radiometry"]["conversion_approved"] is False
        assert baseline["raster_files_verified"] == 16
        assert [s["observed_positive_pixels"] for s in baseline["samples"]] == [1503, 0]
        assert all(s["zero_is_reviewed_negative"] is False for s in baseline["samples"])
        assert baseline["samples"][1]["sentinel_matches"] == 0
        assert baseline["budget"]["remaining_bytes"] == 3768186
        assert baseline["preserved_r5"]["aggregate"]["fp"] == 21387
        run("replay")
        assert sha(out / "baseline.json") == sha(out / "replay.json")
        original_report = sha(out / "baseline.json")
        run("refuse_existing_report", expected=3, output=out / "baseline.json")
        assert sha(out / "baseline.json") == original_report

        with tempfile.TemporaryDirectory(prefix="GTM_Model6_readiness_") as directory:
            staging = Path(directory)
            for record, source in zip(pins["files"], originals):
                destination = staging / record["scope"] / record["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            mutation = next(p for p in pins["files"] if p["path"].endswith("Volusia_Solid_Waste_anagement_Division/plumemask.tif"))
            mask = staging / mutation["scope"] / mutation["path"]
            with mask.open("r+b") as stream:
                byte = stream.read(1)
                stream.seek(0)
                stream.write(bytes([byte[0] ^ 1]))
            fault = run("changed_mask", staging / "repo", staging / "cache", expected=2)
            assert fault["training_ready"] is False
            assert any(c["status"] == "hash_mismatch" for c in fault["checks"])
            shutil.copyfile(CACHE / mutation["path"], mask)
            missing = staging / "cache/emit_followup/swir_20260930/swir22.tif"
            missing.unlink()
            fault = run("missing_swir", staging / "repo", staging / "cache", expected=2)
            assert fault["training_ready"] is False
            assert any(c["status"] == "missing" for c in fault["checks"])
        assert all(sha(p) == before[str(p)] for p in originals)
        status = "passed"
    except Exception as error:
        status = "failed"
        evidence.append({"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report = {"status": status, "checks": evidence,
                  "source_files_unchanged": all(sha(p) == before[str(p)] for p in originals),
                  "artifacts": {p.name: sha(p) for p in sorted(out.glob("*.json"))}}
        (out / "e2e.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "artifact": str(out / "e2e.json")}))


if __name__ == "__main__":
    main()
