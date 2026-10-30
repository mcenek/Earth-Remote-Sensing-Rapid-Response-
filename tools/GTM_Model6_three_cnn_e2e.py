"""Public-CLI rehearsal with persistent arrays, receipts, and viewer HTTP evidence.

Failure cases were declared in docs/GTM_Model6_three_cnn_acceptance_20261021.md
before implementation. This is an engineering check, never a methane benchmark.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/GTM_Model6_three_cnn.py"
PREFIX = "GTM_Model6_three_cnn"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_scene_rejections(scenes_manifest, run_dir, output_dir):
    """Exercise target-free input failures through the public inference command."""
    source = Path(scenes_manifest).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads(source.read_text(encoding="utf-8"))
    # Keep each rejection fixture self-contained and retain its exact CLI output.
    shutil.copytree(source.parent / "arrays", output / "arrays")
    cases = []
    bad = copy.deepcopy(original)
    bad["observations"][0]["target"] = {}
    cases.append(("target_field", bad, "fields"))
    bad = copy.deepcopy(original)
    bad["observations"][0]["input"]["sha256"] = "0" * 64
    cases.append(("bad_hash", bad, "SHA-256 differs"))
    bad = copy.deepcopy(original)
    first = bad["observations"][0]
    with np.load(source.parent / first["input"]["path"], allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["target"] = np.zeros(arrays["input_valid"].shape, dtype=np.float32)
    extra = output / "synthetic_extra_target.npz"
    np.savez_compressed(extra, **arrays)
    first["input"].update(path=extra.name, sha256=sha(extra))
    cases.append(("target_array", bad, "NPZ must contain exactly"))
    bad = copy.deepcopy(original)
    bad["data_kind"] = "measured"
    for row in bad["observations"]:
        row["input"]["grid"]["crs"] = "EPSG:32610"
    cases.append(("false_measured_provenance", bad, "synthetic"))
    bad = copy.deepcopy(original)
    bad["data_kind"] = "measured"
    bad["observations"] = bad["observations"][:2]
    for index, row in enumerate(bad["observations"]):
        old_path = source.parent / row["input"]["path"]
        probe_path = output / f"probe_{index}.npz"
        shutil.copy2(old_path, probe_path)
        row.update(id=f"probe-{index}", source_ids=[f"probe-Sentinel-{index}"])
        row["input"].update(path=probe_path.name, sha256=sha(probe_path))
        row["input"]["grid"]["crs"] = f"EPSG:{32610 + index}"
        row["review"].update(reviewer="rejection-probe", notes="Artificial input for refusal only. Never train.")
    cases.append(("mixed_crs", bad, "mixed measured CRSs"))
    results = []
    for name, value, message in cases:
        path = output / (name + ".json")
        write(path, value)
        argv = [sys.executable, str(CLI), "infer", "--scenes", str(path),
                "--run-dir", str(run_dir), "--output-dir", str(output / (name + "_forbidden"))]
        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
        (output / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        passed = process.returncode == 2 and message.casefold() in process.stderr.casefold()
        results.append({"name": name, "passed": passed, "exit_code": process.returncode, "argv": argv})
    artifact = {"passed": all(r["passed"] for r in results), "checks": results,
                "claim": "Public CLI rejection probes only; no measured data or training."}
    write(output / "input_only_rejections.json", artifact)
    if not artifact["passed"]:
        raise AssertionError("Input-only CLI rejection check failed; inspect retained logs")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for all repeatable evidence")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    log_dir = out / "logs"
    log_dir.mkdir()
    checks, commands = [], []
    receipt_path = out / f"{PREFIX}_e2e.json"
    receipt = {"schema": "GTM_Model6_three_cnn_e2e_v1", "data_kind": "synthetic", "passed": False,
               "checks": checks, "commands": commands, "downloads": 0,
               "claim": "Software acceptance only. No methane science result."}
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2", TF_NUM_INTRAOP_THREADS="2",
               TF_NUM_INTEROP_THREADS="2", OMP_NUM_THREADS="2")

    def check(name, okay, detail=None):
        checks.append({"name": name, "passed": bool(okay), "detail": detail})
        write(receipt_path, receipt)
        if not okay:
            raise AssertionError(name)

    def command(arguments, success=True, script=CLI):
        argv = [sys.executable, str(script), *map(str, arguments)]
        number = len(commands)
        started = time.monotonic()
        log = log_dir / f"{number:02d}.log"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        commands.append({"argv": argv, "exit_code": process.returncode, "seconds": round(time.monotonic() - started, 3),
                         "log": str(log.relative_to(out))})
        check(("accept " if success else "reject ") + " ".join(map(str, arguments[:2])),
              (process.returncode == 0) == success, log.read_text(encoding="utf-8")[-1800:])
        return log

    try:
        data = out / "data"
        manifest = data / "manifest.json"
        run = out / "GTM_Model6_three_cnn_engineering"
        bundles, exports = out / "bundles", out / "exports"
        command(["fixture", "--output", data, "--seed", "41"])
        scenes_dir = out / "input_only"
        command(["scene-fixture", "--manifest", manifest, "--output", scenes_dir])
        rejected = check_scene_rejections(scenes_dir / "manifest.json", run, out / "scene_rejections")
        for item in rejected["checks"]:
            check("input-only rejects " + item["name"], item["passed"])
        command(["preflight", "--manifest", manifest, "--output", out / "preflight.json"])
        original = json.loads(manifest.read_text(encoding="utf-8"))

        def reject_manifest(name, mutate):
            bad = copy.deepcopy(original)
            mutate(bad)
            path = data / (name + ".json")
            write(path, bad)
            command(["preflight", "--manifest", path], success=False)

        reject_manifest("bad_bands", lambda m: m["bands"].reverse())
        reject_manifest("bad_grid", lambda m: m["observations"][0]["target"]["grid"]["affine"].__setitem__(2, 1))
        reject_manifest("bad_hash", lambda m: m["observations"][0]["input"].__setitem__("sha256", "0" * 64))
        reject_manifest("bad_units", lambda m: m["observations"][0]["target"].__setitem__("units", "ppm*m"))
        reject_manifest("bad_split", lambda m: m["observations"][2].__setitem__("site", m["observations"][0]["site"]))
        reject_manifest("bad_overlap", lambda m: [
            m["observations"][2][key]["grid"]["affine"].__setitem__(2, m["observations"][0][key]["grid"]["affine"][2])
            for key in ("input", "target")])
        reject_manifest("bad_kind", lambda m: m["observations"][-1].__setitem__("target_kind", "airborne"))
        with np.load(data / original["observations"][0]["input"]["path"], allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        for name, mutate in (("bad_unknown", lambda a: a["target"].__setitem__((0, 0), 0)),
                             ("bad_support", lambda a: a["target_valid"].__setitem__((0, 0), 1)),
                             ("bad_target_channel", lambda a: a.__setitem__("image", np.concatenate(
                                 [a["image"], np.zeros((*a["image"].shape[:2], 1), dtype=np.float32)], axis=-1)))):
            a = {key: value.copy() for key, value in arrays.items()}
            mutate(a)
            path = data / (name + ".npz")
            np.savez_compressed(path, **a)
            def use_bad(m, path=path):
                m["observations"][0]["input"].update(path=path.name, sha256=sha(path))
            reject_manifest(name, use_bad)
        command(["pipeline", "--manifest", manifest, "--run-dir", run, "--stage", "all", "--epochs", args.epochs,
                 "--base-channels", "4", "--batch-size", "2", "--learning-rate", ".003", "--tile-size", "24",
                 "--halo", "32", "--max-minutes", "3", "--export-dir", exports, "--bundles-dir", bundles, "--publish"])
        command(["replay", "--manifest", manifest, "--run-dir", run, "--stage", "all"])
        # A deployment copy has only weights and receipts, and must work while
        # every original training array and the training manifest are unavailable.
        deployment = out / "GTM_Model6_three_cnn_deployment"
        deployment.mkdir()
        for artifact in [*run.glob("*.weights.h5"), *run.glob("*_training.json")]:
            shutil.copy2(artifact, deployment / artifact.name)
        hidden_manifest = data / "manifest.unavailable.json"
        hidden_arrays = data / "arrays.unavailable"
        manifest.rename(hidden_manifest)
        (data / "arrays").rename(hidden_arrays)
        try:
            command(["infer", "--scenes", scenes_dir / "manifest.json", "--run-dir", deployment,
                     "--stage", "fusion", "--output-dir", out / "inference", "--export-dir", exports,
                     "--bundles-dir", bundles, "--publish"])
        finally:
            hidden_manifest.rename(manifest)
            hidden_arrays.rename(data / "arrays")
        inferred = json.loads((out / "inference" / f"{PREFIX}_fusion_inference.json").read_text())
        check("inference works without training manifest or target arrays", len(inferred["records"]) == 4)
        known = {r["id"]: r for r in json.loads((run / f"{PREFIX}_fusion_results.json").read_text())["records"]}
        matched = 0
        for record in inferred["records"]:
            with np.load(out / "inference" / record["prediction"]["path"], allow_pickle=False) as z:
                check("inference contains no target " + record["id"], set(z.files) == {"probability", "valid"})
                if record["id"] in known:
                    matched += 1
                    with np.load(run / known[record["id"]]["prediction"]["path"], allow_pickle=False) as measured:
                        check("label-free prediction equals paired prediction " + record["id"],
                              np.allclose(z["probability"], measured["probability"], equal_nan=True, atol=1e-5, rtol=0))
        check("label-free replay checks every paired EMIT scene", matched == len(known) == 2)
        fusion = json.loads((run / f"{PREFIX}_fusion_training.json").read_text())
        check("frozen branches unchanged during fusion", all(fusion["frozen_weights_unchanged"].values()))
        fit_groups = set(fusion["fit_groups"])
        parents = [json.loads((run / f"{PREFIX}_{stage}_training.json").read_text()) for stage in ("airborne", "emit")]
        check("fusion fitting uses separate groups", all(not fit_groups.intersection(p["fit_groups"]) for p in parents))
        check("branch normalization excludes fusion and test", all(
            set(p["normalization"][p["stage"]]["fitted_ids"]) == set(p["fit_ids"]) for p in parents))
        command(["train", "--manifest", manifest, "--run-dir", run, "--stage", "emit"], success=False)
        stale = out / "stale_run"
        shutil.copytree(run, stale)
        with (stale / f"{PREFIX}_emit.weights.h5").open("ab") as stream:
            stream.write(b"changed-parent")
        command(["replay", "--manifest", manifest, "--run-dir", stale, "--stage", "fusion"], success=False)
        # Launch the same viewer users run, with this rehearsal's isolated publications.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = subprocess.Popen([sys.executable, str(ROOT / "tools/GTM_Model0_local_viewer.py"),
                                   "--port", str(port), "--bundles-dir", str(bundles), "--poll-seconds", "2"],
                                  cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{port}/"
        try:
            for _ in range(80):
                try:
                    with urlopen(base + "api/experiments", timeout=5) as response:
                        catalog = json.load(response)
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise AssertionError("Viewer did not start")
            for stage in ("emit", "airborne", "fusion"):
                identifier = run.name + "_" + stage
                experiment = next(e for e in catalog["experiments"] if e["id"] == identifier)
                results = json.loads((run / f"{PREFIX}_{stage}_results.json").read_text())
                by_id = {r["id"]: r for r in results["records"]}
                check("viewer has completed scenes " + stage, len(experiment["scenes"]) == len(by_id) > 0)
                check("engineering labels " + stage, "ENGINEERING ONLY" in experiment["status"])
                for scene in experiment["scenes"]:
                    check("synthetic not placed on US map " + scene["id"], scene.get("coordinateSystem") == "pixel")
                    for key in ("rgb", "truthImage", "predictionImage", "grid"):
                        with urlopen(base + scene[key], timeout=5) as response:
                            payload = response.read()
                        check("HTTP asset " + stage + ":" + key, bool(payload))
                    grid = json.loads(payload)
                    with np.load(run / by_id[scene["id"]]["prediction"]["path"], allow_pickle=False) as z:
                        display = np.array([np.nan if v is None else v for v in grid["probability"]]).reshape(z["probability"].shape)
                        check("viewer uses actual saved prediction " + stage + ":" + scene["id"],
                              np.allclose(display, z["probability"], equal_nan=True, atol=0, rtol=0))
                prepared = exports / identifier
                broken = out / ("incomplete_" + stage)
                shutil.copytree(prepared, broken)
                (broken / "scene_0000/prediction.png").unlink()
                published_hash = sha(bundles / identifier / "experiment.json")
                command(["--bundle-dir", broken, "--bundles-dir", bundles], success=False,
                        script=ROOT / "tools/GTM_Model0_publish_viewer.py")
                check("failed publication preserves complete experiment " + stage,
                      published_hash == sha(bundles / identifier / "experiment.json"))
            check("default presentation unchanged", catalog["activeExperimentId"] == "GTM_Model6_mapper_r5_portable")
            unlabeled = next(e for e in catalog["experiments"] if e["id"] == deployment.name + "_fusion_unlabeled")
            check("input-only viewer has no invented score", all("nativeMetrics" not in s and "truthImage" not in s for s in unlabeled["scenes"]))
        finally:
            server.terminate()
            server.wait(timeout=15)
        receipt.update(passed=True, run_dir=str(run), manifest=str(manifest), bundles_dir=str(bundles),
                       note="Browser visual checks and screenshots are recorded separately alongside this receipt.")
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        write(receipt_path, receipt)
    print(json.dumps({"passed": True, "checks": len(checks), "receipt": str(receipt_path)}, indent=2))


if __name__ == "__main__":
    main()
