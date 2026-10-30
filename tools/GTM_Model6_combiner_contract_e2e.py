"""Public-CLI E2E for the synthetic Model6 branch/combiner contract.

Written before the CLI; acceptance is in
reports/research/GTM_Model6_combiner_contract_acceptance_20261001.md.
No model/raster imports, real labels, or unit-test framework.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/GTM_Model6_combiner_contract.py"
R5 = ROOT / "outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r5/GTM_Model6_mapper_mars_pilot_manifest.json"
WINDOWS = (16, 32, 64, 128)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def make_manifest(root):
    rows = []
    for fold in range(5):
        for sibling in range(2):
            ident = f"SYNTH_p{fold}_{sibling}"
            known = [0 < index // 16 < 15 for index in range(256)]
            target = root / "targets" / f"{ident}.json"
            write(target, {"shape": [16, 16], "known": known,
                           "labels": [int(i % 16 >= 8) if known[i] else None for i in range(256)]})
            rows.append({
                "id": ident, "parent_id": ident, "dataset": "SYNTHETIC",
                "site": f"SYNTH_site{fold}", "group": f"SYNTH_group{fold}", "partition": fold,
                "target_scene": f"SYNTH_target{fold}_{sibling}",
                "reference_scene": f"SYNTH_reference{fold}_{sibling}",
                "crs": "SYNTHETIC_CARTESIAN", "affine": [10, 0, fold * 100000, 0, -10, 2000],
                "shape": [200, 200],
                "hashes": {"image": hashlib.sha256((ident + "_imaginary_input").encode()).hexdigest()},
                "synthetic_target": reference(root, target),
            })
    manifest = {"data_kind": "synthetic", "context_windows": list(WINDOWS),
                "prediction_window": 16, "holdout_fold": 0,
                "observations": [r for r in rows if r["partition"] == 0],
                "training_observations": [r for r in rows if r["partition"] != 0]}
    write(root / "manifest.json", manifest)


def make_bundle(root):
    plan = read(root / "plan.json")
    rows = {r["id"]: r for r in plan["observations"]}
    checkpoints, predictions = [], []
    for job in plan["jobs"]:
        # Deliberately not weights or a trained network. The contract must say so.
        checkpoint = root / "checkpoints" / f'{job["job_id"]}.json'
        write(checkpoint, {"kind": "synthetic_branch_stub", "branch": job["branch"],
                           "training_ids": job["training_ids"]})
        provenance = root / "provenance" / f'{job["job_id"]}.json'
        write(provenance, {
            "kind": "synthetic_branch_stub", "job_id": job["job_id"], "branch": job["branch"],
            "context_window": job["context_window"], "training_ids": job["training_ids"],
            "training_groups": job["training_groups"], "independent_training": True,
            "fused_loss_weight": 0, "feature_contract": "sentinel_only",
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "checkpoint_sha256": digest(checkpoint),
        })
        checkpoints.append({"job_id": job["job_id"], "checkpoint": reference(root, checkpoint),
                            "provenance": reference(root, provenance)})
        branch_index = WINDOWS.index(job["context_window"])
        for ident in job["prediction_ids"]:
            row = rows[ident]
            support = [0 < i // 16 < 15 for i in range(256)]
            values = [round((i % 16 - 7.5) / (3 + branch_index)
                            + (i // 16 - 7.5) * (branch_index - 1.5) / 20
                            + (row["partition"] - 2) * .025, 8) if support[i] else None
                      for i in range(256)]
            prediction = root / "predictions" / f'{job["job_id"]}_{ident}.json'
            write(prediction, {
                "job_id": job["job_id"], "id": ident, "branch": job["branch"],
                "checkpoint_sha256": digest(checkpoint), "grid": row["grid"],
                "central_region": row["central_region"], "score_type": "mask_logit",
                "support": support, "logits": values,
            })
            predictions.append({"job_id": job["job_id"], "id": ident,
                                "artifact": reference(root, prediction)})
    write(root / "bundle.json", {"schema_version": 1, "plan_sha256": digest(root / "plan.json"),
                                 "checkpoints": checkpoints, "predictions": predictions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "fixture"
    make_manifest(fixture)
    receipts = []
    originals = {str(p): digest(p) for p in (
        CLI, R5, ROOT / "research/GTM_Model6/GTM_Model6_config.json",
        ROOT / "research/GTM_Model6/GTM_Model6.py",
        R5.with_name("GTM_Model6_mapper_mars_pilot_results.json"))}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def invoke(name, command="check", root=fixture, expected=0, code=None, output=None, fold=0):
        output = output or out / "receipts" / f"{name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-I", str(CLI), command, "--manifest", str(root / "manifest.json"),
                "--output", str(output)]
        if command == "plan":
            argv.extend(["--outer-fold", str(fold)])
        else:
            argv.extend(["--plan", str(root / "plan.json"), "--bundle", str(root / "bundle.json")])
        run = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
        result = read(output) if expected != 3 and output.exists() else None
        passed = run.returncode == expected and (code is None or result.get("error_code") == code)
        receipts.append({"name": name, "argv": argv, "exit_code": run.returncode,
                         "expected_exit_code": expected, "stdout": run.stdout, "stderr": run.stderr,
                         "receipt_sha256": digest(output) if output.exists() else None, "passed": passed})
        if not passed:
            raise AssertionError(receipts[-1])
        return result

    def clone(name):
        destination = out / "faults" / name
        shutil.copytree(fixture, destination)
        return destination

    def change_artifact(root, ref, mutation, rehash=True):
        path = root / ref["path"]
        data = read(path)
        mutation(data)
        write(path, data)
        if rehash:
            ref["sha256"] = digest(path)

    def fault(name, mutate, code, command="check"):
        root = clone(name)
        bundle = read(root / "bundle.json")
        mutate(root, bundle)
        write(root / "bundle.json", bundle)
        return invoke(name, command, root, expected=2, code=code)

    try:
        plan = invoke("plan", "plan", output=fixture / "plan.json")
        assert len(plan["jobs"]) == 20
        assert sum(len(j["prediction_ids"]) for j in plan["jobs"]) == 40
        assert len(plan["combiner_fit_ids"]) == 8 and len(plan["combiner_evaluation_ids"]) == 2
        assert all(not set(j["training_groups"]) & {"SYNTH_group0"} for j in plan["jobs"])
        make_bundle(fixture)
        baseline = invoke("baseline")
        assert baseline["status"] == "contract_passed" and not baseline["training_ready"]
        assert baseline["prediction_tiles_checked"] == 40
        rehearsal = invoke("rehearsal", "rehearse")
        replay = invoke("replay", "rehearse")
        assert rehearsal == replay
        assert digest(out / "receipts/rehearsal.json") == digest(out / "receipts/replay.json")
        assert rehearsal["evaluation_labels_read"] is False
        assert rehearsal["combiner"]["fit_pixels"] == 8 * 224
        assert any(w != 0 for w in rehearsal["combiner"]["weights"])

        # No held target bytes may be read, even for hash checking.
        missing_labels = clone("evaluation_labels_absent")
        for row in plan["observations"]:
            if row["id"] in plan["combiner_evaluation_ids"]:
                (missing_labels / row["synthetic_target"]["path"]).unlink()
        assert invoke("evaluation_labels_absent", "rehearse", missing_labels) == rehearsal

        # Change only outer scores: fitted weights must be exactly unchanged.
        changed_outer = clone("outer_scores_changed")
        bundle = read(changed_outer / "bundle.json")
        for pred in bundle["predictions"]:
            if pred["id"] in plan["combiner_evaluation_ids"]:
                change_artifact(changed_outer, pred["artifact"],
                                lambda d: d.update(logits=[None if x is None else x + 1 for x in d["logits"]]))
        write(changed_outer / "bundle.json", bundle)
        outer = invoke("outer_scores_changed", "rehearse", changed_outer)
        assert outer["combiner"] == rehearsal["combiner"]
        assert outer["outer_predictions"] != rehearsal["outer_predictions"]

        changed_inner = clone("inner_scores_changed")
        bundle = read(changed_inner / "bundle.json")
        for pred in bundle["predictions"]:
            if pred["id"] in plan["combiner_fit_ids"]:
                change_artifact(changed_inner, pred["artifact"],
                                lambda d: d.update(logits=[None if x is None else -x for x in d["logits"]]))
        write(changed_inner / "bundle.json", bundle)
        inner = invoke("inner_scores_changed", "rehearse", changed_inner)
        assert inner["combiner"]["weights"] != rehearsal["combiner"]["weights"]

        # Independent analytic oracle: balanced labels and all-zero features.
        zero = clone("zero_score_oracle")
        bundle = read(zero / "bundle.json")
        for pred in bundle["predictions"]:
            change_artifact(zero, pred["artifact"],
                            lambda d: d.update(logits=[0.0 if v else None for v in d["support"]]))
        write(zero / "bundle.json", bundle)
        zero_result = invoke("zero_score_oracle", "rehearse", zero)
        assert zero_result["combiner"]["weights"] == [0.0] * 4
        assert zero_result["combiner"]["bias"] == 0.0
        assert all(p == .5 for row in zero_result["outer_predictions"]
                   for p in row["probability"] if p is not None)
        assert all(p == .5 for row in zero_result["outer_predictions"]
                   for p in row["mean_probability"] if p is not None)

        def provenance_fault(root, bundle, mutation):
            change_artifact(root, bundle["checkpoints"][0]["provenance"], mutation)

        def prediction_fault(root, bundle, mutation, rehash=True):
            change_artifact(root, bundle["predictions"][0]["artifact"], mutation, rehash)

        for name, training_id in (("own_parent_leak", "SYNTH_p1_0"),
                                  ("sibling_site_leak", "SYNTH_p1_1"),
                                  ("outer_site_leak", "SYNTH_p0_0")):
            fault(name, lambda r, b, ident=training_id: provenance_fault(
                r, b, lambda d: d["training_ids"].append(ident)), "TRAINING_LEAKAGE")
        fault("wrong_training_membership", lambda r, b: provenance_fault(
            r, b, lambda d: d["training_ids"].pop()), "TRAINING_MEMBERSHIP")
        fault("swapped_branch", lambda r, b: prediction_fault(
            r, b, lambda d: d.update(branch="S32")), "BRANCH_MISMATCH")
        fault("fused_gradient", lambda r, b: provenance_fault(
            r, b, lambda d: d.update(fused_loss_weight=1)), "INDEPENDENCE_CONTRACT")
        fault("wrong_context", lambda r, b: provenance_fault(
            r, b, lambda d: d.update(context_window=32)), "BRANCH_MISMATCH")
        fault("mixed_grid", lambda r, b: prediction_fault(
            r, b, lambda d: d["grid"]["affine"].__setitem__(2, 12345)), "GRID_MISMATCH")
        fault("mixed_central_region", lambda r, b: prediction_fault(
            r, b, lambda d: d["central_region"].__setitem__(0, 91)), "GRID_MISMATCH")
        fault("mismatched_support", lambda r, b: prediction_fault(
            r, b, lambda d: (d["support"].__setitem__(17, False),
                            d["logits"].__setitem__(17, None))), "SUPPORT_MISMATCH")
        fault("short_scores", lambda r, b: prediction_fault(
            r, b, lambda d: d["logits"].pop()), "SCORE_CONTRACT")
        fault("wrong_score_type", lambda r, b: prediction_fault(
            r, b, lambda d: d.update(score_type="probability")), "SCORE_CONTRACT")
        fault("stale_prediction_hash", lambda r, b: prediction_fault(
            r, b, lambda d: d["logits"].__setitem__(17, 123), rehash=False), "HASH_MISMATCH")
        fault("stale_checkpoint_hash", lambda r, b: (
            (r / b["checkpoints"][0]["checkpoint"]["path"]).write_bytes(b"changed")), "HASH_MISMATCH")
        fault("stale_provenance_hash", lambda r, b: (
            (r / b["checkpoints"][0]["provenance"]["path"]).write_bytes(b"changed")), "HASH_MISMATCH")
        fault("missing_checkpoint", lambda r, b: b["checkpoints"].pop(), "CHECKPOINT_COVERAGE")
        fault("final_fit_substitution", lambda r, b: prediction_fault(
            r, b, lambda d: d.update(job_id="S16_outer0_final")), "BRANCH_MISMATCH")
        def nonfinite_score(root, bundle):
            ref = bundle["predictions"][0]["artifact"]
            path = root / ref["path"]
            data = read(path)
            data["logits"][17] = "FLOAT_OVERFLOW"
            path.write_text(json.dumps(data).replace('"FLOAT_OVERFLOW"', "1e999"), encoding="utf-8")
            ref["sha256"] = digest(path)
        fault("nonfinite_supported_score", nonfinite_score, "SCORE_CONTRACT")
        fault("wrong_checkpoint_link", lambda r, b: prediction_fault(
            r, b, lambda d: d.update(checkpoint_sha256="0" * 64)), "CHECKPOINT_LINK")
        fault("missing_prediction_file", lambda r, b: (
            (r / b["predictions"][0]["artifact"]["path"]).unlink()), "MISSING_FILE")
        fault("missing_prediction_record", lambda r, b: b["predictions"].pop(), "PREDICTION_COVERAGE")
        fault("duplicate_prediction", lambda r, b: b["predictions"].append(
            copy.deepcopy(b["predictions"][0])), "PREDICTION_COVERAGE")
        fault("escaped_artifact", lambda r, b: b["predictions"][0]["artifact"].update(
            path="../../outside.json"), "UNSAFE_PATH")
        fault("stale_plan_hash", lambda r, b: b.update(plan_sha256="0" * 64), "PLAN_HASH")
        fault("stale_training_target", lambda r, b: (r / "targets/SYNTH_p1_0.json").write_bytes(
            b"changed"), "HASH_MISMATCH", command="rehearse")

        for name, mutation in (
            ("cross_partition_group", lambda rows: rows[1].update(partition=0)),
            ("cross_group_site", lambda rows: rows[1].update(site="SYNTH_site0")),
            ("cross_group_parent", lambda rows: rows[1].update(parent_id="SYNTH_p0_0")),
            ("shared_acquisition", lambda rows: rows[1].update(reference_scene="SYNTH_target0_0")),
            ("duplicate_input", lambda rows: rows[1]["hashes"].update(
                image=hashlib.sha256(b"SYNTH_p0_0_imaginary_input").hexdigest())),
        ):
            root = clone(name)
            manifest = read(root / "manifest.json")
            mutation(manifest["training_observations"])
            write(root / "manifest.json", manifest)
            invoke(name, "plan", root, expected=2, code="GROUP_LEAKAGE")

        stale = clone("stale_manifest")
        manifest = read(stale / "manifest.json")
        manifest["note"] = "manifest bytes changed after schedule freeze"
        write(stale / "manifest.json", manifest)
        invoke("stale_manifest", root=stale, expected=2, code="PLAN_MISMATCH")
        original_report = digest(out / "receipts/rehearsal.json")
        invoke("refuse_existing_output", "rehearse", expected=3, output=out / "receipts/rehearsal.json")
        assert digest(out / "receipts/rehearsal.json") == original_report

        # Actual manifest adapter: inspect metadata only, not model weights or imagery.
        r5root = out / "r5_metadata_only"
        r5root.mkdir()
        shutil.copyfile(R5, r5root / "manifest.json")
        actual_plan = invoke("r5_metadata_plan", "plan", r5root, output=r5root / "plan.json", fold=1)
        assert actual_plan["data_kind"] == "existing_metadata_only"
        assert actual_plan["training_ready"] is False
        assert len(actual_plan["combiner_fit_ids"]) == 99
        assert len(actual_plan["combiner_evaluation_ids"]) == 65
        invoke("refuse_real_rehearsal", "rehearse", r5root, expected=2, code="SYNTHETIC_ONLY")

        status_argv = [sys.executable, "-I", str(ROOT / "research/GTM_Model6/GTM_Model6.py"), "status"]
        status_run = subprocess.run(status_argv, env=env, capture_output=True, text=True, timeout=30)
        status_json = json.loads(status_run.stdout)
        write(out / "receipts/model6_status.json", status_json)
        assert status_run.returncode == 0 and status_json["configuration_valid"]
        assert status_json["combiner_preparation"]["contract_cli_available"]
        assert status_json["combiner_preparation"]["learned_merger_trained"] is False
        assert status_json["combiner_preparation"]["training_ready"] is False
        assert status_json["full_design_implemented"] is False
        assert status_json["mapper_development_diagnostic"]["promotion"].startswith("rejected")
        receipts.append({"name": "model6_status", "argv": status_argv, "exit_code": 0,
                         "passed": True, "receipt_sha256": digest(out / "receipts/model6_status.json")})
        assert all(digest(Path(p)) == before for p, before in originals.items())
        status = "passed"
    except Exception as exc:
        status = "failed"
        receipts.append({"name": "runner_assertion", "passed": False, "error": str(exc)})
    finally:
        report = {"status": status, "scope": "synthetic engineering only; no methane accuracy claim",
                  "sources_unchanged": all(digest(Path(p)) == before for p, before in originals.items()),
                  "commands": receipts, "runner_sha256": digest(Path(__file__)),
                  "implementation_sha256": digest(CLI)}
        write(out / "e2e.json", report)
        print(json.dumps({"status": status, "commands": len(receipts), "artifact": str(out / "e2e.json")}))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
