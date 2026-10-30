"""Plan/check Model6 branch lineage and rehearse a combiner on synthetic scores.

Uses existing mapper observation keys and Model6 config. No model/raster imports,
training launch, download, scientific-gate update, or full-scene export.
See reports/research/GTM_Model6_combiner_contract_acceptance_20261001.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research/GTM_Model6/GTM_Model6_config.json"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_PARENTS = 256
CLAIM = "Engineering contract only; no methane accuracy, upscaling or source-origin claim."


#### Strict, bounded artifact I/O
class Rejected(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise Rejected(code, message)


def digest(path):
    require(path.is_file(), "MISSING_FILE", f"Missing artifact: {path.name}")
    require(path.stat().st_size <= MAX_FILE_BYTES, "RESOURCE_BOUND", f"Artifact too large: {path.name}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "INVALID_JSON", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path):
    digest(path)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              Rejected("INVALID_JSON", f"Nonfinite JSON constant: {value}")))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise Rejected("INVALID_JSON", path.name) from exc


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def artifact(root, ref, budget):
    require(isinstance(ref, dict) and nonempty(ref.get("path")), "UNSAFE_PATH", "Missing artifact path")
    relative = Path(ref["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and path.is_relative_to(root.resolve()), "UNSAFE_PATH",
            "Artifacts must remain inside their bundle")
    actual = digest(path)
    budget[0] += path.stat().st_size
    require(budget[0] <= MAX_BUNDLE_BYTES, "RESOURCE_BOUND", "Bundle exceeds the bounded contract budget")
    require(actual == ref.get("sha256"), "HASH_MISMATCH", f"Hash differs: {relative.as_posix()}")
    return path


#### Existing mapper manifest adapter and nested schedule
def make_plan(manifest_path, outer_fold):
    manifest = read(manifest_path)
    config = read(CONFIG)
    branches = config["next_architecture"]["branches"]
    windows = config["branches"]["context_windows_pixels"]
    width = config["next_architecture"]["prediction_window"]
    require(branches == ["S16", "S32", "S64", "S128"] and windows == [16, 32, 64, 128]
            and width == 16, "CONFIG_CONTRACT", "Review the contract when Model6 context/output sizes change")
    require(manifest.get("context_windows", windows) == windows
            and manifest.get("prediction_window", width) == width, "CONFIG_CONTRACT",
            "Manifest context or output sizes disagree with Model6")
    source_rows = manifest.get("training_observations", []) + manifest.get("observations", [])
    require(0 < len(source_rows) <= MAX_PARENTS, "RESOURCE_BOUND", "Expected 1..256 parent observations")
    rows, identifiers, memberships, partitions = [], set(), {}, {}
    synthetic = manifest.get("data_kind") == "synthetic"
    for source in source_rows:
        ident = source.get("id")
        require(nonempty(ident) and ident not in identifiers, "OBSERVATION_ID", "Missing or duplicate parent ID")
        identifiers.add(ident)
        group, fold = source.get("group"), source.get("partition")
        require(nonempty(group) and type(fold) is int and fold >= 0, "MANIFEST_CONTRACT",
                f"{ident}: group and nonnegative integer partition are required")
        require(group not in partitions or partitions[group] == fold, "GROUP_LEAKAGE",
                f"{group}: connected group crosses partitions")
        partitions[group] = fold
        parent = source.get("parent_id", ident)
        identities = [("parent", parent), ("site", source.get("site")),
                      ("acquisition", source.get("target_scene")),
                      ("acquisition", source.get("reference_scene"))]
        input_hash = source.get("hashes", {}).get("image")
        if input_hash:
            identities.append(("input_hash", input_hash))
        for namespace, identity in identities:
            require(nonempty(identity), "MANIFEST_CONTRACT", f"{ident}: missing {namespace}")
            key = (namespace, identity)
            require(key not in memberships or memberships[key] == group, "GROUP_LEAKAGE",
                    f"{ident}: {namespace} occurs in different connected groups")
            memberships[key] = group
        shape, affine, crs = source.get("shape"), source.get("affine"), source.get("crs")
        require(isinstance(shape, list) and len(shape) == 2
                and all(type(x) is int and 128 <= x <= 4096 for x in shape), "GRID_MISMATCH",
                f"{ident}: parent shape must support an actual S128 context")
        require(isinstance(affine, list) and len(affine) == 6 and all(finite(x) for x in affine)
                and affine[0] * affine[4] - affine[1] * affine[3] != 0 and nonempty(crs),
                "GRID_MISMATCH", f"{ident}: invalid CRS/affine")
        row = {key: source[key] for key in ("id", "site", "group", "partition", "target_scene", "reference_scene")}
        row.update(parent_id=parent, grid={"shape": shape, "affine": affine, "crs": crs},
                   central_region=[(shape[0] - width) // 2, (shape[1] - width) // 2, width, width])
        if synthetic:
            require(source.get("dataset") == "SYNTHETIC", "SYNTHETIC_ONLY", "Every rehearsal parent must be artificial")
            row["synthetic_target"] = source.get("synthetic_target")
        rows.append(row)
    rows.sort(key=lambda row: row["id"])
    folds = sorted(set(partitions.values()))
    require(outer_fold in folds and 3 <= len(folds) <= 10, "FOLD_CONTRACT",
            "Need an outer evaluation fold and at least two separate inner folds; maximum ten")
    fit = [r for r in rows if r["partition"] != outer_fold]
    evaluation = [r for r in rows if r["partition"] == outer_fold]
    jobs = []
    for inner in [f for f in folds if f != outer_fold] + [None]:
        train = [r for r in fit if inner is None or r["partition"] != inner]
        predict = evaluation if inner is None else [r for r in fit if r["partition"] == inner]
        require(train and predict, "FOLD_CONTRACT", "Empty branch training or prediction set")
        for branch, window in zip(branches, windows):
            jobs.append({
                "job_id": f"{branch}_outer{outer_fold}_{'final' if inner is None else 'inner' + str(inner)}",
                "branch": branch, "context_window": window,
                "role": "outer_evaluation" if inner is None else "inner_oof",
                "inner_holdout_fold": inner,
                "training_ids": [r["id"] for r in train],
                "training_groups": sorted({r["group"] for r in train}),
                "prediction_ids": [r["id"] for r in predict],
            })
    return {
        "schema_version": 1, "model_id": "GTM_Model6", "mode": "contract_schedule_only",
        "data_kind": "synthetic" if synthetic else "existing_metadata_only",
        "source_manifest_sha256": digest(manifest_path), "config_sha256": digest(CONFIG),
        "contract_code_sha256": digest(Path(__file__)), "outer_fold": outer_fold,
        "branches": branches, "prediction_window": width, "observations": rows, "jobs": jobs,
        "combiner_fit_ids": [r["id"] for r in fit],
        "combiner_evaluation_ids": [r["id"] for r in evaluation],
        "training_ready": False, "claim_boundary": CLAIM,
        "limits": {"single_central_tile_per_parent": True, "grouping_is_manifest_declared": True,
                   "real_training_runner_integrated": False, "scientific_gates_cleared": False},
    }


#### Check every producer and score before a combiner can consume them
def check_bundle(plan, plan_path, bundle_path):
    bundle = read(bundle_path)
    require(bundle.get("schema_version") == 1, "BUNDLE_CONTRACT", "Unsupported bundle schema")
    require(bundle.get("plan_sha256") == digest(plan_path), "PLAN_HASH", "Bundle refers to another plan")
    root, budget = bundle_path.parent, [0]
    rows = {r["id"]: r for r in plan["observations"]}
    jobs = {j["job_id"]: j for j in plan["jobs"]}
    expected = {(j["job_id"], ident) for j in jobs.values() for ident in j["prediction_ids"]}
    checkpoints, predictions = bundle.get("checkpoints", []), bundle.get("predictions", [])
    require(len(checkpoints) == len(jobs) and {c.get("job_id") for c in checkpoints} == set(jobs),
            "CHECKPOINT_COVERAGE", "Exactly one checkpoint/provenance pair per planned branch fit is required")
    require(len(predictions) == len(expected)
            and {(p.get("job_id"), p.get("id")) for p in predictions} == expected,
            "PREDICTION_COVERAGE", "Predictions are missing, duplicated, or assigned to the wrong producer")
    outer_groups = {rows[i]["group"] for i in plan["combiner_evaluation_ids"]}
    hashes = {}
    for entry in checkpoints:
        job = jobs[entry["job_id"]]
        checkpoint = artifact(root, entry["checkpoint"], budget)
        meta = read(artifact(root, entry["provenance"], budget))
        checkpoint_hash = digest(checkpoint)
        require(meta.get("checkpoint_sha256") == checkpoint_hash, "CHECKPOINT_LINK",
                f'{job["job_id"]}: provenance points to a different checkpoint')
        require(meta.get("job_id") == job["job_id"] and meta.get("branch") == job["branch"]
                and meta.get("context_window") == job["context_window"], "BRANCH_MISMATCH",
                f'{job["job_id"]}: wrong branch/context producer')
        require(meta.get("source_manifest_sha256") == plan["source_manifest_sha256"], "PLAN_HASH",
                f'{job["job_id"]}: producer used a different source manifest')
        require(meta.get("independent_training") is True and meta.get("fused_loss_weight") == 0
                and meta.get("feature_contract") == "sentinel_only", "INDEPENDENCE_CONTRACT",
                f'{job["job_id"]}: expected independent Sentinel-only branch losses')
        if plan["data_kind"] == "synthetic":
            require(meta.get("kind") == "synthetic_branch_stub", "SYNTHETIC_ONLY",
                    "Synthetic scores must not be described as trained real branches")
        training = meta.get("training_ids")
        require(isinstance(training, list) and all(isinstance(i, str) and i in rows for i in training),
                "TRAINING_MEMBERSHIP", f'{job["job_id"]}: unknown or missing training IDs')
        actual_groups = {rows[i]["group"] for i in training}
        forbidden = outer_groups | {rows[i]["group"] for i in job["prediction_ids"]}
        require(not actual_groups & forbidden, "TRAINING_LEAKAGE",
                f'{job["job_id"]}: trained on prediction/evaluation group {sorted(actual_groups & forbidden)}')
        require(len(training) == len(set(training)) and sorted(training) == job["training_ids"]
                and sorted(actual_groups) == meta.get("training_groups"), "TRAINING_MEMBERSHIP",
                f'{job["job_id"]}: training membership differs from the frozen schedule')
        hashes[job["job_id"]] = checkpoint_hash
    scores, supports = {}, {}
    pixels = plan["prediction_window"] ** 2
    for entry in predictions:
        job, row = jobs[entry["job_id"]], rows[entry["id"]]
        data = read(artifact(root, entry["artifact"], budget))
        require(data.get("job_id") == job["job_id"] and data.get("id") == row["id"]
                and data.get("branch") == job["branch"], "BRANCH_MISMATCH",
                f'{row["id"]}: swapped prediction producer/branch')
        require(data.get("checkpoint_sha256") == hashes[job["job_id"]], "CHECKPOINT_LINK",
                f'{row["id"]}: prediction checkpoint differs from the producer')
        require(data.get("grid") == row["grid"] and data.get("central_region") == row["central_region"],
                "GRID_MISMATCH", f'{row["id"]}: grid or central region differs')
        support, logits = data.get("support"), data.get("logits")
        require(data.get("score_type") == "mask_logit" and isinstance(support, list)
                and isinstance(logits, list) and len(support) == len(logits) == pixels
                and all(type(s) is bool for s in support), "SCORE_CONTRACT",
                f'{row["id"]}: expected 16x16 mask logits with explicit boolean support')
        require(all(finite(x) if valid else x is None for x, valid in zip(logits, support)),
                "SCORE_CONTRACT", f'{row["id"]}: finite supported scores and null unknown scores required')
        require(any(support), "SCORE_CONTRACT", f'{row["id"]}: no prediction support')
        require(row["id"] not in supports or supports[row["id"]] == support, "SUPPORT_MISMATCH",
                f'{row["id"]}: branches do not share the same valid output pixels')
        supports[row["id"]] = support
        scores.setdefault(row["id"], {})[job["branch"]] = logits
    result = {
        "schema_version": 1, "status": "contract_passed", "training_ready": False,
        "claim_boundary": CLAIM, "data_kind": plan["data_kind"], "outer_fold": plan["outer_fold"],
        "manifest_sha256": plan["source_manifest_sha256"], "plan_sha256": digest(plan_path),
        "bundle_sha256": digest(bundle_path), "checkpoint_pairs_checked": len(checkpoints),
        "prediction_tiles_checked": len(predictions), "artifact_bytes_checked": budget[0],
        "combiner_fit_ids": plan["combiner_fit_ids"], "combiner_evaluation_ids": plan["combiner_evaluation_ids"],
        "evaluation_labels_read": False,
        "limitations": ["Declared lineage cannot prove a producer's true training history.",
                       "Group metadata is checked for consistency, not independently geolocated.",
                       "Single central tile only; real runner/full-scene export integration is pending.",
                       "Label validity, units, QA and independent paired data remain separate blocked gates."],
    }
    return result, scores, supports


#### Fixed toy combiner: plumbing only, not the research architecture selection
def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def rehearse(plan, manifest_path, result, scores, supports):
    samples, target_hashes, budget = [], {}, [0]
    rows = {r["id"]: r for r in plan["observations"]}
    for ident in plan["combiner_fit_ids"]:
        # Never open or hash an outer evaluation target.
        ref = rows[ident]["synthetic_target"]
        target = read(artifact(manifest_path.parent, ref, budget))
        labels, known = target.get("labels"), target.get("known")
        require(target.get("shape") == [16, 16] and isinstance(labels, list) and isinstance(known, list)
                and len(labels) == len(known) == 256 and all(type(v) is bool for v in known),
                "TARGET_CONTRACT", f"{ident}: invalid toy target/support shape")
        require(all((type(y) is int and y in (0, 1)) if valid else y is None
                    for y, valid in zip(labels, known)), "TARGET_CONTRACT",
                f"{ident}: unknown labels must remain null")
        target_hashes[ident] = ref["sha256"]
        for index, (label, observed) in enumerate(zip(labels, known)):
            if observed and supports[ident][index]:
                samples.append(([scores[ident][b][index] for b in plan["branches"]], label))
    require(samples and {label for _, label in samples} == {0, 1}, "TARGET_CONTRACT",
            "Toy fitting needs observed examples of both labels")
    weights, bias, steps, rate = [0.0] * 4, 0.0, 24, .05
    for _ in range(steps):
        gradients, bias_gradient = [0.0] * 4, 0.0
        for vector, label in samples:
            error = sigmoid(bias + sum(w * x for w, x in zip(weights, vector))) - label
            bias_gradient += error
            for index, x in enumerate(vector):
                gradients[index] += error * x
        weights = [w - rate * g / len(samples) for w, g in zip(weights, gradients)]
        bias -= rate * bias_gradient / len(samples)
    require(all(finite(x) for x in [*weights, bias]), "COMBINER_NONFINITE", "Toy optimizer became nonfinite")
    predictions = []
    for ident in plan["combiner_evaluation_ids"]:
        combined, mean = [], []
        for index, valid in enumerate(supports[ident]):
            if not valid:
                combined.append(None)
                mean.append(None)
                continue
            vector = [scores[ident][b][index] for b in plan["branches"]]
            combined.append(round(sigmoid(bias + sum(w * x for w, x in zip(weights, vector))), 12))
            mean.append(round(sigmoid(sum(vector) / len(vector)), 12))
        predictions.append({"id": ident, "grid": rows[ident]["grid"],
                            "central_region": rows[ident]["central_region"],
                            "probability": combined, "mean_probability": mean})
    result.update(
        status="synthetic_rehearsal_passed", training_targets_sha256=target_hashes,
        combiner={"kind": "synthetic_only_logistic_plumbing", "branches": plan["branches"],
                  "weights": [round(w, 12) for w in weights], "bias": round(bias, 12),
                  "steps": steps, "learning_rate": rate, "fit_pixels": len(samples),
                  "architecture_selected_for_real_research": False},
        outer_predictions=predictions, scientific_metrics_computed=False,
        real_unets_trained=False, scientific_gates_cleared=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "rehearse"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        print(json.dumps({"status": "refused_existing_output"}))
        return 3
    try:
        if args.command == "plan":
            require(args.outer_fold is not None, "ARGUMENT", "--outer-fold is required for plan")
            result = make_plan(args.manifest, args.outer_fold)
        else:
            require(args.plan is not None and args.bundle is not None, "ARGUMENT", "--plan and --bundle are required")
            plan = read(args.plan)
            expected = make_plan(args.manifest, plan.get("outer_fold"))
            require(plan == expected, "PLAN_MISMATCH", "Plan/source/config/code changed; regenerate the schedule")
            if args.command == "rehearse":
                require(plan["data_kind"] == "synthetic", "SYNTHETIC_ONLY",
                        "This command fits only artificial score fixtures, never real imagery/labels")
            result, scores, supports = check_bundle(plan, args.plan, args.bundle)
            if args.command == "rehearse":
                result = rehearse(plan, args.manifest, result, scores, supports)
        code = 0
    except (Rejected, OSError, KeyError, TypeError, AttributeError, OverflowError, ValueError) as exc:
        result = {"status": "rejected", "error_code": getattr(exc, "code", "INVALID_CONTRACT"),
                  "reason": str(exc), "training_ready": False, "claim_boundary": CLAIM}
        code = 2
    try:
        write_new(args.output, result)
    except FileExistsError:
        print(json.dumps({"status": "refused_existing_output"}))
        return 3
    print(json.dumps({"status": result.get("status", result.get("mode")), "output": str(args.output),
                      "error_code": result.get("error_code")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
