"""Reconcile an existing mapper run with its saved pixels, without retraining.

Writes a new audit and optionally a corrected viewer export. Frozen checkpoints,
predictions, manifests and original run reports are never rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_GTM_Model6_mapper_mars_pilot import ROOT, _metrics, _save_bundle, digest, inventory


def audit_saved_run(run: Path) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    manifest = json.loads((run / "GTM_Model6_mapper_mars_pilot_manifest.json").read_text())
    result = json.loads((run / "GTM_Model6_mapper_mars_pilot_results.json").read_text())
    bundle = ROOT / "outputs/GTM_viewer_bundles" / (run.name + "_mean")
    old = json.loads((bundle / "experiment.json").read_text(encoding="utf-8"))
    old_by_id = {row["id"]: row for row in old["scenes"]}
    report_rows, predictions, supports = [], [], []
    mismatch_keys = []
    for row, rec in zip(manifest["observations"], result["observations"], strict=True):
        if row["id"] != rec["id"]:
            raise ValueError("frozen manifest/result order mismatch")
        path = run / f"prediction_{rec['index']:04d}.npz"
        with np.load(path) as z:
            p, y, valid = z["probability"], z["target"], z["valid"].astype(bool)
        m = _metrics(p, {"target": y, "valid": valid})
        for key, value in rec["metrics"].items():
            if value is not None and not np.isclose(m[key], value, rtol=1e-6, atol=1e-8):
                mismatch_keys.append({"id": row["id"], "key": key})
        indices = np.argwhere(valid)
        bounds = [*indices.min(0).tolist(), *(indices.max(0) + 1).tolist()] if len(indices) else None
        viewer_counts = old_by_id[row["id"]]["nativeMetrics"]
        viewer_count = sum(viewer_counts[k] for k in ("tp", "fp", "fn", "tn"))
        report_rows.append({
            "id": row["id"], "label": row["label"], "metrics": m,
            "support_bounds_top_left_bottom_right": bounds,
            "saved_prediction_sha256": digest(path),
            "maximum_supported_probability": float(p[valid].max()) if valid.any() else None,
            "old_viewer_evaluated_pixels": viewer_count,
            "old_viewer_support_matches_saved": viewer_count == m["evaluated_pixels"],
        })
        predictions.append(p)
        supports.append(valid)
    counts = {k: sum(r["metrics"][k] for r in report_rows) for k in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    pooled = {**counts, "iou": tp / max(tp + fp + fn, 1),
              "dice": 2 * tp / max(2 * tp + fp + fn, 1),
              "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
              "evaluated_pixels": tp + fp + fn + tn}
    return {
        "run_id": run.name, "scope": "saved prediction audit; no model training or new holdout",
        "checkpoint_sha256": digest(run / "GTM_Model6_mapper_checkpoint.pt"),
        "checkpoint_matches_result": digest(run / "GTM_Model6_mapper_checkpoint.pt") == result["checkpoint_sha256"],
        "frozen_metric_mismatches": mismatch_keys, "pooled_metrics": pooled,
        "viewer_support_mismatch_scenes": sum(not r["old_viewer_support_matches_saved"] for r in report_rows),
        "positive_scenes": sum(r["label"] == "PLUME" for r in report_rows),
        "negative_scenes": sum(r["label"] == "NO_PLUME" for r in report_rows),
        "observations": report_rows,
    }, predictions, supports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a directory name")
    run = ROOT / "outputs/GTM_Model6" / args.run_id
    output = run.parent / (run.name + "_audit")
    if output.exists():
        raise FileExistsError(output)
    audit, predictions, supports = audit_saved_run(run)
    if audit["frozen_metric_mismatches"] or not audit["checkpoint_matches_result"]:
        raise ValueError("frozen artifacts disagree; export blocked")
    report = "\n".join([
        "# Mapper saved-prediction audit", "",
        f"Run: `{run.name}`. Checkpoint and all 65 saved predictions are unchanged.", "",
        f"- Frozen native metrics recompute: {not audit['frozen_metric_mismatches']}.",
        f"- Old viewer support mismatches: {audit['viewer_support_mismatch_scenes']} scenes.",
        f"- Positive/negative scenes: {audit['positive_scenes']}/{audit['negative_scenes']}.",
        f"- Pooled native IoU: {audit['pooled_metrics']['iou']:.4%}.", "",
        "The original viewer metadata scored the full clear scene, while inference covered only the central 88x88 region of a 200x200 source. A 128-pixel context and 16-pixel output tile leave a 56-pixel margin. The old 64/128-pixel margin descriptions were wrong. The corrected export uses the exact saved support for both metrics and display. Unsupported area is unknown, not a true negative.", "",
        "This correction does not turn the all-negative output into a successful model. The run contains 3 positive and 62 negative scenes; its original mean-per-scene aggregate was not a pooled IoU. A new training-only diagnosis is required before attributing this failure to data scarcity or architecture.", "",
        "This is reused development evidence. It provides no EMIT enhancement, methane super-resolution, source localization or nationwide validation.", "",
    ])
    output.mkdir()
    if args.export:
        items, _ = inventory(128)
        by_id = {i["row"]["id"]: i for i in items}
        manifest = json.loads((run / "GTM_Model6_mapper_mars_pilot_manifest.json").read_text())
        held = []
        for row in manifest["observations"]:
            item = by_id[row["id"]]
            if item["row"]["hashes"] != row["hashes"] or item["row"]["partition"] != manifest["holdout_fold"]:
                raise ValueError("input hashes or held split changed")
            held.append(item)
        bundle = _save_bundle(run, run.name + "_checked", held, predictions, supports, 0.5,
                             holdout_fold=manifest["holdout_fold"], report_text=report,
                             stage="corrected support; failed development run")
        audit["corrected_viewer_bundle"] = str(bundle)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    (output / "audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
