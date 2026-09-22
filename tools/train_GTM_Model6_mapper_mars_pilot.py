"""Run one bounded, held-group MARS mask pilot for the Model6 mapper.

This is an engineering diagnostic for the new independent S16/S32/S64/S128
branches.  It uses only the already-local, fully-labelled MARS masks and does
not train the EMIT enhancement head.  It deliberately does not claim EMIT
upscaling, source localization, CAFO recovery, or nationwide validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL6 = ROOT / "research" / "GTM_Model6"
sys.path.insert(0, str(MODEL6))

from GTM_Model6_mapper import CONTEXT_WINDOWS, MultiScaleMapper, multiscale_mapper_loss  # noqa: E402
from GTM_Model6_mapper_data import sample_batch  # noqa: E402
from GTM_Model6_plume_export import (  # noqa: E402
    candidates,
    digest,
    geometry,
    rgba,
    rgb_image,
    warp,
)
from GTM_Model6_plumes import inventory  # noqa: E402


SEED = 20260922
RUN_ID = "GTM_Model6_mapper_mars_pilot_20260922"
WINDOW = 128
PREDICTION_WINDOW = 16
HOLDOUT_FOLD = 1
BASE_CHANNELS = 8


def _source_hash() -> str:
    value = hashlib.sha256()
    for path in (
        Path(__file__),
        MODEL6 / "GTM_Model6_mapper.py",
        MODEL6 / "GTM_Model6_plumes.py",
        MODEL6 / "GTM_Model6_mapper_data.py",
    ):
        value.update(path.read_bytes())
    return value.hexdigest()


def _metrics(
    probability: np.ndarray,
    item: dict[str, Any],
    threshold: float = 0.5,
    evaluation_support: np.ndarray | None = None,
) -> dict[str, Any]:
    valid = item["valid"].astype(bool)
    if probability.shape != valid.shape or item["target"].shape != valid.shape:
        raise ValueError("prediction, target and validity grids must match")
    if evaluation_support is not None:
        if evaluation_support.shape != valid.shape:
            raise ValueError("evaluation support must match the native grid")
        valid &= evaluation_support.astype(bool)
    if not np.isfinite(probability[valid]).all():
        raise ValueError("nonfinite prediction on evaluation support")
    truth = (item["target"] > 0) & valid
    predicted = (probability >= threshold) & valid
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth & valid).sum())
    fn = int((~predicted & truth & valid).sum())
    tn = int((~predicted & ~truth & valid).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "evaluated_pixels": int(valid.sum()),
        "source_valid_pixels": int(item["valid"].sum()),
        "support_fraction": float(valid.mean()),
        "predicted_fraction": float(predicted[valid].mean()) if valid.any() else None,
        "truth_fraction": float(truth[valid].mean()) if valid.any() else None,
        "negative_activation": float(predicted[valid & ~truth].mean()) if (~truth & valid).any() else None,
    }


@torch.no_grad()
def _dense_predict(
    model: MultiScaleMapper,
    item: dict[str, Any],
    device: str,
    stride: int = 8,
    return_branches: bool = False,
) -> tuple[Any, np.ndarray]:
    """Tile exact native contexts and return probability plus supported centres."""

    h, w = item["x"].shape[-2:]
    radius = max(CONTEXT_WINDOWS) // 2
    half = PREDICTION_WINDOW // 2
    probability = np.zeros((h, w), dtype="float32")
    branch_sums = {f'S{size}': np.zeros((h, w), dtype='float32') for size in CONTEXT_WINDOWS} if return_branches else {}
    weights = np.zeros((h, w), dtype="float32")
    scene_support = np.zeros((h, w), dtype=bool)
    if stride < 1 or stride > PREDICTION_WINDOW:
        raise ValueError("stride must be positive and no larger than the prediction tile")
    if min(h, w) < 2 * radius:
        raise ValueError("scene cannot provide the largest native context")
    centers_r = sorted(set(list(range(radius, h - radius + 1, stride)) + [h - radius]))
    centers_c = sorted(set(list(range(radius, w - radius + 1, stride)) + [w - radius]))
    positions = [(row, col) for row in centers_r for col in centers_c]
    for start in range(0, len(positions), 8):
        batch_positions = positions[start : start + 8]
        contexts: dict[str, torch.Tensor] = {}
        for window in CONTEXT_WINDOWS:
            values = []
            for row, col in batch_positions:
                top, left = row - window // 2, col - window // 2
                values.append(torch.from_numpy(item["x"][:16, top : top + window, left : left + window]).float())
            contexts[f"S{window}"] = torch.stack(values).to(device)
        output = model(contexts)
        tile = output["mask_logits"].sigmoid().cpu().numpy()[:, 0]
        branch_tiles = {name: out['mask_logits'].sigmoid().cpu().numpy()[:, 0]
                        for name, out in output.get('branches', {}).items()} if return_branches else {}
        for patch_index, ((row, col), patch) in enumerate(zip(batch_positions, tile)):
            top, left = row - half, col - half
            probability[top : top + PREDICTION_WINDOW, left : left + PREDICTION_WINDOW] += patch
            weights[top : top + PREDICTION_WINDOW, left : left + PREDICTION_WINDOW] += 1.0
            scene_support[top : top + PREDICTION_WINDOW, left : left + PREDICTION_WINDOW] = True
            for name, values in branch_tiles.items():
                branch_sums[name][top:top+PREDICTION_WINDOW, left:left+PREDICTION_WINDOW] += values[patch_index]
    probability = np.divide(probability, np.maximum(weights, 1.0), where=weights > 0, out=np.zeros_like(probability))
    if return_branches:
        return {'mean': probability, **{name: values / np.maximum(weights, 1)
                                      for name, values in branch_sums.items()}}, scene_support
    return probability, scene_support


def _pooled_metrics(rows):
    tp, fp, fn, tn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn'))
    n = tp + fp + fn + tn
    positive = [r for r in rows if r['tp'] + r['fn'] > 0]
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'evaluated_pixels': n,
            'iou': tp / max(tp + fp + fn, 1), 'dice': 2 * tp / max(2 * tp + fp + fn, 1),
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'predicted_fraction': (tp+fp) / max(n, 1), 'truth_fraction': (tp+fn) / max(n, 1),
            'negative_activation': fp / max(fp+tn, 1),
            'positive_scene_macro_iou': float(np.mean([r['iou'] for r in positive])) if positive else None,
            'positive_scenes': len(positive), 'scenes': len(rows)}


def _save_bundle(
    run: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    predictions: list[np.ndarray],
    supports: list[np.ndarray],
    threshold: float,
    *,
    holdout_fold: int = HOLDOUT_FOLD,
    report_text: str | None = None,
    stage: str = "held-fold diagnostic",
    training_only: bool = False,
    bundle_suffix: str = "mean",
) -> Path:
    bundle = ROOT / "outputs" / "GTM_viewer_bundles" / f"{run_id}_{bundle_suffix}"
    if bundle.exists():
        raise SystemExit(f"Refusing to overwrite viewer bundle: {bundle}")
    bundle.mkdir(parents=True)
    scenes: list[dict[str, Any]] = []
    feature_list: list[dict[str, Any]] = []
    for index, (item, probability, scene_support) in enumerate(zip(rows, predictions, supports)):
        row = item["row"]
        native, display_transform, bounds = geometry(row, probability.shape)
        valid = item["valid"].astype(bool) & scene_support.astype(bool)
        display_valid = warp(valid, row, native, display_transform) > 0.5
        display_probability = warp(
            np.where(valid, probability, np.nan),
            row,
            native,
            display_transform,
            nodata=np.nan,
        )
        display_probability = np.nan_to_num(display_probability)
        display_truth = warp(
            np.where(valid, item["target"], np.nan),
            row,
            native,
            display_transform,
            nodata=np.nan,
        )
        display_truth_mask = np.isfinite(display_truth) & display_valid
        display_rgb = np.stack(
            [warp(item["rgb"][:, :, channel], row, native, display_transform) for channel in range(3)],
            axis=-1,
        )
        stem = f"{index:04d}"
        Image.fromarray(rgb_image(display_rgb)).save(bundle / f"{stem}_rgb.png")
        Image.fromarray(rgba(np.nan_to_num(display_truth), display_truth_mask, "truth")).save(bundle / f"{stem}_truth.png")
        Image.fromarray(rgba(display_probability >= threshold, display_valid, "prediction")).save(bundle / f"{stem}_prediction.png")
        Image.fromarray(rgba(display_probability, display_valid, "probability")).save(bundle / f"{stem}_probability.png")
        grid = {
            "width": int(display_probability.shape[1]),
            "height": int(display_probability.shape[0]),
            "crs": "EPSG:3857",
            "probability": np.round(display_probability, 5).ravel().tolist(),
            "truth": [
                int(value) if good else None
                for value, good in zip(display_truth.ravel(), display_truth_mask.ravel())
            ],
            "valid": display_valid.astype("uint8").ravel().tolist(),
        }
        (bundle / f"{stem}_grid.json").write_text(json.dumps(grid), encoding="utf-8")
        points = candidates(probability, valid, row, threshold)
        for point in points:
            feature_list.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [point["lon"], point["lat"]]},
                    "properties": {"scene": row["id"], **point},
                }
            )
        metrics = _metrics(probability, item, threshold, evaluation_support=scene_support)
        supported_rows, supported_cols = np.where(scene_support)
        support_bounds = ([int(supported_rows.min()), int(supported_cols.min()),
                           int(supported_rows.max()) + 1, int(supported_cols.max()) + 1]
                          if len(supported_rows) else None)
        label = "Reviewed plume" if row["label"] == "PLUME" else "Reviewed no plume"
        scenes.append(
            {
                "id": row["id"],
                "name": f"MARS | {label} | {row['date'][:10]} | {row['id'][-8:]}",
                "task": "plume_segmentation",
                "location": f"{row['lat']:.3f}, {row['lon']:.3f} | United States",
                "date": row["date"][:10],
                "dataset": "MARS",
                "label": row["label"],
                "positiveOnlyReference": False,
                "bounds": bounds,
                "rgb": f"{stem}_rgb.png",
                "truthImage": f"{stem}_truth.png",
                "predictionImage": f"{stem}_prediction.png",
                "probabilityImage": f"{stem}_probability.png",
                "grid": f"{stem}_grid.json",
                "predictionAvailable": True,
                "decisionThreshold": threshold,
                "truthLabel": "MARS reviewed plume mask",
                "truthNote": "MARS reviewed mask; scoring intersects clear Sentinel and actual prediction support",
                "referenceDate": row["date"][:10],
                "notes": ("TRAINING FIT ONLY: this exact patch was used to fit the model; no generalization score. " if training_only
                          else "Reused development fold excluded from this fit; not a new confirmation set. ") + (
                    "Mask-only MARS engineering diagnostic; EMIT enhancement, source localization, "
                    "CAFO recovery, and nationwide validity are not evaluated. Exact context tiling "
                    "leaves unsupported pixels blank. Native metrics use the same prediction support."
                ) + f" Displayed method: {bundle_suffix}. The report headline describes the mean; its table includes all branches.",
                "predictionSupportBounds": support_bounds,
                "source": row["source"],
                "checkpointHash": digest(run / "GTM_Model6_mapper_checkpoint.pt"),
                "predictionHash": digest(run / f"prediction_{index:04d}.npz"),
                "split": (f"TRAINING partition {row['partition']}; fitted example; group {row['group']}" if training_only
                          else f"Held MARS partition {holdout_fold}; group {row['group']}; reused development data"),
                "resolution": f"{probability.shape[1]}x{probability.shape[0]} source grid (10m sampling) / 160x160 EPSG:3857 display",
                "nativeMetrics": metrics,
                "candidatePoints": points[:5],
                "candidateCount": len(points),
            }
        )
    scenes.sort(key=lambda scene: (scene["label"] == "NO_PLUME", scene["date"], scene["id"]))
    experiment = {
        "id": f"{run_id}_{bundle_suffix}",
        "name": f"Model6 mapper | {run_id.removeprefix('GTM_Model6_mapper_')} | {bundle_suffix} | {stage}",
        "lifecycle": "experimental",
        "status": ("TRAINING FIT ONLY · memorized real patches; no held-out accuracy" if training_only
                   else "ACTUAL HELD MARS OUTPUT · engineering diagnostic, not promoted"),
        "description": ("Fixed mean of independent S16/S32/S64/S128 logits." if bundle_suffix == 'mean'
                        else f"Independent {bundle_suffix} branch probability, not the fused mean."),
        "notes": ("TRAINING FIT ONLY: all examples shown were used for fitting. This checks that the optimizer can learn real inputs. " if training_only
                  else "Reused development fold; images were excluded from this fit but are not new confirmation data. ")
                  + f"Displayed method: {bundle_suffix}. No EMIT upscaling or national methane-source result. Report headline: mean; branch scores are in the comparison table.",
        "decisionThreshold": threshold,
        "nativeAggregate": _pooled_metrics([scene['nativeMetrics'] for scene in scenes]),
        "metricScope": "training_fit" if training_only else "reused_development_fold",
        "scenes": scenes,
        "reportUrl": "GTM_Model6_mapper_mars_pilot_report.md",
    }
    (bundle / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    (bundle / "GTM_Model6_mapper_mars_pilot_report.md").write_text(
        report_text if report_text is not None else (run / "GTM_Model6_mapper_mars_pilot_report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (bundle / "GTM_Model6_mapper_candidate_hotspots.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feature_list}), encoding="utf-8"
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--holdout-fold", type=int, default=HOLDOUT_FOLD)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.minutes <= 0 or Path(args.run_id).name != args.run_id:
        parser.error("require a simple run name, positive steps/minutes, and batch size >= 2")
    run = ROOT / "outputs" / "GTM_Model6" / args.run_id
    if run.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run}")
    run.mkdir(parents=True)
    source_dir = run / 'source'
    source_dir.mkdir()
    for source in [Path(__file__), MODEL6/'GTM_Model6_mapper.py', MODEL6/'GTM_Model6_mapper_data.py',
                   MODEL6/'GTM_Model6_plumes.py', MODEL6/'GTM_Model6_plume_export.py']:
        shutil.copyfile(source, source_dir/source.name)
    (run/'predeclared_protocol.json').write_text(json.dumps({
        'seed': SEED, 'steps': args.steps, 'threshold': .5, 'learning_rate': 3e-4,
        'holdout_fold': args.holdout_fold, 'batch_size': args.batch_size, 'primary_method': 'mean',
        'scope': 'reused development cohort; sampler/independent-loss repair; no new confirmation',
        'source_hashes': {p.name: digest(p) for p in source_dir.iterdir()},
        'training_sanity_required': 'GTM_Model6_mapper_training_sanity_20260922/receipt.json',
    }, indent=2))
    sanity = ROOT/'outputs/GTM_Model6/GTM_Model6_mapper_training_sanity_20260922/receipt.json'
    if not sanity.is_file() or not json.loads(sanity.read_text()).get('gate_passed'):
        raise ValueError('real training-patch sanity check must pass before another held-fold diagnostic')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    started = time.monotonic()
    deadline = started + args.minutes * 60
    items, excluded = inventory(128)
    mars = [item for item in items if item["row"]["dataset"] == "MARS"]
    train_items = [item for item in mars if item["row"]["partition"] != args.holdout_fold]
    held_items = [item for item in mars if item["row"]["partition"] == args.holdout_fold]
    if not train_items or not held_items:
        raise SystemExit("MARS train/held split is empty")
    if not any(item["target"].any() for item in train_items):
        raise SystemExit("MARS training split has no positive masks")
    if not any(item["target"].any() for item in held_items):
        raise SystemExit("held fold has no positive mask for a useful visual check")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiScaleMapper(input_channels=16, base_channels=BASE_CHANNELS, prediction_window=PREDICTION_WINDOW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    log: list[dict[str, Any]] = []
    telemetry: dict[str, Any] = {}
    # Separate branch losses meet Martin's independent-training requirement;
    # the fixed fusion is evaluated but sends no cross-branch training gradient.
    model.train()
    for step in range(args.steps):
        if time.monotonic() >= deadline:
            raise TimeoutError("compute cap reached before diagnostic completed")
        contexts, y, support = sample_batch(train_items, rng, args.batch_size, telemetry=telemetry)
        contexts = {name: value.to(device) for name, value in contexts.items()}
        y, support = y.to(device), support.to(device).bool()
        output = model(contexts)
        zeros = torch.zeros_like(y)
        loss_values = multiscale_mapper_loss(
            output,
            y,
            support,
            zeros,
            torch.zeros_like(support),
            enhancement_weight=0.0,
            fusion_weight=0.0,
        )
        loss = loss_values["loss"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite mapper loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (step + 1) % 20 == 0:
            log.append({"step": step + 1, "loss": float(loss.detach().cpu()),
                        "positive_pixels": int((y.bool() & support).sum()),
                        "positive_tiles": int((y.bool() & support).flatten(1).any(1).sum())})
    model.eval()
    checkpoint = run / "GTM_Model6_mapper_checkpoint.pt"
    torch.save(
        {
            "model_id": "GTM_Model6_multiscale_sentinel_emit_mapper",
            "input_channels": 16,
            "base_channels": BASE_CHANNELS,
            "branches": ["S16", "S32", "S64", "S128"],
            "state_dict": model.cpu().state_dict(),
            "seed": SEED,
            "holdout_fold": args.holdout_fold,
            "training_log": log,
            "training_ids": [item['row']['id'] for item in train_items],
            "prediction_window": PREDICTION_WINDOW,
            "fusion_mode": "logit_mean",
            "branch_training": "independent losses; no fused training gradient",
        },
        checkpoint,
    )
    model.to(device)
    predictions: list[np.ndarray] = []
    method_predictions: dict[str, list[np.ndarray]] = {f'S{s}': [] for s in CONTEXT_WINDOWS}
    baseline_rows = {name: [] for name in ('all_negative', 'all_positive', 'spectral_baseline', 'training_spatial_prior')}
    positive_parents = [item for item in train_items if item['target'].any()]
    spatial_prior = np.mean([i['target'] for i in positive_parents], axis=0)
    prediction_supports: list[np.ndarray] = []
    result_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, item in enumerate(held_items):
            all_predictions, scene_support = _dense_predict(model, item, device, return_branches=True)
            probability = all_predictions['mean']
            predictions.append(probability)
            for name in method_predictions:
                method_predictions[name].append(all_predictions[name])
            prediction_supports.append(scene_support)
            metrics = _metrics(probability, item, evaluation_support=scene_support)
            np.savez_compressed(
                run / f"prediction_{index:04d}.npz",
                probability=probability,
                target=item["target"].astype("float32"),
                valid=(item["valid"] & scene_support).astype("uint8"),
                context_support=scene_support.astype('uint8'),
                **{name: p for name, p in all_predictions.items() if name != 'mean'},
            )
            per_method = {name: _metrics(p, item, evaluation_support=scene_support) for name, p in all_predictions.items()}
            for name, baseline in {
                'all_negative': np.zeros_like(probability), 'all_positive': np.ones_like(probability),
                'spectral_baseline': 1/(1+np.exp(-item['x'][16])), 'training_spatial_prior': spatial_prior,
            }.items():
                baseline_rows[name].append(_metrics(baseline, item, evaluation_support=scene_support))
            result_rows.append({"index": index, "id": item["row"]["id"], "metrics": metrics, 'per_method': per_method})
    aggregate = _pooled_metrics([row['metrics'] for row in result_rows])
    manifest = {
        "task": "bounded MARS mask diagnostic for multiscale Sentinel-to-EMIT mapper contract",
        "status": "completed_engineering_diagnostic_not_promoted",
        "seed": SEED,
        "device": device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "context_windows": list(CONTEXT_WINDOWS),
        "prediction_window": PREDICTION_WINDOW,
        "window": WINDOW,
        "base_channels": BASE_CHANNELS,
        "holdout_fold": args.holdout_fold,
        "training_records": len(train_items),
        "held_records": len(held_items),
        "augmentation": "mirror plus rotations 0/15/30/45, sampled online",
        "fusion_mode": "logit_mean",
        "branch_supervision": "mean of independent branch losses; no fused training gradient",
        "learning_rate": 3e-4,
        "sampling": "balanced positive/reviewed-negative parents; positives guaranteed after full-parent transform",
        "sampling_telemetry": telemetry,
        "feature_contract": "17 stored Sentinel-derived features: mapper inputs are first 16; seventeenth analytic evidence is baseline-only",
        "spatial_prior_definition": "mean training-positive mask in crop pixel coordinates; shortcut baseline, not a geographic prior",
        "edge_support": "exact S128 context and 16-pixel output: outer 56-pixel margin excluded from scoring",
        "emit_enhancement_trained": False,
        "source_origin_trained": False,
        "nationally_validated": False,
        "cafo_validated": False,
        "source_hash": _source_hash(),
        "excluded_inventory_records": len(excluded),
        "observations": [item["row"] for item in held_items],
        "training_observations": [item["row"] for item in train_items],
    }
    (run / "GTM_Model6_mapper_mars_pilot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "status": "completed_engineering_diagnostic_not_promoted",
        "aggregate": aggregate,
        "aggregate_definition": "pooled native counts on observed inference support; not mean of scene IoUs",
        "methods": {name: _pooled_metrics([r['per_method'][name] for r in result_rows]) for name in ['mean', *method_predictions]},
        "baselines": {name: _pooled_metrics(rows) for name, rows in baseline_rows.items()},
        "observations": result_rows,
        "training_log": log,
        "checkpoint_sha256": digest(checkpoint),
        "runtime_seconds": time.monotonic() - started,
        "device": device,
    }
    (run / "GTM_Model6_mapper_mars_pilot_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_lines = [
        "# GTM_Model6 multiscale mapper MARS pilot",
        "",
        "Status: **completed engineering diagnostic; not promoted**.",
        "",
        "This run trained the independent S16/S32/S64/S128 mask branches on the existing fully-labelled MARS masks and evaluated one held geographic partition. The EMIT quantitative enhancement head was disabled because the current EMIT cohort has positive-only support and no quantitative target. This is not evidence of EMIT upscaling, source-origin localization, CAFO recovery, or nationwide screening.",
        "",
        f"- Device: `{device}`; steps: `{args.steps}`; held fold: `{args.holdout_fold}`; train/held records: `{len(train_items)}/{len(held_items)}`",
        f"- Checkpoint SHA256: `{digest(checkpoint)}`",
        f"- Source bundle hash: `{manifest['source_hash']}`",
        "",
        "## Reused development fold: pooled native aggregate",
        "",
        "| IoU | Dice | Precision | Recall | Predicted fraction | Negative activation |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {aggregate['iou']:.4%} | {aggregate['dice']:.4%} | {aggregate['precision']:.4%} | {aggregate['recall']:.4%} | {aggregate['predicted_fraction']:.4%} | {aggregate['negative_activation']:.4%} |",
        "",
        "| Method | Pooled IoU | Precision | Recall | Background activation |",
        "|---|---:|---:|---:|---:|",
        *[f"| {name} | {m['iou']:.3%} | {m['precision']:.3%} | {m['recall']:.3%} | {m['negative_activation']:.3%} |"
          for name, m in {**result['methods'], **result['baselines']}.items()],
        "",
        "The local viewer bundle contains every held scene, fixed threshold 0.50, RGB, reviewed MARS mask, probability layer, prediction mask, georeferenced display bounds, and candidate hotspot markers. Exact S128 context with 16-pixel output tiles leaves the outer 56-pixel margin unsupported and blank in scoring/display. Inspect both positive and negative scenes before treating the aggregate as useful.",
        "",
        "The next scientifically meaningful run requires time-aligned EMIT quantitative enhancement and representative controls. Until then the mapper's second head remains a tested contract, not a trained result.",
        "",
    ]
    report = "\n".join(report_lines)
    (run / "GTM_Model6_mapper_mars_pilot_report.md").write_text(report, encoding="utf-8")
    bundle = _save_bundle(run, args.run_id, held_items, predictions, prediction_supports, 0.5, holdout_fold=args.holdout_fold)
    for name, values in method_predictions.items():
        _save_bundle(run, args.run_id, held_items, values, prediction_supports, .5,
                     holdout_fold=args.holdout_fold, bundle_suffix=name,
                     stage='independent branch; reused development fold')
    print(
        json.dumps(
            {
                "run": str(run),
                "bundle": str(bundle),
                "aggregate": aggregate,
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
