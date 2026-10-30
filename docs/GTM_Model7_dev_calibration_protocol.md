# Development cutoff calibration, before final evaluation

The original fixed-0.50 comparisons remain unchanged. Visual review of the complete MARS development set showed that positive-scene IoU and a 5% clear-pixel false-alarm limit can accept a model with very poor pooled precision. This additional development-only calibration asks whether saved scores rank plume pixels well enough for a stricter decision. It does not retrain models or establish national reliability.

## Failure cases fixed before implementation

- Never read final-evaluation images, labels or model predictions for cutoff or checkpoint choice. Use only independently replayed TRAIN/DEV runs, sealed manifests and their saved DEV probability files. Do not overwrite original receipts or predictions.
- Verify trainer, model, protocol, manifest, checkpoint and every used DEV array/probability hash. Recompute native-grid confusion from actual scores and native support, not the old coarse summaries. Reject inconsistent grids, changed support, nonfinite scores or values outside zero to one.
- Evaluate the fixed cutoff list 0.05, 0.10, ..., 0.95, 0.975, 0.99, 0.995 for each of the four saved milestones and each model. No scene-specific cutoffs, smoothing or label-dependent support changes.
- For each model within each run, choose the highest pooled native DEV IoU among candidates with pooled precision at least 0.50, pooled recall at least 0.30, pooled IoU at least 0.20, and no-plume pixel false-positive rate at most 0.001. At least half of the positive scenes must recover at least 10% of their supported labeled plume pixels. Ties prefer lower negative FPR, then earlier checkpoint, then higher cutoff.
- If none qualifies, retain the highest pooled-IoU candidate only as a failed diagnostic. Report the failed gate explicitly. Do not relax the gate after seeing results. Report all candidates, all-negative/all-positive controls and positive-scene recovery counts. A six-positive development split is small and repeatedly used; none of these scores is independent validation.
- Freeze checkpoint, normalization, source implementation and per-model cutoff before final inference. Compare the same saved score arrays at 0.50 and calibrated cutoffs. The viewer must use each model's frozen cutoff for masks, counts, error overlays and threshold reset; fixed-scale score colors remain unchanged.
- Fresh-process inference replay and independent source/grid reconstruction remain mandatory before publication. Keep failed cases visible. If neither model passes development, preserve its final evaluation for later rather than consuming it for another failed diagnostic.

This is a stricter experimental screen. Even a pass does not establish an acceptable national alert rate, source localization, emission-rate estimation, quantitative EMIT reconstruction or performance at unrepresented source types.
