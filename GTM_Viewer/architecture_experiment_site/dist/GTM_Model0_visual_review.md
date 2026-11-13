# Visual review — 18 September 2026

Two historical U-Net artifacts now have real geographic outputs in the viewer: Compact ResUNet and Physics-feature ResUNet, on two Texas scenes from the legacy validation folder. These are diagnostic checkpoint replays, not a rerun of the frozen foundation-model benchmark.

## Finding: the saved thresholds overpredict heavily

| Model | Scene | Saved threshold | Fraction of displayed scene predicted positive | Reference coverage | TP | FP | FN | TN | Display IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact | T14RMU, 2023-02-03 | 0.01 | 100% | 9.20% | 816 | 692 | 0 | 0 | 0.541 |
| Compact | T13RFQ, 2023-04-10 | 0.01 | 99.97% | 10.93% | 480 | 1311 | 0 | 0 | 0.268 |
| Physics | T14RMU, 2023-02-03 | 0.05 | 99.89% | 9.20% | 816 | 692 | 0 | 0 | 0.541 |
| Physics | T13RFQ, 2023-04-10 | 0.05 | 98.55% | 10.93% | 470 | 1309 | 10 | 2 | 0.263 |

Counts are on the 128×128 display grid and only score pixels with a supported EMIT reference. A high recall here can coexist with almost no ability to reject reference-negative pixels. Predicted pixels outside reference coverage are unknown, not automatically false positives. The stored thresholds produce broad masks that do not localize the reference plumes adequately in these examples. Use the threshold control to inspect behavior, not to retune a final test score.

## What was replayed

- Existing model.keras artifacts, loaded without retraining using Keras 3.14 and its Torch backend. No cross-backend numerical equivalence test against the original TensorFlow runtime was performed.
- Native inputs use the documented five Sentinel bands and artifact normalization; the physics model adds its six documented spectral features.
- Four native 128×128 windows are stitched over each 256×256 source. The source is approximately 20 m; it is not an exact metric-grid resampling. No overlap/blending is applied, so tile-boundary discontinuities remain inspectable.
- Display rasters share one EPSG:3857 grid. Reference masks use EMIT CH4 >300 ppm·m only on finite, non-nodata pixels. Missing reference pixels remain null.
- Checkpoint/source hashes are visible per scene. The selected dates are Sentinel dates; EMIT dates and source Sentinel product-level metadata are not verified. Validation-folder membership alone does not establish independent site holdout. These two examples are not a representative US evaluation.

## What is still missing

The newer frozen ERSRR, spatial/foundation successor and MethaneS2CM experiments have authentic summary reports, but their matched geographic probability rasters were not available locally. Their scores remain visible under Experiments and the previous chart; this interface does not substitute old U-Net outputs for those models. Acquire or regenerate those exact outputs and import them with the documented experiment bundle contract.

## Next research decision

Before another architecture sweep, recover the best foundation-model checkpoint and its exact held-out image/reference cohort, audit radiometry/date/resolution contracts, and add confirmed negative scenes. Evaluate localization and source-level false alarms on independent sites. Broad positive masks and sparse reference coverage cannot establish a reliable US methane detector.

## Interface inspection

Both map and comparison views were inspected with the real exports. Switching Compact/Physics retains the selected scene and restores the corresponding artifact threshold. Linked zoom settled to the same scale on all three maps. On Physics/T13RFQ, raising the exploratory threshold to 0.50 changed the display counts to TP 280, FP 588, FN 200 and TN 723 (IoU 0.262): fewer extra predictions, but many more missed reference pixels. The broad masks do not become a convincing localized result just by raising the threshold. The reference-only coverage and missing-layer states remain explicit.
