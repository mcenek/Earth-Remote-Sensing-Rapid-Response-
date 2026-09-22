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

## Correction: v5.1 outputs were already local

The earlier search missed `v5_1_location_test_predictions.npz` and `v5_location_test_packed.h5`. The local viewer now defaults to authentic cached v5.1 predictions with eight matched 32x32 target/reference examples. No inference or training was needed. The cache has point coordinates but no affine/CRS footprint, so these examples use image pixel coordinates; geographic bounds are not invented. Other summary-only models remain explicitly labeled.

## v5.1 localization finding

Across all 20,789 cached test crops at the frozen pixel threshold 0.40, 51.571% have no positive pixels and 44.774% are entirely positive. Almost every scene-level true positive has a full positive pixel mask. Pixel probabilities in those crops are nearly spatially uniform. These are numeric cache properties, not a renderer filling a rectangle. Published scene AP 0.818 and pixel IoU 0.1852 measure different capabilities: the improved scene classifier does not establish reliable plume localization.

The evaluator retains dense model output and uses separate calibrated scene decisions; no explicit scene-to-pixel broadcast was found. The ultimate training/checkpoint/input-packing cause remains unproven. Preserve the frozen test and audit the spatial head, spatial loss and training label pipeline on development/calibration data before another architecture sweep.

Full audits: [v5.1 spatial audit](assets/GTM_Model51_spatial_audit.md), [legacy U-Net audit](assets/GTM_Model1_box_audit.md).

## Interface inspection

Both map and comparison views were inspected with the real exports. Switching Compact/Physics retains the selected scene and restores the corresponding artifact threshold. Linked zoom settled to the same scale on all three maps. On Physics/T13RFQ, raising the exploratory threshold to 0.50 changed the display counts to TP 280, FP 588, FN 200 and TN 723 (IoU 0.262): fewer extra predictions, but many more missed reference pixels. The broad masks do not become a convincing localized result just by raising the threshold. The reference-only coverage and missing-layer states remain explicit.


## Spatially constant baseline check — 2026-09-18

A post-hoc diagnostic replaced each crop's pixel probabilities with their mean over observable pixels, then applied the unchanged pixel threshold0.40. It used no truth to construct predictions, no training, no new imagery, and no threshold selection. This is a diagnostic of the frozen cache, not a new confirmatory experiment or proposed model.

- Original dense mask IoU:0.1852062.
- Constant-within-crop mask IoU:0.1849650.
- Mask agreement across21,219,466 observable pixels:99.8289% (36,299 differ).
- Original dense pixel precision:19.8353%; approximately80.2% of predicted-positive pixels are false positives against the dataset reference.
- Predicting every pixel positive in every crop yields IoU0.1296339.

The spatial variation in the frozen probabilities adds negligible IoU at the fixed operating threshold compared with a crop-constant map. This does not prove the encoder has learned no spatial features, nor identify whether training, checkpoint construction or packing caused the behavior. It does establish that the exported segmentation masks are not a demonstrated plume-localization success.

Scene precision86.3% is the proportion of flagged *crops* that are positive, not the proportion of predicted plume pixels that are correct. Scene recall37.8% means roughly62.2% of positive crops are missed. Scene AP0.818 measures scene ranking across cutoffs; it is not81.8% segmentation accuracy. These were measured on a roughly balanced crop benchmark, not a nationwide background-dominated search.

Reproduce with `tools/GTM_Model51_constant_crop_diagnostic.py`; exact counts are in `reports/research/GTM_Model51_constant_crop_diagnostic.json`.
