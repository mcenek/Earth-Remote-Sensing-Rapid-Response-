# ERSRR v5.1 frozen dense-cache spatial audit

Audit date: 2026-09-18. Source is the authentic local cache `v5_1_location_test_predictions.npz` (20,789 32x32 crops), stored as float16 probabilities. No model inference was run.

## Whole-crop behavior at pixel threshold 0.40

The cache has 99.678% observable pixels. The independent dense pixel decision is `probability >= 0.40`; the scene gate is separate at score `0.7335791607366197`.

Across all 20,789 crops, the predicted-mask fraction has:

| statistic | fraction of crops |
|---|---:|
| zero predicted pixels | 51.571% |
| at least 95% predicted | 47.636% |
| exactly 100% predicted | 44.774% |

The mean predicted-pixel fraction is 48.131%. These fractions are effectively unchanged when restricted to observable pixels (minor difference from 0.322% unobservable pixels). Thus the viewer's nearly whole-crop examples reflect the cache's dense output, not a fill or alpha rendering issue.

## Spatial variation and scene outcomes

Scene-level categories use the cache's `label` and independent v5.1 scene decision: TP = label 1/decision 1, FP = label 0/decision 1, FN = label 1/decision 0, TN = label 0/decision 0.

| category | crops | crop mask p50 | crop mask p95 | exactly 100% | probability spatial std p50 | probability p50 |
|---|---:|---:|---:|---:|---:|---:|
| TP | 3,949 | 100% | 100% | 99.975% | 0.00148 | 0.6528 |
| FP | 627 | 100% | 100% | 99.681% | 0.00148 | 0.6450 |
| FN | 6,504 | 99.707% | 100% | 47.694% | 0.00273 | 0.4243 |
| TN | 9,709 | 0% | 100% | 16.830% | 0.00155 | 0.0720 |

TP and FP crops are especially scene-uniform: their median spatial standard deviation is only about 0.0015 probability units, with TP all-pixel probability median 0.653 and FP 0.645. This is strong evidence that the dense head is largely expressing a scene-level baseline, with weak within-crop spatial modulation. It does not by itself prove the architecture is irreparably broken; it does establish that these cached outputs are not spatially resolved plume masks.

For context, the scene-gate counts are 4,576 positive and 16,213 negative. The scene gate removes many broad FN/TN responses, but the dense pixel gate is explicitly independent and still marks nearly every pixel in positive-scene crops.

## Aggregate pixel metrics

To match the published evaluation convention, the cached float16 values are decoded to float32 and then thresholded with `probability >= 0.40` on observable pixels:

| TP | FP | FN | TN | IoU | Dice |
|---:|---:|---:|---:|---:|---:|
| 2,025,788 | 8,187,251 | 724,974 | 10,281,453 | 0.185206 | 0.312530 |

Thresholding the raw float16 values directly changes 5,024 near-cutoff pixels and gives IoU 0.185194 / Dice 0.312513; the float32-decoded results above are the values to compare with the published report. The dominant error is FP area, not a display artifact.

The cache's truth is a dense 32x32 target mask, unlike the sparse EMIT support in the legacy 256x256 viewer export. This audit therefore evaluates the cache's own observable mask semantics and does not transfer its metrics to the legacy model.

## Interpretation and bounded next checks

The authentic cache confirms a real whole-crop behavior: almost half of crops are >=95% positive, and TP/FP outputs are nearly spatially constant. The frozen IoU≈0.185 is consequently compatible with broad scene-uniform responses: 8.19M false-positive pixels versus 2.03M true-positive pixels dominate the score.

This is evidence about the v5.1 frozen dense outputs and pipeline behavior. It is not sufficient alone to identify whether the weak spatial head arose from training, checkpoint construction, input packing, or an intentional scene-conditioned design. The next diagnostic should compare the cached dense probabilities against the exact packed H5 input and checkpoint inference path on a small set, preserving float16 quantization and both independent gates.

Evidence: `Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MethaneS2CM/l2a_location_split_32x32/v5_1_location_test_predictions.npz` and `ersrr-ordinal-win/outputs/GTM_viewer_bundles/ersrr_v5_1/experiment.json`.


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
