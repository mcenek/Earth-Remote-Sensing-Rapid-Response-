# Real Sentinel and EMIT recovery, October 23

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

Real CNN 2 training has started. We used the cached Volusia Sentinel and EMIT pair. Artificial data checked the new code, but did not answer the research question. Treating incomplete airborne data as a blocker for the EMIT branch was too restrictive.

## What was missing

| Cached data | Actual limitation | Action taken or needed |
|---|---|---|
| Volusia, June 1, 2025 | Real annotation and all five Sentinel bands existed, but had not been prepared for the new trainer. No independent negative scene. | Prepared aligned input, native annotation and explicit support. Trained CNN 2 locally. |
| 102 legacy paired TIFFs | Continuous signed methane retrieval, rather than binary labels. Some large date offsets, nodata mismatch and already resampled targets. | Completed the audit. Keep these for a separately defined retrieval-regression experiment. Do not silently make binary masks. |
| Nine additional US external pairs | Final cloud gates pass and 45 linked assets are intact. Cached masks are catalog outlines. Native EMIT targets are missing at expected paths. | Saved exact inventory and hashes. Reuse Sentinel assets; retrieve only compatible missing EMIT targets and support. |
| Other native V002 products | Real retrievals exist. The close-time Georgia crop does not contain the plume peak. Other US examples have much larger date offsets. | Georgia may support a retrieval-fitting experiment, but cannot demonstrate recovery of the missing plume peak. |
| Airborne branch | No matched measured airborne cohort found locally. | Still needed for CNN 1 and the full CNN 3 combination. It does not block standalone CNN 2. |

There are 23 useful near-time legacy tiles globally, from 16 connected acquisition/location groups. Only two are in the US, California and Louisiana. A third near the border is in Mexico and is excluded from the US count. Do not count image crops or rotations as independent sites.

## The real run

The Sentinel and EMIT acquisitions are 12.742 minutes apart. The five-band input has 165 x 183 pixels at 20 m. The native provider annotation has 55 x 61 cells at 60 m. Training compares each mean 3 x 3 prediction block with one native annotation cell. This avoids scoring interpolated reference pixels as new measurements.

Of 3,355 native cells, 3,221 have both measured retrieval support and clear Sentinel support. These contain 1,448 annotated plume cells and 1,773 weak background cells. The other 134 are unknown. Weak background is unannotated, not a verified absence of methane. The model receives Sentinel bands only; the annotation enters the loss and fitting crop selection.

CNN 2 completed 160 epochs. Each epoch contains 12 annotated crops and 12 all-weak-background crops. Eight rotation/mirror variants are sampled. All 3,840 crop draws come from this one fitting image, including only 39 distinct background origins. They are not 3,840 independent observations. The raw Sentinel DN values use fitting-only normalization; this diagnostic does not resolve the wider radiometric calibration question.

| Native 60 m agreement at fixed 0.50 | Initial model | Final model |
|---|---:|---:|
| Annotation IoU | 44.95% | 63.39% |
| Annotated cells recovered | 1,448 | 1,285 |
| Extra cells against weak background | 1,773 | 579 |
| Missed annotation cells | 0 | 163 |
| Predicted positive area | 100% | 57.9% |

The initial model marks every supported cell. Its score also represents the all-positive baseline. The final output is spatially varying and improves its fitting agreement. It still has large extra regions and misses. This establishes that the branch can fit this annotation. It does not establish methane detection on a new site, useful source identification, or validated 20 m upscaling. It may memorize the site or use terrain features.

The first shorter run collapsed to background. A later attempt marked the whole image. Both remain saved. The final 160-epoch run uses balanced crop sampling and class-balanced BCE plus Dice. This fitting diagnostic was developed after seeing those failures; it is not a preregistered independent comparison.

## Inspect and repeat

- [Open the HTML review](../../outputs/GTM_Model6/GTM_Model6_real_emit_20261023/review_v2/index.html). It is a single portable file with embedded images. It shows the actual inputs, annotation, initial output, final output, error map, learning curve and input-to-loss diagram.
- [Open the live local viewer](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model6_real_emit_20261023_cnn2_fit&scene=Volusia-20250601-fit#compare). Select **Side by side**, then **Fit scene**. Cyan is the reference outline. Orange is the thresholded prediction. **Disagreement** shows the errors.
- [Runbook and repeat commands](../../docs/GTM_Model6_three_cnn_runbook.md#october-23-real-cnn-2-fitting-run).
- [Data preparation protocol](../../docs/GTM_Model6_real_emit_pilot_20261023.md).
- [Missing-target inventory](../../outputs/GTM_Model6/GTM_Model6_real_emit_20261023/real_cache_inventory.json).

The accepted run is `outputs/GTM_Model6/GTM_Model6_real_emit_20261023/cnn2_fit_ready`. Fresh-process checkpoint replay reproduced all saved probabilities exactly. The renderer independently recalculated counts and checked source hashes, native aggregation, supported cells and unchanged run artifacts. Evidence is saved in `review_v2/verification.json`. Actual browser inspection covered threshold and score maps, disagreements, side-by-side alignment, all report figures and the retrieval disclosure. Browser evidence is in `review_v2/browser_verification.json` and its screenshots.

## Next real-data pass

1. Use the saved nine-pair inventory to retrieve the missing EMIT measurements and their valid support. Keep acquisition identity, native resolution and units. A CH4PLM raster is continuous; positive values alone are not a plume mask.
2. Separate a binary annotation experiment from a signed-retrieval regression experiment. Use the correct target and loss for each. Keep L1C and L2A radiometry separate until their handling is explicit.
3. Establish independent locations and dates before augmentation. Include documented clear, measured background controls. Unmeasured areas and unannotated catalog exteriors remain unknown.
4. Fit CNN 2 on multiple scenes and inspect unseen positives, misses and background false detections in the same viewer. Compare with empty, full-mask and simple spectral baselines.
5. Add CNN 1 only when matched airborne labels exist. Test CNN 3 against CNN 2 alone before claiming a useful combined model.

No imagery, model weights or packages were downloaded during this recovery. The shared imagery ledger remains 101,089,414 of 104,857,600 bytes. Only 3,768,186 bytes remain, so any next acquisition must fit that budget or use an explicitly revised one. No external journal write, message, commit or push was made.
