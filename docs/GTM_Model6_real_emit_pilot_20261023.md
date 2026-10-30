# Real Sentinel and EMIT pilot, October 23

## Purpose

Train CNN 2 on cached real observations now. Missing airborne data stops CNN 1 and the full fusion experiment, but it does not stop CNN 2. The artificial runs checked software only.

The first run uses the Volusia landfill observation from June 1, 2025. Sentinel follows EMIT by 12.742 minutes. All five Sentinel bands are already cached. No imagery download is needed.

This is a single-site fitting diagnostic, not an independent test. It asks whether the EMIT branch can learn a real provider annotation. It cannot establish detection accuracy across other places or dates.

## Failure cases, recorded before implementation

1. A changed source file must stop preparation. Verify all raster hashes against the original acquisition receipts.
2. Zero in the provider mask must not become proof of methane absence. Train the explicit class **provider-annotated plume**, with zero as weak unannotated background only where retrieval and clear Sentinel support exist. Preserve this interpretation in the data, run, figures, and viewer. No confirmed-negative scene is available in this pair.
3. Do not use a CMR scene bounding polygon or the old filled-box masks as plume truth. Use the actual provider pixel annotation.
4. Preserve the native 60 m annotation grid. Reproject the Sentinel input to an aligned 20 m grid, using area averaging for continuous bands and nearest sampling for SCL. Supervise the mean of each 3 by 3 output block. A fine output is not validated fine detail.
5. Keep raw Sentinel DN. The cached asset offset metadata conflicts. Fit per-band input normalization on this scene and label the input as DN, not calibrated reflectance. A uniform band offset cancels under this normalization, but this does not resolve calibration for future cross-scene training.
6. Do not require a temporal reference image or NIR band for this five-band CNN 2. Neither is an input to the current branch.
7. Retain finite retrieval support, valid Sentinel pixels, and SCL classes 4, 5, and 6. Exclude cloud, shadow, nodata, and partially supported native cells. Unknown cells must never become negative labels.
8. Do not split overlapping tiles from this one scene into training and test sets. All crops and D4 rotations/mirrors belong to fitting. No validation-selected checkpoint or held-out score is claimed.
9. Save initial and final weights, full-scene predictions, crop sampling counts, source hashes, runtime, loss history, and the numerical policy. Check reload predictions in a fresh process. Never repair an old result in place.
10. Show Sentinel, native annotation, initial prediction, final prediction, and errors on the same extent. Inspect the actual outputs. Report empty or full-box predictions as failures.
11. Limit fitting to a small CPU run. Reuse the branch architecture and leave previous accepted runs unchanged. Do not promote this fitting diagnostic as the viewer's best scientific result.

## What still needs data

Additional independent, close-time pairs and documented annotation background are needed for a detection benchmark. Actual airborne supervision is needed for CNN 1. A frozen CNN 1 and CNN 2 are needed before the proposed CNN 3 fusion comparison.

Legacy TIFFs also contain real measurements. They require a separate signed retrieval regression target, because continuous EMIT enhancement is not a binary mask. Some are many days apart, and their target was already resampled. These limits must be recorded rather than silently converted into good-looking masks.
