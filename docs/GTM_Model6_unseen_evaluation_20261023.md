# Frozen CNN 2 on real unseen scenes

## Question

Does the existing Volusia-only CNN 2 produce useful predictions at other real locations, including observations reviewed as having no plume?

Keep the accepted weights, Volusia normalization, tiling and 0.50 cutoff fixed. Do not fit, select a checkpoint, recalibrate or adjust the cutoff from these results. Do not change earlier accepted artifacts. This is a diagnostic of this checkpoint. Other project models have previously used MARS, so this is not a new untouched project-wide benchmark.

## Selection fixed before inference

Use the cached MARS publication training manifest. Select PLUME first, then NO_PLUME. Order each class by sample ID. Take the first eight available observations from each class with at least 90% clear, valid target input, distinct location identities, and at least 25 km between selected sites. Exclude any site within 25 km of the Volusia fitting footprint. Do not use predictions to choose examples. Preserve every exclusion and the full selected list before inference. If fewer observations qualify, retain all that qualify and report the count.

MARS uses observed Sentinel L1C target bands, not EMIT-simulated Sentinel data. This differs from the L2A fitting image. Use the documented DN values, five target-date bands in the original order, and the frozen fitting normalization. Report this as a domain shift test, not a clean same-distribution EMIT accuracy benchmark. Producer-reviewed NO_PLUME means no plume in that annotation task on that observation, not permanent proof of no methane emissions.

Also run the cached repaired Georgia L2A observation with its continuous EMIT retrieval for visual context. It has no verified binary plume annotation. Do not score it as a segmentation mask or invent a negative label.

## Failure cases recorded before implementation

1. Wrong weights or changed fitting receipts. Pin the accepted receipt and final checkpoint hashes. Hash all accepted training files before and after evaluation.
2. Test leakage. Reject fitting identities, identical input assets, shared selected sites and geographic overlap with Volusia. Keep labels outside the model call. Use only the stored fitting normalization.
3. Changed band order, target/reference date mix, input level, scale or missing bands. Read the producer README and raster band descriptions. Use target bands B02, B03, B04, B11 and B12 only. Keep radiometric differences visible.
4. Invalid/cloud pixels counted as negatives. Use cloud and input support. Preserve unknown as unknown. Check known label codes and exclude unsupported cells before scoring.
5. Continuous retrieval or a catalog bounding polygon converted into plume truth. Only score producer pixel annotations with their declared meaning. Show Georgia retrieval separately, without pixel accuracy.
6. Hidden resampling or lost small reference plumes. MARS labels are on their original 10 m delivered grid, unlike the Volusia 60 m annotation. Average each exact 2 by 2 input block to 20 m. Require all four original cells clear and valid. For MARS annotation comparison, repeat each 20 m output cell over its original four 10 m annotation cells with no smoothing or threshold change. Score only supported original annotation cells. Preserve all original reference pixels, record this exact grid mapping, and do not call repeated pixels extra model detail. Training still used 60 m supervision. Georgia uses its real retrieval grid for visual context only and has no segmentation score.
7. No-plume scenes excluded, silently changed into positives, or a zero denominator reported as perfect accuracy. Keep selected negatives, count false-positive cells, report activation by scene, and use null for undefined positive-class metrics. Give all-negative and all-positive controls.
8. Selection of flattering outputs or a new threshold after viewing results. Include all selected cases, empty and full masks, worst cases and the fixed threshold. Keep any later experiment separate.
9. Arrays or reports do not match actual inference. Save predictions and hashes, replay in a fresh process, reconstruct counts independently and verify checkpoint and fitting artifacts remain unchanged.
10. Viewer hides the model output or invents a ground-truth panel. Publish actual prediction arrays with input, available reference, provenance, dates, support and limitations. Inspect positive and no-plume cases in the running viewer, including error overlays. Show continuous observations in a separate aligned report panel where the viewer has no continuous-reference renderer.
11. Excessive download or runtime. Reuse cached imagery; download zero bytes. Run frozen CPU inference only, without tuning or training. Do not alter the persistent download allowance.

## Evidence

Save a sealed selection manifest, source hashes, prepared arrays, inference receipt, fresh-process replay, independent native counts, full contact sheet, an HTML review, published viewer bundle and browser screenshots. Results belong to an unseen-scene diagnostic, separate from Volusia fitting and artificial software checks.
