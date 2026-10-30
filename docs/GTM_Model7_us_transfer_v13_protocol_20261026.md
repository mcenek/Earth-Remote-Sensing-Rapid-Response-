# Native-resolution US transfer study

Written before new image reconstruction, implementation or fitting for this study. October 26, 2026.

## Question and limits

Can a fresh, deeper temporal U-Net with the native image grid recover missed plumes without excessive extra area? Does correctly transformed wind help the same model? These are development experiments, not evidence of nationwide accuracy. Do not force success by expanding masks or choosing an easy subset.

V12 failed: no eligible SELECT choice, held-out diagnostic F1 roughly 34-40%. Its 121,937-parameter network uses 100 by 100 averaged 20 m inputs. The released MARS network uses a substantially deeper native-grid model, different radiometry, and wind/cloud inputs. It cannot initialize this study because it has seen all v12 holdout positives. Use fresh weights only. Differences from historical v11/v12 are a package comparison, not proof that one individual change caused improvement.

## Fixed data boundaries

Source manifest: `outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json`, SHA256 `1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a`. FIT is its 958 TRAIN observations (268 plume, 690 clear). Development selection is its 198 DEV observations (46 plume, 152 clear). Both are historically exposed. Do not call them a new validation set. Do not use v12 ASSESS for selection or confirmation again. Do not open protected final-test arrays.

Reconstruct current and selected clear-reference native 200 by 200 six-band images from pinned original TIFFs. Read the first six channels of each date's own source image, preserving B02/B03/B04/B08/B11/B12 order. Never substitute the concatenated original reference bands for the separately selected clear reference. Validate source hashes, shapes, CRS, transform, radiometry and clouds. Keep existing 200 by 200 binary labels and native support exactly; no dilation, filling, smoothing, shape filtering or case removal. Cache only under the new ignored study directory, with source/cache hashes. No DEV-fitted normalization.

## Two bounded fresh fits

Common inputs: current six and reference six bands divided by 5000, clipped to [0,2]; validity-aware current/reference MBMP ratio; binary union cloud indicator. Two wind channels occupy the same input slots in both models. Metadata wind_u/wind_v must be finite, clipped to [-20,20] and divided by 8. Methane flux, detection strength, geometry, source category and labels cannot be inputs.

1. `native_zero_wind`: wind channels always zero.
2. `native_wind`: use metadata wind. Transform its vector with the image D4. A counterclockwise quarter-turn maps (u,v) to (-v,u); a horizontal mirror negates u.

Fresh U-Net: widths 32/64/128/256/256, four downsampling levels, paired bilinear decoder, two 3 by 3 convolutions per block, BatchNorm and GELU, one logit per native pixel. Check arbitrary 200 by 200 shape. Both models start from identical seeded weights with both wind first-layer input slices zero. No pretrained methane weights.

Seed 20261026. AdamW learning rate 0.0003, weight decay 0.0001, batch 8 with four plume and four clear scenes. Sample sites uniformly within each class, then observations within the site. Save the identical 38,400-draw scene/D4 schedule for both fits. Full scenes in training and inference. Weighted BCE positive weight 2 plus 0.1 pooled-positive Dice on native valid support. Gradient cap 1. Float32, no AMP or TF32. At most 4,800 updates per fit; checkpoints 1,200/2,400/3,600/4,800; maximum 40 minutes per fit. Benchmark a real batch first. Stop/report if the budget is unsuitable. No third fit, seed search or loss sweep after results.

## Failure controls before fitting

Reject changed source/manifest, wrong band order, copied current as reference, bad geometry/support, changed native labels/support, nonfinite inputs/loss/gradients, unbalanced class schedule, incorrect vector D4, answer-bearing metadata inputs, mismatched common initial weights, unsuitable runtime and checkpoint replay drift. Check real positive and clear forward/backward passes, weight changes, native loss against independent NumPy and CPU serialization before fitting. Reuse helpers only when their contracts match. Do not edit old scripts or receipts.

## Fixed selection and usefulness checks

Evaluate all four fixed checkpoints on all 198 DEV scenes. Cutoffs: .05 through .95 step .05, plus .975/.99/.995. Compute native pooled TP/FP/FN/TN across plume AND clear observations, precision/recall/F1, site/scene recall, clear-pixel false-positive rate, clear-scene activation, complete misses and marked area.

Eligible choices require P >= .70, R >= .30, clear-pixel FPR <= .001, clear activation <= .15, and at least half of positive observations recovering >=10% of their labels. Rank by F1, positive-site mean recall, positive-scene mean recall, lower clear FPR, earlier step, then higher cutoff. If no choice qualifies, freeze the failed diagnostic at update1200/.70 and retain v11. Useful target: F1 >= .70, P >= .70, R >= .65 and the same clear-error guards. Report rejected unconstrained best separately.

Freeze both model/cutoff choices before opening prospective assessment imagery. Bind weights, preprocessing, inputs, split, schedule and thresholds. Replay saved predictions in a fresh process before publication.

## Prospective US readiness and download limits

Independently search pinned 2025 EMIT training metadata for unused US events farther than 25 km from exposed/protected source locations. Freeze IDs/order before queries. Require Sentinel within one hour, overlapping footprint, declared cloud <=20%, required bands and an earlier clear reference. Native annotation/retrieval need explicit support and units/validity limitations. An all-zero annotation is not automatically verified methane-free truth.

Respect the existing cumulative 100 MiB ledger `outputs/cafo_pilot_2026-09-11/range_cache`; never reset it or create another allowance. Reserve metadata bytes before requests. Do not acquire imagery without a frozen plan, byte estimate and remaining capacity. No large archives or full scenes. If budget or valid supervision blocks confirmation, record the specific gap and do not replace independent cases with training examples. New candidate imagery remains assessment-only until both models are frozen.

## Actual-output deliverables

Publish every DEV scene for both fresh models alongside historical v11, with native score/mask/errors and source-bound counts. Preserve misses and clear scenes. Root inspects all plumes, largest clear false alarms and all US positives in the viewer. Verify model/scene switching, cutoff restoration, images, error/score modes and private Tailscale access. Retain contact sheets, served-asset checks and concise report.

If prospective US data becomes usable, show every reserved case and disclose native resolution, timing and alignment. Do not label EMIT retrieval input as Sentinel-only inference, or binary masks as emission-rate accuracy. Completion records must distinguish technical verification from scientific usefulness.
