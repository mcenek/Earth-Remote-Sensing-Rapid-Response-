# Broader development and bounded U-Net revision

Written before this round's implementation, new fits or reserved test images. This follows Joshua's request to broaden development from cached real data, reserve new test sites and continue improving the U-Net with visual checks.

## Failure modes and controls

- A new date at an old site is not a fresh site. Exclude all known fitted, evaluated or viewed locations, acquisitions and spectral hashes from a new final reservation, including the now-consumed 17-site cohort. Freeze final selection in metadata before fitting. Do not open final pixels until model and cutoff selection is complete. Report exhaustion if no fresh cached positive sites remain.
- Previously exposed but TRAIN-disjoint sites may enter DEV. Disclose this reuse. Do not move an existing TRAIN site to DEV or call a training example independent. Check site, acquisition, content hash and geographic separation, including references.
- Preserve the existing 958-observation TRAIN arrays, TRAIN-only normalizer and balanced US/auxiliary and positive/negative draw schedule. Only DEV membership changes. Any incompatible DEV rows are excluded by fixed source/split rules with reasons, not model performance. A metadata-only TEST reservation cannot affect normalizer or draws.
- Do not relax false-alarm gates to create an apparent improvement. Do not select a model using a displayed final plume, a good single scene, or large-plume pooled metrics alone. Preserve misses and false detections.
- Do not alter prior immutable trainers, checkpoints, manifests or outputs. New architectures need source-bound reconstruction, train/inference parity, finite gradients, valid loss support and explicit input/output shape checks before full fitting. No unit tests after implementation; use real-data E2E verification with repeatable receipts.
- Binary MARS labels are not quantitative EMIT concentrations, and repeated 20 m output is not learned 10 m upscaling. No CAFO coordinate becomes a methane label. No new imagery downloads.

## Data and comparison

The data preparation protocol and manifest receipts will record exact broader DEV counts and TEST feasibility before fitting. Prefer combining all compatible train-disjoint earlier evaluation cohorts, including the consumed 17-site set, over keeping the six-plume DEV selection bottleneck. Deduplicate targets and physical sites for weighting; assess both classes and countries. Freeze the resulting DEV manifest and the independent TEST metadata reservation before training.

Evaluate frozen v5 step3600/cutoff0.95 on the broader DEV first. Keep its frozen result as the primary reference. Also allow its cutoff to be recalibrated on broader DEV, with unchanged weights, as a separately labeled calibration-only comparator. Never present that as architecture learning. The unchanged residual CNN may remain a viewer control.

## Bounded model candidates

At most two new U-Net fits, each 4,800 updates, milestones1200/2400/3600/4800, seed20261024, learning rate0.0005, AdamW weight decay0.0001, gradient cap1, batch8, four real plume and four real no-plume samples per batch, exact old domain-balanced draw schedule and D4 transforms. Full100x100 inputs and native annotation counts remain. Maximum30minutes per fit, fail on nonfinite loss/gradient or source/split drift.

1. `early_fusion`: a U-Net with16/32/64 feature widths. Its initial input combines standardized current bands, reference bands, their signed difference and absolute difference (24 channels). Use two convolution blocks per scale, GroupNorm4, SiLU activation, two average-pool stages and bilinear decoding with full-resolution skip connections. This tests whether preserving signed temporal evidence before the first shared encoder, together with more capacity and stable optimization, helps. It jointly changes fusion, capacity and normalization, so it cannot isolate one causal factor.
2. `spectral_fusion`: the same U-Net and training loss, with two extra full-resolution channels for centered-log temporal B12-minus-B11 and B12-minus-B08 contrasts. Recover the log units using the frozen TRAIN band scales before forming differences; scale each contrast by the corresponding TRAIN band-scale quadrature, never by DEV/TEST statistics. These are candidate nuisance-rejecting spectral features, not measured methane concentrations. No unconstrained direct linear output skip.

Both keep v5 loss: valid-pixel-weighted BCE positive weight2 plus0.1 pooled positive-scene Dice. Invalid cells contribute neither loss nor counts. No new loss tuning. A changed architecture and improved DEV transfer are separate claims from quantitative methane identification.

## Selection, stopping and test

Use the fixed cutoffs0.05..0.95 by0.05,0.975,0.99,0.995 on broader DEV only. Gate remains pooled native precision>=0.50,recall>=0.30,IoU>=0.20,clear pixelFPR<=0.001,and at least half of positive scenes recovering>=10% of annotation. Rank eligible model/checkpoint/cutoff by mean per-positive-site recall (average positive-scene recall within each physical site, then average sites), then mean positive-scene recall, then pooled IoU, lower clear FPR, earlier step, higher cutoff. Frozen v5 and its calibration-only comparator compete with both fits. If none qualifies, publish the best diagnostic by the same ranking but clearly mark it failed; do not promote it or consume final sites merely to collect another failed result.

Before any final pixels: freeze selected source, checkpoint, preprocessing, cutoff, DEV and TEST reservation hashes. Require an eligible selection with better DEV objective than the original frozen comparator to justify final evaluation. Evaluate frozen baseline and selected model on identical accepted new sites, preserving all QA exclusions without replacement. If fresh sites are exhausted, stop the independent-test stage, disclose the limitation and keep reserved metadata without fabricating novelty. Do not fit again after final outcomes.

## Verification and actual outputs

Preflight on real balanced TRAIN samples must independently check band/feature construction, normalizer identity, both-class and invalid-support loss contributions, finite gradients,100x100 output shape and checkpoint round trip. Train only after this passes. Replay all saved DEV milestones in a separate CPU process and independently recount native confusion. Compare all evaluated cases against source arrays. Publish baseline, calibration-only and new candidate predictions with fixed-scale scores, masks, annotations and errors; retain all cases. Root personally inspects plume misses, small plumes, multiple countries and no-plume false alarms in actual contact sheets and browser. Verify HTTP assets and viewer model/scene switching, score/error controls and frozen native scores; save screenshots and receipts.
