# Local CPU comparison protocol

Recorded October 3, 2026, before fitting or inspecting the new comparison scores.

## Questions

1. Does the saved four-window average outperform its individual branches on identical native pixels?
2. Do fixed branch-agreement rules reduce false activation without losing all plume overlap?
3. Do the neural models beat a simple linear predictor fitted only on the original training parents?

## Fixed experiment

Use the R5 manifest, all 99 original training parents and all 65 original evaluation parents. This is reused development data, with only three evaluation plume scenes in one connected positive group. It is not a new independent validation set. Use the cached inputs from the earlier evidence run only after matching IDs, source hashes, grid metadata, groups and partitions with R5.

- Compare S16, S32, S64, S128, the saved mean, arithmetic probability mean, agreement of at least three branches, agreement of all four branches, always-negative and always-positive controls.
- Fit two class-balanced L2 logistic models: cached feature 15 only (temporal SWIR contrast), and cached features 0 through 15. Feature 16, the extra evidence channel, is excluded. These are per-pixel classifiers with no learned spatial context.
- Use seed 20261003. Sample at most 512 positive and 512 negative valid pixels per training parent without replacement. Standardize from sampled training pixels only. Weight each class to one half of the objective. Use L2 strength 0.01, at most 30 Newton steps with training-objective backtracking, and a fixed decision cutoff of 0.50. The resulting scores are not calibrated methane probabilities.
- Fit once. No evaluation-driven thresholds, hyperparameters, augmentation choices, early stopping or model selection. No additional run if these models fail.
- Score every method on the exact same saved R5 valid mask. Verify it equals source validity intersected with the saved context support. Recompute and match every saved branch confusion count before accepting the comparison.
- Report pooled native pixel IoU, precision, recall, false-positive pixels and activation on reviewed negative scenes. Do not call any positive pixel a verified source. Scene activation is a diagnostic, not a validated scene decision.
- Save every native prediction. Show all three positive cases, the two largest R5 false alarms, and the two largest false alarms from each new linear model (deduplicated). State this diagnostic selection rule.
- Use native image coordinates for image comparisons. Do not stretch native UTM pixels onto a geographic map. Reference outlines and unsupported areas must be visible. Original benchmark files stay unchanged.

## Failure cases and E2E acceptance before implementation

1. Missing or mismatched cache IDs, source hashes, grids, labels, groups or partitions must stop the run.
2. Shared training/evaluation IDs, sites, groups, source image hashes or target/reference acquisitions must stop the run.
3. Wrong array shape, nonfinite supported input/score, nonbinary target/support, no valid class samples, stale R5 scores, or mismatched cached and saved targets must stop the run.
4. Existing output directories must be refused. Inputs must retain their original hashes.
5. Fit must work with evaluation target arrays removed. Fitted numeric parameters must remain identical. Repeating the same input must produce identical fitted parameters and predictions.
6. A balanced zero-feature fixture must produce zero weights, zero intercept and score 0.5. At the fixed inclusive cutoff its evaluation counts must equal an all-positive control.
7. Use public CLI subprocesses for these checks. Save fixture inputs, invocations, outcomes and hashes as a repeatable E2E artifact before running the real cohort.
8. Inspect real output images in the local browser. Scores alone do not complete the experiment.

## Limits

Use the current NumPy/Pillow runtime, at most two BLAS threads, streaming one cached parent at a time, and a five-minute compute deadline. Do not use the GPU or install packages. No imagery, weights, external API, paid service, label change, external journal edit, GitHub push or message is part of this comparison. The new three-CNN architecture and scientific data-readiness gates remain unchanged.
