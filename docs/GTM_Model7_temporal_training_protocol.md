# Model 7 temporal comparison training protocol

This protocol is fixed before either model sees the sealed test split. The input is the local `GTM_Model7_temporal_cohort_v1` manifest of real Sentinel-2 L1C target/reference pairs and reviewed native labels. It excludes EMIT, airborne imagery, and synthetic labels. Test scenes are unseen by these new models; this does not claim they are untouched by the wider project.

## Failure cases and E2E checks, specified before implementation

1. Reject the wrong schema, empty/missing splits, duplicate scene IDs, a `group_id` crossing splits, unsafe array paths, changed array or source SHA-256, malformed grids, nonbinary labels, invalid supported values, nonfinite supported input, and disagreement between the 20 m support/target and native 2×2 aggregation. Producer positives outside clear support remain in the source label but are excluded by the scoring mask. Training must not create a run on validation failure.
2. Fit normalization from valid train pixels of **both dates only**. Verify finite center/scale and saved values in both checkpoints. A test-label sentinel change must not affect initialization, sampled training crops, normalizer, or optimizer steps; the trainer does not load test labels before the final checkpoint is sealed. Test labels are read only for final metrics.
3. Positive and reviewed negative train scenes must both exist. The same seeded crop schedule is used for both architectures: half positive-scene draws with a supported plume pixel centered within a 48-pixel crop, half negative-scene draws. Apply the same D4 transform to both dates, target, and valid. Fail if sampled crop support is empty or train pixels are nonfinite.
4. Compare `residual` and `siamese` using the same manifest, draws, loss, threshold, and fixed update count. Save initial, best, and final checkpoints, learning curves, and a receipt. `best.pt` is an explicit byte-for-byte alias of `final.pt`; there is no checkpoint selection. The grouped dev split contains negatives only, so dev false activation is reported and cannot select a checkpoint. A fixed 0.5 threshold is the sole evaluation threshold.
5. An E2E run checks each actual run receipt, hashes and counts, all expected test prediction files, finite supported 100×100 probabilities, exact 2× repeat to 200×200, native-grid confusion counts against the sealed labels, pooled metrics, and both positive and negative controls. A fresh process replays saved checkpoints and compares probabilities within the declared tolerance.
6. Negative controls report any-pixel activation and false-positive area rate under the fixed threshold. This does not use a scene-level hard gate. Native probabilities are repeated coarse-grid values, not calibrated 10 m predictions. Native scoring support is the 2× repeat of fully valid 20 m cells.
7. Stability amendment before valid production runs: the invalid source-change attempt showed a train-loss spike from about 1 to 16. Apply the same global gradient norm cap of 1.0 to both architectures before each optimizer step. Reject nonfinite gradient norms. The E2E receipt verifies the common cap in both run configurations. This change uses train-only evidence and leaves the sealed test split untouched.

## Fixed experiment

- Residual temporal CNN: standardized target/reference six-band inputs represented as target six bands plus signed target-minus-reference six bands, width 8, stem 3×3, three residual blocks with dilation 1, 2, and 4, 1×1 logit head.
- Siamese U-Net: shared encoder widths 8/16/32 at three scales. Each scale fuses target features, signed feature difference, and absolute feature difference by 1×1 reduction; upsample and skip convolutions decode a 100×100 logit grid. The last scale uses 4× padding when needed.
- Train normalization is per band mean and standard deviation over supported train pixels from both dates; minimum scale is `1e-6`, standardized values are clipped to ±8, unsupported inputs become zero.
- Torch float64 CPU, one thread, deterministic algorithms, MKLDNN disabled. AdamW, learning rate 0.001, weight decay 0.0001, global gradient norm cap 1.0. Masked BCEWithLogits with train-only positive weight at most 10, plus 0.1 Dice on positive crops. No hard gate or class-balanced full-box BCE.
- Crop 48×48, 48 draws/epoch, batch 8, D4 train augmentation, seed 20261023. Fixed 60 epochs (360 optimizer updates) with a 15-minute safety limit per model. The sealed-cohort warm benchmark measured 0.0730 seconds/update for residual and 0.0677 for Siamese; 360 updates project 26.3 and 24.4 seconds of optimizer time, excluding data and dev evaluation. This common schedule was frozen before either production run. An exceeded limit or nonfinite loss is a failed comparison, never a shorter valid run. Full-scene dev false activation is reported every two epochs. The exact same precomputed crop schedule is used for both architectures.
- Initial checkpoint is saved before fitting. Final is the primary checkpoint after all fixed updates. `best.pt` aliases final for artifact compatibility, and the receipt explicitly records that no selection occurred. Predictions and metrics use final only.

## Run and verify

From the repository root, after cohort sealing:

```powershell
$py = 'C:\Users\joshu\miniconda3\envs\hyworld2\python.exe'
& $py tools/GTM_Model7_train_temporal.py train --model residual --manifest outputs/GTM_Model7_temporal_20261023/data/manifest.json --output outputs/GTM_Model7_temporal_20261023/residual
& $py tools/GTM_Model7_train_temporal.py train --model siamese --manifest outputs/GTM_Model7_temporal_20261023/data/manifest.json --output outputs/GTM_Model7_temporal_20261023/siamese
& $py tools/GTM_Model7_temporal_e2e.py --manifest outputs/GTM_Model7_temporal_20261023/data/manifest.json --runs outputs/GTM_Model7_temporal_20261023/residual outputs/GTM_Model7_temporal_20261023/siamese --output outputs/GTM_Model7_temporal_20261023/e2e_receipt.json
```

The E2E receipt is a repeatable artifact. Each run snapshots the trainer source, manifest, and configuration under `source/`. The run receipts record exact source/manifest hashes, environment, settings, timings, parameter count, normalization, split counts, learning curves, and native-grid metrics. Each run stores all test probabilities in `predictions/<scene-id>.npz` with `probability20`, `probability_native`, and `valid20`. Only a fresh process running `replay` loads a saved checkpoint to confirm deterministic probabilities.

## Completed pilot outcome, October 23

Both fixed runs completed and the E2E receipt passed. On the same 3 plume and 24 no-plume held-out observations, residual CNN IoU was 0.874%, precision 0.880%, and recall 55.954%. Temporal U-Net IoU was 0.631%, precision 0.640%, and recall 32.333%. The residual activated in 23/24 no-plume scenes and the U-Net in 16/24. Their false-positive area rates on negatives were 9.269% and 9.326%. Neither run meets useful plume-screening quality. This short, single-seed pilot does not establish an architecture's upper limit.

Independent original-TIFF reconstruction and native confusion recomputation are in `outputs/GTM_Model7_temporal_20261023/review/verification.json`. Fresh-process checkpoint replay and corruption/split checks are in `e2e_receipt.json` in the run root. All held-out cases, including failures, are in the two viewer experiments `GTM_Model7_temporal_20261023_residual` and `GTM_Model7_temporal_20261023_siamese`. Each links to the same head-to-head HTML report.

The release's official test pool remains untouched. The selected test scenes are unseen to these two models, but may have been examined in earlier project work. The three test plumes cover three sites within one connected geographic component. Some temporal references postdate the target. These runs measure segmentation of MARS release annotations, not reconstruction of measured EMIT concentrations or validated upscaling.

The train-only residual diagnostic found nonzero parameter changes and weak class discrimination, without a concrete gradient or crop-alignment bug. Positive-rich crop sampling plus a positive loss weight of 10 may encourage excessive activation. Any follow-up must document a new objective and use training/development evidence for decisions. The current test results must not become a checkpoint or threshold selection set disguised as an independent benchmark.
