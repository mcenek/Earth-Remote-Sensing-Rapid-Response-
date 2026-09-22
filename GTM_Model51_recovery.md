# ERSRR v5.1 recovery notes

Read-only inventory of the authentic MethaneS2CM v5.1 artifacts found locally on 2026-09-18.

## Actual prediction artifact

`EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MethaneS2CM/l2a_location_split_32x32/v5_1_location_test_predictions.npz` exists locally (86,376,601 bytes; SHA-256 `3f371aa0a349dc8e0f291ce3efdb2aebc105f31aabfa0786d7dd17c114adaab9`). It contains 20,789 location-test crops and these arrays:

* `sample_id`, `label`, `group_id`, `observable`, `truth`
* `ersrr_v5_1_scene_score`, `ersrr_v5_1_scene_decision`, `ersrr_v5_1_probability`
* frozen v4.3 and released MARS-S2L comparator scores/decisions/probabilities
* `packed_sha256`, `acquisition_sha256`

The v5.1 probability array is real frozen dense output: shape `(20789, 32, 32)`, dtype `float16`. The frozen dense display threshold is `0.40`. The independently calibrated scene-score gate is `0.7335791607366197` (development FPR target 0.05); do not derive scene score from max pixel probability.

The sealed test input/target pack also exists at `.../v5_location_test_packed.h5` (1,075,369,707 bytes; SHA-256 `7e0c7d06cdf6fde8eb81c6feea179f9cb6b1e6d887797a5ebadf99037670caca`). The local raw crop directories are absent; use the packed H5 if actual input imagery/targets are needed. `test.csv` beside it maps IDs to original logical paths and includes coordinates, e.g. `80234/s2.tif`, but those per-crop files are not present.

## Provenance and metrics

The campaign protocol is `reports/experiments/methanes2cm_v5_1_campaign_protocol.json` (architecture `ersrr_methanes2cm_tri_temporal_context_v5_1`, commit `39da5c20`, seeds 1101/2202/3303, 12 epochs, frozen calibration). The ensemble report is `reports/experiments/methanes2cm_v5_1_ensemble_validation.json`; the one-shot location-test report is `reports/experiments/methanes2cm_v5_1_location_test.json`.

On the 20,789-crop sealed location test, v5.1 recorded scene AP **0.8180196480**, AUROC **0.8275743356**, recall **0.3777862815**, FPR **0.0606617647**, pixel AP **0.2082942863**, Dice **0.3125309890**, and IoU **0.1852069502**. The post-hoc report confirms the 5% development rule transferred to 6.07% test FPR / 37.78% recall without retuning.

## Checkpoint status

No v5.1 model checkpoint was found in the three sibling repositories. The protocol records seed-1101 checkpoint SHA-256 `7b648548cc62ca3f6d428df2cf427e373fba5a7bdcf03aabada68bf6f1cfc446`, but the corresponding file is not present locally. The prediction cache plus packed H5 therefore provides a concrete, authentic viewer route without needing checkpoint inference. Existing local `.pt` files are MARS/Gaussian or ordinal recovery artifacts and must not be labeled v5.1.

The calibration cache is also present at `.../v5_1_ensemble_calibration.npz` (13,559,029 bytes; SHA-256 `6e6ad2334b046292ca2c88d9364215f8964f0b49a28478e5d04cbc72a7ea7510`).
