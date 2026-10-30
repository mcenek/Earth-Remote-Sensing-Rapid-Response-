# Reproduce the training-pipeline correction

Run these commands from the repository folder on the Windows training machine. Use the existing `.venv`. The commands use cached real Sentinel-2 pairs and binary MARS plume annotations. They do not download imagery. The source arrays and previous experiments stay unchanged.

## What changed

The expanded sampler reduced US positive training draws from 9,600 to 2,043 at the old update budget. The corrected sampler gives each batch two US plume scenes, two US no-plume scenes, two international plume scenes and two international no-plume scenes. It preserves the old US draw order and adds the international examples. There are 4,800 updates, with 9,600 draws in each group.

The corrected run trains on full 100x100 scenes and predicts on full 100x100 scenes. Earlier training used plume-centered 48x48 crops. Full scenes change context, spatial normalization and the amount of background in the loss. The controlled comparison measures this complete change; it does not isolate these mechanisms individually.

The separate centering pilot removes each input date's median log brightness for each band. Global centers and scales still come from TRAIN only. This loses absolute scene brightness and must be compared with the unchanged full-scene control. It is a preprocessing experiment, not a correction to the producer's labels.

## Train and verify

Use a new, empty output folder for a new run. The existing output folders contain the recorded experiment and cannot be overwritten by these commands.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_train_repaired_domain_v4.py train --arm balanced_full_scene --output outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene
.venv/Scripts/python.exe tools/GTM_Model7_domain_v4_e2e.py --run outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene
.venv/Scripts/python.exe tools/GTM_Model7_calibrate_repaired.py --run outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene --output outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_calibration
```

The centering pilot expects the full-scene control beside it, so it can check that the draw schedules are identical:

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_train_repaired_centered_v5.py --output outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered
.venv/Scripts/python.exe tools/GTM_Model7_centered_v5_e2e.py --run outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered
.venv/Scripts/python.exe tools/GTM_Model7_calibrate_repaired.py --run outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered --output outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration
```

## Save actual predictions

Choose one frozen run. The frozen trainer, preprocessing, source hashes, weights and cutoff must stay together. Do not apply the new preprocessing to old weights.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_repaired_inference.py infer --freeze outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_calibration/freeze.json --split dev --output outputs/GTM_Model7_pipeline_fix_20261024/full_scene_dev_inference
.venv/Scripts/python.exe tools/GTM_Model7_repaired_inference.py infer --freeze outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_calibration/freeze.json --split dev --output outputs/GTM_Model7_pipeline_fix_20261024/full_scene_dev_inference --verify
```

`GTM_Model7_review_repaired.py` reconstructs source images, checks native counts and makes the viewer bundles. `GTM_Model7_prediction_publication_e2e.py` then checks the actual served images, score grids, frozen cutoffs and native totals. Use the recorded review and HTTP receipts in the run folder to inspect the published result.

## Read the result correctly

- Review all six DEV plume cases and the no-plume cases. Inspect the error view as well as the prediction overlay.
- The 86 DEV scenes are an exposed development set. Results on them are not fresh validation.
- The 55 international scenes used again after freezing are an exposed regression check. Do not use them to choose another cutoff or model.
- The original US TEST was not reopened for this correction. It has earlier project exposure and is not a fresh holdout.
- A 20 m binary mask shown on a 10 m label grid is not methane super-resolution or a quantitative EMIT prediction.
- A no-plume annotation does not prove zero methane. Cloud-mask checks do not prove an ideal reference pair.
- Passing a numeric development check does not approve a model for nationwide inference.

The diagnosis, centering and earlier calibration protocols define the stopping rules. Keep failed cases and failed runs visible in the evidence. Do not continue a search on the same six DEV plumes after this bounded pilot.

## Recorded outcome

The strongest tested improvement came from full-scene training followed by per-date, per-band brightness centering. Restoring US exposure alone did not rescue both models. Adding examples was useful only after changing the training recipe. The source audit found no systematic band swap, target/reference swap or damaged label join. Some reference pairs still have visible haze and brightness differences.

Six controlled arms ran on the same GPU with the same architectures, optimizer, loss and seed. At the frozen DEV selections:

| Training arm | Residual CNN native IoU | Temporal U-Net native IoU |
| --- | ---: | ---: |
| US only, original normalization | 28.82% | 2.16% |
| Same US draws, combined normalization | 36.10% | 4.75% |
| Expanded sampler, longer training | 21.90% | 8.50% |
| US/international balanced, 48x48 crops | 21.26% | 13.92% |
| Same balanced draws, full 100x100 scenes | 27.98% | 5.94% |
| Same full scenes, brightness centering | 34.64% | 43.04% |

These selections use the predeclared DEV threshold grid. Fixed-0.50 milestone counts are also saved in `outputs/GTM_Model7_pipeline_fix_20261024/diagnosis_report.json`. Full scenes jointly change context, normalization and foreground/background proportions. This comparison does not isolate those mechanisms. This is one seed, not a statistical uncertainty estimate. Earlier CPU fits are historical baselines, not matched GPU optimizer controls.

The final residual checkpoint is step 4800 and the U-Net checkpoint is step 3600. Both use cutoff 0.95. The freeze SHA256 is `ead439af495367347dfd8d8ba70870440882c40164b9f99ad66856a090007bc8`.

| Evaluation | Residual before / after IoU | U-Net before / after IoU |
| --- | ---: | ---: |
| Exposed DEV, 6 plume + 80 no-plume | 11.96% / 34.64% | 4.57% / 43.04% |
| Exposed external regression, 15 plume + 40 no-plume | 4.93% / 8.58% | 2.38% / 22.95% |

Both corrected models pass the existing numeric DEV gate. Both still fail the external geographic gate. External U-Net precision is 71.69%, but recall is only 25.23%. Its no-plume scenes with any false prediction fell from 30/40 to 9/40. Residual external recall is 9.96%, with 8/40 no-plume scenes activated. Small false detections count in these scene totals. Neither model is ready for nationwide inference.

The entire 958-observation TRAIN collection is retained. Each final fit uses 38,400 draws, with 9,600 in each of US plume, US no-plume, international plume and international no-plume. All global scaling comes from TRAIN. The external choices were frozen before reopening its results. No new imagery was downloaded and no annotations were changed.

## Reproduce the final predictions

The earlier inference example deliberately reproduces the uncentered control. Use the centered freeze for the corrected result:

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_repaired_inference.py infer --freeze outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration/freeze.json --split dev --output outputs/GTM_Model7_pipeline_fix_20261024/centered_dev_inference
.venv/Scripts/python.exe tools/GTM_Model7_repaired_inference.py infer --freeze outputs/GTM_Model7_pipeline_fix_20261024/balanced_full_scene_centered_calibration/freeze.json --split dev --output outputs/GTM_Model7_pipeline_fix_20261024/centered_dev_inference --verify
```

Use a new output directory for a new export. The recorded outputs are evidence and must be preserved. The `.venv` renderer also needs matplotlib; keep the recorded NumPy 2.4.4 and Torch 2.11.0+cu128 versions for replay. Installing the plotting dependency does not require another model fit.

Final viewer IDs are `GTM_Model7_pipeline_v5_20261024_dev_siamese` and `GTM_Model7_pipeline_v5_20261024_dev_stable_residual`. Replace `dev` with `external` for the 55-case regression. Select the same scene in either model. Compare the orange mask with the cyan annotation boundary, then select the error display. Gray or empty predictions are retained as evidence.

Root personally checked contact sheets, live paired image panels and the main georeferenced viewer, including plume recovery, misses, clear scenes, both models, scores, errors and cutoff reset. Saved probabilities replay exactly. Native annotation counts and served masks agree. The HTTP E2E checks cover 1,206 DEV assets and 772 external assets. Final publication versions and browser screenshots are recorded in `outputs/GTM_Model7_pipeline_fix_20261024/completion_receipt.json`.
