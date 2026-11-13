# US transfer work, October 26

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

## What is established

The retained v11 model can trace parts of real methane plume annotations. On the 198 development observations, it has 53.06% pixel F1, 71.48% precision and 42.19% recall. It marks pixels in 18 of 152 provider no-plume observations. These observations have been used in earlier project work. They are not an independent US test.

Only three US plume observations are in that development group, alongside 60 US controls. Their descriptive F1 is 80.98%. Do not use that small subgroup to claim national accuracy. Of the 97 US plume observations in fitting, 95 belong to one connected West Texas group.

These runs predict binary plume masks from paired Sentinel images. They do not measure emission rates or demonstrate quantitative EMIT upscaling. A CAFO coordinate is a possible source location, not a dated methane plume label.

## Completed native-grid experiment, v13

Two fresh U-Nets were trained for 4,800 updates each, with and without wind. Both used the same original source data, weights, draw schedule and loss. Neither produced an eligible development choice. Even the highest unconstrained F1 was only 27.67% without wind and 24.30% with wind, with excessive false detections.

The predefined failure display uses update 1,200 and cutoff 0.70. It is not the best achievable F1. Both failed masks, and the unchanged v11 control, are available in the local viewer. The report explicitly retains all misses and false detections.

CPU/GPU replay also exceeded its numerical tolerance. That failure is preserved. No accepted numerical receipt or promoted model was created. CPU and GPU still give the same scientific rejection, zero eligible choices and identical masks at the predefined failure cutoff.

Evidence:

- `outputs/GTM_Model7_us_transfer_v13_20261026/selection/rejection_summary.json`: all 88 choices per arm, CPU and GPU, with error guards and hashes.
- `evaluation/summary.json`: actual saved failed fallback masks for all 198 observations per model.
- `review_v2/`: source-checked report and map bundles. The baseline has its own retained-control status.
- `contacts/`: all 46 plume observations, the largest ten clear false alarms and all three US positives.
- `training_audit/`: TRAIN-only loss and signal audit.

Root inspected every plume contact and the ten clear false alarms. The map and report show the same shapes and missed plumes. The failure is not an absent overlay.

## Controlled correction experiment, v14

The next experiment starts with the fixed v11 score. A small U-Net learns a bounded correction on the original annotation grid. Its final layer starts at zero, so all initial masks must match the teacher. This passed on all 1,156 fitting and development observations.

Two fits differ only in pixel weighting. One keeps the original loss. The other gives rare plume pixels more weight. Scene balancing alone did not balance pixels: plume pixels are 0.83% of pooled fitting support and 1.52% under the fixed balanced draws.

The protocol was written before preparation or fitting. It fixes two runs, four checkpoints, 22 cutoffs, clear-error limits and a teacher fallback. A replacement must improve F1 by at least two percentage points, keep recall, limit clear false alarms and avoid adding more false pixels than true plume pixels. No mask growth, filling, seed search or case removal is allowed.

Status: both 4,800-update fits are complete. The original-loss fit took 940.5 seconds; the balanced-pixel fit took 1,003.4 seconds. All four saved checkpoints in both arms replayed exactly on CPU in a fresh process. Each arm had zero eligible choices out of 88. An independent root audit recomputed all 176 choices' counts and error guards and confirmed the decision to retain v11.

The highest unconstrained F1 was 52.59% for original loss at update 1,200 and cutoff 0.70, and 52.88% for balanced pixels at update 1,200 and cutoff 0.95. Both are below v11's 53.06%. Their precision fell to 59.51% and 61.70%. They activated 45 and 29 provider no-plume observations, compared with 18 for v11. Neither reaches the 70% useful target.

The frozen selection correctly falls back to the unchanged teacher. Therefore the selected-result comparison contains three identical teacher masks. That is not evidence that the correction models learned identical outputs. A separate, explicitly rejected diagnostic comparison displays the actual best unconstrained correction masks, alongside v11, for all 198 observations.

The correction study verified source data, optimization and reproducible inference. It did not establish a better detector. Keep the failed results visible and do not keep launching new fits on the same exposed validation labels to manufacture progress.

Evidence is under `outputs/GTM_Model7_native_residual_v14_20261026/`: `selection/`, `evaluation/`, `root_selection_audit.json`, `review/` for the unchanged selected fallback, and `review_diagnostics/` for actual rejected predictions. The corresponding `mask_changes` receipts record gained and lost true pixels and added and removed false pixels.

Protocol: `docs/GTM_Model7_native_residual_v14_protocol_20261026.md`.

## Apply the retained model to another compatible local pair

Run from the repository root. Use the existing virtual environment.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_predict_pair.py `
  --current CURRENT.tif --reference REFERENCE.tif `
  --current-cloud CURRENT_cloud.tif --reference-cloud REFERENCE_cloud.tif `
  --current-utc 2024-10-23T17:00:00Z --reference-utc 2024-10-18T17:00:00Z `
  --radiometry mars_cached_l1c_dn_uint16 --output outputs/NEW_PAIR
```

Use actual acquisition times. Both images must follow the verified MARS cached-DN convention: aligned 200 by 200 crops, a projected metre CRS, 10 m spacing, uint16 DN, NoData zero and named B02/B03/B04/B08/B11/B12 bands. Both CloudSEN12 masks are required. The earlier reference must be a different observation. Its methane absence still needs review.

The command saves georeferenced probability, mask and validity TIFFs, an NPZ and a receipt. The score is an uncalibrated model output. Invalid score pixels are NoData. The 20 m prediction is repeated to the 10 m output grid; this adds no spatial detail. No label enters inference. The output directory must be new.

This is a crop inference tool, not a validated nationwide scanner. It rejects arbitrary JP2, L2A and shifted-grid inputs. The radiometry declaration is a caller assertion; it cannot establish the producer's processing history. Five checks cover real exposed plume/clear flows and malformed inputs. Exact old-score parity is an engineering check, not new accuracy evidence.

Publish that saved prediction in the running viewer:

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_publish_pair.py `
  --prediction-dir outputs/NEW_PAIR `
  --output outputs/NEW_PAIR_VIEWER `
  --id GTM_Model7_new_pair_001 --name "US pair 001 | unverified prediction" --publish
```

Use new output directories and a new experiment ID. The publisher checks the source images, model, output files, support and frozen mask against the inference receipt. It does not fit a model or use a label. Current and earlier RGB images use one shared stretch. The map shows the actual saved mask and score. When no methane annotation exists, the comparison panel shows the earlier image, disables truth and error controls, and says that accuracy is not measured.

Open `http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_new_pair_001&scene=pair#compare`. Refresh the experiment source if needed. On another device, replace the host with the existing private Tailscale host and keep port 8766. Publication does not restart the viewer.

Two already exposed development samples verify this path. Sample A has 1,076 marked native pixels; sample B has zero. Their IDs are `GTM_Model7_v11_exposed_dev_sample_a_engineering_20261026` and `GTM_Model7_v11_exposed_dev_sample_b_engineering_20261026`. They demonstrate the unlabeled inference workflow, not a new test result. The original labeled comparisons remain available separately. Three focused publisher checks passed, including file tampering and the refusal to overwrite an existing publication.

The commands require the local frozen checkpoint, normalization and source receipts. Those ignored research artifacts are not supplied by a fresh code-only clone.

## New US data ready for the next step

Metadata screening found six possible paired observations across four EMIT overpasses. Three distinct events have current and earlier-reference Sentinel assets selected: July 31, WM Security on July 29 and Okeechobee on July 28, 2025. The closest time difference is 12.1 minutes. These are acquisition candidates, not validated training or test pairs.

The selected native EMIT files total about 1.65 MB. Their contents have not been opened. The complete Sentinel crop cost remains unknown. The new range reader charges transfers against the existing ledger and passed seven offline tests. A small window required the whole test JP2 file, so real crop efficiency must be checked before acquiring all bands.

The shared 100 MiB ledger has 130,938 bytes left. The question asking for up to 256 MiB additional acquisition is pending. No new allowance was created and no further imagery was transferred. Preserve the accumulated charges if an increase is approved.

Before model use, verify native EMIT labels, full footprints, timing, radiometry, alignment, local cloud quality and reference quality. Reserve new positive and control sites before fitting or choosing cutoffs. Existing unresolved historical site names prevent calling the shortlist a pristine final test.

The pinned MARS producer code uses Earth Engine's harmonized L1C collection and describes cached DN as top-of-atmosphere reflectance times 10,000. This establishes the intended convention. It does not establish each historical TIFF's processing history or the selected 2025 JP2 offsets. Read the exact products' metadata before conversion; do not apply a blanket offset or switch to L2A silently.

The released CloudSEN12 path needs all 13 L1C bands. Our six-band acquisition shortlist still lacks B01/B05/B06/B07/B8A/B09/B10 for each date. The cloud weight is already local, but its pinned package was not importable in the inspected environments, and resampling parity with the producer is unverified. Thus the pilot is metadata-prepared, not acquisition-complete or ready for valid inference. Freeze the extra-band allowlist and finish these checks before using it. See `readiness/cloud_mask_readiness.md`.

An additional metadata audit checked the 579 already cached official-test scenes. All were scored in an earlier project evaluation. No US site is more than 25 km from the current fitting and development cohort. Three positive observations elsewhere are geographically separate, but none has a verified local earlier clear reference. This archive cannot supply the missing fresh US validation by renaming its split. No protected arrays or labels were opened for that audit.

Detailed acquisition evidence is under `outputs/GTM_Model7_us_transfer_v13_20261026/readiness/`.

The fixed acquisition and evaluation rules are in `docs/GTM_Model7_us_pair_pilot_protocol_20261026.md`. The pilot must retain rejected pairs and must not treat an unknown area outside a positive annotation as negative ground truth.

## Local verification commands

```powershell
.venv/Scripts/python.exe -m pytest tests/test_bounded_jp2_v13.py tests/test_predict_pair.py -q
.venv/Scripts/python.exe tools/GTM_Model7_native_review_recipe_v14.py
.venv/Scripts/python.exe tools/GTM_Model7_review_native_v13.py --recipe outputs/GTM_Model7_native_residual_v14_20261026/review_recipe.json --output outputs/GTM_Model7_native_residual_v14_20261026/review
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_native_residual_v14_20261026/review --output outputs/GTM_Model7_native_residual_v14_20261026/http_verification.json
```

The last three commands require completed, frozen and replayed v14 results. They refuse stale or incomplete scientific artifacts. Do not rerun a builder into an existing destination. The viewer remains on localhost port 8766 and the existing private Tailscale address. No public exposure or server restart is needed.


## Fixed comparison with cached released models

The v15 protocol was written before inference. Both local released checkpoints ran at their published 0.5 cutoff, without retraining or downloads. Each model replayed exactly in a fresh CPU process over all 198 development observations. There were no valid scores exactly equal to the cutoff.

| Detector | Precision | Recall | F1 | Clear observations marked |
| --- | ---: | ---: | ---: | ---: |
| Retained v11, cutoff 0.65 | 71.48% | 42.19% | 53.06% | 18/152 |
| Released MARS-S2L, cutoff 0.5 | 56.32% | 55.14% | 55.72% | 17/152 |
| Released CH4Net, cutoff 0.5 | 7.89% | 2.62% | 3.93% | 21/152 |

MARS-S2L recovers more labeled plume area but has 26,003 false pixels, versus 10,237 for v11. Its clear-pixel error is also higher despite activating one fewer clear scene. This is not a qualifying replacement. CH4Net misses nearly all plume pixels on these inputs.

Both released configurations list 38,366 training examples and overlap these development locations: 113 observations are at released training-location IDs, and 11 are at released validation-location IDs. MARS uses our separately selected earlier reference, which differs from its original release benchmark. The 85 observations whose IDs are not listed as released training locations still belong to historically exposed project development data; the ID check does not establish geographic or imagery independence. No national accuracy claim or model promotion follows from this study.

Evidence: `outputs/GTM_Model7_released_comparison_v15_20261026/`, including `preflight/`, `inference/`, `replay/`, and the source-reconstructed `review/`. The protocol is `docs/GTM_Model7_released_comparison_v15_protocol_20261026.md`. The viewer comparison retains every observation, including misses and false alarms.


Root personally checked every plume, the ten largest clear false alarms and the three US plume observations in 13 contact pages. The report and map were also checked interactively. MARS recovers some complete misses in Turkmenistan and Uzbekistan, but still has complete misses elsewhere and visible excess markings on US clear scenes. All 198 inputs were independently rebuilt from the original sources. All 4,161 served assets matched the saved exports; three prediction overlays also matched through private Tailscale HTTPS.

Open the visual comparison:
`http://127.0.0.1:8766/bundle-assets/GTM_Model7_released_v15_20261026_candidate/_versions/125dbef2a5164af5b25f8945647ecc89/index.html`

Open the retained detector over a US plume:
`http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_released_v15_20261026_baseline&scene=MARS_9123854a-64f6-4791-9255-63ad109e0896#compare`


## Cloud runtime preparation completed

The missing CloudSEN12 package now runs in a separate local package folder. The shared virtual environment and Torch were not changed. Official package transfers totaled 4,048,639 bytes, and the existing 26.8 MB cloud weight loaded strictly on CPU. No imagery, product metadata or new weights were transferred.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_cloud_runtime_v16.py infer `
  --input EXISTING_ALIGNED_13_BAND_L1C_DN.tif `
  --output NEW_CLOUD_MASK.tif --radiometry l1c_dn_div10000
```

The command requires exactly the 13 named bands on an aligned 200 by 200, 10 m projected grid. It rejects scaled, offset-tagged and L2A inputs. It zeros all channels at invalid pixels before cloud prediction and assigns class 4 afterward. The output keeps the source grid and uses NoData 4. The pair predictor and publisher now accept this proper invalid code as well as the historical MARS NoData 0 metadata; in both cases zero class values remain clear.

Five cloud-runtime checks, six pair-inference checks and three publisher checks passed. Root also checked the new class-4 format through actual inference on a real cached source pair and verified its saved receipt. The cloud model itself was tested only with synthetic plumbing data. This does not establish real cloud accuracy or producer resampling parity. Missing bands, exact source radiometry, product identity and real-data quality still need verification.

The isolated package loader bypasses unused plotting and SAFE-reader imports with scoped import shims. It does not support those interfaces or change official cloud prediction source. Runtime files, package hashes, synthetic receipts and root integration evidence are under `outputs/GTM_Model7_us_transfer_v13_20261026/readiness/cloud_runtime_v16/`. Copy the required artifacts with the code when moving machines; a code-only checkout is insufficient.
