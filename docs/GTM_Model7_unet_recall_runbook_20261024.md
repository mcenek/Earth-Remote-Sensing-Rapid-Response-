# U-Net recall experiment, October 24

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

This run tests why the temporal U-Net misses plumes. It uses real cached MARS Sentinel-2 image pairs and binary annotations. It does not train quantitative EMIT methane reconstruction. The scientific rules were fixed before fitting in [the protocol](GTM_Model7_unet_recall_protocol_20261024.md).

## What changes

The baseline uses full 100 x 100 scenes, date-wise band-median removal in log brightness, and TRAIN-only scaling. Keep those corrections. Keep the exact 38,400 training draws, four equal domain/class groups, seed, optimizer, and 4,800 updates. Half of each training batch is explicitly labeled no-plume data.

- `scene_dice` gives each positive scene equal weight in the Dice term and raises that term from 0.1 to 0.5. Weighted BCE is unchanged. These are joint changes; this run cannot separate their individual effects.
- `delta_scene_dice` also adds a zero-initialized linear path from the six current-minus-reference input bands to the output. It starts with exactly the same backbone predictions.
- The residual CNN is a retained control. It is not retrained or selected again.

The audit found 92 of 271 training plume components with no pixel score above 0.50 at the frozen baseline checkpoint. This shows that lowering the display cutoff cannot fix all misses. Components are descriptive measurements, not independent sites. The exposed DEV set has only six positive scenes, so it provides weak evidence for selecting a model.

## Data separation

Training contains 958 observations: 268 plume and 690 no-plume. DEV contains six plume and 80 no-plume observations. DEV was already used in previous research and is not an independent test.

The separate metadata reservation contains 17 sites, 28 plume observations, and 64 no-plume observations. These are candidates before fixed source QA, not a promise that all pass. No replacement is allowed after QA. The selected model and cutoff must be fixed before opening these images. The reservation excludes known prior local project fit, evaluation and viewer sites, target/reference acquisitions, spectral hashes, and sites within 25 km. Historical source inventory metadata is disclosed separately. This does not establish absence from an upstream publication's training data or inaccessible experiments.

The final comparison uses frozen v5 and the selected model on identical accepted observations. If selection retains v5 itself, run it once and disclose that there is no improved-model comparison. No final-site outcomes may choose a checkpoint, cutoff, architecture or reference. Once evaluated, these sites are consumed and cannot be presented as fresh in a later iteration.

## Run and verify

Run from the repository root with `.venv/Scripts/python.exe` on Windows. Existing output folders are preserved; use new folders for repetitions. See the scripts' `--help` for required paths.

1. `tools/GTM_Model7_recall_v6_e2e.py preflight` reconstructs the loss independently from a real balanced batch, checks invalid support and negatives, and verifies initialization and finite gradients.
2. `tools/GTM_Model7_train_repaired_recall_v6.py --arm scene_dice --output <run>` trains the loss candidate. Repeat with `delta_scene_dice` in a separate folder.
3. `tools/GTM_Model7_recall_v6_e2e.py replay --run <run> --output <replay>` replays all four DEV milestones on CPU and checks native counts. Run for both candidates.
4. `tools/GTM_Model7_calibrate_recall_v6.py --run <run> --output <calibration>` applies the fixed DEV gate and scene-recall ranking. Run for both candidates.
5. `tools/GTM_Model7_select_recall_v6.py --candidate <first-freeze> --candidate <second-freeze> --reservation <reservation.json> --output <selection>` compares both candidates against the baseline at its original frozen cutoff. This writes the final selected `freeze.json` and selection receipt.
6. Only after step 5, run `outputs/GTM_Model7_fresh_site_reservation_20261024/prepare_reserved.py --output <fresh-data> --freeze <selection/freeze.json> --freeze-sha256 <SHA256>`. Then run `verify_reserved_e2e.py --output <fresh-data>` from the same directory. These scripts rebuild accepted arrays from the cached TIFFs.
7. Run `tools/GTM_Model7_diversity_external_inference.py --freeze <freeze> --evaluation-manifest <fresh-data/manifest.json> --output <predictions>`. Repeat the command with `--verify`, then with `--summary`. Use separate folders only if baseline and selected model differ. The tool only accepts those two frozen choices on this final cohort. Each includes the unchanged residual control.
8. Run `tools/GTM_Model7_review_repaired.py` with the matching freeze, inference, evaluation manifest, new ID/title, scope file, `--balanced-display`, and `--publish`. This rebuilds source arrays again, recounts native pixels, and exports every prediction, score, error image, annotation and map grid.
9. Run `tools/GTM_Model7_prediction_publication_e2e.py --review <review> --output <http-receipt>` against the running local viewer. It checks served bytes, static versus interactive masks, both classes, and native totals. Inspect actual contact sheets and browser controls personally. Save screenshots and the browser receipt.

## Reading the viewer

Orange is the predicted mask. Cyan is the released annotation. The error view shows correct overlap in teal, extra predictions in pink, and missed annotation in purple. Scores use a fixed zero-to-one scale and are not calibrated probabilities. The frozen cutoff is shown with each run. Changing it in the viewer is exploration, not a new validated result.

A blank predicted mask can be an actual miss. A no-plume annotation means no released plume annotation for that observation; it is not proof that the location emits no methane. A 20 m prediction repeated on a 10 m annotation grid is not learned upscaling. National source mapping, emission rates, and CAFO detection accuracy remain separate unproven goals.

## Completion evidence

The final metrics, accepted counts, viewer links, visual checks and limitations are saved under `outputs/GTM_Model7_unet_recall_20261024/`. Keep failed candidates and all final cases visible. Do not promote a model for statewide inference solely because one displayed plume looks good.

## Completed result

Neither new candidate improved the frozen selection objective. The retained model is exactly v5, step 3600, cutoff 0.95, with identical weights. There is no new improved U-Net. Both candidates completed 4,800 updates using the same 958 TRAIN observations, with four plume and four no-plume observations per batch.

Development results below use the same exposed 86 observations, including only six positives. They are model-selection evidence, not independent accuracy.

| Model or setting | Native IoU | Pixel recall | Mean positive-scene recall | No-plume pixel false alarms | Decision |
|---|---:|---:|---:|---:|---|
| Retained v5, 3600 / 0.95 | 43.04% | 55.88% | 31.26% | 0.0702% | Retained |
| Scene Dice, 3600 / 0.975 | 39.72% | 44.74% | 25.09% | 0.0331% | Eligible, worse selection objective |
| Image-difference candidate, best IoU setting, 4800 / 0.975 | 34.84% | 52.38% | 29.27% | 0.1423% | Fails false-alarm limit |

No image-difference checkpoint/cutoff passed every development gate. Its published diagnostic uses 1200 / 0.05, selected for recall among rejected settings: recall 85.22%, precision 3.47%, IoU 3.45%, all 80 clear scenes activate. This diagnostic is not its best-IoU setting and is not a viable model.

Only after selection, opened the reserved final cohort. All 92 observations at 17 sites passed fixed source QA: 28 plume and 64 no-plume, zero exclusions, replacements or downloads. The sites are over 25 km from known previous local project sites, with no shared target/reference acquisitions or spectral hashes. They are international locations in six countries, not a new US benchmark. This evaluation is now consumed.

| Final model | Native IoU | Pixel precision | Pixel recall | Plume scenes with at least 10% pixel recovery | No-plume scenes with any extra prediction |
|---|---:|---:|---:|---:|---:|
| Retained temporal U-Net | 9.99% | 54.23% | 10.91% | 8 / 28 | 5 / 64 |
| Unchanged residual control | 9.67% | 48.65% | 10.77% | 7 / 28 | 7 / 64 |

The U-Net had zero correctly recovered pixels on 20 of 28 plume scenes. It fails geographic evaluation and is not suitable for statewide inference. Source reconstruction, native-grid recounts, exact fresh-process replay and HTTP publication passed. Root reviewed actual masks, scores, misses, clear cases and false alarms. These checks rule out a missing-overlay explanation for the observed misses. They do not prove one universal scientific cause. Loss aggregation, spatial resolution, limited model capacity, reference quality and weak methane signal remain possible contributors.

The training-component audit found 92 of 271 components with every score below 0.50, and only five of 17 small components reaching 0.50. Lowering the display threshold alone cannot repair this. The baseline BCE is weighted by valid pixel count, with positive weight 2, plus 0.1 batch-pooled positive Dice. An earlier audit description of equal-class-mean BCE was incorrect and has been corrected; numeric audit counts were unchanged.

### Open the actual predictions

- [All 92 cases, both models, score and error views](http://127.0.0.1:8766/bundle-assets/GTM_Model7_recall_v6_20261024_fresh_retained_siamese/_versions/0ae9a6c9cfc5465aba552ba247425747/index.html)
- [Georeferenced viewer, partial plume example](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_recall_v6_20261024_fresh_retained_siamese&scene=MARS_9048f20f-fad5-4d2c-832d-c125b4ad0413#compare)
- [Scene Dice candidate on DEV](http://127.0.0.1:8766/bundle-assets/GTM_Model7_recall_v6_20261024_scene_dice_dev_siamese/_versions/28f3378aa1cf4634bd108494c97fd61f/index.html)
- [Rejected image-difference diagnostic on DEV](http://127.0.0.1:8766/bundle-assets/GTM_Model7_recall_v6_20261024_delta_dev_siamese/_versions/26e13c5182c84eca89efd4b1f06e3d73/index.html)

Use the scene menu to inspect every case. Scene 8 in the paired report is a complete plume miss; scene 18 is a partial detection; scene 31 is a correct no-plume result; scene 41 is a no-plume false alarm. The one-based numbers refer to this exact saved report. A single example is not the aggregate result. Access requires the existing private Tailscale connection; the local server is unchanged on port 8766.

Completion receipts: `completion_receipt.json`, `diagnosis_report.json`, `browser_e2e.json`, `fresh_http_e2e_v2.json`, `fresh_review_v2/verification.json`, and `fresh_inference/replay.json` in the output folder. The v2 final report supersedes v1 wording. `preflight_v6_final` is authoritative; preserve the older preflight and interrupted attempt as provenance only.

Next research needs a separate, geographically broader development cohort before further model selection, plus reserved final sites. Do not tune on these 17 sites and then call them fresh again. Preserve the current failures as regression cases. Quantitative EMIT upscaling and US source detection remain unproven.
