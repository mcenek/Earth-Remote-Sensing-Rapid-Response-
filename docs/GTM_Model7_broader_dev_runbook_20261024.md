# Broader development and U-Net improvement, 24 October 2026

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

The revised U-Net improves on the broader development set. It still misses whole plumes and produces false detections. It is not ready for a United States survey. New-site plume recall remains unknown.

## Data and separation

- Rebuilt 198 development observations from cached real source files: 46 plume and 152 no-plume observations, 55 physical sites, 12 countries. These sites have been evaluated in earlier project work. They are explicitly exposed development data, not a fresh test set. No development rows enter gradient updates.
- Retained the same 958 training observations: 268 plume and 690 no-plume. Verified original identity, source arrays, reference images, support masks and native labels. No imagery downloads.
- A metadata audit found no unused paired positive sites that pass the recorded prior-exposure, 25 km distance, acquisition and hash checks. Of 43 positive candidates, 35 were previously used, seven were too near exposed sites and the remaining site shared a consumed acquisition. The empty positive reservation is saved; no fresh-positive result is claimed.
- Separately reserved 20 explicit NO_PLUME observations at ten previously unused sites, before their pixels were opened. Model selection and both training runs were complete before these final pixels were prepared. Fixed two-date support checks accepted 17 observations at nine sites; three failed the 90% support rule. No replacements were made.
- The new clear-site test spans nine country/location labels and includes no US sites. Minimum distance from prior sites is 25.026 km. These final observations are now consumed and must not be used for further tuning while retaining a test claim.

## What changed

Compared two bounded fits, each with 4,800 updates, the same 38,400 training draws, seed, TRAIN-only scales, full 100 by 100 windows and D4 rotations/reflections. Each batch contains four plume and four no-plume observations. Retained valid-pixel BCE with positive weight 2 plus 0.1 pooled positive Dice.

1. Early fusion: current six bands, reference six bands, their signed difference and absolute difference enter a 24-channel U-Net with widths 16/32/64, GroupNorm and SiLU.
2. Spectral fusion: the same model plus two centered log-band contrasts, B12 minus B11 and B12 minus B08.

Fusion, width and normalization changed together. This comparison cannot identify which one caused the improvement. The residual CNN is a frozen control and was not retrained.

## Selection and results

Selected early fusion at update 1,200 and cutoff 0.70. Later fitting improved training performance but did not improve the broader development gate. Selection considered 177 predeclared checkpoint/cutoff rows, including the old frozen model and calibration-only control. Eligibility required precision at least 50%, recall at least 30%, IoU at least 20%, clear-pixel false-positive rate at most 0.10%, and recovery of at least 10% of annotation pixels in at least half of the plume observations. Eligible candidates were ranked by mean positive-site recall, mean positive-scene recall, IoU, lower clear false-positive rate, earlier update, then cutoff.

| Same 198 development observations | Old v5 U-Net | Selected U-Net |
|---|---:|---:|
| Pooled plume overlap, IoU | 17.06% | 28.13% |
| Plume-pixel precision | 62.52% | 65.18% |
| Plume-pixel recall | 19.00% | 33.10% |
| Mean positive-site recall | 12.39% | 29.75% |
| Observations recovering at least 10% of plume pixels | 13/46 | 23/46 |
| Clear observations with any positive pixel | 26/152 | 20/152 |
| Clear-pixel false-positive rate | 0.0889% | 0.0849% |

These are development results used for selection, not independent test performance. The old model's earlier 43% figure used a much smaller, different development set and is not comparable to this table.

Neither recalibrating the old weights nor the spectral arm passed the broader gate. The spectral arm's highest IoU was 29.64%, but precision was 48.06% and clear-pixel false-positive rate was 0.2572%. Published rejected diagnostics are the predeclared maximum-recall fallbacks, not their highest-IoU settings. They are clearly marked REJECTED.

| New-site, no-plume test only | Old v5 U-Net | Selected U-Net |
|---|---:|---:|
| False-positive pixels / 672,612 evaluated pixels | 2,844 | 1,328 |
| Clear-pixel false-positive rate | 0.4228% | 0.1974% |
| Observations with any false detection | 3/17 | 2/17 |
| Physical sites with any false detection | 2/9 | 2/9 |

False-positive pixels fell by 53%, but the new 0.1974% rate exceeds the 0.10% development limit. Fewer activated observations did not reduce the number of activated physical sites. There are no positive test examples, so recall cannot be estimated. No training or threshold changes were made after this test.

## Visual checks and viewer

Root inspected all 46 development plume cases, representative negatives and the largest false alarms, and all 17 fresh no-plume cases. Some Egypt, Iran and Russia plume shapes improve. Complete misses remain in Kazakhstan, Libya and Venezuela. The new model also creates real false blobs in clear Egypt and Iran scenes. Reference haze and surface changes remain visible. The missing detections are present in the saved arrays, not explained by an absent overlay.

- Main viewer: http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_broader_v7_20261024_selected_siamese&scene=MARS_1017caf2-e6a8-482b-b176-fbf2beaa7c71#compare
- Selected development report: http://127.0.0.1:8766/bundle-assets/GTM_Model7_broader_v7_20261024_selected_siamese/_versions/2090c85c3e0a408cb85a64542fcd0e3c/index.html
- Fresh clear-site, old-versus-new comparison: http://127.0.0.1:8766/bundle-assets/GTM_Model7_broader_v7_20261024_fresh_clear_selected/_versions/dee3e73587d44ea8a43487b8570fd122/index.html

Use Model and Scene to change cases. Use Display for scores, Disagreement for missed/false pixels, and Reset to artifact threshold to return to the frozen cutoff. Native scores are the saved benchmark; map resampling can change exploratory display scores. The selected development paired report compares the selected U-Net with the retained residual CNN. For old-U-Net versus selected-U-Net comparisons, use the separately named main-viewer runs or saved contact sheets. The fresh clear report directly compares both U-Nets.

## Verification and reproduction

Failure modes and gates were written before implementation in the three broader-development and negative-only protocols. Preflight checked real balanced batches, feature channels, valid support, finite loss/gradients and checkpoint replay. Fresh CPU processes replayed all four 198-case development milestones per fit exactly. Source reconstruction and native-count checks passed for every published case. Five HTTP publication checks verified 11,336 served assets, including prediction grids and masks. Live browser checks covered model/scene changes, mask/scores/errors, threshold reset and direct-link reload. Initial publication briefly fell back to the cached catalog; after catalog refresh, direct-link reload retained the correct run. No viewer-source fix was made or needed for that transient.

All paths below are relative to the repository. Use the existing .venv Python. Verification commands only, with no refitting or reselection:

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_broader_v7_inference.py --freeze outputs/GTM_Model7_broader_unet_20261024/calibration/freezes/selected/freeze.json --evaluation-manifest outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json --output outputs/GTM_Model7_broader_unet_20261024/selected_dev_inference --verify
.venv/Scripts/python.exe tools/GTM_Model7_negative_v7_inference.py --gate outputs/GTM_Model7_broader_unet_20261024/negative_test_gate.json --manifest outputs/GTM_Model7_broader_unet_20261024/fresh_clear_data/manifest.json --output outputs/GTM_Model7_broader_unet_20261024/fresh_clear_inference --verify
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_broader_unet_20261024/selected_review --output outputs/GTM_Model7_broader_unet_20261024/selected_http_e2e.json
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_broader_unet_20261024/fresh_clear_review_v2 --negative-only --output outputs/GTM_Model7_broader_unet_20261024/fresh_clear_http_e2e_v2.json
```

Training metadata records source hashes, dependencies, inputs, draw schedule and configuration. Preserve frozen trainer sources. The completion receipt pins the manifests, selection, checkpoints via their freezes, source tools, verification receipts and screenshots. Reports and arrays remain under outputs/GTM_Model7_broader_unet_20261024. Earlier report versions are preserved; fresh_clear_review_v2 is current.

## Scientific limits and next step

This pass predicts binary MARS-S2L plume annotations from six-band Sentinel-2 L1C imagery. It does not perform quantitative NASA EMIT methane upscaling. Repeating a 20 m prediction onto a 10 m annotation grid does not recover 10 m detail. EMIT in a site name does not change the source product. A no-plume annotation does not prove zero methane emission.

Retain the selected checkpoint as the improved development candidate, preserve failures and stop final-set tuning. Before another promotion, obtain or identify independent reviewed plume-positive paired sites, including US conditions, and reserve them prospectively. CAFO presence alone cannot replace a dated plume label. No reliable national methane map or source-detection accuracy has been established. No Git commit/push, research-journal edit or external message was made in this pass.
