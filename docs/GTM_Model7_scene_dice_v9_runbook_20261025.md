# U-Net follow-up: equal overlap loss for each plume scene

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

## Decision

Keep v7 as the selected model. Save v9 as a development comparison. This change does not solve the small-plume problem, and it fails the written replacement rule.

We completed one local training run. No new imagery was downloaded. No final-test arrays were opened. The research journal was not changed. Nothing was committed or pushed in this pass.

## What we tested

The previous loss combined the plume pixels from all positive images in a batch. Large plumes could therefore have more influence on its overlap term. The training audit found that the largest quarter of positive images held 54.85% of positive pixels. Small training plumes were also missed more often. This suggested a hypothesis; it did not prove a cause.

For v9, each positive image contributes equally to the overlap term. Clear images still contribute to the pixel classification loss. We kept the same U-Net, initial weights, exact image draw order, augmentation, optimizer, data and 4,800-update budget as v8. The fit took 271 seconds, including its evaluations.

Training used 958 real observations: 268 plume and 690 no-plume observations. Each batch had four of each. Evaluation used the same 198 development observations: 46 plume and 152 no-plume observations, from 55 sites in 12 countries. These observations were excluded from gradient fitting, but they have project history and are used for selection. They are not an independent test.

The protocol was written before implementation and fitting: [v9 protocol](GTM_Model7_scene_dice_v9_protocol_20261025.md).

## Results on the same development set

| Measure | Selected v7 | Previous trial, v8 | New trial, v9 |
|---|---:|---:|---:|
| Plume overlap, IoU | 28.13% | 36.24% | 33.65% |
| Plume-pixel precision | 65.18% | 66.49% | 72.18% |
| Plume-pixel recall | 33.10% | 44.34% | 38.67% |
| Mean positive-site recall | 29.75% | 37.59% | 32.20% |
| Recover at least 10% of the plume label | 23/46 | 24/46 | 26/46 |
| Entire plume missed, zero correct pixels | 21/46 | 18/46 | 15/46 |
| Clear observations with any positive pixel | 20/152 | 23/152 | 24/152 |
| False-positive pixels on clear observations | 5,140 | 5,780 | 5,184 |
| Clear-pixel false-positive rate | 0.08489% | 0.09546% | 0.08562% |

Checkpoints and cutoffs were selected under the same written rules: v7 update 1,200 / 0.70, v8 update 2,400 / 0.75, and v9 update 2,400 / 0.80. These are comparisons of the full selection procedures. The paired viewer compares v8 with v9; v7 remains available as its own run.

v9 finds some additional plume fragments and reduces extra area relative to v8. It also loses correct plume area and has an extra activated clear observation. It passes the ordinary development limits but fails replacement: its clear false-positive count is 44 pixels higher than v7. We did not relax the rule after seeing the result. This narrow difference alone does not establish a meaningful population difference.

At the same update 2,400 and cutoff 0.75 as v8, v9 has IoU 35.71%, recall 42.40%, mean positive-site recall 35.73%, and clear false-positive rate 0.10650%. It activates 30/152 clear observations. Thus, the selected v9 reduction in false-positive area cannot be credited to the loss alone; the higher selected cutoff also matters.

## Small-plume check

For the 25 smallest positive training images, at update 2,400 and cutoff 0.50, zero-overlap misses fall from 15 to 12. Mean recall rises from 28.57% to 32.01%.

This does not carry through to the smallest development plumes. Of the five with fewer than 100 positive 20 m pixels, none reaches 10% recovery at the selected cutoff. Four remain complete misses. At the matched 0.75 cutoff, one reaches 10%, with worse clear-scene false alarms overall. The local training benefit is not evidence of reliable transfer.

The full area breakdown is saved in `outputs/GTM_Model7_scene_dice_v9_20261025/size_breakdown.json`. It is descriptive and did not select the checkpoint or cutoff.

## Visual findings

Root inspected all 46 positive observations in paired contact sheets, plus the largest clear-scene false alarms and representative clear observations. Live browser checks covered improvements, regressions, complete misses, clear false alarms, and mask, score and error displays.

- Iran, 2024-07-07, `642bb155`: overlap improves from 2.82% to 21.77%. Most of the plume is still missed.
- US, 2024-10-23, `9123854a`: overlap falls from 78.99% to 74.24%. Extra pixels fall from 241 to 24, but the predicted plume is smaller.
- Venezuela, 2024-04-28, `02e43c1e`: overlap falls from 27.54% to 0.54%. This is a substantial regression.
- Kazakhstan, 2024-08-22, `54ad1a4f`: both trials miss the whole plume.
- Egypt clear scene, 2024-09-27, `36acacd4`: extra pixels rise from 220 to 584.
- US clear scene, 2023-11-06, `eaf034e7`: a false blob remains, with 824 extra pixels in v8 and 728 in v9.

The failures are present in the saved predictions. They are not missing overlays.

## View the results

1. Connect to Tailscale.
2. Open the [paired v8 and v9 comparison](http://127.0.0.1:8766/bundle-assets/GTM_Model7_scene_v9_20261025_candidate/_versions/806833955e664fb8870468efb3aecda1/index.html).
3. Select a scene. Use **Errors against annotation** to see correct overlap, extra prediction and missed plume pixels. Use **Model score, zero to one** to see the full score field. Scores are not calibrated probabilities.
4. Use **Open candidate in viewer** for the map, or open the [v9 map view](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_scene_v9_20261025_candidate&scene=MARS_9123854a-64f6-4791-9255-63ad109e0896#compare).
5. Use **Full research archive** and **Refresh now** if the new runs are not listed. v9 is a diagnostic trial, not the default selected model.

The paired report contains all images and is about 59 MB. Its header retains the renderer's October 24 template date; this experiment ran on October 25. The dated folder, protocol and receipts identify the run.

## Verification and repeatable artifacts

The real-data preflight checked the new loss against an independent calculation, invalid-pixel handling, clear-image contribution, augmentation alignment, gradients, initialization and source identities. The final preflight is under `preflight/final`. An earlier preflight from before the trainer filename correction was preserved and was not used to authorize the fit.

All four checkpoints reproduced all 198 development score maps in fresh CPU inference with zero probability error, identical support and exact native counts. The selected frozen inference also reproduced exactly. Export independently rebuilt source images and native counts. The HTTP check verified 2,774 published assets, including the mask and score-grid agreement. The remote HTTPS report matched local bytes with certificate verification enabled.

The local viewer process was no longer running when HTTP verification first ran. It was started again on loopback port 8766 with hidden-window logs in this run's folder. No Tailscale access settings were changed. The subsequent HTTP and remote checks passed.

From the repository root, repeat the saved-inference and publication checks:

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_broader_v7_inference.py --freeze outputs/GTM_Model7_scene_dice_v9_20261025/selection/freeze.json --evaluation-manifest outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json --output outputs/GTM_Model7_scene_dice_v9_20261025/inference --verify
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_scene_dice_v9_20261025/review --output outputs/GTM_Model7_scene_dice_v9_20261025/http_recheck.json
```

Receipts, source hashes, checkpoint files, frozen scores, native counts, contact sheets and browser screenshots remain in `outputs/GTM_Model7_scene_dice_v9_20261025`. No unit tests were added.

## What this establishes

Equal scene overlap loss is not sufficient to fix missed plumes. It is not a replacement for v7 under the present rule. The next study should investigate why small real plumes lack usable detection signal and obtain prospectively reserved positive and clear sites before claiming improved transfer. More tuning on this development set cannot create an independent result. Prior clear-only test observations remain consumed and must not be used to tune.

These labels are binary MARS-S2L Sentinel-2 L1C annotations, not quantitative NASA EMIT methane maps. Repeating 20 m scores on a 10 m annotation grid does not establish learned 10 m detail. No-plume labels do not prove zero emission. A CAFO location is not a dated observed plume. This work does not establish nationwide detection, source identification or methane upscaling.
