# Hard-negative replay: result and visual review

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

October 25, 2026. V11 reaches **71.48% pixel precision**, with **42.19% recall** and **53.06% plume F1**. This is a useful development improvement over retained v7. It does not meet the predefined 70% F1 target or establish performance on new sites. V11 is the new guarded development selection; it is not approved for national inference.

## What changed

We inspected recurring false alarms in the training images. Forty-nine clear training sites had false predictions on at least two observations. These sites accounted for 63.9% of v8's clear training false pixels at update 2,400 and cutoff 0.50.

V11 changes how clear training examples are selected. Each batch still has four plume images and four clear images. It keeps all positive draws, transforms and batch positions from v8. It keeps two original clear draws and replaces the other two with draws from the 49 difficult clear sites. All clear observations at those sites are eligible. Development errors do not define this pool. All 223 clear training sites remain represented.

The model, initial weights, input features, loss, optimizer and 4,800 updates match v8. One fit took 286.8 seconds, including checkpoint scoring. There are 958 training observations: 268 plume and 690 clear. The protocol and failure checks were specified before implementation and fitting.

No predicted masks were enlarged, filled or dilated. No labels define inference crops. No scenes were removed from evaluation.

## Full development result

All 198 observations remain: 46 plume and 152 clear, from 55 sites in 12 countries. They are separate from fitting but have prior project exposure and are used for selection. They are not an independent test.

| Native annotation-grid measurement | Previously retained v7 | V8 | Selected v11 |
| --- | ---: | ---: | ---: |
| Plume F1 | 43.90% | 53.20% | 53.06% |
| Plume IoU | 28.13% | 36.24% | 36.11% |
| Pixel precision | 65.18% | 66.49% | 71.48% |
| Pixel recall | 33.10% | 44.34% | 42.19% |
| Correct plume pixels, TP | 20,126 | 26,960 | 25,655 |
| Extra prediction pixels, FP | 10,750 | 13,588 | 10,237 |
| Missed plume pixels, FN | 40,680 | 33,846 | 35,151 |
| Total marked pixels | 30,876 | 40,548 | 35,892 |
| Clear FP pixels | 5,140 | 5,780 | 4,256 |
| Clear observations activated | 20/152 | 23/152 | 18/152 |
| Entire plume missed | 21/46 | 18/46 | 19/46 |
| Observations recovering half their plume | 11/46 | 17/46 | 14/46 |
| Mean positive-site recall | 29.75% | 37.59% | 32.68% |

Against v7, v11 finds 27.47% more correct plume pixels, with 4.77% fewer false pixels overall and 17.20% fewer clear false pixels. It passes the fixed replacement guards: higher site recall, no lower IoU, no higher clear pixel false-alarm rate, and at most 20 activated clear observations. This is more than an arbitrary increase in marked area. Some individual cases still have oversized predictions.

Against v8, v11 is a conservative tradeoff. F1 is almost unchanged. Precision improves and clear false alarms fall, but recall and site recall fall. The viewer compares v8 and v11 so this tradeoff is visible.

The selected checkpoint is update 2,400 at cutoff 0.65. Only one of the 88 declared checkpoint/cutoff pairs passes the selection guards. The selection rule is stricter than v8's original rule, so selected results compare complete procedures, not the isolated effect of the sampler. At matched update 2,400 / cutoff 0.75, v11 F1 is 50.29%, precision 76.10%, recall 37.56%, with 2,464 clear FP pixels and 15 activated clear observations. Its site recall fails the fixed guard. The unconstrained best F1 is 54.98% at cutoff 0.40, with 10,580 clear FP pixels and 36 activated clear observations. It is rejected.

The predefined useful target remains unmet: F1 at least 70%, recall at least 65%, mean positive-site recall at least 50%, and at least 23 plume observations recovering half their labels. Precision and clear-error limits pass. Of the five smallest labeled plumes, four have zero overlap. This model is not yet a reliable general plume detector.

## Actual visual checks

Root inspected all 46 plume predictions and clear contact pages 6, 10, 11, 14, 15, 16, 17 and 22. These include the largest clear false alarms and new regressions. Browser checks covered all three US positives, whole misses, clear improvements and clear regressions. Teal means correct overlap, pink means extra prediction, and purple means missed plume.

- US `9123854a`: v8 IoU 78.99%; v11 78.91%. Extra pixels fall from 241 to 84, while some correct area is lost. Both trace a substantial part of the actual annotation.
- US `cde9d349`: IoU rises from 83.82% to 86.38%, although extra pixels rise from 26 to 72.
- US `53ce2910`: IoU rises from 49.19% to 52.71%, with extra pixels falling from 529 to 439. The prediction remains too wide.
- China clear `c8569e36`: false pixels fall from 732 to 96. Syria clear `40c50f2a`: 688 to zero.
- Uzbekistan `df4817f5`: v8 has 298 correct pixels; v11 misses the whole plume. Venezuela `02e43c1e`: correct pixels fall from 616 to 72.
- Turkmenistan clear `4be5c331`: false pixels rise from 48 to 428. Kazakhstan clear `69737c23`: zero to 280.
- US clear `eaf034e7`: the largest remaining clear false alarm follows a straight surface feature, with 892 extra pixels. The image alone cannot establish its physical cause or prove zero emissions.

The US-only F1 is 80.98%, including all 60 clear US observations. There are only three positive US observations, from three sites, and these sites have prior exposure. This is descriptive, not national validation. The report includes every country's results and all 198 cases.

## Open the results

Connect to Tailscale. Open the [v8 versus v11 comparison](http://127.0.0.1:8766/bundle-assets/GTM_Model7_hard_v11_20261025_candidate/_versions/3b808c1b44e64324b2fa8a92c721d5cf/index.html). Select a scene. Select **Errors against annotation** to inspect misses and extra predictions. Select **Model score, zero to one** to inspect scores before the cutoff. Scores are not calibrated probabilities.

Open the [map comparison](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_hard_v11_20261025_candidate&scene=MARS_9123854a-64f6-4791-9255-63ad109e0896#compare). The model menu contains both v8 and v11. Each entry restores its own saved cutoff. The report includes actual learning histories at fixed cutoff 0.50, the full-set table and the country table. V11 is the selected research candidate. The GUI home page keeps its existing portable-viewer default.

## Verification and repeat commands

Preflight checked actual training inputs, source hashes, exact initialization, model/preprocessing/loss parity, TRAIN-only hard-pool membership, preserved positive draws, D4 alignment, malformed data, finite gradients, weight changes and CPU checkpoint replay. All four checkpoints replayed all 198 DEV score maps exactly in fresh CPU inference. Selected inference also replayed with zero probability error.

Publication rebuilt all 198 inputs from original source TIFFs, independently checked native confusion counts, and verified 2,774 assets over HTTP. The private HTTPS report matched local bytes with normal certificate verification. Browser checks verified mask, score and error views, scene/model changes, saved cutoffs and direct reload.

Run from the repository root. Use new output directories for replay audits that require a fresh directory.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_hard_v11_e2e.py replay --run outputs/GTM_Model7_hard_negative_v11_20261025/fit --output outputs/GTM_Model7_hard_negative_v11_20261025/replay_recheck
.venv/Scripts/python.exe tools/GTM_Model7_broader_v7_inference.py --freeze outputs/GTM_Model7_hard_negative_v11_20261025/selection/freeze.json --evaluation-manifest outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json --output outputs/GTM_Model7_hard_negative_v11_20261025/inference --verify
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_hard_negative_v11_20261025/review --output outputs/GTM_Model7_hard_negative_v11_20261025/http_recheck.json
```

The fit protocol, replay receipts, selection freeze, predictions, contact sheets, screenshots and completion receipt are under `outputs/GTM_Model7_hard_negative_v11_20261025`. No new imagery downloads, final-test access, journal writes, external messages or Git push occurred.

## Limits and next decision

These are binary MARS-S2L Sentinel-2 L1C labels, not quantitative NASA EMIT methane maps. Repeating 20 m scores on a 10 m annotation grid is not learned 10 m upscaling. Clear labels do not prove zero methane emissions. Known CAFOs are not dated plume labels.

Two bounded fits were completed: deeper v10 failed, while TRAIN-only hard-negative v11 gave a cleaner development selection. Additional fitting and cutoff choices on these same exposed sites cannot establish broader usefulness. No certifiably unused paired positive cached sites remain. Before another accuracy claim, reserve newly obtained paired plume and clear sites, freeze the model and scoring rule, then evaluate once with actual overlay review. Separate small-plume recovery and recurring surface-feature false alarms in that evaluation.
