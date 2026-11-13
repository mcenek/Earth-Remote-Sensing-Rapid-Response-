# Deeper context trial: result and visual review

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

October 25, 2026. The useful 70% plume-F1 target was not met. Retain v7. The new deeper model is a failed diagnostic, not a replacement.

## What was tested

One 4,800-update fit used a residual U-Net with three pooling stages and 1,163,185 parameters. V8 has two stages and 121,937 parameters. Inputs, preprocessing, loss, optimizer, initial seed, training observations and exact image/augmentation draws stayed fixed. The fit took 431.7 seconds, including checkpoint scoring. The protocol was written before implementation and training.

There are 958 TRAIN observations, including 268 plume and 690 clear images. All 198 exposed DEV observations remain in the comparison: 46 plume and 152 clear. DEV is separate from fitting but has prior project exposure and is used for selection. It is not an independent test.

## Result

| Native annotation-grid measurement | Retained v7 | V8 | V10 failed diagnostic |
| --- | ---: | ---: | ---: |
| Plume F1 | 43.90% | 53.20% | 44.91% |
| Plume IoU | 28.13% | 36.24% | 28.96% |
| Precision | 65.18% | 66.49% | 58.02% |
| Recall | 33.10% | 44.34% | 36.63% |
| Correct plume pixels, TP | 20,126 | 26,960 | 22,276 |
| Extra prediction pixels, FP | 10,750 | 13,588 | 16,116 |
| Missed plume pixels, FN | 40,680 | 33,846 | 38,530 |
| Total marked pixels, TP+FP | 30,876 | 40,548 | 38,392 |
| Clear FP pixels | 5,140 | 5,780 | 9,072 |
| Clear observations activated | 20/152 | 23/152 | 39/152 |
| Entire plume missed | 21/46 | 18/46 | 16/46 |
| Observations recovering half their plume | 11/46 | 17/46 | 15/46 |

V10 had zero eligible rows among the fixed four checkpoints and 22 cutoffs. The displayed update 1,200 / cutoff 0.70 is the predeclared failed fallback. It was not chosen as the best score. Even the unconstrained best v10 F1 is only 47.56%, at update 3,600 / cutoff 0.70, with 20,888 clear FP pixels. No threshold choice in the declared grid reaches the target.

At the matched v8 update 2,400 / cutoff 0.75, v10 F1 is 39.83%, with 20,628 clear FP pixels. The architecture change did not improve this comparison.

At fixed cutoff 0.50, v10 TRAIN IoU grows to 51.19%, while DEV IoU is 28.22%. Higher fitting performance did not transfer. The TRAIN-only gradient diagnostic also found the weighted Dice contribution is substantial: median gradient norm ratio 0.409 versus BCE, with median cosine 0.866 over the first sixteen saved v8 batches. These findings do not establish a capacity-only cause or justify a loss-weight sweep.

## What the actual predictions show

Root inspected all 46 plume cases, the eight largest clear TRAIN errors, and representative clear DEV scenes including the largest false alarms. Browser checks covered all three US positives, a complete missed plume, and clear failures in Egypt, Libya and the US. Error images use teal for correct overlap, pink for extra prediction and purple for missed plume.

- US `9123854a`: v8 IoU 78.99%; v10 62.98%. V10 removes extra pixels but misses substantially more of the plume.
- US `cde9d349`: v8 IoU 83.82%; v10 71.48%. Both locate the plume; v8 traces more of its extent.
- US `53ce2910`: v8 IoU 49.19%, with 529 extra pixels. Its shape is too wide. This weaker case stays visible.
- Iran `642bb155`: v8 has a small correct fragment; v10 misses the whole plume.
- Egypt clear `36acacd4`: false pixels increase from 220 to 1,304 near the image edge.
- Libya clear `47f609f7`: v8 has no false pixels; v10 adds an 896-pixel blob.
- US clear `321f13e1`: a false blob remains beside a lake, 816 pixels in v8 and 832 in v10. The image alone does not prove its cause.

The US-only v8 F1 is 79.85%, including all 60 clear US observations, not just the three plume images. Counts are TP 8,704, FP 2,540 and FN 1,852; five clear observations activate. The three positives come from only three sites. This is a promising descriptive subgroup, not confirmation of national accuracy and not a substitute for the failed all-198 target. The report includes every country's counts and all cases to prevent selective reporting.

## Open and check

Connect to Tailscale and open the [v8 versus v10 paired report](http://127.0.0.1:8766/bundle-assets/GTM_Model7_context_v10_20261025_candidate/_versions/6798f687af6140e796427182812680db/index.html). Select a scene and choose **Errors against annotation**. Use **Model score, zero to one** to inspect scores before the fixed cutoff. Scores are not calibrated probabilities. The report includes actual learning histories at cutoff 0.50 and the full-set F1 table.

Open the [map](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model7_context_v10_20261025_candidate&scene=MARS_9123854a-64f6-4791-9255-63ad109e0896#compare). Choose the baseline v8 entry to compare it with the failed v10 entry. The complete research archive has both; the retained research selection remains v7. The normal home page keeps its existing portable-viewer default.

## Repeatable verification

Real-data preflight checked all 958 TRAIN inputs, frozen feature/loss parity, D4 alignment, malformed support/labels/flags, stage shapes, finite nonzero gradients, changed weights and exact CPU checkpoint replay. All four checkpoints reproduced all 198 DEV score maps exactly in fresh CPU inference. Selected frozen inference also replayed exactly. Publication independently rebuilt the original source TIFF inputs and native counts, then verified 2,774 assets over HTTP, including static mask versus score-grid agreement. The private HTTPS report matched local bytes with normal certificate verification.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_broader_v7_inference.py --freeze outputs/GTM_Model7_context_v10_20261025/selection/freeze.json --evaluation-manifest outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json --output outputs/GTM_Model7_context_v10_20261025/inference --verify
.venv/Scripts/python.exe tools/GTM_Model7_prediction_publication_e2e.py --review outputs/GTM_Model7_context_v10_20261025/review --output outputs/GTM_Model7_context_v10_20261025/http_recheck.json
```

Artifacts are under `outputs/GTM_Model7_context_v10_20261025`. No mask enlargement, filling, dilation, new imagery download, final-test access, external message, journal write or Git push occurred.

## Bounded follow-up

Recurring false alarms also occur in TRAIN: 49 clear sites with at least two activated observations account for 63.9% of v8's clear TRAIN false pixels. The separate v11 protocol fixes one sampler-only replay experiment using these TRAIN sites. It keeps four positive examples per batch, two original uniform clear draws, and two draws from difficult clear sites. No DEV errors define the pool. Success limits stay unchanged.

The data are binary MARS-S2L Sentinel-2 L1C annotations, not quantitative NASA EMIT concentration maps. Repeating 20 m scores on a 10 m label grid is not learned 10 m upscaling. Released no-plume labels do not establish zero methane emissions. No unused paired positive cached test sites remain; new positive and clear sites must be reserved before acquisition to verify broader transfer.
