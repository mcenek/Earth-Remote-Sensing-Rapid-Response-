# One training-only hard-negative replay trial

Written before implementation, fitting and candidate predictions on October 25, 2026.

## Evidence and question

The v10 deeper context model improved fitting-data overlap but failed transfer and every fixed selection guard. Do not promote it. The TRAIN-only loss gradient check found the weighted overlap term already contributes a median 0.409 of the BCE gradient norm, with median cosine 0.866 on the first sixteen saved v8 batches. This does not justify saying overlap loss is missing or starting a loss sweep.

The completed v8 update 2,400 model at fixed cutoff 0.50 marks false-positive pixels on 190/690 clear TRAIN observations and 122/223 clear TRAIN sites. Forty-nine clear TRAIN sites have false positives on at least two observations. They account for 41,768 of 65,364 clear TRAIN false-positive pixels. Root inspected the eight largest TRAIN false alarms against independently rebuilt source TIFFs and released no-plume labels. The saved predictions mark surface regions despite broadly similar current/reference images. RGB alone cannot prove methane absence. Treat the released labels as binary annotation truth and disclose that limit.

Question: can replay of recurring difficult clear TRAIN sites reduce false alarms enough to permit useful plume recovery? This is a sampler hypothesis. It is not evidence of an eventual 70% result or independent transfer.

## Fixed one-fit intervention

- Return to the v8 early-fusion U-Net, widths 16/32/64, two pools, 121,937 parameters. Same 24 channels, scene-centered log preparation, frozen TRAIN-only normalizer, initial seed 20261024 and exact initial tensor parity with v8. Same AdamW learning rate 0.0005, weight decay 0.0001, gradient clip 1, float32, no AMP/TF32, full 100x100 scenes and D4 transforms. Same pooled valid BCE positive weight 2 plus 0.1 pooled positive Dice. No loss or architecture change.
- Manifest SHA 1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a. TRAIN 958 observations: 268 plume and 690 clear, separate from 198 exposed DEV observations: 46 plume and 152 clear. No new imagery, final arrays, CAFO surrogate labels or downloads.
- Derive the hard pool ONLY from the completed, source-bound v8 update2400 TRAIN receipt at cutoff 0.50. A physical site qualifies if at least two distinct clear TRAIN observations have native FP greater than zero. Require exactly 49 eligible sites. Freeze site IDs, member clear TRAIN observations, native counts and all hashes before fitting. At a hard site, sample uniformly from all its clear TRAIN observations, not only its activated observations. Never use a DEV error to form this pool.
- Start with the exact v8 saved draw list. For each eight-observation batch, keep all four positive draws, all transforms, coordinates and class flags unchanged. Keep the first two clear draws in their original batch positions unchanged. Replace only the observation IDs of the last two clear draws, using a separate random.Random(20261025), first choosing uniformly among sorted hard-site IDs, then uniformly among all sorted clear TRAIN observation IDs at that site. Preserve every original batch position and all original D4 transforms. No adaptive mining during the fit.
- One fit from scratch, 4,800 updates, checkpoints at 1,200/2,400/3,600/4,800, thirty-minute wall cap. No extra seeds, loss sweeps or sampling variants. Stop on numerical or integrity failures. A lower threshold or larger predicted area alone is not success.

## Preflight failure modes and real-data verification

Before the fit, reject changed protocol/trainer/dependency/manifest/v8 receipt/checkpoint/schedule hashes, DEV/final members in the hard pool, duplicate or non-clear pool members, site-count drift, changed positive draws or augmentation, changed retained clear slots, malformed support/labels/class flags, nonfinite features/loss/gradients and empty per-scene support. Require all TRAIN clear sites remain represented in the fixed schedule. Compare prepared features and pooled loss with v8 on real four-plume/four-clear batches. Verify initialization exactly against v8 initial weights, one real optimization update, nonzero finite gradients and independent CPU save/load replay. Save all controls and pool/schedule provenance in a source-bound preflight receipt before fitting. No unit tests after implementation; use the real data-to-prediction path and saved E2E artifacts.

After fitting, fresh CPU replay all four checkpoints across all 198 DEV observations with maximum probability error 1e-6, identical support and exact native counts. Replay selected frozen inference independently before publication. The fit may access TRAIN/DEV only. Previously consumed tests stay untouched.

## Fixed success, selection and visual rules

Reuse the v10 prospectively fixed target, with no relaxation: pooled native plume F1 at least 0.70 on ALL 198 exposed DEV observations, precision at least 0.70, recall at least 0.65, mean positive-site recall at least 0.50, at least 23/46 positive observations recovering half their label, clear FP at most 5,140 and at most 20/152 activated clear observations. This is a development usefulness target, not a national accuracy certificate.

Evaluate the same 22 cutoffs at all four milestones. Eligibility requires historical P>=0.50, R>=0.30, IoU>=0.20, clear FPR<=0.001 and at least half positive observations recovering >=10%; plus clear FP<=5140, clear activated<=20 and positive-site recall>=0.297463372680822. Rank eligible candidates by F1, positive-site recall, positive-scene recall, negative clear FPR, negative update, then cutoff. If none qualify, publish update1200/cutoff0.70 as an explicitly failed diagnostic. Save the unconstrained best F1 and failed gates for diagnosis only.

Compare against frozen v7 and v8. Report the matched update2400/cutoff0.75 v8 comparison separately. Retain v7 unless higher site recall, nondecreasing IoU and no greater clear FPR all hold. Save TP/FP/FN, total marked pixels, whole misses, half-recovered scenes, clear activations and unchanged descriptive size/country bins. The US subgroup has only three positive observations; its high v8 F1 cannot replace the all-198 target.

Root must inspect all 46 positive predictions and the largest clear false alarms. Keep all 198 cases available in the paired report and map. Independently rebuild source TIFFs/native counts, verify HTTP masks against saved score grids, check browser scene/model/display switching and existing private Tailscale links. No dilation, flooding, morphology, label-shaped inference crops or excluded difficult cases. No viewer source changes unless needed for a real blocker. No external journal writes/messages, commits or pushes in this pass.

## Limit

These are binary MARS-S2L Sentinel-2 L1C labels, not quantitative EMIT concentration maps. DEV was repeatedly exposed and used for selection. No certifiably unused positive paired cached test sites remain. New prospectively reserved positive and clear sites are needed before any broad transfer claim. Another failed fit must be reported honestly; do not move the target to training scores or an easy subgroup.
