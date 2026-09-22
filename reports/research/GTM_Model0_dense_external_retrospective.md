# GTM Model 0: dense and external experiment retrospective

Audit date: 2026-09-18. Scope: all historical dense, foundation/spatial, and external experiments located in `reports/experiments`, excluding the MethaneS2CM v5 family. No downloads or training were performed. Metrics below are copied from the checked-in JSON/Markdown receipts; code paths were inspected for split, threshold, validity-mask, and metric behavior.

## Executive finding

The historical record supports a modest but real claim: the compact residual U-Net learns some same-cohort pixel ranking above a constant-prior control at the 300 ppm·m target, with strict group-held-out calibration. It does not support reliable high-enhancement segmentation, robust cross-sensor confirmation, or broad external geographic generalization. The foundation/spatial Prithvi work improves the full MARS post-test score in some replays, but the required held-site gate fails; its CloudSEN12 result is a negative-only false-positive safety test. CH4Net is a useful frozen comparator and performs poorly on this stricter cohort. Stanford gives the clearest metered external operating-point stress test, but is one site and has no pixel truth.

This is therefore a differentiated retrospective: the dense baseline is evidence of limited in-domain signal, the external tests expose severe transport and observability limits, and several apparently strong numbers are not valid substitutes for dense segmentation evidence.

## Report inventory and coverage

The experiment directory contains **459 entries: 211 Markdown reports and 245 JSON reports**, plus three temporal-evaluation directories. Name-based coverage is: 399 MARS-family entries, 29 MethaneS2CM v5 entries, 44 Prithvi-related entries, 12 CloudSEN12 entries, 7 Stanford entries, 4 EMIT entries, 2 CH4Net entries, 4 U-Net entries, 16 baseline-named entries, and 12 UNEP entries. The overlap is intentional (for example, Prithvi reports can also be MARS reports). The non-MARS/non-MethaneS2CM file inventory is recorded by the audit as:

- `baseline_results.{json,md}`
- `unet_unweighted_300_results.{json,md}` and `unet_high_enhancement_results.{json,md}`
- `ch4net_released_model_baseline.{json,md}`
- four CloudSEN12 fresh/spatial safety reports (each JSON plus Markdown)
- two EMIT V002 reports (confirmation and post-hoc diagnostic)
- Stanford Evanston, Stanford controlled-release, and their pre-authorization/pre-outcome/synthetic-smoke receipts
- temporal synthetic/rehearsal receipts, which are preparation/smoke artifacts rather than dense methane outcomes

MARS-family Prithvi and released-model reports were also audited because they are the repository's foundation/spatial experiments. The MethaneS2CM v5 reports were not used as evidence here.

## 1. Legacy compact U-Net and classical baselines

Evidence: [`BASELINE_RESULTS.md`](../experiments/BASELINE_RESULTS.md), [`baseline_results.json`](../experiments/baseline_results.json), [`UNET_UNWEIGHTED_300_RESULTS.md`](../experiments/UNET_UNWEIGHTED_300_RESULTS.md), [`unet_unweighted_300_results.json`](../experiments/unet_unweighted_300_results.json), [`UNET_HIGH_ENHANCEMENT_RESULTS.md`](../experiments/UNET_HIGH_ENHANCEMENT_RESULTS.md), [`unet_high_enhancement_results.json`](../experiments/unet_high_enhancement_results.json), and `tools/run_unet_experiment.py`.

The compact U-Net protocol is the cleanest dense experiment in the historical set. `run_unet_experiment.py` filters the six-band rasters by temporal gap and valid coverage, derives a binary EMIT target, creates leakage-safe connected scene groups, uses outer group-held-out folds, disjoint inner group validation for early stopping and threshold calibration, and fits channel normalization on training indices only. The report states 65 scenes and 32 leakage-safe scene components. This is materially better evidence than a random crop split.

At **>300 ppm·m**, the sampled metrics are:

| model | channels | AUPRC | AUROC | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| raw residual U-Net | 5 | 0.4168 | 0.5659 | 0.4988 | 0.3546 |
| physics-feature residual U-Net | 11 | 0.4079 | 0.5556 | 0.4939 | 0.3502 |

The physics augmentation is slightly worse on every listed sampled metric. In the JSON's full-pixel five-fold summary, the physics model is AUPRC 0.4166 ± 0.0732, F1 0.5142 ± 0.0920, IoU 0.3513 ± 0.0853, AUROC 0.5664 ± 0.0151, recall 0.9713, specificity 0.0308; the sampled summary is the 0.4079/0.5556/0.4939/0.3502 row above. The Markdown table is therefore not wrong, but it reports sampled aggregates while the JSON also contains full-pixel and scene-macro aggregates. These should not be compared interchangeably.

The constant prior is an essential negative control: pixel AUPRC 0.3548, AUROC 0.5000, scene F1 0.4988, and IoU 0.3548. Thus raw U-Net AUPRC exceeds prevalence by about 0.062, and AUROC exceeds chance by about 0.066. The gain is real but modest. The prior has recall 1.0 and specificity 0.0; its F1/IoU are prevalence artifacts, not localization skill. Classical raw logistic is stronger on pixel ranking (AUPRC 0.4481, AUROC 0.5995) but its scene-macro AP/AUROC are only 0.3714/0.5045, with mean specificity 0.0720. Physics logistic and histogram gradient boosting do not improve the raw logistic ranking. At >1000 ppm·m, raw U-Net falls to sampled AUPRC 0.0953, AUROC 0.5793, F1 0.1203, IoU 0.1016; physics U-Net falls further to 0.0767/0.5504/0.0868/0.0622.

Interpretation: the raw compact U-Net provides limited evidence of learnable low-threshold pixel ordering, not a useful high-concentration plume segmenter. The very high calibrated recalls in the JSON (often near 1.0) are accompanied by specificity near zero and thresholds as low as 0.01; they are operating-point behavior, not evidence that masks are selective. The separate two-scene legacy prediction replay also shows broad masks on those inspected scenes, so dense IoU must remain the primary spatial caveat; that observation is not generalized to the later v5.1 audit cohort.

## 2. Released MARS-S2L and CH4Net controls

Evidence: [`MARS_RELEASED_MODEL_BASELINE.md`](../experiments/MARS_RELEASED_MODEL_BASELINE.md), [`MARS_RELEASED_MODEL_FULL_STRICT_BASELINE.md`](../experiments/MARS_RELEASED_MODEL_FULL_STRICT_BASELINE.md), and [`CH4NET_RELEASED_MODEL_BASELINE.md`](../experiments/CH4NET_RELEASED_MODEL_BASELINE.md), with inference implementation in `tools/evaluate_released_marss2l.py`.

The released MARS-S2L checkpoint is a fixed-rule comparator on 579 scenes / 150 frozen 25 km groups (67 positives, 512 negatives): scene recall 0.642, specificity 0.922, FPR 0.078, AUROC 0.822, AP 0.650, validity-aware pixel AP 0.4943, IoU 0.3608, Dice 0.5303. On the full 4,401-scene strict cohort (67 positives, 4,334 negatives), the same checkpoint is recall 0.642, specificity 0.905, FPR 0.095, AUROC 0.818, AP 0.352, pixel AP 0.2481, IoU 0.1329, Dice 0.2346. The AP change is driven by the negative prevalence change; these are not contradictory model reruns.

CH4Net, evaluated with the authors' fixed 0.5 / 100-pixel rule and no ERSRR recalibration, gives scene recall 0.164, specificity 0.912, FPR 0.088, AUROC 0.597, unweighted scene AP 0.1578, and validity-aware pixel AP 0.00693, IoU 0.00754, Dice 0.01496. The Markdown headline says “scene AP 0.158,” while the JSON also exposes a scene-representative weighted AP of 0.0222 and weighted precision 0.0281. The unweighted AP is the appropriate headline for the stated 579-scene table; the weighted quantity must not be substituted without saying so.

These checkpoints are useful negative/positive controls because the inference code reconstructs the released U-Net topology, sensor inputs, validity logic, and fixed connected-component rule. They are not official paper aggregate metrics: the reports explicitly say the checkpoint saw the authors' official training split, so ERSRR did not use its internal official-train validation subset for recalibration. Dense masks are also not equivalent to scene detection: MARS has moderate pixel overlap while CH4Net's pixel overlap collapses despite nonzero scene ranking.

## 3. Foundation and spatial Prithvi experiments

Evidence: [`MARS_PRITHVI_SCENE_PROBE.md`](../experiments/MARS_PRITHVI_SCENE_PROBE.md), [`MARS_PRITHVI_SCENE_PROBE_PAPER_POSTTEST.md`](../experiments/MARS_PRITHVI_SCENE_PROBE_PAPER_POSTTEST.md), [`MARS_SPATIAL_PRITHVI_ENSEMBLE.md`](../experiments/MARS_SPATIAL_PRITHVI_ENSEMBLE.md), [`MARS_SPATIAL_PRITHVI_ENSEMBLE_PAPER_POSTTEST.md`](../experiments/MARS_SPATIAL_PRITHVI_ENSEMBLE_PAPER_POSTTEST.md), [`MARS_PRITHVI_SPATIAL_HEAD.md`](../experiments/MARS_PRITHVI_SPATIAL_HEAD.md), [`MARS_PRITHVI_DOMAIN_ADAPTIVE_V2.md`](../experiments/MARS_PRITHVI_DOMAIN_ADAPTIVE_V2.md), [`MARS_PRITHVI_LORA_SCENE_MODEL.md`](../experiments/MARS_PRITHVI_LORA_SCENE_MODEL.md), [`MARS_PRITHVI_100M_CLS_PROBE.md`](../experiments/MARS_PRITHVI_100M_CLS_PROBE.md), [`MARS_DENSE_PRITHVI_MASK_PRESERVATION.md`](../experiments/MARS_DENSE_PRITHVI_MASK_PRESERVATION.md), and [`MARS_DENSE_PRITHVI_TEACHER_PILOT.md`](../experiments/MARS_DENSE_PRITHVI_TEACHER_PILOT.md).

The favorable numbers are real but have a narrow status. A Prithvi scene probe selected on cross-fitted development folds improved held-fold AP by +0.02506 and +0.03765. The exact-paper post-test replay reports full AP 0.67338, AP delta +0.03236, recall delta +0.03089, FPR delta -0.03682, and IoU delta +0.05560. The spatial-Prithvi ensemble reports full AP 0.676102 versus exact v3 0.641020, AP delta +0.035082, recall delta +0.031440, and IoU delta +0.055598. The IoU lift must be attributed correctly: `configs/mars_spatial_prithvi_ensemble_paper_protocol.json` keeps the dense-mask endpoint as the inherited released-MARS mask (`architecture.dense_mask`), while the Prithvi contribution is the scene score/suppression gate (`evaluator.gate_counts`). These IoU deltas are therefore not evidence that Prithvi learned a new dense mask.

However, the required held-site view fails in both reports. Scene-probe test-only-sites AP is 0.46779 with AP-delta CI [-0.01780,+0.05003]; spatial ensemble test-only-sites AP is 0.467027 with AP-delta CI [-0.023301,+0.052032]. The reports explicitly reject the Prithvi complement/ensemble as a final successor. Thus the full-cohort post-test lift is a development/post-test result, not evidence of stable geographic generalization.

The spatial patch head alone is slightly harmful (AP delta -0.00065, recall delta -0.00289, CI [-0.00149,+0.00011]). Domain-adaptive Prithvi v2 is rejected before external scoring: pilot AP delta -0.002024 and matched-FPR recall delta -0.000656, with harm increasing at residual strengths 0.25, 0.50, and 1.00. The LoRA scene model is also rejected (selection AP delta -0.000489; reused holdout -0.000514), as is the 100M CLS probe (-0.000288 AP, -0.000656 recall). Dense mask preservation on fold 2 shows IoU delta +0.009969, while the dense teacher fusion pilot has AP delta -0.000644 and was rejected. These are useful ablations, not independent external confirmations.

The key limitation is task mismatch: most Prithvi evidence is a scene-level ranker or score blend. A scene AP increase cannot establish reliable pixel localization. The dense reports retain IoU from the inherited released-MARS dense-mask path, while the held-site gate fails for the scene contribution; together they do not establish a new Prithvi dense-segmentation capability.

## 4. CloudSEN12 external negatives

Evidence: [`CLOUDSEN12_FRESH_TEST_SCENE_HEAD_EVALUATION.md`](../experiments/CLOUDSEN12_FRESH_TEST_SCENE_HEAD_EVALUATION.md), [`CLOUDSEN12_FRESH_SPATIAL_PRITHVI_EVALUATION.md`](../experiments/CLOUDSEN12_FRESH_SPATIAL_PRITHVI_EVALUATION.md), [`CLOUDSEN12_FRESH_NEGATIVE_CDF_SPATIAL_PRITHVI.md`](../experiments/CLOUDSEN12_FRESH_NEGATIVE_CDF_SPATIAL_PRITHVI.md), and [`CLOUDSEN12_FRESH_TAIL_CALIBRATED_SPATIAL_PRITHVI.md`](../experiments/CLOUDSEN12_FRESH_TAIL_CALIBRATED_SPATIAL_PRITHVI.md), plus their JSON receipts and `tools/evaluate_cloudsen12_fresh_*`.

The available CloudSEN12 cohort contains 368 rows, with 6 unavailable. It is negative-only safety evidence, so it cannot estimate recall, AP, or plume segmentation. The initial spatial-Prithvi test has 4 current and 4 candidate false positives (1.087% each); all safety gates fail. The scene-head candidate reduces false positives from 4 to 2 (1.0870% to 0.5435%); its symmetric adversarial full-cohort FPR bound is 2.1390% versus 2.6738% current, and the predeclared gates pass. Later negative-CDF and tail-calibrated replays reuse the already-inspected fresh cohort; they retain four false positives and are explicitly marked “not an untouched test.”

The CloudSEN12 script correctly checks frozen input hashes, group identity, all-zero labels, and unavailable-row worst-case bounds. The report also says its controlled zero cloud proxy is identical for both heads and is not producer spatial cloud truth. Therefore the result supports a limited external-negative operating-point safety statement, not cloud-mask truth, false-positive performance in all environments, or dense methane localization.

## 5. EMIT V002 external confirmation

Evidence: [`EMIT_V002_EXTERNAL_CONFIRMATION.md`](../experiments/EMIT_V002_EXTERNAL_CONFIRMATION.md), [`EMIT_V002_EXTERNAL_POSTHOC_DIAGNOSTIC.md`](../experiments/EMIT_V002_EXTERNAL_POSTHOC_DIAGNOSTIC.md), their JSON receipts, `tools/evaluate_emit_v002_external.py`, and `tools/analyze_emit_v002_external_posthoc.py`.

This is a sealed positive-only cross-sensor cohort of 55 independent groups. ERSRR five-seed mean recall is 0.011 and mean EMIT-mask IoU 0.001; released MARS-S2L recall is 0.018 and IoU 0.000. Recall delta is -0.007 with 95% CI [-0.055,+0.029]. Because there are no negatives, this cohort cannot estimate FPR, AP, precision, specificity, or AUROC. Because EMIT and Sentinel-2 can be separated by up to six hours, the mask is not simultaneous Sentinel-2 truth.

The post-hoc diagnostic makes the transport failure clearer: median offset 1.266 h, 23/55 at most one hour; external direction-free MBMP mask AUC median 0.550 versus 0.799 for strict MARS positives; robust absolute contrast median 0.167 versus 1.363; 52 scenes are missed by both released MARS and every ERSRR seed. Even beyond 25 km from ERSRR fit locations, released MARS recall is 0.032 and ERSRR seed-mean recall 0.019. These are diagnostic comparisons, not retuned model results. They support a conclusion of weak cross-sensor/time-shift observability and poor confirmation transport, not proof that every external EMIT target is wrong or every model component failed.

## 6. Stanford controlled releases

Evidence: [`STANFORD_LARGE_CONTROLLED_RELEASE_ONE_SHOT.md`](../experiments/STANFORD_LARGE_CONTROLLED_RELEASE_ONE_SHOT.md), [`STANFORD_EVANSTON_ONE_SHOT.md`](../experiments/STANFORD_EVANSTON_ONE_SHOT.md), their JSON receipts, and `tools/evaluate_stanford_*` / `tools/score_stanford_*`.

The larger one-site temporal controlled-release stress test has 169 rows: 86 primary negatives, 8 primary positives, and 75 subthreshold challenges. With unchanged thresholds, released MARS-S2L gives AP 0.095588, recall 0.125, FPR 0.151163. Gaussian DOFA gives AP 0.087243, recall 0, FPR 0.081395; spatial Prithvi gives AP 0.087295, recall 0, FPR 0.011628. The released model has the best AP/recall but a high FPR; Prithvi reduces FPR by rejecting all eight primary positives at this operating point. No pixel masks exist, so IoU is not estimable. This is temporal operating-point evidence at one site, not geographic generalization.

The Evanston one-shot is even more limited: 9 events at one location, 1 primary positive, 8 challenges, and zero primary negatives. Released MARS detects the one positive (recall and precision 1.0, exact 95% interval [0.025,1.0]) and two of eight challenges. Prithvi detects zero of the positive and zero challenges. AP, AUROC, FPR, and specificity are not estimable because there are no negatives. Reporting the 1.0 recall as broad success would be invalid.

## 7. Other legacy and positive-only records

The UNEP post-2024 report is positive-only and explicitly says it cannot estimate AP, FPR, precision, specificity, or AUROC. On 135 auxiliary training-domain positives, released fixed-0.5 gives recall 0.6444 and pixel IoU 0.4440; current Sentinel mask gives recall 0.5778 and IoU 0.3945. On four isolated development groups, both detect 3/4, while IoU is 0.4308 versus 0.3235. This is a positive-mask overlap diagnostic, not an external negative-control result.

The 18-sample MARS contract pilot is also a pipeline smoke test (6 train/6 validation/6 test, three positives and three negatives), with test recall 0.333 for MBMP and 0.667 for pixel logistic and pixel IoU 0.000. Its own report correctly says one mistake changes recall/FPR by 0.333 and that it is not an accuracy estimate. Temporal attention receipts are synthetic/rehearsal artifacts and contain no real external dense outcome.

## Metric and contamination audit

Several reporting hazards recur:

1. **Scene ranking versus dense localization.** Scene AP/AUROC can improve while pixel IoU remains low. MARS full strict AP 0.352 and pixel IoU 0.1329, CH4Net scene AP 0.1578 and pixel IoU 0.0075, and the Prithvi full replay are direct reminders. The dense promise must be judged with validity-aware pixel AP/IoU/Dice and spatial output inspection.
2. **Positive-only or negative-only cohorts.** EMIT and UNEP positives cannot support FPR/AP; CloudSEN negatives cannot support recall/AP; Evanston has no negatives; Stanford has no pixel masks. The reports generally state these boundaries, and the retrospective preserves them.
3. **Prevalence and weighting.** The prior dummy's AUPRC equals positive prevalence by construction. CH4Net's weighted scene AP (0.0222) differs sharply from unweighted AP (0.1578); the headline uses unweighted AP. MARS AP changes substantially between 579-scene and 4,401-scene cohorts because negative prevalence changes.
4. **Sampled versus full metrics.** U-Net Markdown reports sampled-pixel metrics, while JSON also contains full-pixel and scene-macro values. They are not interchangeable. Thresholded F1/IoU can be inflated by calibrated near-all-positive masks when specificity is near zero.
5. **Post-test reuse.** CloudSEN tail/CDF reports explicitly reuse the opened fresh cohort. Exact MARS paper Prithvi replays are labeled post-test, and their test-only-sites gate fails. They are useful diagnostics and hypothesis generation, not untouched confirmation.
6. **Training-domain exposure.** Released MARS/CH4Net checkpoints were trained on the authors' official split; ERSRR therefore forbids internal validation recalibration. The compact U-Net uses strict train-only normalization and group-held-out calibration. EMIT post-hoc proximity audits show 24/55 external scenes within 25 km of ERSRR fit locations, so “external” does not automatically mean geographically novel.

## Honest product-level conclusion

The historical dense experiments provide **limited research evidence of low-threshold same-cohort pixel ranking under a frozen group-held-out protocol**, with the raw compact U-Net as a transparent baseline and classical raw logistic as a strong low-capacity comparator. They do not justify claiming reliable high-enhancement plume segmentation, calibrated plume probabilities, or broad external recall.

Foundation/spatial Prithvi is best described as a **scene-score complement with an encouraging full-cohort post-test lift but failed held-site confirmation**, while CloudSEN12 provides **negative-only safety evidence**. EMIT V002 and Stanford are valuable stress tests: EMIT exposes cross-sensor/time-offset failure, and Stanford exposes the one-site precision/recall tradeoff under metered releases. CH4Net is a reproducible released-model control with very weak dense overlap on the strict cohort. The appropriate next dense claim requires untouched mixed positive/negative geographic groups with simultaneous or defensible time alignment, explicit mask truth, prevalence-preserving AP/FPR, and a predeclared scene-to-pixel operating rule.

## Evidence limitations

This audit is report- and code-based. It did not rerun inference, inspect every raster visually, independently recompute every JSON metric, or validate the external source labels. The experiment inventory is filename-based and may miss a legacy artifact whose name does not contain the audited family tokens. The foundation-model discussion includes MARS-family Prithvi reports because they are part of the requested spatial/foundation history; MethaneS2CM v5 was intentionally excluded. Checkpoint files and some feature caches are not all tracked, so claims are limited to the receipts, hashes, and source paths preserved in the repository.
