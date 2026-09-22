# GTM Model 0 / MARS experiment retrospective

## Scope and verdict

This is a read-only forensic audit of every root-level file in `reports/experiments` whose basename starts with `mars` or `MARS`: **397 artifacts, 212 JSON result records and 185 Markdown reports, covering 103 named experiment families**. JSON and Markdown pairs are treated as one experiment where both exist; JSON-only artifacts are included as well. The three `temporal_evaluation_rehearsal_*` directories were not included because they are not `mars*`/`MARS*` report paths. The inventory is exhaustive; metric discussion below is an evidence-weighted reading of the reports and scorer code, not a claim that every JSON field was independently recomputed. No training, downloads, or report-input changes were performed.

The honest conclusion is narrow. The MARS-S2L task is a scene-level plume-presence classification problem with a segmentation auxiliary output. Several experiments demonstrate better plume-mask learning or better ranking on internal development data, and a few improvements are statistically positive on a subset of development folds. Those are useful engineering findings. They do not establish a better deployed detector. On the frozen 579-scene / 150-group ERSRR cohort, the released MARS-S2L checkpoint is the strongest simple reference: scene recall 0.642, specificity 0.922, FPR 0.078, AUROC 0.822, AP 0.650, pixel AP 0.4943, IoU 0.3608, Dice 0.5303 (`MARS_RELEASED_MODEL_BASELINE.md`). ERSRR v3 averages recall 0.319 at FPR 0.037 and loses 0.322 recall; v4.3 reduces FPR but loses recall and pixel overlap (`MARS_V3_STRICT_CAMPAIGN.md`, `MARS_V4_3_STRICT_COMPARISON.md`).

The exact paper replay is a different cohort and must not be mixed with the strict ERSRR cohort. The reconstructed paper comparator is full-view AP 0.64102, recall 0.79151 at FPR 0.07069, pixel IoU 0.32437; test-only-site AP 0.45027, recall 0.77533, FPR 0.07551, IoU 0.17156 (`MARS_SUCCESSOR_PAPER_TEST.md`). Successor variants improve some full-view numbers but all reported successor post-tests fail at least one test-only-site or paired-gate requirement. Therefore no report supports the claim that a successor is a confirmed paper-level improvement on unseen sites.

## Evaluation contract and what is being measured

The MARS reports use two related targets:

* **Scene classification/ranking:** one score per crop/scene, evaluated with AP, AUROC, recall at a frozen FPR target, FPR, specificity, precision, and group/site bootstrap intervals. This is the operational detection claim.
* **Pixel segmentation:** a mask evaluated with pixel AP, IoU, and Dice. A good mask can coexist with poor scene ranking because the scene score is a separate aggregation/routing rule. Many pilot reports improve dense-mask IoU while AP and recall are unchanged or worse.

The v3 scorer freezes checkpoint, proposal thresholds, operating threshold, and predictions before strict scoring (`tools/evaluate_mars_v3.py`, especially its fixed-threshold checks and group bootstrap). The v3 campaign resamples frozen 25-km groups and five seeds (`tools/aggregate_mars_v3_strict.py`). The v4.3 scorer computes AP/AUROC from scores, binary operating metrics at frozen thresholds, pixel metrics from the observable mask, and paired bootstrap over the 150 groups (`tools/evaluate_mars_v4_3_strict.py`). The exact paper audit explicitly prohibits test tuning and requires full and test-only views with paired-site AP/IoU support (`tools/audit_mars_paper_benchmark.py`).

These are appropriate safeguards, but later reports contain many **development-only**, **fold-confirmation**, **label-free post-test**, and **candidate-specific post-test** results. A number in a post-test replay is evidence about that frozen candidate, not an independent confirmation. A validation AP or synthetic-bank AP is not a strict scene-classification result.

## Family-by-family findings

### Baselines and frozen comparators

`MARS_PILOT_BASELINES.md` is a smoke/pilot reference, not a scientific comparison. The predeclared spatial baseline ladder is more useful: scene raw-logistic and physics HGB obtain strict-test recall 0.075 and 0.104 respectively, while pixel logistic obtains recall 0.000, pixel AP 0.0091, IoU 0.0000 (`MARS_DEV_SCENE_BASELINES.md`, `MARS_DEV_PIXEL_BASELINES.md`). The class-enriched development tranche has 67 positives and 512 negatives; its calibration is not representative of deployment prevalence.

The released model is the valid frozen baseline for same-cohort comparisons. It uses the authors' fixed 0.5/100-pixel rule, with no ERSRR threshold tuning. The reports correctly warn that this is not the paper's official aggregate test metric. The `ch4net_released_model_baseline` and non-MARS baseline files are contextual comparators and are not evidence about MARS successor superiority.

### v3: strong internal validation, weak strict generalization

The v3 validation reports (seeds 101, 202, 404, 505 plus the aggregate) show validation AP roughly 0.794–0.828, AUROC 0.923–0.940, recall about 0.820–0.848 at approximately 5% FPR, and validation Dice 0.549–0.592 (`MARS_V3_*_VALIDATION.md`). Those numbers are legitimate internal-development results, with thresholds chosen on validation groups.

The strict per-seed results collapse: seed 202 AP 0.195, recall 0.299, FPR 0.030, IoU 0.0966; seed 303 AP 0.112, recall 0.269, FPR 0.024, IoU 0.0794; seed 404 AP 0.197, recall 0.388, FPR 0.044, IoU 0.0653; seed 505 AP 0.096, recall 0.284, FPR 0.035, IoU 0.1062 (`MARS_V3_SEED*_STRICT_EVALUATION.md`). The campaign mean recall/FPR is 0.319/0.037 versus released 0.642/0.095; the lower FPR is real, but it is purchased with a large recall loss (`MARS_V3_STRICT_CAMPAIGN.md`). Post-hoc strata by wind, plume area, and target/reference interval are diagnostics only (`MARS_V3_STRICT_POSTHOC_DIAGNOSTIC.md`).

### v4 and v4.1–v4.3: mask gains do not become detector gains

The score/physics v4 nested development run is negative: nested AP 0.183, AUROC 0.699, recall 0.507 at 5% FPR, versus released-development AP 0.352/AUROC 0.818/recall 0.642 (`MARS_V4_NESTED_DEVELOPMENT.md`). It was correctly rejected before confirmatory use.

v4.1 improves internal mask learning (epoch 20 validation AP 0.787, AUROC 0.952, recall 0.875 at an 8% FPR operating point, positive-pixel Dice 0.704), but misses the preregistered v3 AP/recall gates (`MARS_V4_1_SEED606_EPOCH20_VALIDATION.md`). v4.2's three-seed mean is AP 0.7953, AUROC 0.9492, recall@5% FPR 0.8125, Dice 0.7158; it passes AUROC/Dice but fails AP and recall against the v3 internal reference (`MARS_V4_2_VALIDATION_CAMPAIGN.md`). The nested morphology selector reports AP 0.862/AUROC 0.964/recall 0.828 at observed FPR 0.049, but was rejected under the predeclared gate (`MARS_V4_2_NESTED_SCORING.md`).

The v4.3 ensemble passes its internal validation and group-held development gates (all-validation AP 0.8126, AUROC 0.9536, recall 0.8337, FPR 0.0500, Dice 0.7235; `MARS_V4_3_ENSEMBLE_VALIDATION.md`). On the already-opened strict cohort it has lower recall by 0.1194, lower IoU by 0.0563, and lower Dice by 0.0924, even though AP and AUROC are higher and FPR is lower (`MARS_V4_3_STRICT_COMPARISON.md`). This is a classic classification-versus-segmentation and operating-point tradeoff, not a clean win.

### Scene heads, OOF rankers, hard-example and site routing families

Many scene-head experiments report small positive development deltas, generally around +0.001 to +0.013 AP, while preserving recall through a protected gate. Examples include crossfold bagging (+0.00820 AP), trust-region bagging (+0.00152), hard-positive bagging (+0.00152), XGBoost head (+0.00245), context inner-fold (+0.01297), and hard-scene inner-fold (+0.00650). Their own reports reject them before paper scoring or show failure on another fold (`MARS_CROSSFOLD_BAGGED_SCENE_HEAD.md`, `MARS_CROSSFOLD_BAGGED_TRUST_REGION.md`, `MARS_CROSSFOLD_HARD_POSITIVE_BAGGING.md`, `MARS_XGBOOST_SCENE_HEAD.md`, `MARS_CONTEXT_SCENE_RANKER_INNER_FOLD2.md`, `MARS_HARD_SCENE_RANKER_INNER_FOLD2.md`).

OOF scene-ensemble and context reports are useful for protocol development, but their fold-0/folds-2–4 estimates are not independent paper tests. The scene stacker, OOF context minimum blend, OOF ranker, counterfactual ranker, causal residual, dense AP residual, and invariant-site families repeatedly fail either low-prevalence, sensor-stratified, per-fold, or paired-site gates. Representative failures: dense AP residual AP delta -0.015748 and matched-FPR recall delta -0.007528; causal residual becomes harmful at strength 0.10; context fold 0 is rejected despite an inner-fold gain (`MARS_DENSE_AP_RESIDUAL_RANKER.md`, `MARS_CAUSAL_RESIDUAL_RANKER_FOLDS34.md`, `MARS_CONTEXT_SCENE_RANKER_FOLD0.md`).

Site/geospatial priors are especially vulnerable to shortcut learning. The geospatial prior is -0.00005 AP; temporal/site risk priors have point gains but unstable or non-positive confidence bounds; unseen-low-prevalence routing fails its lower-bound gate (`MARS_GEOSPATIAL_SITE_PRIOR.md`, `MARS_SITE_RISK_PRIOR.md`, `MARS_TEMPORAL_SITE_PRIOR.md`, `MARS_UNSEEN_LOW_PREVALENCE_ROUTER.md`). These should not be described as sensor-general detection improvements.

### Prithvi, DOFA, DINOv3 and protected fusion

Spatial-Prithvi, Prithvi scene probes, physical probes, LoRA/domain-adaptive variants, dense Prithvi teachers, and mask-preservation experiments separate representation/mask quality from scene ranking. The clearest legitimate positive is mask preservation: +0.009969 IoU on development fold 2 with positive paired-site interval [+0.002480,+0.016141], while this was explicitly not a scene-model promotion (`MARS_DENSE_PRITHVI_MASK_PRESERVATION.md`). Teacher fusion and encoder probes generally have zero or negative AP deltas despite positive mask deltas.

There is a material provenance trap in the exact spatial-Prithvi ensemble replay. `configs/mars_spatial_prithvi_ensemble_paper_protocol.json` declares its `dense_mask` as the **released MARS-S2L probability mask with frozen sensor thresholds, gated by the unchanged v3 stronger scene score at cutoff 0.75**. The evaluator (`tools/evaluate_mars_spatial_prithvi_ensemble_paper_cache.py`) applies the gate to inherited `candidate_pixels`/`candidate_scores`; it does not obtain a new Prithvi-drawn spatial mask. Thus its IoU increase versus the reconstructed paper comparator is a real system-level segmentation/filtering improvement under the stated operating rule, but not evidence that Prithvi learned new plume geometry. The same caveat applies to any replay whose protocol says “unchanged v3 stronger scene score” or “released MARS-S2L probability mask.”

DOFA-v2 protected fusion passes folds 3/4 (+0.001208 AP, recall unchanged, paired AP lower bound +0.000362) and train-fitted normalization passes the same development gates (+0.001412 AP, lower bound +0.000526), but the fixed fold-2 confirmation is only +0.000203 AP with CI [-0.000343,+0.000933] and is rejected (`MARS_DOFA_V2_PROTECTED_FUSION_FOLDS34.md`, `MARS_DOFA_V2_TRAIN_FITTED_NORMALIZATION_FOLDS34.md`, `MARS_DOFA_V2_FOLD2.md`). DOFA anchored protected ensemble likewise passes folds 3/4 (+0.002230 AP) but fails fold 2 (+0.000313, CI [-0.000541,+0.001131]).

The Gaussian+DOFA protected ensemble is a development passer (+0.002449 AP, recall unchanged) and is authorized only for separate external/new-cohort confirmation (`MARS_DOFA_GAUSSIAN_PROTECTED_ENSEMBLE.md`). Its candidate-specific strict post-test is explicitly not a fresh holdout (`MARS_GAUSSIAN_DOFA_STRICT_SPATIAL_ONE_SHOT.md`). DINOv3 fusion pilots show AP deltas from -0.001832 to +0.000739 and are all rejected before external scoring (`MARS_DINOV3_*`).

### Gaussian synthetic and simulation studies

The Gaussian plume bank is a real audit of synthetic geometry, not a detector benchmark. It contains 20,000 disjoint indexed templates and passes the declared distribution/alignment gates (`MARS_GAUSSIAN_PLUME_BANK_AUDIT.md`). The Gaussian contrast dense confirmation shows validation dense-evidence AP 0.8092 and pixel IoU 0.2711 after transfer, but the learnability/exposure audits show that later variants fail the synthetic disjoint-validation gate (`MARS_GAUSSIAN_CONTRAST_VIT_DENSE_CONFIRMATION.md`, `MARS_GAUSSIAN_CONTRAST_VIT_DENSE_EXPOSURE_AUDIT.md`, `MARS_GAUSSIAN_CONTRAST_VIT_LEARNABILITY_AUDIT.md`).

This evidence supports debugging the architecture and loss path. It cannot support real-scene classification claims because synthetic dense-evidence AP is not scene AP on MARS. The scene-aligned cache replay also failed bitwise/protocol reproduction: strength reproduced within 1e-9, pooled and sensor AP did not (`MARS_GAUSSIAN_SCENE_ALIGNED_CACHE_REPLAY_FAILURE.md`). The bounded stochastic replicate passed its own reproducibility checks, but real-development cross-fit still failed promotion (+0.002507 AP with paired CI crossing zero; `MARS_GAUSSIAN_SCENE_ALIGNED_CROSSFIT.md`, `MARS_GAUSSIAN_SCENE_ALIGNED_REPRODUCIBILITY.md`).

### Joint, MIL, temporal, NDMI, physics and instance-guided pilots

Joint presence/segmentation development did not clear its promotion gate (`MARS_JOINT_DEVELOPMENT.md`). Joint MIL reports 97 abstain, 224 no-plume, and 63 plume validation decisions and correctly reserve the strict audit for error-stratum planning (`MARS_JOINT_MIL_DEVELOPMENT.md`, `MARS_JOINT_MIL_VALIDATION_AUDIT.md`). Calibration-aware finalization passed its internal gates, but the exact paper post-test still has an unresolved gate (`MARS_JOINT_CALIBRATION_AWARE_PAPER_POSTTEST.md`).

Temporal-site prior, temporal suppression, gated temporal-spatial boost, temporal-spatial ensemble, NDMI bitemporal fusion, and large-history prior all fail at least one stability or confirmation gate. The gated temporal-spatial confirmation is particularly clear: AP/recall deltas -0.02486/-0.02098 on confirmation and -0.01132/-0.00872 whole-view (`MARS_GATED_TEMPORAL_SPATIAL_BOOST.md`). Physics-guided teachers, instance-guided teachers, physical patch transfer, and dual-teacher rescue improve occasional mask or dense metrics but fail scene gates or have confidence intervals crossing zero (`MARS_PHYSICS_GUIDED_TEACHER*.md`, `MARS_INSTANCE_GUIDED_TEACHER_PILOT.md`, `MARS_DUAL_TEACHER_RESCUE_FOLDS34.md`).

### External data: CloudSEN12 and UNEP

CloudSEN12 negative augmentation has essentially no inner gain (+0.00008 AP), while spatially augmented negatives show a small folds-2/3/4 gain (+0.00229 AP; lower CI +0.00023) and fold-1 confirmation (+0.00295; lower CI +0.00079). Both branches are nevertheless rejected before sealed-negative or paper replay (`MARS_CLOUDSEN12_NEGATIVE_AUGMENTED_XGBOOST.md`, `MARS_CLOUDSEN12_SPATIAL_AUGMENTED_XGBOOST.md`). UNEP positive augmentation is frozen for a paper-cache evaluation, but the exact post-test fails a paper gate (`MARS_UNEP_POSITIVE_AUGMENTED_XGBOOST.md`, `MARS_UNEP_POSITIVE_AUGMENTED_PAPER_POSTTEST.md`). External-data heads therefore provide hypotheses, not confirmed generalization.

### Sensor ordinal, selective proposals, offshore and deployment diagnostics

The completed sensor-aware ordinal result is decisively negative: pooled AP delta **-0.786371** (bootstrap lower bound -0.835886), matched-FPR recall delta **-0.776772**, dense IoU delta **-0.636630** (lower bound -0.682844), with all seven promotion gates false (`MARS_SENSOR_ORDINAL_FOLDS34.md`). Fold endpoints were epoch 11 and 23; the report correctly limits enhancement values to ordinal ordering and makes no physical-unit claim. A separate held attempt failed infrastructure before a held result (`MARS_SENSOR_ORDINAL_HELD_ATTEMPT_FAILURE.md`), but that earlier failure must not replace the completed negative scientific result.

The selective proposal-verifier transformer is also negative: AP delta **-0.000024**, paired 25-km-group interval [-0.000055,-0.000006], with 42 raised rows (0 positives, 42 negatives), despite unchanged matched-FPR recall (`MARS_SELECTIVE_PROPOSAL_TRANSFORMER_FOLDS34.md`).

The v6 product-aware scene pilot is near-zero and inconclusive-to-negative (+0.000182 AP, recall unchanged, paired interval [-0.000107,+0.000427]); the v6.1 error-correcting residual is harmful (-0.001376 AP, interval [-0.002920,-0.000085]) (`MARS_V6_SCENE_PILOT.md`, `MARS_V6_ERROR_CORRECTING_PILOT.md`). Sensor mask-threshold confirmation can pass an isolated folds-0/1 development confirmation at threshold 0.70, while the extended audit fails another fold or sensor gate (`MARS_MASK_THRESHOLD_FOLDS01_CONFIRMATION.md`, `MARS_MASK_THRESHOLD_FOLDS01_EXTENDED.md`). Offshore fine-tuning and mask-threshold diagnostics do not establish a new independent external benchmark.

### Exact paper successor post-tests

These are the most tempting files to overclaim because they show full-view gains. The full/test-only values are:

| Candidate | Full AP | Full recall | Test-only AP | Test-only recall | Decision |
|---|---:|---:|---:|---:|---|
| Frozen successor v1 | 0.64693 | 0.81688 | 0.46710 | 0.80176 | failed at least one gate |
| Sensor-mask v2 | 0.64693 | 0.81688 | 0.46710 | 0.80176 | failed at least one gate |
| Scene-ensemble v3 | 0.67380 | 0.82350 | 0.46550 | 0.79736 | failed at least one gate |
| XGBoost head | 0.67599 | +0.03530 recall delta | 0.46400 | +0.02203 recall delta | test-only gate failed |
| Adaptive Prithvi | 0.67700 | +0.03420 delta | 0.46550 | +0.02643 delta | test-only gate failed |
| All-development refit | 0.67295 | +0.03089 delta | 0.46006 | +0.01762 delta | test-only gate failed |

Sources: `MARS_SUCCESSOR_PAPER_TEST*.md`, `MARS_XGBOOST_SCENE_HEAD_PAPER_POSTTEST.md`, `MARS_ADAPTIVE_PRITHVI_PAPER_POSTTEST.md`, and `MARS_ALL_DEVELOPMENT_SCENE_REFIT_PAPER_POSTTEST.md`. Full-view gains are therefore plausible candidate-specific post-test findings, but they are not independent confirmation; the low-prevalence unseen-site view and paired confidence gates remain decisive.

Pixel-support caveat for the exact replays: the evaluator compares stored `baseline_pixels` with candidate counts after `gate_counts` (`tools/evaluate_mars_spatial_prithvi_ensemble_paper_cache.py` and `tools/evaluate_mars_scene_gated_masks_paper_cache.py`). A cache QA comparison found 35 rows where baseline TP+FN differs from candidate TP+FN, with 666 fewer candidate reference-positive pixels over the full 43,529 rows. The candidate is therefore not demonstrably evaluated on identical positive-pixel support in every row; the reported IoU is the frozen count-based comparison produced by the protocol, not a strictly common-support pixel IoU recomputation. This flags an interpretation limitation in the IoU gain; it does not by itself explain the gain or invalidate the separately hash-verified AP/IoU headline.

## Reporting mistakes and misleading interpretations to avoid

1. **Calling validation AP a test result.** v3/v4 validation AP and synthetic-bank AP are development metrics. They cannot be compared directly with strict or paper-test metrics.
2. **Calling a mask improvement a detection improvement.** Positive IoU/Dice deltas in Prithvi, DINOv3, Gaussian, and teacher pilots often accompany zero/negative scene AP or recall. Report both heads separately.
3. **Mixing cohorts.** The 579-scene/150-group ERSRR strict cohort, the development folds, the CloudSEN12/UNEP external data, and the exact MARS-S2L paper benchmark answer different questions.
4. **Treating opened strict results as a new test.** v3 informed later v4 research; v4.3 is explicitly a development benchmark on an already-opened cohort.
5. **Using post-test selection language.** “Full-view AP improved” in a candidate-specific paper replay does not mean the candidate was selected without exposure to the outcome. The reports themselves correctly label several as post-test diagnostics; that label must remain in any downstream summary.
6. **Ignoring low-prevalence and site-group uncertainty.** Point gains of one or two AP thousandths routinely have paired-site intervals crossing zero. Group bootstrap, sensor strata, and unseen-site gates are more informative than pooled point estimates.
7. **Reporting FPR reduction as a free win.** v3/v4.3 reduce FPR while losing recall and/or IoU. The operating point must state the recall/FPR tradeoff and the frozen threshold source.
8. **Calling an infrastructure failure a model rejection.** The sensor-ordinal held attempt is explicitly an infrastructure failure before a held result; it supplies no scientific metric.

## Strongest legitimate positive evidence

The strongest positive evidence is methodological and bounded: (a) frozen released-model reproduction provides a solid comparator; (b) v4.1–v4.3 reliably improve internal positive-pixel Dice and AUROC; (c) the dense Prithvi mask-preservation result has a positive paired IoU interval on a development fold; (d) several protected scene heads show small, predeclared development-fold AP gains; and (e) the synthetic Gaussian bank audit passes geometry/distribution checks, proving the synthetic generator is at least internally controlled. None of these establishes superior real-world scene detection on unseen sites.

## Final assessment

MARS experiments support a useful research direction—simulation- and representation-assisted segmentation with explicit hard-negative/site-aware evaluation—but they do not support a claim that GTM Model 0 or its successors outperform released MARS-S2L on a clean unseen-site benchmark. The most defensible product statement is: **segmentation representations and internal ranking can improve, while scene-level generalization remains unresolved; the released checkpoint remains the strongest frozen same-cohort operational baseline, and every apparent paper-level successor gain is bounded by failed test-only or paired-gate evidence.**

## Coverage inventory

Coverage was generated from the exact filename rule `reports/experiments/*` with basename regex `^(mars|MARS)` and included every matching Markdown and JSON artifact. Family counts below count Markdown reports; each family may have a paired JSON and some JSON-only companions:

`adaptive_prithvi(2), all_development(1), anchored_full(1), causal_residual(1), cloudsen12_negative(1), cloudsen12_spatial(1), context_scene(2), counterfactual_scene(1), crossfold_bagged(2), crossfold_hard(1), dense_ap(1), dense_prithvi(2), dev_pixel(1), dev_scene(1), dinov3_methane(4), dinov3_multiseed(1), dofa_anchored(2), dofa_gaussian(1), dofa_v2(5), dual_teacher(1), encoder_scene(1), gated_temporal(1), gaussian_contrast(5), gaussian_dofa(1), gaussian_plume(2), gaussian_scene(3), gaussian_vit(2), geospatial_site(1), group_crc(1), hard_scene(1), instance_guided(1), instance_scene(1), joint_calibration(2), joint_development(1), joint_external(2), joint_mil(2), large_history(1), low_prevalence(1), mask_routing(1), mask_threshold(3), ndmi_bitemporal(1), ndmi_patch(1), offshore_finetune(1), offshore_mask(1), oof_context(4), oof_scene(4), paper_model(1), paper_released(1), paper_residual(6), paper_v3(1), physical_patch(1), physics_guided(2), pilot_baselines(1), prior_reference(1), prithvi_100m(1), prithvi_domain(2), prithvi_lora(1), prithvi_physical(1), prithvi_scene(2), prithvi_spatial(1), protected_bisensor(4), recall_anchor(1), released_model(2), residual_endpoint(1), rex_spatial(1), robust_prithvi(1), robust_site(1), scene_domain(1), scene_gated(2), scene_ranker(2), scene_stacker(2), selective_proposal(1), sensor_mask(1), sensor_ordinal(1), simulation_augmented(1), site_relative(2), site_risk(1), source_aligned(2), spatial_hard(2), spatial_prithvi(2), spatial_scene(1), spatial_successor(1), successor_fold1(1), successor_paper(3), target_mixture(1), target_weighted(1), target_xgboost(1), temporal_site(2), temporal_spatial(1), unep_positive(2), unseen_low(1), v3_proposal(1), v3_seed101(3), v3_seed202(3), v3_seed303(1), v3_seed404(3), v3_seed505(3), v3_smoke(1), v3_strict(2), v3_validation(1), v4_1(4), v4_2(7), v4_3(3), v4_nested(1), v4_seed606(1), v4_smoke(1), v5_transfer(1), v6_error(1), v6_scene(1), xgboost_scene(2)`.
