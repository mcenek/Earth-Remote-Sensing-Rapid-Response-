# Honest research retrospective

Audit date: 2026-09-18. This is a retrospective, not a new model-selection experiment.

## Direct answer

**Yes: my scientific communication, prioritization and visual quality control were inadequate. No: the evidence does not justify saying every experiment was worthless or that every reported number was fabricated.** There is useful scene-classification signal, some reproducible false-positive filtering, and a few development-only dense improvements. There is not a demonstrated reliable methane upscaler, and v5.1's displayed masks genuinely fail to localize plumes.

The central mistake was letting progress on ranking scenes and rejecting false alarms stand in for the requested product: a credible spatial methane map. I then made the opposite mistake by accepting a blanket restart before fully separating the stronger MARS system from the poor v5.1 and legacy examples in the viewer. Both conclusions were too broad.

## What was checked

- Inventoried all 470 Markdown/JSON experiment artifacts, including nested rehearsals, in the current checkout. These are report files, protocols, seed runs and diagnostics, **not 470 independent experiments**. There are 397 root-level MARS artifacts grouped into 103 filename families.
- Compared the two other research checkouts: 452 and 455 artifacts. Neither contains additional experiment reports missing from the current checkout. Six byte-level differences in the canonical copy disappear after normalizing formatting; there are no conflicting result contents among common reports.
- Reviewed the experiment families and important JSON endpoints, selection protocols, training/evaluation code and viewer provenance. The companion reports provide the longer family inventory.
- Newly replayed all eight existing v5.1 viewer cases from the three original, hash-verified checkpoints. Checked their packed reference/validity masks against the saved prediction cache and inspected the resulting montage.
- Independently recomputed the MARS ensemble's full and test-only-site AP/IoU from hash-verified caches: all eight headline values match the recorded report exactly. Also recovered four native MARS diagnostic examples with reference masks and before/after scene gating. This does not constitute a rerun of the entire model chain.
- Did not rerun every historical training job, independently recompute every confidence interval, or establish the physical correctness of all producer labels. Historical metrics below remain recorded evidence unless explicitly marked as newly recomputed. No new imagery was downloaded.

Inventory: [CSV](GTM_Model0_experiment_inventory.csv), [JSON including cross-checkout comparison](GTM_Model0_experiment_inventory.json).

## What the experiments actually showed

Scores in different rows use different cohorts and endpoints. They are not a cross-dataset leaderboard. A failed superiority gate can mean an inconclusive improvement or an unfavorable tradeoff; it does not always mean an unusable model.

| Family | Legitimate positive finding | Limitation / verdict |
|---|---|---|
| Original capstone model and early upscaling proposals | Existing data collection, model and map scaffolding are useful starting assets. | I have not established that the original team's model is better on a matched evaluation. Attractive maps cannot establish that either. The root architecture proposal's expected PSNR/IoU/detection-limit gains were predictions, not measured project outcomes. |
| Legacy 5-band residual U-Net and physics-feature U-Net | On the grouped >300 ppm·m cohort, raw U-Net sampled pixel AUPRC 0.4168 beats the constant prior's 0.3548. | Raw logistic regression is stronger on this ranking metric: 0.4481. Raw U-Net reported IoU 0.3546 is essentially the prior's 0.3548. Physics channels did not improve the result. Very low thresholds and near-zero specificity explain broad masks. Small ranking signal, no convincing selective mapping. |
| Released MARS-S2L and CH4Net comparators | Released MARS supplies a reproducible methane-specific reference, with strict 579-scene IoU 0.3608 and recall 0.642. | Expanding to 4,401 scenes lowers AP from 0.650 to 0.352 and IoU to 0.1329 as negatives increase. CH4Net transfers poorly here (IoU 0.00754). These are external pretrained baselines, not architectures we invented. |
| MARS v3 | Development AP approximately 0.794–0.828; substantial internal learnability. | Strict campaign mean recall falls to 0.319 versus released MARS 0.642, with lower false positives. Generalization failure/tradeoff, not a clean improvement. |
| MARS v4 / v4.1 / v4.2 / v4.3 | Internal dense learning improves: v4.2 positive-pixel Dice 0.7158 versus v3 0.5770; v4.3 ensemble validation Dice 0.7235. | v4.2 fails AP/recall promotion. v4.3 on the opened strict cohort loses 0.1194 recall and 0.0563 IoU versus its comparator. Real internal promise; no successful strict replacement. |
| MARS spatial classifiers, Prithvi probes and ensemble | Full paper-cohort post-test AP 0.676102 versus 0.641020; full IoU 0.379964 versus 0.324365. Test-only-site IoU 0.292462 versus 0.171562. | Scene AP improvement on test-only sites has CI crossing zero; overall superiority gate fails. Crucially, dense outputs are inherited released-MARS masks with sensor thresholds and a v3 scene gate. This is useful system filtering, not newly learned Prithvi plume geometry or upscaling. |
| Dense Prithvi mask adapter | A distinct development-fold-2 experiment has IoU gain +0.009969, paired interval [+0.002480,+0.016141], with the scene rule unchanged. | A modest genuine dense-development lead worth preserving. It is neither the source of the ensemble's larger headline IoU gain nor independently confirmed across new sites. |
| DOFA / Gaussian / protected ensemble | Gaussian+DOFA improves development AP by +0.002449, interval [+0.000489,+0.004068]. Strict post-test AP 0.374894 versus 0.352123, FPR 0.056991 versus 0.095062. | Recall also falls from 0.641791 to 0.611940; superiority gate fails. It modifies scene scores and makes no localization improvement claim. Gaussian synthetic training here is different from predicting an emission-source Gaussian in the new design. |
| Other MARS rankers, routing, teachers and foundation adaptations | Some small fold-specific improvements; useful negative findings about site shortcuts and brittle calibration. | Most lose gains on another fold, sensor or prevalence regime. DINOv3, LoRA, domain adaptation, large pooled Prithvi features and several teacher variants do not establish a better detector. Repeated small gains on reused folds are not repeated independent confirmations. |
| Selective transformer and v6 pilots | Concrete bounded experiments rather than untested model-name speculation. | Selective proposal AP delta approximately −0.000024; v6 scene pilot +0.000182 with CI crossing zero; v6.1 −0.001376 with negative interval. No supported upgrade. |
| Sensor-aware ordinal replacement | Implemented and ultimately completed, despite earlier infrastructure failures. | Completed held-fold result: AP delta −0.786371, IoU delta −0.636630; all seven gates fail. This configuration failed badly. That does not isolate ordinal supervision as the sole cause. |
| MethaneS2CM v5 / v5.1 | v5 development AP 0.8225; v5.1 ensemble 0.8658. Frozen location-test AP 0.8180 shows useful crop discrimination. | Frozen pixel IoU 0.1852 and newly verified whole-crop outputs fail useful localization. Scene precision 86.3% is not pixel precision, which is 19.84%. L1C-trained zero-shot comparators versus L2A-trained v5.1 do not isolate architecture quality. |
| Calibration / group CRC | Explicit operating-point and uncertainty analysis; CRC reduces group-balanced FPR. | CRC group-balanced recall falls by about 0.1661 and is rejected. Threshold/calibration improvements cannot create missing spatial detail. |
| External negatives, EMIT and controlled releases | CloudSEN scene-head test reduces false positives from four to two on 368 available negative scenes. Independent stress tests reveal weaknesses rather than hiding them. | Negative-only evidence cannot measure recall. Positive-only EMIT recall is approximately 1.1% ERSRR versus 1.8% released MARS, with cross-sensor timing uncertainty. Stanford's one-site test: Prithvi and Gaussian+DOFA detect 0/8 primary positives at frozen rules; released MARS detects 1/8. Poor external sensitivity; no broad reliability claim. |
| CAFO, temporal attention, TESSERA, new multiscale reconstruction | CAFO acquisition/provenance and local viewer infrastructure; implemented temporal CNN/attention with synthetic rehearsal; written reconstruction design. | Three CAFO input crops are not a trained methane detector. Temporal synthetic success is not a real-imagery result. TESSERA and the new reconstruction model have no completed result to assess. |

Detailed sources and additional seeds/ablations: [MARS family audit](GTM_Model0_MARS_retrospective.md), [legacy/dense/external audit](GTM_Model0_dense_external_retrospective.md). V5 sources: [protocol](../experiments/METHANES2CM_V5_PROTOCOL.md), [v5 development](../experiments/METHANES2CM_V5_SEED1101_VALIDATION.md), [v5.1 campaign](../experiments/METHANES2CM_V5_1_CAMPAIGN_PROTOCOL.md), [ensemble](../experiments/METHANES2CM_V5_1_ENSEMBLE_VALIDATION.md), [location test](../experiments/METHANES2CM_V5_1_LOCATION_TEST.md), [post-hoc transport](../experiments/METHANES2CM_V5_1_LOCATION_TEST_POSTHOC.md), [CRC](../experiments/METHANES2CM_V5_1_GROUP_CRC_TRANSPORT.md).

## Are the bad maps a display or reporting error?

### V5.1: the maps are genuinely bad

The earlier full-cache diagnostic covered 20,789 crops. Replacing each crop's pixel probabilities with that crop's mean reproduces 99.8289% of threshold decisions; IoU changes only from 0.1852062 to 0.1849650. This model is mostly deciding which crop to light up, not where the plume is within it.

The new independent checkpoint replay strengthens that conclusion:

- All three checkpoint SHA256 hashes match the frozen ensemble report.
- Reference masks and observation masks match the packed H5 exactly for the eight existing viewer cases.
- CPU float32 replay and historical CUDA/float16 cache produce identical thresholded masks in every case. Maximum probability difference across them is approximately 0.000604, consistent with a small numerical rather than geometric discrepancy.
- Both example scene true positives paint the entire crop despite reference plume coverage of approximately 24.5% and 9.9%. Scene false positives also paint the whole crop.

This checks checkpoint → input loader → dense output → saved cache for those examples. The full-cache diagnostic checks prevalence of the failure. It does not prove why training produced it or validate every upstream dataset label.

[Replay evidence JSON](GTM_Model51_checkpoint_replay_audit.json) · [Replay montage](GTM_Model51_checkpoint_replay_audit.png) · [full-cache diagnostic](GTM_Model51_constant_crop_diagnostic.json).

Code inspection found a plausible process problem: the campaign selected checkpoints lexicographically by scene AP, AUROC and scene recall, with pixel Dice last. V5.1 adds a 65% context / 35% mask-derived scene logit. Scene loss still reaches shared features. Calling this “segmentation-first” did not make dense quality the acceptance criterion. This is a documented mismatch, not a proven causal diagnosis of the collapse.

### Other families: the viewer was not a complete comparison

The viewer displayed the replayable legacy U-Nets and v5.1. Stronger MARS-family systems were summary-only or absent as a distinct best-system entry. Their absence is my failure to recover and present the relevant evidence; it is not proof that their masks look like v5.1. Conversely, a promising table is not a substitute for recovering and checking those outputs.

The earlier asset audit incorrectly said there were no local dense caches. The v5.1 exporter also incorrectly said its checkpoint files were unavailable locally; they exist in the canonical sibling checkout. Both are definite reporting/inventory errors. I corrected the local viewer's checkpoint provenance and retired-mapping status while preserving prediction arrays and scores.

The MARS ensemble protocol's `architecture.dense_mask` and evaluator's `gate_counts` explain a separate presentation trap: a higher system IoU can result from retaining the original masks on some scenes and suppressing them on others. That can be a real and useful gain without sharper boundaries or an upscaling model.

### Recovered MARS evidence and remaining parity caveats

The full and test-only-site headline AP/IoU reproduce exactly from the existing hash-verified caches: [recomputation](GTM_Model0_mars_headline_recomputation.json). This is positive evidence that the table is reporting its cached quantities correctly.

The four newly recovered native examples show a spatial plume-shaped response, a missed positive, a false alarm removed by the scene gate, and background. They are deliberately selected illustrative strata, not an unbiased estimate or new confirmation set. [Actual prediction/reference montage](assets/GTM_Model0_mars_dense_examples.png) · [sample IDs and replay receipt](assets/GTM_Model0_mars_dense_examples.json). Columns are corrected RGB, dataset reference mask, inherited mask before gating, and mask after gating. Gray is unobserved support; negative scenes without separate pixel annotations are identified explicitly. These are released-MARS-derived outputs, not newly learned Prithvi plume geometry.

Unlike v5.1, exact native/cache mask parity is not yet established here: the two nonempty CPU replay masks contain 89 and 38 more predicted pixels than their cached candidate counterparts. Both differences are disclosed in the receipt. Do not call this an exact full-system replay or silently substitute these examples' counts into the historical benchmark. The other two examples match cached counts.

A second audit caveat is independent of that CPU replay: 35 of the 43,529 cached comparison rows have different baseline/candidate reference-positive totals (TP+FN), with 666 fewer positive reference pixels in the candidate totals. The reported IoU is thus not demonstrably evaluated on identical pixel support in every row. The candidate evaluator applies observable-pixel support while the comparator retains reconstructed published counts. A full common-support rescore is needed before treating the dense gain as a strictly controlled pixel-level comparison. This finding neither proves the whole gain is an artifact nor erases the separately recomputed scene AP, but it must be disclosed.

## What I did poorly

1. **Allowed task drift.** Much of the work optimized crop ranking and false-alarm control, while the intended deliverable was a spatial methane reconstruction. I did not keep those deliverables visibly separate enough.
2. **Delayed decisive visual checks.** Eight small checkpoint replays expose v5.1's problem immediately. We should have required that before promoting its scene metrics or launching further architecture variants.
3. **Overstated success and then overstated failure.** “Best” lacked a consistent endpoint/cohort. Later, seeing v5.1 fail did not justify scrapping every independent MARS result without review.
4. **Missed existing assets.** The local-cache and checkpoint-availability claims were wrong. The viewer was not the promised complete best-experiment comparison.
5. **Tolerated diminishing-return complexity.** Numerous tiny development-fold gains were pursued while basic spatial usefulness and external sensitivity remained unresolved. Formal gates were valuable, but did not replace a clear product-level stop condition.
6. **Made unsupported architectural expectations sound too concrete.** The old proposal's projected PSNR, IoU and methane detection limits were not this project's measurements. They should not have informed confidence in deliverable quality.

I do not find evidence in the inspected records and replay that the headline v5.1 numbers were invented. Many source reports explicitly say FAIL, post-test, positive-only or development-only. The major failure was translating those qualified findings into a coherent and accurate account of progress for you.

## What should survive the restart

Preserve the released MARS baseline, its sensor thresholds, the useful scene-gating result, and the small dense-Prithvi adapter as separate baselines or hypotheses. Preserve the data manifests, group splits, audit tools and real local viewer. Retire v5.1 as a plume mapper and the failed ordinal configuration. Do not interpret failed superiority as proof that every component has no value.

Also correct two assumptions before new training:

- Augmentation is not entirely new. The legacy U-Net used quarter-turn augmentation; v5.1 and MARS used quarter-turns and mirroring. Martin's proposed 15–45° transforms differ, but multiplying views is not a cure for missing supervision, geographic shift or scene-uniform outputs.
- The project does not only contain 102 images. It has much larger historical crop benchmarks with different targets and limited independent geography. However, those classification/segmentation crops do not automatically provide finer-resolution methane truth for an upscaler.

My recommendation is a controlled restart of the **reconstruction objective**, not deletion of all prior research: a simple measurable upscaling baseline; train/target separation; tiny-set visual learnability; a fair released-MARS comparison for detection; and only then Martin's independently trained multiscale model. We have not demonstrated accurate methane super-resolution, Gaussian source localization or nationwide methane mapping. Those remain research goals, not achievements.
