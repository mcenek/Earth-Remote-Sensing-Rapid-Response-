# GTM_Model6 — Sentinel-to-EMIT plume detection design

Native Google Doc: [GTM_Model6 — Sentinel-to-EMIT Plume Detection & Source Review Design](https://docs.google.com/document/d/1g7JtVt-BammYKuymoBimqXqjnRcdK1oFsgo-WAWY-H0/edit)

## Current measured status — 2026-09-22

The [follow-up verification report](../reports/research/GTM_Model6_followup_verification_2026-09-22.md) supersedes the earlier mapper status: positive sampling could lose its plume, export metrics used inconsistent support, and the unsupported margin was misdescribed. Those faults are corrected. One bounded R5 development run now yields visible localized predictions but still fails: **2.1485% pooled IoU, 2.2487% precision, 32.5182% recall**, versus 1.697% IoU for its spectral baseline. All three positive scenes and the two worst false-positive controls were visually inspected, and all 325 scene/method comparisons match saved native arrays. The 65-case fold has been reused and is not a confirmation cohort. [Viewer](http://127.0.0.1:8766/?experiment=GTM_Model6_mapper_mars_pilot_20260922_r5_mean&scene=MARS_2c6ef011-0729-42e2-b64f-10ff8207778b#compare).

The [native pairing audit v3](../reports/research/assets/GTM_Model6_20260922/native_pair_readiness.md) finds zero of the original six cached Sentinel pairs satisfying both the six-hour timing policy and peak containment. The 2025 Texas crops contain the peak but are 21–27 hours apart; the original near-time Georgia crop misses it. A subsequent [targeted Georgia repair](../reports/research/assets/GTM_Model6_20260922/georgia_acquisition.json) acquired the five native Sentinel bands plus SCL around the verified peak, 31 minutes before EMIT, using 9.02 MiB total transfer. This repairs one input footprint, not an independent training cohort. Local [QA v2](../reports/research/assets/GTM_Model6_20260922/georgia_qa.md) confirms alignment and 99.36% finite/nodata common support; this is not a full quality mask. Resolve the legacy Sentinel already-applied BOA flag versus the nonzero literal offset, recover and verify the separate provider plume outline (CH4PLM > 0 is not a binary truth mask), and obtain an appropriate temporal reference plus independent observations before quantitative training. R5 itself reused existing MARS mask data; its enhancement head is disabled.

## Earlier full-scene mask pilots — 2026-09-19

Two bounded Sentinel-only pilots have completed: 40 independently trained checkpoints total (four context sizes × five outer folds × two variants). All 173 held-group scene predictions per variant are available in the local viewer, including every negative control. Neither model is promoted. No imagery was downloaded for these runs.

| Four-branch mean | Reviewed MARS pixel precision | Reviewed MARS pixel IoU | Mean negative-scene pixels flagged | EMIT positive support covered |
|---|---:|---:|---:|---:|
| Unconstrained segmentation | 1.38% | 1.33% | 8.90% | 28.52% |
| Spectral-evidence constrained follow-up | 2.89% | 2.67% | 5.06% | 7.28% |

These are native-source-grid scores at the fixed 0.50 cutoff. Negative-pixel coverage is not scene-level false-positive rate or national prevalence. EMIT exterior is unknown, so EMIT precision/IoU are not measured. Individual MARS cases show localized plume shapes, including one with native IoU 54.87%, but surface-pattern activations and misses dominate the aggregate. Review of every EMIT mean prediction shows failure to consistently recover the catalog plume. The follow-up barely changes pooled MARS IoU relative to its analytic spectral baseline (2.66%); it is a post-hoc development comparison, not independent confirmation.

The follow-up supplies a scene-wide robust z-score of decreasing temporal B12/B11 log ratio as an evidence plane and limits the learned logit correction to ±1.5. It reduces broad activation, but leaves texture/edge responses and misses EMIT support. Training stopped after this failure gate. A learned merger, Gaussian source-origin head, quantitative enhancement/upscaling result and nationwide scan have not been validated or delivered. Fine-grid segmentation is not demonstrated methane super-resolution.

Next: audit independently located US positive pairs and representative reviewed negatives, obtain quantitative EMIT pixels where needed, and compare the previous capstone checkpoint on compatible common scenes. Keep 20%-train/80%-held rotations as a separately specified data-efficiency stress test; the completed pilot instead used ordinary five-fold group holdout. More rotations do not add independent sites. Full implementation and nationwide validation remain incomplete.

Evidence: [pilot report](../reports/research/assets/GTM_Model6_20260922/earlier_unconstrained_report.md), [follow-up report](../reports/research/assets/GTM_Model6_20260922/earlier_evidence_report.md), [local viewer](http://127.0.0.1:8766/?experiment=GTM_Model6_plume_01_mean#compare).

The [bounded research gate](../reports/research/GTM_Model6_research_gates_2026-09-22_v2.md) records why the existing nine-EMIT-positive cohort is not being retrained without new quantitative targets and provenance.

## Martin traceability and corrected scope

Martin's recorded objective is a nationwide US **Sentinel-only candidate detector** supervised by valid EMIT methane observations. Known CAFOs are acquisition anchors and a corroboration/recovery endpoint; oil, industrial and unlisted detections remain review candidates. They are not a closed-set label and facility presence is not a date-specific plume. The detailed source-linked reconciliation is [here](../reports/research/SUMMER_RESEARCH_MARTIN_TRACEABILITY_2026-09-22.md).

Keep three tracks separate:

1. **Plume mapping:** Sentinel target/reference inputs predict plume extent and, only with quantitative EMIT support, enhancement. EMIT is a target/evaluation source and never a nationwide inference input.
2. **Facility recovery:** the provisional Iowa DNR inventory is used to select and associate observations. Huang attribution, coordinate datum/UTM zone and accuracy semantics are unverified; the inventory has no date-specific methane labels or reviewed facility controls.
3. **Nationwide screening:** a frozen model is tiled over the US and emits georeferenced candidate maps plus provenance for human/source review. Coverage and candidate workload are reported separately from methane precision/recall.

Martin's one-fold-train/four-fold-validation rotations are a future CAFO/facility experiment. The completed Model6 five-fold run used the ordinary approximately 80% fit/20% holdout direction and is not that experiment. It also used reviewed MARS masks and positive-only EMIT footprint support, not quantitative EMIT enhancement targets. It must remain labeled a failed exploratory mask pilot, not an upscaler or CAFO result.

The supported next model is therefore a multiscale Sentinel-to-EMIT mapper: independent S16/S32/S64/S128 branches, a fixed mean before any learned out-of-fold merger, dense mask loss plus masked quantitative enhancement loss, and a Gaussian source head only after verified source/wind labels. Synthetic Gaussian plumes are pretraining/plumbing evidence only. A fine file grid is a reporting grid, not proof of fine methane information.

The dual-head branch contract is implemented in [`research/GTM_Model6/GTM_Model6_mapper.py`](../research/GTM_Model6/GTM_Model6_mapper.py). Each branch receives its exact native S16/S32/S64/S128 context and returns a common 16×16 centre tile; the fixed merger averages branch logits followed by sigmoid. The loss API supports an optional fused loss, but R5 sets its weight to zero and minimizes separate branch losses to preserve independent training. The [CPU synthetic smoke](../reports/experiments/GTM_Model6_mapper_synthetic_smoke_2026-09-22_v3.json) confirms contract wiring only. A separate four-real-patch memorization diagnostic passed before R5; it is training fit, not generalization evidence.

The earlier R4 one-fold MARS diagnostic remains frozen at `outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r4`; its corrected viewer export is suffixed `r4_checked_mean`. Its fixed 0.50 output really is all-negative (IoU 0.0). R5 fixes positive sampling, lowers the learning rate and disables fused-loss gradients. These simultaneous changes prevent single-cause attribution. Both use exact context and leave a **56-pixel** unsupported margin around the 200×200 file; only the central 88×88 region is scored. The new viewer displays frozen native scores separately from resampled display agreement.

## 1. Objective and claim boundary

The primary objective is a nationwide US candidate detector: use Sentinel-2 imagery at inference time to predict methane plume extent and enhancement-like evidence, then rank candidate methane sources for review. EMIT CH4 products are the supervision source when they are valid, georeferenced and time-aligned. EMIT targets are never supplied as inference inputs for the nationwide detector. CAFO and oil-location inventories provide corroborating context and candidate-site association; they are not plume or emission-origin labels.

The 16/32/64/128 branches are compact, independent spatial-context models. The bounded detection run trains dense plume segmentation heads. An enhancement head is deferred unless trustworthy quantitative methane supervision is available; plume masks alone do not justify one. A candidate heatmap peak is a ranked review location, not an emission origin. A Gaussian source model is allowed only when source and wind supervision exists.

The EMIT-only controlled reconstruction is a diagnostic for alignment, loss behavior, and whether a branch can recover withheld observed structure after deliberate degradation. It is not the product goal and cannot establish nationwide Sentinel detection, fine-resolution methane truth, or source-localization success.

No detection success, national generalization, fine-resolution methane recovery, or emission-origin accuracy is promised before the protocol below produces held-out evidence. Current measurable outcomes are exploratory and do not prove the objective.

## 2. Evidence and data contract

Build a parent-observation manifest before training. It must record site, acquisition and scene IDs; Sentinel and EMIT timestamps and lag; footprints; CRS and affine transforms; native resolutions; band order and units; EMIT enhancement, sensitivity, uncertainty and nodata; valid-observation masks; source-coordinate provenance; and file/code hashes.

The current evidence and newly audited local resources are:

- The legacy audit selected 13 conservative US observations in 11 spatial groups from 102 historical TIFF candidates. They are resampled exports, not recovered native EMIT grids. Several historical joins have multi-day or multi-week lags, and QA/provenance are incomplete.
- A preliminary external cohort contains 70 timely L1C target/reference pairs at 200×200, with 12 US cases. NASA CMR outlines were queried from the official source for screening; they are irregular outlines, not bounding boxes. Areas outside an EMIT polygon remain unlabeled. Nine eligible cases enter the completed exploratory mask pilots; quantitative CH4 rasters are not available for this cohort.
- Three native US acquisitions have quantitative CH4ENH/CH4SENS/CH4UNCERT/CH4PLM sets (`20241020T170504`, `20241130T180310`, `20250922T204933`) and six cached Sentinel L2A files. Native grid equality, bounded geometry support and cached hashes are now recorded in the v3 readiness manifest. The Georgia and Texas 2024 crops miss the native maximum; both 2025 Texas crops contain it but fail the declared six-hour timing policy. Finite/nodata support is not a validated quality mask or independent split. Partial/lagged pairs remain diagnostic, and no complete quantitative supervised cohort is approved.
- `MARS_publication_v3_training_samples.jsonl` contains 1,000 complete local US records: 36 PLUME and 964 NO_PLUME. This supports a bounded mask-supervision cohort, subject to review and grouped splitting; it does not establish detection performance.
- After QA and eligibility filtering, the first pilot has 173 records: 164 MARS (36 PLUME and 128 capped NO_PLUME controls) plus nine EMIT positive-only cases, connected into 16 groups. Two large groups contain all 36 reviewed MARS positives, producing deliberately unequal five-fold group sizes of 82, 67, 9, 8 and 7 records. This imbalance is a limitation, not evidence of balanced validation.
- CAFO and oil inventories identify facilities or wells. They do not establish a dated plume, a negative scene, or the exact upwind source.

Resolve US eligibility from actual footprints, not MGRS tokens or filenames. Require time-matched Sentinel/EMIT pairs for the primary supervised cohort, retain the actual lag, and review plume transience. Missing or invalid methane pixels are unknown, never background negatives. Negative controls require a valid observation and a defensible reference. Do not silently combine incompatible products or resolutions.

Native Sentinel inputs must be preserved at their actual product grid and georeferencing. A 20 m output grid may be a reporting grid, but it is not evidence of 20 m methane information. Upsampling EMIT supervision to a finer grid does not validate fine-resolution methane. An optional EMIT-conditioned refinement experiment must remain a separately named diagnostic and must never be used to claim S2-only inference.

## 3. Splits and eight training transforms

Create grouped splits by physical site, acquisition and overlapping parent footprint before any augmentation. Keep all transformed siblings with their parent. No transformed view is an independent observation.

Use Martin's eight deterministic training views per training parent: rotations of 0°, 15°, 30° and 45°, each with and without a horizontal mirror. This is the declared augmentation protocol, not right-angle D4. Apply the same transform to Sentinel bands, target/reference masks, EMIT polygons, validity masks and source coordinates; transform wind vectors as vectors. Use validity-aware interpolation for continuous fields and nearest-neighbor sampling for categorical masks. Preserve the valid support and reject a view that truncates required target context.

Primary validation and test imagery stay untransformed. The first pilot uses five outer group-held-out folds with the other groups for fitting and a fixed 0.5 cutoff; it performs no inner threshold calibration or model selection. A future inner OOF calibration or merger requires more independent positive groups. A separate robustness analysis may evaluate all eight views, invert them, and summarize per parent. Test-time averaging must be predeclared and compared with the unaugmented prediction. It cannot create extra held-out cases.

## 4. Model and inference contract

Train four independent compact U-Nets, named S16, S32, S64 and S128, with ordinary 3×3 convolutions, skip connections, group normalization and bilinear upsampling. The sizes are actual native context windows, not single filters: at the actual 10 m file grid they span 160 m, 320 m, 640 m and 1.28 km. Each branch consumes an exact S16/S32/S64/S128 crop around the same 16×16 prediction tile; the data pipeline must reject an insufficient crop rather than resize a smaller crop to fabricate context.

Each branch uses 16 features derived from Sentinel target/reference B2, B3, B4, B11 and B12: five scene-centered target log bands, five reference log bands, five temporal log ratios and a differential SWIR contrast. QA/validity excludes invalid losses and outputs; label or validity masks are not model input channels. Real 200×200 context at the 10 m file grid is retained throughout the completed runs. Sentinel SWIR is 20 m and EMIT is nominally 60 m, so file-grid context does not validate methane resolution. The evidence follow-up has one additional analytic Sentinel-derived plane as specified above. The completed mask pilot predicts:

1. a dense plume-probability mask; and
2. optionally, a calibrated candidate heatmap for ranking review locations.

The next dual-head mapper implementation additionally exposes a signed-log quantitative EMIT enhancement field. That head is a contract and synthetic-wiring result only until the manifest contains valid quantitative EMIT enhancement, sensitivity, uncertainty and support. It must not be trained or scored from the positive-only footprint masks used by the completed pilot.

The nationwide forward pass contains Sentinel only. EMIT arrays, EMIT support and EMIT-derived thresholds are labels or evaluation metadata, never input channels. The model must retain native predictions and georeferencing. Use exact-context Sentinel tiles with overlapping 16×16 output tiles, stitch only where every required context exists, and expose edge support and unknown areas. Do not fill the outer unsupported band by padding or interpolation and call it a prediction.

For the completed bounded mask run, use pixel BCE plus Dice on reviewed MARS masks and positive-only BCE on the reviewed EMIT polygon. EMIT pixels outside the irregular polygon are ignored, not negative labels. The next mapper may train its enhancement head only with trustworthy quantitative supervision; the first pilot fixes the output cutoff at 0.5, and any future threshold calibration must be a separately declared inner OOF procedure.

The primary output is a plume extent map plus ranked candidate peaks. A peak is not an emission origin. Fit a Gaussian source head only with verified source coordinates and wind, report uncertainty in map units, and compare against centroid and wind-aware baselines. Without those labels, source output is a review heuristic and source accuracy is unmeasurable.

## 5. Merger policy

First compare every branch and a fixed uniform four-branch mean. A learned merger is optional and train-only: freeze branches, generate out-of-fold branch predictions inside the outer training partition, and fit a small spatial merger there. It must not see predictions from models trained on the same samples, and it must not tune thresholds on held-out data.

The merger must expose branch weights and be compared with the mean, best single branch, and simple baselines. If the independent cohort is too small for honest inner cross-fitting, keep the uniform mean and mark learned fusion blocked. A merger cannot rescue branches that do not show useful spatial behavior.

## 6. Baselines and evaluation

Freeze the split, preprocessing, threshold and metrics before inspecting held-out scores. Include, as applicable, all-negative and constant-crop baselines, nearest/bilinear interpolation, a Sentinel-only simple model, the prior capstone baseline, each single scale, the uniform mean, and any learned merger. Include a shuffled-Sentinel pairing ablation to test whether Sentinel contributes signal.

Evaluate on original native inputs and georeferenced outputs. Report per-site and per-acquisition results as well as pooled values: plume IoU/Dice, precision/recall, area-weighted false positives on observed support, enhancement MAE/RMSE at the observed EMIT grid, candidate recall and localization distance where source truth exists, and calibration/abstention behavior. Stratify by plume size, background, valid coverage and region. Unknown support is excluded from negative scoring and shown explicitly.

Every checkpoint requires a complete review set containing positives, valid negative controls, false positives, false negatives and seeded random cases. The review must include original Sentinel, native EMIT target, prediction, residual, plume probability, validity/unknown support, candidate heatmap, source/wind overlays where available, native-resolution zoom, cross-sections and stitched maps. Use common physical scales for comparable enhancement layers and retain actual float arrays. A colored footprint is observed support, not a learned plume outline.

Reject promotion for whole-box behavior comparable to a constant baseline, systematic seams or edge artifacts, hallucinated high-frequency detail, unsupported registration, invalid background handling, or failure against the relevant simple baseline. Visual review is mandatory evidence alongside frozen metrics, not a selection step after choosing the best-looking scene.

## 7. Bounded implementation passes

### Pass 1 — manifest and visual audit

Inventory local Sentinel and native EMIT files; verify US geometry, pairing, lags, bands, native grids, QA, units and independent groups. Freeze parent splits and produce baseline overlays. Stop when required metadata or valid target support is missing; document the missing evidence.

### Pass 2 — bounded Sentinel mask proof

Implement the five-band Sentinel target/reference input, 16 temporal features, and one S32 Sentinel mask branch. Use all 36 reviewed PLUME records plus a capped 128 NO_PLUME negative-control subset and the nine eligible EMIT cases, grouped by 25 km and acquisition before augmentation. Train with pixel BCE+Dice on reviewed MARS masks and positive-only BCE on EMIT polygons. The first pilot uses five outer group-held-out folds, the other groups for fitting, and a fixed 0.5 cutoff; it has no inner threshold calibration or model selection. This is an engineering proof, not a detection claim.

### Pass 3 — independent spatial context

Train S16, S32, S64 and S128 under the same grouped outer splits and declared compute limits. Export native predictions and stitched maps for each. Stop unsupported branches and do not fill missing context by resampling.

### Pass 4 — uniform merger and candidate ranking

Evaluate the fixed mean before any learned merger. Add train-only OOF fusion only when the cohort supports it. Add candidate heatmaps for review; add Gaussian and wind-aware origin fitting only with source/wind supervision. CAFO/oil overlap is a corroboration analysis, not a target label.

### Pass 5 — frozen nationwide evaluation

Run the frozen Sentinel-only system on held-out US sites/acquisitions and, only when coverage and labels support it, broader contiguous Sentinel mosaics. Report detection and candidate metrics, unknown support, baseline comparisons and visual failures. A result with too few independent sites is an engineering demonstration, not nationwide validation.

## 8. Controlled diagnostics and reproducibility

The EMIT-only controlled reconstruction may use degraded native EMIT as input to recover a withheld native EMIT grid. It must remain labeled diagnostic, separate from the Sentinel-only detector, and cannot validate 60→20 m detail. The optional EMIT-conditioned refinement has the same status.

Save parent split IDs, the 0°/15°/30°/45° transform definitions, preprocessing and degradation parameters, data/code/checkpoint hashes, thresholds, metrics, native prediction arrays, validity masks and visual-review decisions in 'experiment.json' bundles. Name artifacts 'GTM_Model6_S16_*', 'S32_*', 'S64_*', 'S128_*', 'Fusion_*' and 'Source_*'; distinguish exploratory, diagnostic and confirmatory stages.

No new download, bulk national acquisition or training campaign is implied. Reuse local data first and retain the shared 100 MiB CAFO imagery cap. Current native and legacy outcomes remain evidence for pipeline behavior only. Better hardware does not remove a failed data, split, metric or visual gate.

## 9. References

Project evidence: 'reports/research/GTM_Model51_constant_crop_diagnostic.json', 'reports/research/GTM_Model51_spatial_audit.md', 'research/GTM_Model6/reviews/GTM_Model6_reconstruction_review.md', 'research/GTM_Model6/reviews/GTM_Model6_native_review.md', 'research/GTM_Model6/reviews/GTM_Model6_provenance_audit.md', and the completed plume reports linked above. The mask pilots are implemented; later merger, quantitative upscaling and emission-origin components remain proposed.

- [U-Net](https://arxiv.org/abs/1505.04597) — architectural precedent, not methane-performance evidence.
- [NASA EMIT methane enhancement catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_EMIT_L2B_CH4ENH) — product and unit context; native metadata governs each observation.
