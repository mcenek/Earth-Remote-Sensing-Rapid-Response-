# GTM_Model6 research journal

## Follow-up verification and failed R5 — 2026-09-22

Found and corrected actual implementation faults: old positive-center clamping produced empty targets in 56/186 requested-positive draws; all 65 r4 viewer exports used a different native support from frozen scoring. Recomputing original NPZs confirmed r4 was really all-negative. The correct unsupported margin is 56 pixels around a 200×200 file, leaving central 88×88 prediction support. Original results are preserved; corrected r4 export is `r4_checked_mean`.

Added shared full-scene transformations, registered exact context crops, balanced positive/reviewed-negative batches and separate branch losses. A four-real-patch training-fit check passed (training IoU 1.0, explicitly not accuracy). One predeclared 400-step R5 follow-up took about 38.1 seconds and used no new imagery. On the reused 65-case MARS development fold, mean pooled IoU/precision/recall are 2.1485%/2.2487%/32.5182%; spectral baseline IoU is 1.697%. Native counts match all 325 exported scene/method comparisons. Visual review of all three plumes and two largest false positives confirms partial overlap but severe surface/facility activations. No promotion or additional training sweep.

The viewer now defaults to R5, retains all 65 cases and four branches, labels training-fit and archived diagnostics, and separates frozen native scores from resampled display scores. [Detailed report and visual evidence](../../reports/research/GTM_Model6_followup_verification_2026-09-22.md).

The corrected native EMIT [v3 readiness manifest](../../reports/research/assets/GTM_Model6_20260922/native_pair_manifest.json) verifies hashes, native grid alignment and exact geometry support. Zero of six cached pairs meets both ±6-hour timing and peak containment. Both 2025 Texas crops do contain the peak, contrary to earlier blanket wording; their 21–27-hour lags fail the declared policy. The near-time Georgia crop misses the peak. A targeted local cache search found no larger matching raster in the inspected locations; the [crop-repair plan](../../reports/research/assets/GTM_Model6_20260922/georgia_crop_plan.json) records the actual peak and desired bounds without downloading data. Quantitative target QA, corrected pairing and independent observations remain required for EMIT supervision.

## Near-time native-pair repair and QA — 2026-09-22

After the no-download model checks above, acquired one 2.56 km Georgia crop from the exact Sentinel scene 31 minutes before EMIT. All five bands and SCL retain native 10/20 m grids and acquisition hashes. A 5.12 km request exceeded the 12 MiB attempt estimate and was not fetched; the smaller crop and metadata/header probe received 9,458,642 bytes (9.02 MiB). The unchanged shared 100 MiB ledger now charges 40.5 MiB, with 59.5 MiB remaining.

[QA v2](../../reports/research/assets/GTM_Model6_20260922/georgia_qa.md) verifies all crop hashes, exact footprint and ENH/SENS/UNCERT alignment: 2,166/2,180 native pixels have finite common support. The actual maximum is 0.341 m from the published peak coordinate. The observation preview shares one UTM extent and keeps continuous measurements separate from predictions.

Root review invalidated the draft QA before use: wrong display alignment, rounded windows, unjustified CH4PLM > 0 masking and a second Sentinel offset followed by clipped RGB. CH4PLM is continuous enhancement, equal to ENH in this crop; 96.4% positive-valued support is not a plume outline. The frozen Sentinel metadata says both BOA offset already applied and offset -0.1; raw DN is preserved and conversion is unapproved. QA v2 uses a labeled raw-DN RGB display stretch. No new quantitative training was started. Next: reconcile radiometry, verify provider GeoJSON outline and native target QA, identify a temporal reference, and build independent positive/negative groups. Additional offline acquisition/QA tests pass; no bulk download or sweep.

## Low-compute comparison preparation — 2026-09-19

Restored the previous team's April web-model definition from commit `ce6844d4410b32ae4137ada9f7d44a271df9c192`; existing `checkpoint_new.weights.h5` matches the original Git blob. Pinned per-tile normalization, mask/regression heads and source hashes. The similarly named older checkpoint has different decoder shapes and must not be substituted. No Keras model load or inference occurred.

Added a paired native-grid comparison harness with explicit reference semantics, source/grid/hash checks, common-support coverage, IoU/area/recall/background metrics and four-panel offline overlays. Ten tiny serial tests passed; three already-evaluated cached cases verified 100% shared reference support. These previews compare the two Model6 means to test the harness, not the previous team model. The prepared legacy manifest contains missing output paths deliberately; comparison fails until real predictions exist.

Active checkout is `ersrr-ordinal-win`; older sibling supplies historical artifacts. About 3 GiB RAM was available and the selected environment lacks TensorFlow, so the future baseline CPU command has explicit resource/runtime gates. No training, GPU inference, downloads, full-dataset evaluation or service launch. [Commands, evidence and remaining gates](../GTM_Baselines/README.md). Local file-URL browser preview was policy blocked; HTML structure and generated PNGs were inspected directly.

## Completed plume pilots — 2026-09-19

The corrected Sentinel-only objective has two actual trained runs: `GTM_Model6_plume_01` and `GTM_Model6_plume_evidence_02`. Each contains 20 checkpoints (four independent 16/32/64/128 context branches across five held-group partitions), all 173 held predictions, source snapshots and hashes. Total training/evaluation time was about 306 seconds; no imagery was downloaded. Eight transforms are applied to training patches on the fly. The 200×200 10 m input file grid retains real context; SWIR is nominally 20 m and EMIT 60 m.

Both fail promotion. Unconstrained mean MARS pixel precision/IoU are 1.38%/1.33%, mean negative-scene pixel activation 8.90%, EMIT positive-support recall 28.52%. The constrained follow-up gives 2.89%/2.67%, 5.06% negative activation, and 7.28% EMIT coverage. The analytic spectral baseline already gives MARS IoU 2.66%, so the learned follow-up has not established useful improvement. All these are development comparisons, not national validation.

Root inspected every EMIT mean prediction in both runs plus a localized MARS overlay in the real viewer. EMIT contours are irregular, but predictions generally miss them, overactivate background, or follow roads/fields/surface changes. The MARS scene `MARS_2c6ef011-0729-42e2-b64f-10ff8207778b` has a recognizable predicted plume and native IoU 54.87%; it is explicitly a selected success, not representative performance. All cases remain available, including every no-plume control. Shared 160×160 web-map display scores are labeled exploratory and differ slightly from native 200×200 evaluation.

Only two connected groups contain the 36 reviewed MARS positives. The unequal held partitions are 82/67/9/8/7 records; this is ordinary group holdout, not Martin's 20%-train/80%-held stress test. No inner tuning, learned merger, quantitative enhancement head, Gaussian emission-origin head or nationwide run was performed. Candidate map points are predicted plume hotspots, not verified origins. EMIT exterior stays unknown in loss and UI; positive-support coverage is not precision/IoU.

Training stopped at the failure gate. Next useful work is independent, closely timed US positives and representative negatives, quantitative EMIT targets for the later upscaling question, and the exact capstone checkpoint on compatible common scenes. More hardware and rotations do not add independent truth. Reports live under the two run directories; the local viewer has both means, all eight individual trained branches, and baseline selectors.

## Reset and objective — 2026-09-19

The objective is a nationwide US Sentinel-2 inference system supervised by valid, georeferenced and time-aligned EMIT methane observations. The system should estimate plume extent and enhancement-like evidence, then rank candidate methane sources for review. CAFO and oil inventories are corroborating context only; they are not plume labels, negative labels or exact emission-origin labels. EMIT targets are never inference inputs for this objective.

The active design is the Sentinel-to-EMIT plume detection document at ../../docs/GTM_Model6_multiscale_reconstruction_design.md. It specifies independent compact S16/S32/S64/S128 spatial-context models with segmentation heads, Martin's eight views (0°/15°/30°/45°, each mirrored and unmirrored), grouped site/acquisition splits before augmentation, frozen held-out thresholds and metrics, native georeferenced evaluation, and mandatory visual review. The first pilot uses a fixed 0.5 cutoff with no inner threshold calibration or model selection. A uniform branch mean is the first merger baseline. A learned train-only OOF merger is conditional on sufficient independent data.

The EMIT-only controlled reconstruction is retained as a diagnostic for alignment and withheld-observation recovery. It is not the product goal, does not validate Sentinel-only detection, and does not establish fine-resolution methane truth. An optional EMIT-conditioned refinement must remain separately labeled diagnostic.

## Evidence boundary

Earlier reconstruction evidence consists of 13 legacy US observations in 11 spatial groups and three native US acquisitions (Georgia and two Texas acquisitions). The legacy observations are resampled exports with incomplete QA and some long acquisition lags. The native study has 12 checkpoints and 15 held windows, but only evaluates controlled EMIT-only reconstruction. These cohorts are too small to support nationwide generalization, a learned merger, or detection success claims. The completed 173-case detector pilot is described above.

The new local audit also found 70 timely L1C target/reference pairs at 200×200, including 12 preliminary US cases screened with official NASA CMR outlines; outside-polygon EMIT remains unlabeled. Six native US L2A files are cached, but the Georgia and Texas 2024 crops miss peaks or truncate plumes. The local MARS v3 manifest has 1,000 complete US records (36 PLUME, 964 NO_PLUME). After QA, the pilot cohort is 173 records: 164 MARS (36 positives and 128 capped controls) plus nine EMIT positive-only cases, connected into 16 groups. Partition sizes are 82, 67, 9, 8 and 7, with two large groups containing all 36 reviewed positives.

CAFO/oil overlap may be reported as corroboration after candidate ranking. Candidate heatmap peaks are review locations, not emission origins. Gaussian source fitting requires verified source and wind supervision.

## Pass status

| Pass | State | Required evidence |
|---|---|---|
| 1. Data contract | Exploratory mask manifest complete; quantitative paired cohort fails readiness | Native v3 audit admits zero cached pairs under strict timing/peak policy; correct Georgia crop and validate target QA/splits |
| 2. Supervised proof | Bounded mask pilots completed, failed quality gate | No quantitative EMIT-trained mapper/upscaler result; four-patch memorization is training-only evidence |
| 3. Four spatial scales | Implemented and trained in two full-scene pilots and bounded mapper diagnostics | Exact-context R5 uses central 88×88 support, all methods visually reviewable; no branch promoted |
| 4. Fusion/candidates | Fixed mean implemented; learned OOF merger and Gaussian source head deferred | R5 mean fails precision/visual gate; candidate peaks are unverified review locations |
| 5. Held-US evaluation | Development-group metrics and visual review completed; national/fresh confirmation not started | Reused groups are not a confirmation cohort; reserve independent sites/acquisitions before national claims |

## Recorded diagnostics

The legacy reconstruction diagnostic trained an S32 residual U-Net on 13 US legacy observations with eight training views and group-held-out partitions. EMIT-only MAE was 121.66 versus bilinear 132.01; Sentinel+EMIT was 135.18. This is an engineering diagnostic on resampled legacy targets and provides no Sentinel detection evidence. The current Sentinel mask-run outcomes are recorded in the completed plume-pilot section above; they are not national validation.

The native reconstruction diagnostic trained independent 16/32/64/128 EMIT-only branches over three US acquisitions. The fixed mean was 264.07 MAE versus nearest 264.25 overall and improved vetted plume-support MAE by 3.35% against nearest. This is a small controlled 120→60 m reconstruction result, not validated 60→20 m methane detail and not a national detector result.

Both reviews require observed support to be distinguished from unknown background. Neither run establishes representative false-positive rates, source origins, national generalization or detection success.

## Martin traceability and dual-head mapper gate — 2026-09-22

The source-linked [summer reconciliation](../../reports/research/SUMMER_RESEARCH_MARTIN_TRACEABILITY_2026-09-22.md) separates the three deliverables Martin described: (1) Sentinel-only plume mapping supervised by valid EMIT observations, (2) CAFO facility recovery using geographic rotations and reviewed controls, and (3) nationwide candidate screening followed by source review. The provisional Iowa CSV is strongly identified as an Iowa DNR export; Huang attribution, coordinate datum/UTM zone and `Accuracy` semantics remain unverified. Facility membership is not a methane label.

The read-only [research gate](../../reports/research/GTM_Model6_research_gates_2026-09-22_v2.md) found 77 records in its capped local inventory (68 MARS and nine US EMIT positive-only records). Those nine EMIT cases are the same cases already used by the completed pilots. Native quantitative CH4ENH/CH4SENS/CH4UNCERT/CH4PLM products are also present for three US acquisitions, but the six cached Sentinel L2A crops are not a complete paired mapper cohort: the Georgia and Texas 2024 crops miss or truncate peaks, and support/timing/grid/split provenance is not frozen. The gate rejects another sweep on the current positive-only cohort until the paired native manifest, quantitative support and CAFO controls are available. This is a scientific data-readiness gate; bounded implementation/testing is authorized. The exact legacy source/checkpoint hashes pass, but runtime loading remains unavailable because TensorFlow is absent and free RAM was below the six-GiB safety gate.

The mapper contract was repaired after review. `GTM_Model6_mapper.py` now requires exact S16/S32/S64/S128 context tensors, returns a common 16×16 centre tile, fuses branch logits by arithmetic mean, and exposes explicit branchwise plus fused losses. Empty support reports zero supported pixels; nonfinite values are rejected only on supported targets. Six contract tests and the [replacement synthetic smoke](../../reports/experiments/GTM_Model6_mapper_synthetic_smoke_2026-09-22_v3.json) pass on CPU (seed 20260922, 24 steps, training-only IoU 1.0). This is plumbing evidence, not real imagery or held-out scientific evidence.

The repaired one-fold MARS mask diagnostic `GTM_Model6_mapper_mars_pilot_20260922_r4` trained with the eight Martin transforms and exact context tiling, then evaluated a held partition with a fixed 0.50 threshold. It produced no positive pixels (held-fold IoU 0.0) and is retained as failed engineering evidence. The outer 64-pixel band is explicitly unsupported because S128 context is unavailable; no threshold tuning or EMIT upscaling claim was made.

## Run-entry requirements

For each real run, record its ID and purpose, parent cohort, code/config/data identities, grouped split manifest, 0°/15°/30°/45° transform definition, actual outputs, frozen thresholds, quantitative baselines, native visual review link, failures and next decision. Preserve actual float predictions, masks, georeferencing and unknown support. Distinguish measured evidence, diagnostic behavior, hypotheses and implementation status. A failed run is retained with its failure reason.
