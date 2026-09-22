# GTM_Model6 — active Sentinel-to-EMIT plume detection study

The product objective is nationwide US Sentinel-only inference supervised by valid EMIT observations: plume extent and ranked candidate source locations. CAFO and oil inventories are corroboration context, not labels. [Design](../../docs/GTM_Model6_multiscale_reconstruction_design.md).

Latest (2026-09-22): corrected mapper R5 produces visible masks but **fails the quality gate**. Pooled native IoU is 2.1485%, precision 2.2487% and recall 32.5182% on one reused 65-scene development fold (three plumes, 62 negative controls). Positive sampling and export-support faults are fixed; all 325 scene/method counts match saved arrays. [Verification and visual review](../../reports/research/GTM_Model6_followup_verification_2026-09-22.md) · [Open R5 mean](http://127.0.0.1:8766/?experiment=GTM_Model6_mapper_mars_pilot_20260922_r5_mean&scene=MARS_2c6ef011-0729-42e2-b64f-10ff8207778b#compare). All four branches and all scenes are available. This is MARS mask supervision, not quantitative EMIT upscaling. The original pairing defect was subsequently repaired for one Georgia crop; the [native readiness audit](../../reports/research/assets/GTM_Model6_20260922/native_pair_readiness.md) admits zero of six cached pairs under its timing/peak policy.

Data repair completed after that audit: a 2.56 km Georgia crop now covers the verified native EMIT peak, with five Sentinel bands and SCL from 31 minutes before EMIT. [Acquisition receipt](../../reports/research/assets/GTM_Model6_20260922/georgia_acquisition.json). The transfer was 9.02 MiB including the header probe; shared charges remain 40.5/100 MiB. Local [QA v2](../../reports/research/assets/GTM_Model6_20260922/georgia_qa.md) verifies 2,166/2,180 native pixels and the common footprint. It flags unresolved Sentinel offset metadata and confirms continuous CH4PLM values are not a binary plume mask. It is one input-data repair, not a quantitative model result or an approved training cohort.

Earlier: two full-scene Sentinel-only plume pilots, neither promoted. Each has 20 independent checkpoints and 173 held-group predictions. Their pooled MARS IoUs are 1.33% and 2.67%; EMIT positive support coverage is 28.52% and 7.28%. These use different evaluation support and cohorts from R5, so their headline scores cannot establish a ranking against R5. [Pilot report](../../reports/research/assets/GTM_Model6_20260922/earlier_unconstrained_report.md) · [Follow-up report](../../reports/research/assets/GTM_Model6_20260922/earlier_evidence_report.md).

Training uses `GTM_Model6_plumes.py --run-id UNIQUE --steps 600 --minutes 20`; `--evidence` selects the constrained follow-up. `GTM_Model6_plume_export.py --run-id UNIQUE` exports it. These are explicit commands, not an automatic campaign; do not repeat sweeps without a new falsifiable data or modeling hypothesis. Real 200×200 context and 16 Sentinel-derived features are used (17 for evidence). No imagery was downloaded. No learned merger, quantitative enhancement head, Gaussian origin head or national inference is validated.

The read-only [research gate](../../reports/research/GTM_Model6_research_gates_2026-09-22_v2.md) records the current capped cohort, exact legacy artifact check, native quantitative EMIT availability, paired-Sentinel data-readiness status and the decision not to repeat training on the same nine EMIT-positive cases. Bounded implementation/testing is authorized; the false gate is scientific data readiness, not user permission.

The next dual-head contract is implemented in [`GTM_Model6_mapper.py`](GTM_Model6_mapper.py): independent exact S16/S32/S64/S128 context branches emit a common 16×16 plume-logit tile and signed-log EMIT enhancement, with documented logit-mean fusion and branchwise supervision. Its [replacement synthetic smoke receipt](../../reports/experiments/GTM_Model6_mapper_synthetic_smoke_2026-09-22_v3.json) is plumbing evidence only; no EMIT quantitative target or nationwide holdout was used. The repaired one-fold MARS diagnostic is `outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r4`; it is retained as a failed engineering output with fixed-threshold IoU 0.0.

The latest real-data path adds [`GTM_Model6_mapper_data.py`](GTM_Model6_mapper_data.py): shared full-scene rotation/mirroring, exact registered context crops and guaranteed positive targets. R5 uses 16 input features, independent branch losses with fused weight zero, and a fixed mean. Output support is central 88×88 (56-pixel margin), not a full-scene prediction. `tools/train_GTM_Model6_mapper_mars_pilot.py` requires the training-only learnability receipt before a new bounded run and refuses to overwrite existing runs. Further sweeps on this failed development fold are not the next research step.

Earlier diagnostic: [native four-branch results and visual review](reviews/GTM_Model6_native_review.md). Twelve checkpoints, three US acquisitions; near-tie with nearest baseline. This is EMIT-only controlled reconstruction.

Earlier legacy run: a single S32 residual U-Net reconstructs 16→32 legacy aggregates. This is not validated native-grid methane super-resolution. The newer plume pilot has a frozen exploratory manifest, not a nationally representative or confirmatory cohort.

[Measured results and visual review](reviews/GTM_Model6_reconstruction_review.md) · [Open actual predictions](http://127.0.0.1:8766/?experiment=GTM_Model6_emit_only_f0#compare).

`GTM_Model6_reconstruction.py --run-id UNIQUE --steps 600 --minutes 25` runs the bounded diagnostic; `GTM_Model6_export.py --run-id UNIQUE` exports its completed outputs. Both use the existing Torch/rasterio environment. Training refuses to overwrite a run. The separate `GTM_Model6.py status` command distinguishes these diagnostic artifacts from the unimplemented full design.

## Files to use

- [Configuration](GTM_Model6_config.json): proposed protocol, not measured results.
- [Source catalog](GTM_Model6_data_sources.json): local candidate data and known limitations.
- [Journal](GTM_Model6_journal.md): current decisions and pass status.
- [Manifest contract](manifests/README.md): information required before training.
- [Visual-review template](reviews/GTM_Model6_visual_review_template.md): required evidence before model promotion.
- [Design](../../docs/GTM_Model6_multiscale_reconstruction_design.md).
- [Archive](../../archive/README.md).

Run `python research/GTM_Model6/GTM_Model6.py status` from the repository root, or invoke that script by absolute path from elsewhere. `validate` checks configuration consistency, not data quality or scientific readiness. Neither command imports historical training code.

## Five passes

1. Verify existing local observations, US eligibility, pairing, nodata, native grids and independent site groups. Produce the first approved manifest and input/reference visual audit.
2. Completed: Martin's 0°/15°/30°/45° transforms with and without mirroring, 16 features derived from five Sentinel bands at target/reference dates, grouped 25 km/acquisition splits, five outer group-held-out folds and a fixed 0.5 cutoff with no inner calibration/model selection. The cohort is 173 records: 36 reviewed PLUME, 128 NO_PLUME and nine EMIT positive-only cases across 16 connected groups. All reviewed positives occur in only two groups.
3. Completed: four independent spatial branches and fixed uniform mean in each of two bounded pilots. The full 200×200 source file context supports all windows.
4. Deferred after failures: learned merger and Gaussian origin model. More independent positive groups and appropriate source/wind labels are required.
5. Held-group metrics and visual outputs are available; national/mosaic validation is not. Neither pilot met promotion criteria. The next evidence is more independent, well-paired US plume observations and a compatible common-scene capstone comparison.

No pass is marked complete merely because code exists. TESSERA is deferred until the Sentinel baseline workflow succeeds.

## Implementation boundaries

New model-specific code belongs here, with names beginning `GTM_Model6_`. Add modules when they implement real behavior; do not create empty model/trainer placeholders that suggest a working experiment. Shared existing utilities may be reused after verifying their input and evaluation contracts. Keep the historical model files unchanged.

Future run artifacts belong in `outputs/GTM_Model6/<run-id>/`: checkpoint, configuration snapshot, source/parent hashes, split manifest identity, training log, native predictions, metrics and visual review. Published viewer bundles belong in `outputs/GTM_viewer_bundles/<run-id>/` and must identify Model6 and its actual stage. A new run cannot overwrite a frozen historical run.

The run manifest must distinguish Sentinel inference inputs, EMIT supervision, any degraded diagnostic input and withheld target; preserve units, native resolution, transform, valid support and uncertainty. The default configuration deliberately does not point to a historical checkpoint.
