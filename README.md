# ERSRR — Sentinel-to-EMIT methane plume detection restart

**Active research: GTM_Model6. The objective is nationwide US Sentinel-only plume extent and candidate-source detection supervised by EMIT; current reconstruction runs are diagnostics and do not prove detection success.**

Latest (2026-09-22): **the corrected mapper produces visible predictions but still fails the quality gate.** Fixed positive-patch sampling and inconsistent viewer scoring support; ran one bounded R5 development follow-up with four independent context branches. Its pooled IoU is 2.1485%, precision 2.2487% and recall 32.5182% on 65 reused MARS cases. All three plume cases and the two worst false-positive controls were visually inspected. [Verification report and actual outputs](reports/research/GTM_Model6_followup_verification_2026-09-22.md) · [Open R5 mean](http://127.0.0.1:8766/?experiment=GTM_Model6_mapper_mars_pilot_20260922_r5_mean&scene=MARS_2c6ef011-0729-42e2-b64f-10ff8207778b#compare). R5 reused existing imagery; a subsequent targeted data repair transferred 9.02 MiB under the existing cap. This is mask supervision only; quantitative EMIT upscaling and source localization remain unproven.

Data progress after the training run: [one corrected Georgia crop](reports/research/assets/GTM_Model6_20260922/georgia_acquisition.json) now covers the native EMIT peak with near-time Sentinel imagery. This used 9.02 MiB including the header probe and kept cumulative charges at 40.5/100 MiB. Quantitative target QA, a temporal reference and independent observations remain necessary before an EMIT-trained mapper/upscaler run.

Earlier full-scene pilots have four independent context models across five group-held-out folds, with all 173 cases exported per run. Their pooled MARS IoUs are 1.33% and 2.67%; EMIT support coverage is 28.52% and 7.28%. Both failed. Their cohort and scored support differ from R5, so the numbers do not establish a ranking against R5. [Pilot report](reports/research/assets/GTM_Model6_20260922/earlier_unconstrained_report.md) · [Follow-up report](reports/research/assets/GTM_Model6_20260922/earlier_evidence_report.md).

Earlier EMIT-only diagnostic: four context models and their mean across three native US acquisitions nearly tied nearest-neighbor overall. [Native diagnostic review](research/GTM_Model6/reviews/GTM_Model6_native_review.md). This remains controlled 120→60 m reconstruction, not Sentinel detection or validated 20 m detail.

The first run used 13 US observations in 11 spatial groups. EMIT-only reconstruction reduced mean held-out MAE from 132.01 to 121.66 (7.8%); adding Sentinel worsened MAE to 135.18. These are deliberately degraded legacy aggregates, not native EMIT or independent 20 m truth. See the [run and visual review](research/GTM_Model6/reviews/GTM_Model6_reconstruction_review.md) and [actual predictions](http://127.0.0.1:8766/?experiment=GTM_Model6_emit_only_f0#compare).

The goal is nationwide US Sentinel-2 plume extent and candidate methane-source ranking, supervised by valid EMIT observations. The design uses Martin's 0°/15°/30°/45° rotations with and without mirroring, independent 16/32/64/128 context branches, segmentation heads, and mandatory native georeferenced review. EMIT is never an inference input. CAFO/oil locations corroborate candidates; Gaussian source localization requires source and wind supervision.

## Start here

New software preparation: [previous-team baseline and paired comparison commands](research/GTM_Baselines/README.md). The exact April web-model weights are recovered and hash pinned; the comparison harness is verified on tiny fixtures and three existing cached scenes. Actual legacy inference is deferred for RAM and TensorFlow runtime availability, so this is not a legacy-vs-Model6 performance result.

1. [Active workspace and next steps](research/GTM_Model6/README.md)
2. [Architecture design](docs/GTM_Model6_multiscale_reconstruction_design.md) · [Google Doc](https://docs.google.com/document/d/1g7JtVt-BammYKuymoBimqXqjnRcdK1oFsgo-WAWY-H0/edit)
3. [Current research journal](research/GTM_Model6/GTM_Model6_journal.md)
4. [Honest retrospective](reports/research/GTM_Model0_honest_research_retrospective.md)
5. [Historical archive index](archive/README.md)

The current [Martin traceability report](reports/research/SUMMER_RESEARCH_MARTIN_TRACEABILITY_2026-09-22.md) and [bounded research gate](reports/research/GTM_Model6_research_gates_2026-09-22_v2.md) explain the corrected objective, blockers and why the same nine EMIT-positive cases are not being retrained.

## Active commands

From this checkout, with a working Python 3.11+ interpreter:

```powershell
python research/GTM_Model6/GTM_Model6.py status
python research/GTM_Model6/GTM_Model6.py validate
.\GTM_Model0_OpenViewer.cmd
```

The Model6 commands use only the Python standard library. They read the active configuration and local source locations; they do not download imagery, load old checkpoints, or train. The viewer starts on localhost:8766. It distinguishes the untrained Model6 study from archived experiments.

The first implementation pass is a verified US observation manifest: pairing times, native EMIT resolution, band order, nodata, independent site groups and real spatial context. Existing imagery remains in its original locations and is referenced through the [source catalog](research/GTM_Model6/GTM_Model6_data_sources.json). Available files are not automatically approved training examples.

## Repository layout

| Location | Role |
|---|---|
| `research/GTM_Model6/` | Only active model research workspace: configuration, status, journal, manifests and visual-review protocol |
| `GTM_Viewer/` and `GTM_Model0_OpenViewer.*` | Shared local experiment viewer |
| `docs/GTM_Model6_multiscale_reconstruction_design.md` | Active design |
| `archive/` | Historical navigation and the previous project README |
| `EarthRemoteSensingRapidResponse/` | Existing datasets, adapters and historical models; not the active training entry point |
| `tools/`, `configs/`, `reports/experiments/` | Historical experiments and reusable audited utilities; Model6 does not automatically invoke them |
| `reports/research/` | Audit evidence and historical preparation notes |
| `outputs/GTM_Model6/` | Ignored run artifacts: checkpoints, predictions, manifests, metrics and complete visual review sheets |
| `outputs/GTM_viewer_bundles/` | Real viewer exports, including archived experiments |
| `ERSRR_Website/` | Historical web API; not the local viewer or Model6 inference service |

Historical files remain at their original paths because their imports, manifests and hashes are evidence. Archived does not mean deleted or scientifically worthless. The released MARS baseline and limited successful components remain comparison candidates; v5.1 is retired as a plume mapper.

## Research rules

- US-only primary study. Split original sites/acquisitions before augmentation; eight views are not eight independent observations.
- Use controlled EMIT reconstruction only as a diagnostic. Do not feed a target back as its own input. A 20 m grid is not proof of 20 m methane information.
- Missing methane pixels are unknown, not background. Facility coordinates are not dated methane or exact source labels.
- Keep original validation images; transformed validation is a separate per-parent robustness analysis.
- No promotion without reference overlays, negative/failure examples and stitched-map review.
- No automatic training, bulk download or publication. The existing cumulative CAFO imagery cap remains 100 MiB.
- Preserve raw data and frozen historical outcomes. A previously opened test set is not a fresh confirmation set.

The original capstone was created by Eduardo Gonon, Kincaid Larson, Kevin Nguyen and Hung-Nghi Vu for Dr. Cenek, advised by Dr. Nuxoll. Their work and subsequent experiments remain credited in the archive.
