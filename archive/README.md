# Historical research archive

The clean-slate Model6 study starts at [research/GTM_Model6](../research/GTM_Model6/README.md). Previous experiments are retired from the active workflow. Their files stay in place to preserve imports, artifact identities, relative paths and frozen evaluation evidence.

## Read the evidence first

- [Retrospective and corrected interpretation](../reports/research/GTM_Model0_honest_research_retrospective.md)
- [MARS experiment families](../reports/research/GTM_Model0_MARS_retrospective.md)
- [Legacy models and external tests](../reports/research/GTM_Model0_dense_external_retrospective.md)
- [Complete report inventory](../reports/research/GTM_Model0_experiment_inventory.csv)
- [Previous project README, verbatim snapshot](GTM_Model0_previous_README.md). Its relative links use the original repository root, not this archive directory.

## What is retained

| Historical material | Original location | Interpretation |
|---|---|---|
| Original capstone model/API | `EarthRemoteSensingRapidResponse/ERSRR_Model.py`, `ERSRR_Website/` | Prior implementation; not the new upscaler |
| Legacy U-Net and classical experiments | `tools/run_unet_experiment.py`, `reports/experiments/` | Limited ranking signal; weak selective mapping |
| MARS, Prithvi, DOFA, Gaussian work | `tools/*mars*`, `configs/mars_*`, `reports/experiments/MARS_*` | Preserve useful baselines and qualified findings; no global success claim |
| MethaneS2CM v5/v5.1 | `EarthRemoteSensingRapidResponse/methanes2cm_*`, `tools/*methanes2cm*` | Classification evidence; v5.1 mapping failure confirmed |
| Sensor ordinal | `.research/mars_sensor_ordinal/`, corresponding reports | Completed configuration failed; earlier infrastructure notes are not final status |
| Temporal attention | `EarthRemoteSensingRapidResponse/temporal_attention_*` | Implemented/synthetic rehearsal, not real-data validation |
| CAFO pilot | `outputs/cafo_pilot_*/`, acquisition reports | Reusable inputs/provenance; not methane targets or a trained detector |
| Old paper planning | `docs/RESEARCH_LEDGER.md`, `PAPER_OUTLINE.md`, `PUBLICATION_ROADMAP.md` | Historical claims and decisions, not current roadmap |
| Generated dossier | `reports/ERSRR_RESEARCH_REPORT.html`, `tools/build_research_report.py` | Historical report generator; not the active project front page |

The canonical sibling checkout also holds ignored checkpoints and imagery needed for historical replay. Do not remove it merely because the active work happens here. Old public-site artifacts are not the requested local experiment viewer.

No old training commands are executed by the active Model6 status/configuration entry point. Historical scripts remain explicitly runnable for audit work; they have not been rewritten in a way that would invalidate their recorded code hashes.
