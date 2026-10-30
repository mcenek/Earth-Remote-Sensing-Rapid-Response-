# ERSRR: methane plume research

The goal is to screen the United States for possible methane plumes using Sentinel-2 imagery, with EMIT observations providing training and validation evidence. Known CAFOs and oil facilities can corroborate candidates; a facility location alone is not a methane measurement.

**Active work: GTM_Model8.** A fresh 13.6 million parameter U-Net uses the released MARS-S2L architecture and preparation. Its 16 channels combine current and reference satellite bands, a methane-sensitive ratio, wind and clouds. It predicts a plume probability map. The frozen split has 30,304 training, 1,102 validation and 60 evaluation observations, with plume and no-plume cases.

The first saved epoch scores 22.1% pixel F1 and misses many plumes. The original training schedule continues. Validation scene average precision selects checkpoints; it is not pixel F1. The dataset contains Sentinel-2 and Landsat scenes from several countries, not all US imagery. Only six reserved observations are in the US. National accuracy and quantitative methane upscaling remain unproven.

Read the [current status](docs/GTM_Model8_current_status_20261030.md), [baseline design](docs/GTM_Model8_mars_baseline_20261030.md), [Linux setup](docs/GTM_Model8_server_setup_20261030.md), [facility evaluation plan](docs/GTM_Model8_facility_validation_plan.md) and [receiving-agent instructions](handoff/skills/methane-server-research/SKILL.md).

Earlier multiscale and three-CNN code remains for audit. The R5 pilot failed its visual quality gate. Historical implementations are not accepted results.

## Open the results

From the repository root, using Python 3.11 or newer:

```text
python tools/GTM_Model0_local_viewer.py --open
```

Windows users can also run `GTM_Model0_OpenViewer.cmd`. The viewer runs at `http://127.0.0.1:8766` and needs no model runtime or imagery download.

When Model8 bundles are present, open `http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model8_mars_20261030_holdout_baseline#compare`. Switch to `GTM_Model8_mars_20261030_holdout_released_reference` for the same scenes. The server handoff includes all 60 saved first-pass cases for both models. A plain clone contains the viewer and older checked-in examples, not the full dataset or generated prediction inventory.

- **Historical Friday visual review:** six saved cases, linked image comparison, architecture diagram and a four-page handout.
- **Original team vs ours:** the original team's saved rendering beside Model6 examples. They are different scenes with different evidence, so this is a visual comparison, not an accuracy ranking.
- **Experiment viewer:** connected US map, acquisition timeline and linked image comparison, with reference masks, saved predictions and native scores. Checked-in samples work on a fresh clone; complete local exports appear automatically as they are published.

See [viewer instructions](GTM_Viewer/README.md) for the page addresses, offline review and packaging command. The [R5 verification report](reports/research/GTM_Model6_followup_verification_2026-09-22.md) explains the failure cases behind the scores.

For a backend machine, follow the [live viewer setup](docs/GTM_Model0_live_viewer.md). It uses the same Python server and saved result format; no frontend build is required.

## Research and code

| Location | Purpose |
|---|---|
| [Model8 status](docs/GTM_Model8_current_status_20261030.md) | Active baseline and remaining checks |
| `tools/GTM_Model8_server.py` | Portable Linux checks and training commands |
| [Model6 workspace](research/GTM_Model6/README.md) | Earlier multiscale and three-CNN work |
| [Earlier architecture design](docs/GTM_Model6_multiscale_reconstruction_design.md) | Historical stages and validation plan |
| `GTM_Viewer/` | Local viewer and curated, portable evidence |
| `tools/` | Data, export and verification commands; run only the intended command |
| `reports/research/` | Dated findings and supporting evidence |
| [Historical archive](archive/README.md) | Prior experiments and the original project overview |
| `EarthRemoteSensingRapidResponse/`, `ERSRR_Website/` | Earlier model, acquisition and web-app implementations |
| `output/`, `outputs/` | Ignored local receipts, runs, checkpoints and generated exports |

Inspect the earlier Model6 setup without training or downloads:

```text
python research/GTM_Model6/GTM_Model6.py status
python research/GTM_Model6/GTM_Model6.py validate
```

The Volusia input review is historical. Use the current Model8 documentation for the active training contract and data limitations.

## Research rules

Split physical sites and acquisitions before augmentation. Eight rotated/mirrored views are not eight independent observations. Preserve the original validation images and report transformed validation separately. Unknown methane pixels are not background. A finer display grid does not prove finer methane information.

Inspect actual predictions, misses, false positives and stitched maps before accepting a result. Historical paths, raw data and frozen outcomes remain intact so earlier claims can be checked. See the [research retrospective](reports/research/GTM_Model0_honest_research_retrospective.md).

The original capstone was created by Eduardo Gonon, Kincaid Larson, Kevin Nguyen and Hung-Nghi Vu for Dr. Cenek, advised by Dr. Nuxoll.
