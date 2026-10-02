# ERSRR: methane plume research

The goal is to screen the United States for possible methane plumes using Sentinel-2 imagery, with EMIT observations providing training and validation evidence. Known CAFOs and oil facilities can corroborate candidates; a facility location alone is not a methane measurement.

**Active work: GTM_Model6.** Martin's design trains four models with 16, 32, 64 and 128 pixel context windows, then trains a second model to combine their predictions. The saved R5 pilot used a fixed average instead. It failed its quality gate: 2.15% pooled plume overlap and 2.25% precision across 65 reused development scenes. Quantitative methane upscaling and emission-origin estimates remain unproven.

## Open the results

From the repository root, using Python 3.11 or newer:

```text
python tools/GTM_Model0_local_viewer.py --open
```

Windows users can also run `GTM_Model0_OpenViewer.cmd`. The viewer runs at `http://127.0.0.1:8766` and needs no model runtime or imagery download.

- **Friday visual review:** six saved cases, linked image comparison, architecture diagram and a four-page handout.
- **Original team vs ours:** the original team's saved rendering beside Model6 examples. They are different scenes with different evidence, so this is a visual comparison, not an accuracy ranking.
- **Experiment viewer:** map overlays, reference masks, saved predictions and native scores. Checked-in samples work on a fresh clone; complete local exports appear when present.

See [viewer instructions](GTM_Viewer/README.md) for the page addresses, offline review and packaging command. The [R5 verification report](reports/research/GTM_Model6_followup_verification_2026-09-22.md) explains the failure cases behind the scores.

## Research and code

| Location | Purpose |
|---|---|
| [Model6 workspace](research/GTM_Model6/README.md) | Active configuration, manifests, status and research journal |
| [Architecture design](docs/GTM_Model6_multiscale_reconstruction_design.md) | Model stages, data requirements and validation plan |
| `GTM_Viewer/` | Local viewer and curated, portable evidence |
| `tools/` | Data, export and verification commands; run only the intended command |
| `reports/research/` | Dated findings and supporting evidence |
| [Historical archive](archive/README.md) | Prior experiments and the original project overview |
| `EarthRemoteSensingRapidResponse/`, `ERSRR_Website/` | Earlier model, acquisition and web-app implementations |
| `output/`, `outputs/` | Ignored local receipts, runs, checkpoints and generated exports |

Check the local research setup without training or downloads:

```text
python research/GTM_Model6/GTM_Model6.py status
python research/GTM_Model6/GTM_Model6.py validate
```

The Volusia input review has a paired Sentinel/EMIT observation and five target bands. Exact label meanings, retrieval units, radiometric conversion, temporal references and independent reviewed examples still need resolution before a new training run.

## Research rules

Split physical sites and acquisitions before augmentation. Eight rotated/mirrored views are not eight independent observations. Preserve the original validation images and report transformed validation separately. Unknown methane pixels are not background. A finer display grid does not prove finer methane information.

Inspect actual predictions, misses, false positives and stitched maps before accepting a result. Historical paths, raw data and frozen outcomes remain intact so earlier claims can be checked. See the [research retrospective](reports/research/GTM_Model0_honest_research_retrospective.md).

The original capstone was created by Eduardo Gonon, Kincaid Larson, Kevin Nguyen and Hung-Nghi Vu for Dr. Cenek, advised by Dr. Nuxoll.
