# Real unseen-scene check, October 23, 2026

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

The current CNN 2 checkpoint fails this transfer test. It learned to fit the Volusia annotation, but it does not trace useful plumes on the scored unseen scenes. The full-box outputs in the viewer are real saved model predictions.

## Open the results

- [Open a reviewed no-plume case](http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model6_real_unseen_20261023_cnn2&scene=MARS_002c14c4-9972-4488-a1d6-254ad8662de4#compare).
- [Open all 16 cases and the complete explanation](http://127.0.0.1:8766/bundle-assets/GTM_Model6_real_unseen_20261023_cnn2/_versions/1bf3fabf4d4942ae986889d1c215f782/index.html).
- On this PC, use the same comparison URL with `http://127.0.0.1:8766` as the host. Remote access needs the existing tailnet connection and this PC awake with the viewer running.

In Compare, choose **Unseen real scenes | frozen CNN 2**. Use **Scene** to switch between seven plume cases, eight reviewed no-plume cases and Georgia. Select **Side by side**, **Fit scene** and **Disagreement**. Cyan outlines the source annotation. Pink marks extra predictions. The saved benchmark remains at 0.50 if an exploratory display threshold is moved.

## What was tested

The checkpoint was trained only on Volusia. The weights, fitting normalization, tiling and 0.50 cutoff stayed fixed. EMIT or MARS labels did not enter the model as input. No retraining or threshold search occurred.

Selection was written before inference: US samples in sample-ID order, plume cases first, at least 90% clear input and at least 25 km separation between selected MARS sites and from the fitting footprint. Seven plume sites met these rules; eight no-plume sites were selected. All selected cases were kept. Inputs are actual Sentinel target-date bands B2, B3, B4, B11 and B12, not artificial satellite data. The delivered 10 m imagery was averaged in 2 by 2 blocks for 20 m model input. Each saved 20 m prediction repeats over four original annotation cells for scoring, without smoothing.

The 15 scored MARS observations use **L1C imagery and reviewed MARS annotations**. Fitting used **L2A imagery and an EMIT provider annotation**. This input and reference difference matters. Frozen fitting normalization gives large shifts in input values. This run cannot separate those effects from single-site overfitting or other model limits. It is a transfer diagnostic, not a clean same-distribution EMIT benchmark or nationwide accuracy estimate. Other project models used MARS before; these scenes are unseen to this Volusia-only checkpoint.

NO_PLUME means that the source producer identified no plume in that observation. It does not mean the facility never emits. The source is UNEP-IMEO/MARS-S2L at revision `c26b1d7e31a0c5241fa37c9140802622c215eb32`, under CC-BY-NC-SA-4.0 for this research use.

## Result

| Check | Result |
|---|---:|
| Reviewed plume scenes | 7 |
| Reviewed no-plume scenes | 8 |
| No-plume scenes with strong false alarms | 8 of 8 |
| No-plume background marked positive | 99.89615% |
| Pooled annotation IoU, all 15 scored scenes | 1.16926% |
| All-positive control IoU | 1.16861% |
| New imagery downloaded | 0 bytes |

Pooled counts are TP 7,006, FP 592,178, FN 0 and TN 332 over 599,516 source annotation cells. All seven positive scenes are full-box predictions. Seven no-plume scenes are full positive; the eighth is 99.16% positive. The apparent 100% recall results from marking almost everything. It is not useful plume detection. On the no-plume scenes alone, FP is 319,360 and TN is 332.

Georgia is a separate real L2A observation with continuous EMIT, about 31 minutes apart. Its repaired crop includes the published peak. Prediction covers 21.81% of valid input. There is no binary reference, so no segmentation accuracy or false-positive rate is claimed for it. The HTML report shows the actual measured enhancement beside the prediction. The map viewer disables binary truth and error controls for this case. Raw DN is retained because the BOA-offset metadata remains ambiguous.

CNN 1 and CNN 3 still lack trained real-data outputs. This evaluation covers CNN 2 only. Fine output pixels do not establish validated methane upscaling or source localization.

## Evidence and verification

All paths below are relative to the repository root.

- Sealed protocol: `docs/GTM_Model6_unseen_evaluation_20261023.md`. Do not edit it after this run.
- Prepared scene list and source hashes: `outputs/GTM_Model6/GTM_Model6_unseen_20261023/data/manifest.json` and `selection.json`.
- Saved probabilities and frozen-run receipt: `outputs/GTM_Model6/GTM_Model6_unseen_20261023/prediction/`.
- Fresh-process replay: `outputs/GTM_Model6/GTM_Model6_unseen_20261023/replay/replay.json`. All 16 predictions match exactly, maximum absolute difference 0.
- Independent raster reconstruction and counts: `outputs/GTM_Model6/GTM_Model6_unseen_20261023/review/verification.json`. Original MARS TIFFs were used to rebuild band inputs, cloud support and labels. Accepted training and prediction files stayed unchanged.
- Portable HTML and complete contact sheet: `outputs/GTM_Model6/GTM_Model6_unseen_20261023/review/index.html` and `all_scenes.png`.
- Actual viewer screenshots and checks: `outputs/GTM_Model6/GTM_Model6_unseen_20261023/review/browser/`. Positive, negative and unscored Georgia states were inspected. All 80 report images loaded. Health, catalog and HTML report returned HTTP 200 through the existing private Tailscale proxy. The live catalog contains all 16 cases.

The published experiment is `GTM_Model6_real_unseen_20261023_cnn2`, version `1bf3fabf4d4942ae986889d1c215f782`. It is marked **FAILED TRANSFER**, experimental and not promoted to presentation examples. No network exposure or sharing settings changed.

Manifest SHA-256: `a425ded1462519d4c79091f6ecb29c836031fdaa8bfc06c51ab5816c8c5379dd`.

Frozen weights SHA-256: `1c57f154f7f3a8f24e67c812575a7c6b192b9120b40f60afd79ad1fd272d02e4`.

Inference receipt SHA-256: `ba5e7772010480195508d48c69eeafb410d0f4ded074e6ad9583fd948da9bf2b`.

## Repeat the check

From the repository root in PowerShell, use the existing native Torch environment. This repeats inference against the saved source cohort without replacing predictions. The verification directory must be new each time.

```powershell
$torchPython = 'python'
$cohort = 'outputs/GTM_Model6/GTM_Model6_unseen_20261023'
& $torchPython tools/GTM_Model6_evaluate_unseen.py replay `
  --manifest "$cohort/data/manifest.json" `
  --training-run outputs/GTM_Model6/GTM_Model6_real_emit_20261023/cnn2_fit_ready `
  --output "$cohort/prediction" `
  --verification-output "$cohort/replay_repeat"
```

To rebuild the independent raster review into a new directory, use the existing Python 3.11 interpreter and geospatial packages. Omit `--publish` to preserve the published version.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.venv\Lib\site-packages'
$geoPython = 'python'
& $geoPython tools/GTM_Model6_review_unseen.py `
  --manifest "$cohort/data/manifest.json" `
  --predictions "$cohort/prediction" `
  --replay "$cohort/replay/replay.json" `
  --output "$cohort/review_repeat"
```

## What this changes

Do not present the Volusia fitting score as a working detector. Before the next training experiment, make input radiometry consistent, use multiple independent training sites with reviewed no-plume examples, and fix a separate evaluation set before fitting. Keep this failed diagnostic as a regression record. Do not tune against these results and then describe the same observations as untouched testing.

No imagery or weights were downloaded, and no external journal edits, messages, commits or pushes were made for this evaluation.
