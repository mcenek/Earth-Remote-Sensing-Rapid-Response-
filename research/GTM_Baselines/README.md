# Previous-team baseline and paired comparison

This software slice restores a reproducible **web-model baseline definition** and a saved-prediction comparison harness. It does not claim the previous team beats Model6. No model inference, GPU work, training, downloads, full-dataset evaluation or service launch was performed here.

## What is pinned

- Source: `ce6844d4410b32ae4137ada9f7d44a271df9c192:ERSRR_Website/server.py`, the April 13, 2026 checkpoint commit.
- `GTM_Model0_capstone_20260413.py` contains the six unchanged model definitions extracted from that source, with its default hyperparameters: 256×256×5 input, 8-pixel patches, 64-dimensional embeddings, four attention heads, eight transformer layers and separate `mask_output` / `regression_output` heads. Server startup, Earth Engine authentication and training entrypoints are excluded.
- The existing sibling `checkpoint_new.weights.h5` matches Git blob `0b34e01a4dbd117dec49611e4959ae123e9b4a1a` exactly; SHA-256 is in the contract. Do not substitute `checkpoint.weights.h5`: its decoder tensor shapes differ.
- This **web baseline** uses per-tile `max + 1e-6` normalization, as historical `predict_tiles` did. The standalone training code used a saved `x_max.npy` and had different behavior. Those are separate baseline variants, not interchangeable preprocessing.
- Raw input bands are B2/B3/B4/B11/B12. Smaller scenes are zero-padded on the bottom/right to 256, then outputs are cropped to the original grid; no scene is stretched. Invalid observed input is zero-filled and excluded from scoring. No padding is scored. These small L1C crops are a transfer diagnostic, not proof of identical training radiometry.
- The mask head is compared directly. Normalized regression and its mask-gated product are saved separately, never presented as calibrated methane enhancement or substituted for mask probability.
- Hashes and HDF5 tensor metadata have been inspected. **Keras strict weight loading and prediction have not been executed in this pass.** The wrapper fails if loading disagrees; no skipped weights or random fallback are allowed. Historical training membership is unknown, so a paired fresh-holdout claim is unavailable.

`GTM_Model0_capstone_contract.json` pins source, checkpoint, normalization and claim limits. The older checkout is an artifact source; `ersrr-ordinal-win` remains the active software checkout. Existing models, runs and holdouts are unchanged.

## Commands

Run from `C:/Users/joshu/PROJECTS/REMOTE-SENSING/ersrr-ordinal-win`. The existing Python runtime may need its installed DLL permissions. Checks, preparation and cache comparison use installed dependencies. The selected environment has Keras but no TensorFlow; the future predictor requires a compatible TensorFlow CPU runtime. No package was installed in this pass.

```powershell
$py = 'C:/Users/joshu/.pyenv/pyenv-win/versions/3.11.9/python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) '.venv/Lib/site-packages')
& $py tools/GTM_Model0_legacy_baseline.py check
& $py -m unittest discover -s tests -p test_GTM_Model0_paired_comparison.py
```

Three real scenes have already been prepared under `outputs/GTM_Model0_pair_prepared_01`: the first lexicographic scene ID in each of MARS PLUME, MARS NO_PLUME, and eligible EMIT. All were previously evaluated development cases. The source TIFFs, raw five-band inputs and current mean predictions are present and hash checked. Frozen group IDs and reference masks are retained.

The **next actual legacy-vs-current comparison** is below. First select a compatible TensorFlow CPU runtime and make at least 6 GiB system RAM available. The snapshot during this pass was about 2.9–3.0 GiB and TensorFlow was absent, so model execution was deferred. Available RAM and runtime presence are checked before importing Keras/TensorFlow. CPU threads are capped at one, and GPU visibility is disabled. No automatic installation or backend substitution occurs.

```powershell
# Run with a compatible TensorFlow runtime and enough RAM; model executes on CPU.
foreach ($casePrefix in @('00', '01', '02')) {
    & $py tools/GTM_Model0_legacy_baseline.py predict `
        --scene "outputs/GTM_Model0_pair_prepared_01/${casePrefix}_scene.json" `
        --output "outputs/GTM_Model0_pair_prepared_01/${casePrefix}_capstone.npz" `
        --metadata-output "outputs/GTM_Model0_pair_prepared_01/${casePrefix}_capstone.json"
    if ($LASTEXITCODE -ne 0) { throw 'Baseline prediction failed; stop before comparison.' }
}
& $py tools/GTM_Model0_compare_plumes.py `
    --manifest outputs/GTM_Model0_pair_prepared_01/legacy_manifest.json `
    --output outputs/GTM_Model0_capstone_comparison_01
```

Prediction and comparison outputs refuse overwrites. For a partially completed loop, resume only missing scene outputs, keeping existing hashes. Missing legacy caches fail comparison; they never become all-negative masks. A failed runtime load requires a compatibility investigation, not silent retraining or conversion.

To prepare other **already evaluated** scenes, specify one to four IDs explicitly:

```powershell
& $py tools/GTM_Model0_prepare_comparison.py `
    --run outputs/GTM_Model6/GTM_Model6_plume_evidence_02 `
    --output outputs/GTM_Model0_pair_prepared_NEW `
    --scene-id emit25km-0024
```

## Harness behavior

The version-1 manifest supplies one shared evidence NPZ per scene (`truth`, `valid`, `rgb`), exact grid/scene/source identity, fixed thresholds, and two prediction NPZ/metadata pairs (`probability`, `valid`). Metadata pins model identity, prediction hash and provenance. The two internal slots are `legacy` and `current`; displayed captions always show the actual model IDs. A supplied `--baseline-run` can populate the left slot from another completed Model6 cache strictly for harness development; that is never called a capstone result.

The harness requires identical native shape, CRS, affine, scene ID and input-source identity. It performs no automatic alignment or label generation. It processes scenes serially, defaults to four scenes, caps each invocation at sixteen, caps grids at 512×512 and each uncompressed per-scene archive at 32 MiB. Runs are rejected rather than silently truncated.

Metrics use the **same intersection** of input validity, reference-known support and both predictions' validity. Common coverage must be at least 95% unless explicitly lowered and recorded; each model's individual coverage is also reported. Differences in coverage cannot silently improve apparent accuracy.

For reviewed masks it reports TP/FP/FN/TN, IoU, Dice, precision, plume recall, background pixel false-positive rate, signed/absolute pixel area error and relative area error. Projected square-metre error is available only with explicit pixel-area metadata. Equal area can conceal incorrect geometry; IoU is retained alongside it. Empty-positive IoU/recall and undefined ratios are null, not perfect scores. Pixel false-positive rate is not scene-level FPR or US prevalence.

For positive-only EMIT references, exterior is NaN/unknown. Only support recall is measured; FP/TN/precision/IoU/area error are null. Exterior predictions remain visible and their count is reported without labeling them false methane. Cyan is the shared reference, orange the prediction; purple marks unknown reference or missing model coverage. All physical scoring uses source grids; no display reprojection enters metrics.

## Verification delivered

- Ten serial behavior tests passed, including the low-memory gate before any model load.
- A three-case real-cache rehearsal completed using the two existing Model6 means, with 100% common reference coverage for every case. No aggregate model superiority claim follows from this selected sample.
- [Offline report](../../outputs/GTM_Model0_pair_cache_review_02/index.html) and [exact metrics/provenance](../../outputs/GTM_Model0_pair_cache_review_02/comparison.json).
- Eight generated overlay/reference PNGs were visually inspected across the three cases. They show the already-known surface false alarms, missed plume support and correctly visible unknown EMIT exterior.
- The browser's URL policy blocked opening the local `file:` HTML. No alternate browser route was attempted. HTML structure was checked separately and PNGs inspected directly; a live browser layout check remains unperformed.

Remaining gates: sufficient free RAM, compatible TensorFlow CPU runtime, successful strict Keras load, bounded actual legacy predictions and visual review, and investigation of historical training membership/product normalization before any scientific holdout claim. The exact legacy weight file and the three selected source scenes are **not missing**.
