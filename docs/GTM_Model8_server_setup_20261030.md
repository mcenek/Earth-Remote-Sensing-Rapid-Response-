# Model 8 Linux server handoff

This launches the same frozen MARS-S2L baseline from new random weights on a Linux NVIDIA server. It keeps the existing 30,304 FIT, 1,102 validation, and 60 reserved evaluation scene IDs. The 60 scenes never select epochs, thresholds, or model settings. The fixed review pixel cutoff is 0.50. This is exploratory public-data evaluation, not an independent US or national validation.

The server launcher is `tools/GTM_Model8_server.py`. It does not change the active Windows controller or its model fingerprint. Its `doctor` and `commands` actions use only the Python standard library and do not import Torch. `doctor` reports missing files and packages; a passing doctor means the heavy trainer preflight is still required.

## Required layout

Unpack the private handoff without changing file bytes. It contains `ersrr-ordinal-win/` for project code, `source/marss2l-code/` for the pinned upstream source snapshot, and `preview-outputs/GTM_viewer_bundles/` for saved viewer examples. The full MARS-S2L raster cache is supplied separately. Choose a separate output root for the new Linux fit. The frozen trainer expects two local paths under the code checkout. Create links so those paths resolve to the explicit roots:

```bash
GTM_PACKAGE=/srv/ersrr/handoff
GTM_CODE="$GTM_PACKAGE/ersrr-ordinal-win"
GTM_DATA=/srv/ersrr/data/MARS-S2L
GTM_SOURCE="$GTM_PACKAGE/source/marss2l-code"
GTM_OUTPUTS=/srv/ersrr/linux-outputs

mkdir -p "$GTM_CODE/EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external" "$GTM_OUTPUTS"
ln -s "$GTM_DATA" "$GTM_CODE/EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
ln -s "$GTM_SOURCE" "$GTM_CODE/EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/marss2l-code"
ln -s "$GTM_OUTPUTS" "$GTM_CODE/outputs"
```

If a link path already exists, inspect it instead of replacing it. Keep data, new outputs, checkpoints, and credentials outside Git. The MARS-S2L dataset is the pinned Hugging Face revision `c26b1d7e31a0c5241fa37c9140802622c215eb32`; the bundled upstream `UNEP-IMEO-MARS/marss2l` source snapshot is commit `f7d264c2c845dfba1cb27f76ef6026275f8d8758`. Use that bundled source. Its pinned `loaders.py` has CRLF line endings, and the trainer hashes those exact bytes. Python reads CRLF source on Linux. Do not normalize line endings, fresh-clone then edit files to match hashes, or disable the source checks. The archive also preserves the frozen plan and manifest bytes. Copy the verified local MARS data cache or retrieve the immutable dataset revision separately. The frozen asset catalogs and verifier under `tool-diagnostics/mars-baseline-20261030` and `tools/GTM_Model8_verify_assets.py` can check transferred objects. The server needs the FIT and validation rasters and the released `trained_models/MARSS2L_20250326/{best_epoch,config_experiment.json}` for the reviewer. The trainer checks pinned upstream source file hashes, metadata, and raster paths. The reviewer checks released reference hashes.

The bundled `preview-outputs/GTM_viewer_bundles/` are prior saved examples for inspection. They are separate from `$GTM_OUTPUTS`, which begins a fresh Linux fit. Do not resume the Windows checkpoint in this environment: Python, dependency lock, Torch/CUDA, and source-location fingerprints differ, and the trainer rejects a mismatched fingerprint.

The saved preview can be opened without a model runtime: `python "$GTM_CODE/tools/GTM_Model0_local_viewer.py" --bundles-dir "$GTM_PACKAGE/preview-outputs/GTM_viewer_bundles" --host 127.0.0.1 --port 8766`. Use the server launcher viewer command below for new Linux predictions.

The frozen plan is `tool-diagnostics/mars-baseline-20261030/plan.json` with SHA-256 `3b29c62bb250dd9155b1c2af90637b605f2709f7ec6c64b18662a0c9ed2a6345`. The training manifest is `train_manifest.json` in that directory. Transfer the exact files from the research branch. Do not regenerate the split on the server.

## Proposed Linux environment

Use Python 3.12 in a new venv. The upstream source declares Python 3.12 or newer. Its own `environment/requirements-test.lock` pins the Linux candidate environment, including Torch 2.12.0, torchvision 0.27.0, and marshsi 0.1.6. That full lock was generated on Python 3.12; it is not the tested Windows Python 3.11 overlay. The local `requirements-minimal-locked.txt` includes Windows-only `win32-setctime` and should not be treated as a tested Linux install recipe. Confirm the GPU driver and the Torch CUDA wheel on the destination. The doctor checks that `nvidia-smi -L` sees a GPU, and trainer preflight must establish actual CUDA support, BF16, source parity, data controls, and available memory.

```bash
cd "$GTM_CODE"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r "$GTM_SOURCE/environment/requirements-test.lock"
GTM_LOCK="$GTM_SOURCE/environment/requirements-test.lock"
GTM_COMMON=(--code-root "$GTM_CODE" --data-root "$GTM_DATA" --source-root "$GTM_SOURCE" --output-root "$GTM_OUTPUTS" --runtime-lock "$GTM_LOCK")
python tools/GTM_Model8_server.py doctor "${GTM_COMMON[@]}"
python tools/GTM_Model8_server.py commands "${GTM_COMMON[@]}"
python tools/GTM_Model8_server.py preflight "${GTM_COMMON[@]}" --batch-size 32 --validation-batch-size 8 --precision bf16 --num-workers 2
```

The install command is a proposed starting point, not a verified server environment. Resolve any wheel or driver mismatch on the destination before running preflight. Keep the exact lock file used for that run; the trainer hashes it into its fingerprint. Start with the stated batch and worker settings only if preflight passes. If hardware requires different settings, repeat preflight with those exact settings before training. The trainer rejects a changed fingerprint.

## Training and saved review

After a passed preflight, this invokes the existing trainer with its original frozen manifest and source checks. It writes only under `$GTM_OUTPUTS/GTM_Model8_mars_baseline_20261030/fit`:

```bash
python tools/GTM_Model8_server.py train "${GTM_COMMON[@]}" --batch-size 32 --validation-batch-size 8 --precision bf16 --num-workers 2 --draws 65536 --epochs 170 --patience 30
```

The trainer saves `epoch_000.pt` after the first completed epoch and `best_epoch.pt` selected by validation average precision. This standalone launcher does not automate concurrent review. To publish the predeclared interim snapshot after `epoch_000.pt` appears, use the exact first-pass checkpoint. After successful training completion, publish the validation-selected final checkpoint. Each reviewer invocation publishes validation or holdout bundles into `$GTM_OUTPUTS/GTM_viewer_bundles`. It creates a new review directory and refuses to overwrite one; inspect a partial directory before retrying. The reviewer also loads the pinned released reference and checks its hashes.

```bash
python tools/GTM_Model8_server.py review "${GTM_COMMON[@]}" --stage first_pass --role validation
python tools/GTM_Model8_server.py review "${GTM_COMMON[@]}" --stage first_pass --role holdout

# Run these only after the fit completes successfully:
python tools/GTM_Model8_server.py review "${GTM_COMMON[@]}" --stage final --role validation
python tools/GTM_Model8_server.py review "${GTM_COMMON[@]}" --stage final --role holdout

python tools/GTM_Model8_server.py viewer "${GTM_COMMON[@]}" --host 127.0.0.1 --port 8766
```

Open `http://127.0.0.1:8766/?catalog=all&experiment=GTM_Model8_mars_20261030_holdout_baseline#compare` through an SSH tunnel or your private network. The saved viewer bundles include the model predictions and released reference on the same scenes. The first pass is interim. Do not use its holdout result to alter training or pick a different checkpoint.

## Remaining destination checks

Run doctor, the frozen preflight, a short bounded performance audit if the GPU differs, and the source-only tests on the server before a full fit. Verify available disk space for the dataset, checkpoints, and saved viewer bundles. A local Windows dry run cannot prove Linux package installation, CUDA compatibility, raster integrity, or full training throughput.
