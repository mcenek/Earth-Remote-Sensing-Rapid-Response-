# Model 8 Linux server handoff

This launches the same frozen MARS-S2L baseline from new random weights on a Linux NVIDIA server. It keeps the existing 30,304 FIT, 1,102 validation, and 60 reserved evaluation scene IDs. The 60 scenes never select epochs, thresholds, or model settings. The fixed review pixel cutoff is 0.50. This is exploratory public-data evaluation, not an independent US or national validation.

The server launcher is `tools/GTM_Model8_server.py`. It does not change the active Windows controller or its model fingerprint. Its `doctor` and `commands` actions use only the Python standard library and do not import Torch. `doctor` reports missing files and packages; a passing doctor means the heavy trainer preflight is still required.

## Campus server launch

The campus login is now working through Horizon, the Engineering Linux desktop, and SSH to the GPU host. The October 30 inventory found a roughly 48 GB RTX PRO 5000 Blackwell GPU, about 184 GiB RAM, 24 CPUs, and `/usr/bin/python3.12`. System `python3` is 3.14.4. Keep the system Python, driver, shared packages, and Martin's TensorFlow helper unchanged. This experiment uses PyTorch in its own `~/gtm-model8/venv` environment. Do not install this runtime into `~/tf-env`.

Use the existing clone and research branch. Code is edited and checked on the development machine, pushed to this branch, then fetched on the server:

```bash
cd ~/Earth-Remote-Sensing-Rapid-Response-
git switch research/gtm-model6-restart-2026-09-22
git pull --ff-only
python3 tools/GTM_Model8_server_prepare.py probe
```

First check whether a verified MARS cache is already available. Use its actual path if so. The new account initially contained only the clone, and no shared dataset directory was visible under `/data`. The small bundled catalog describes only the existing frozen experiment's required assets. It contains 67,687 objects totaling 32,607,815,901 bytes. It includes input rasters, the FIT methane patterns used by the released augmentation, provider metadata, and the released reference checkpoint. It does not request the entire public release. Downloading requires an explicit byte limit:

```bash
mkdir -p ~/gtm-model8/data/MARS-S2L
python3 tools/GTM_Model8_fetch_assets.py --data-root ~/gtm-model8/data/MARS-S2L
python3 tools/GTM_Model8_fetch_assets.py --data-root ~/gtm-model8/data/MARS-S2L --download --max-network-bytes 35000000000
```

The first command is a local inventory. The second transfers missing objects from the immutable provider revision and verifies each object's size and hash. Check its completion receipt before setup. Keep the acquisition process in `tmux` if the connection may close. The 35 GB limit applies to this acquisition tool, including retry body bytes. Python package downloads are separate. The tool must stop on repeated failures rather than repeatedly downloading the release.

The source archive under `handoff/vendor/` preserves the frozen upstream source bytes and its license. It avoids a fresh checkout changing recorded line endings. The setup script is dry by default. After the asset inventory is complete, inspect the plan, then execute:

```bash
python3 tools/GTM_Model8_server_bootstrap.py --data-root ~/gtm-model8/data/MARS-S2L
python3 tools/GTM_Model8_server_bootstrap.py --data-root ~/gtm-model8/data/MARS-S2L --execute --install-runtime --start-training
```

Setup creates or checks the separate Python 3.12 venv, installs the upstream Linux candidate lock there, verifies source and data paths, checks for other GPU work, and runs the frozen preflight. Successful preflight is required before a fresh training controller is launched in `tmux` as `gtm-model8-linux`. Logs and status are kept under `~/gtm-model8/outputs/GTM_Model8_mars_baseline_20261030/`. A live tmux session alone does not prove training is progressing. Check the trainer's `run_status.json`, epoch logs, and GPU use.

The controller publishes the first and final saved predictions after fitting. To inspect the predeclared first-epoch snapshot sooner, run its `publish` action from another shell after `epoch_000.pt` exists. This must use the same roots and runtime lock as the fit. The status command and logs distinguish a fit failure from a prediction-export failure. Keep the original split, checkpoint rule, pixel cutoff, and exposure caveats below.

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
