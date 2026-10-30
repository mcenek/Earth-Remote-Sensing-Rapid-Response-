# Current methane research status

Updated October 30, 2026, 12:12 Pacific. This is a dated handoff, not a live dashboard.

## What is running

The active local run trains a fresh MARS-S2L U-Net with 13,587,585 parameters. It uses real Sentinel-2 and Landsat scenes, real clear controls, and the release's physical insertion of measured training plumes into real backgrounds. Validation and evaluation use real observations.

The frozen split has 30,304 training observations, 1,102 validation observations and 60 reserved evaluation observations. Sites, nearby site groups and known acquisition IDs are separate across roles. Evaluation has 20 plume and 40 no-plume observations. Only two plume and four no-plume observations are in the US. The regions have earlier project exposure, so this is an exploratory model-unseen-location test, not independent national validation.

At this update, twelve epochs have completed, numbered 0 to 11. The best validation scene average precision is 0.784824 at epoch 8. This selects the checkpoint. It is not pixel F1 and does not prove good plume outlines. The original schedule remains a 170 epoch cap with patience 30. No settings were changed after inspecting the first held-out predictions.

| Model, same 60 scenes | Pixel F1 | Precision | Recall | Clear observations with any mark |
|---|---:|---:|---:|---:|
| Fresh baseline, first epoch | 22.1% | 64.1% | 13.3% | 10 / 40 |
| Released MARS reference | 65.7% | 61.8% | 70.2% | 6 / 40 |

The released reference has exact training-location overlap for 19 observations. It is context, not a fair architecture ranking. The fresh interim model misses most plume area. It is not ready for a nationwide scan. All 60 predictions, including misses and false alarms, are retained.

## What the server should do

Use the [Linux setup guide](GTM_Model8_server_setup_20261030.md) and packaged source snapshot. Inspect existing data and the GPU first. Verify hashes and run the frozen preflight before a full fit. Use a fresh Linux output directory. A different Python or Torch environment must not silently resume the Windows checkpoint.

Keep the existing local fit running. Do not start an identical second fit without a reason. Once hardware and runtime are verified, use validation data to plan one controlled larger-U-Net comparison. Record capacity, memory, throughput and training budget. Keep preprocessing, labels and split fixed. Select the model on validation evidence before reserved evaluation. A larger model is a proposed next experiment, not a completed result.

Publish actual probability maps, fixed-cutoff masks, reference images, truth, errors and native metrics. Inspect aligned plume cores, missed plumes, false alarms and complete maps. A high scene score alone is insufficient.

## Facility checks and more data

Apply [facility association checks](GTM_Model8_facility_validation_plan.md) after inference. Keep CAFO coordinates out of plume-mask labels. Report point containment and distances separately from methane segmentation accuracy. A nearby facility is a candidate explanation, not measured methane confirmation.

The new [Stanford controlled-release record, version 17](https://purl.stanford.edu/qh001qt3946/version/17) describes 2024-25 releases at two western US sites, with release timing, gas flow, wind and other measurements. Its assets and matching Sentinel observations need inspection. Do not assume paired satellite images and pixel plume labels are already available. No large files from that record were downloaded for this handoff.

Additional possible airborne data await product, provenance and permitted-use confirmation. They are not in the verified training cache. Satellite EMIT and airborne observations must remain distinct.

Ready: source and tests, frozen IDs, local training, first-pass predictions, portable viewer snapshots, Linux commands and a research-agent skill. The package preserves byte hashes across platforms, including the upstream source's recorded line endings.

Pending: completed local fit and final visual review; Linux installation, CUDA and data checks; a controlled capacity experiment; matched US facility evaluation; and independent US validation. Training on MARS scenes is not training on every part of the US. National inference has not been completed.
