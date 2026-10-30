---
name: methane-server-research
description: "Set up or continue ERSRR MARS-S2L U-Net research on the campus Linux GPU server, verify transferred data, and publish real plume and no-plume predictions in the experiment viewer. Use for this project's server training and visual evaluation handoff."
---

# ERSRR server research

Read [current status](../../../docs/GTM_Model8_current_status_20261030.md), [Linux setup](../../../docs/GTM_Model8_server_setup_20261030.md) and [facility evaluation](../../../docs/GTM_Model8_facility_validation_plan.md). The package's START_HERE.md has the exact Git revision and archive checksums.

## Outcome and current state

Develop useful plume predictions on real US imagery. Keep actual predictions visible beside inputs and ground truth. The current Model8 is a fresh 13,587,585-parameter, 16-channel MARS-S2L U-Net. It has not established quantitative methane super-resolution, emission rates or nationwide accuracy.

The Windows fit is running. Do not stop it or modify its frozen trainer, source, labels, split or settings. This handoff prepares a fresh Linux environment and subsequent controlled experiments. Avoid an unnecessary duplicate fit. Inspect saved predictions before making claims.

The split has 30,304 FIT, 1,102 validation and 60 evaluation observations, including 20 plume and 40 no-plume cases. Only six evaluation observations are US scenes. Regions have prior project exposure. This is exploratory model-unseen-location evaluation, not pristine independent validation.

## Set up and check

Use Horizon to reach the campus Linux VM, then SSH to the GPU host. Use the account and host from university instructions. User authentication stays with the user. Do not put passwords, SSH private keys, session cookies or access tokens in Git or the package.

Extract without changing line endings. Use bundled source/marss2l-code. Frozen source hashes include recorded CRLF bytes; a normalized Linux clone can differ. Do not change hash constants to bypass checks. Preserve the plan and manifest bytes too.

Use explicit code, data, source and output roots from the guide. Inspect paths before creating links. Keep a new Linux venv and output root. Do not copy a Windows venv or resume Windows checkpoints under a different fingerprint.

Inventory server data before transfer. Prefer the verified pinned cache. Transfer missing objects from supplied catalogs, count bytes, preserve hashes and keep a receipt. Do not download a duplicate full dataset. Stanford and possible airborne observations are follow-up sources, not verified inputs to this fit.

Run tools/GTM_Model8_server.py doctor, then the frozen preflight. Doctor only checks prerequisites, not CUDA, raster integrity or learning. Check source parity, input/label independence, real plume and clear controls, checkpoint roundtrip and memory. Inspect the GPU and measure throughput before choosing batch settings. Retain the exact runtime lock and fingerprint.

## Train and evaluate honestly

Upstream source revision: f7d264c2c845dfba1cb27f76ef6026275f8d8758. Dataset revision: c26b1d7e31a0c5241fa37c9140802622c215eb32. Do not regenerate the split. Sites, nearby groups and current/reference acquisitions stay separate across roles.

The baseline uses real clear scenes and measured plume patterns inserted through released physical augmentation. Validation and evaluation are real. CAFO coordinates are not plume-mask labels. Training input must remain independent of the true label.

Validation scene average precision selects checkpoints. It is not pixel F1. First-pass baseline pixel F1 is 22.1%, with many misses. The released reference has location overlap, so its score is not a fair architecture comparison.

Only publish the predeclared first epoch and validation-selected final checkpoint for this baseline. Pixel cutoff is 0.50. Do not tune on reserved scenes or expand masks to increase recall. A capacity comparison needs a new experiment, a validation decision and disclosure of earlier test exposure.

## Make results usable

Use the reviewer and atomic publisher. Retain saved probabilities, reconstructed inputs, labels, valid support and checkpoint hashes. Inspect plume cores, misses, false alarms and no-plume cases in the viewer. Include every evaluated scene. Keep native metrics separate from display resampling and later grouping.

Start tools/GTM_Model0_local_viewer.py with saved bundles. Viewing needs no GPU. Use SSH tunnelling or the user's private network for remote access. Verify real images and model switching in the browser. A server HTTP response does not prove remote access works.

After inference, use containment and distance checks from the facility plan. A hottest pixel or centroid is not a proven emission origin. The three-mile example is not a fixed rule. Save important new decisions to memory. Keep private meetings and accounts out of public Git history. Do not edit the shared journal unless asked.
