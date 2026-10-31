# Continue methane research after server setup

Joshua asked the existing campus Codex session to continue the research after setup. This is the continuation handoff. Finish the current setup first. Then read this file and check the live jobs. The older status document is a dated snapshot and predates the working runtime and download queues described here.

## Objective

Produce useful methane plume predictions from real satellite imagery. Keep the inputs, reference images, EMIT truth, probabilities and fixed-cutoff predictions visible in the experiment viewer. Evaluate real plume scenes and real clear scenes at locations excluded from fitting. A strong scene score alone does not prove useful plume outlines or US-wide performance.

## Existing server work

Use the existing research branch `research/gtm-model6-restart-2026-09-22` in `~/Earth-Remote-Sensing-Rapid-Response-`. Preserve other people's edits. Do not reset the checkout or change main.

These paths are relative to the campus account's home directory:

| Item | Location |
| --- | --- |
| Data cache | `~/gtm-model8/data/MARS-S2L` |
| Python 3.12 runtime | `~/gtm-model8/venv/bin/python` |
| Pinned upstream source | `~/gtm-model8/source/marss2l-code` |
| Upstream runtime lock | `~/gtm-model8/source/marss2l-code/environment/requirements-test.lock` |
| Outputs | `~/gtm-model8/outputs` |
| Baseline acquisition log | `~/gtm-model8/acquisition.log` |
| Full-release acquisition log | `~/gtm-model8/full-release-acquisition.log` |
| Guarded launch log | `~/gtm-model8/launch.log` |
| Bootstrap receipt | `~/gtm-model8/outputs/GTM_Model8_server_bootstrap.json` |
| Runtime checks | `~/gtm-model8/gpu-probe.json` and `runtime-freeze.txt` |
| Saved absolute cache path | `~/gtm-model8/MARS_DATA_LOCATION.txt` |

The real CUDA BF16 matrix calculation and `pip check` already passed in the private runtime. It reports Torch 2.12.0+cu130, CUDA 13.0, and an RTX PRO 5000 Blackwell with 48,935 MiB VRAM. The host has about 184 GiB RAM. The global `python3` is Python 3.14; missing ML packages there do not establish that the private runtime is broken. Use the explicit private interpreter for ML and its tests.

Do not change system Python, GPU drivers, university TensorFlow environments or `/usr/local/bin/tf-venv-setup`. Do not install system packages, reboot, kill other people's GPU jobs, or change access permissions. Diagnose within this account's isolated environment.

Three tmux sessions were present at the last desktop check:

- `gtm-model8-data`: the baseline's required files. It uses a cumulative 35,000,000,000 response-byte cap.
- `gtm-model8-launch`: already waits for that acquisition, verifies its receipt, prepares paths, runs doctor and frozen preflight, checks GPU availability, then starts `gtm-model8-linux`.
- `gtm-mars-full`: waits for the acquisition's cache claim, then hashes and reuses files and fetches the remainder of the complete release with its separate cumulative 66,000,000,000-byte cap.

At 19:50 Pacific on October 30 the baseline had about 21.3 GB transferred and the launcher still waited for assets. These are historical observations. Read the current receipts and processes before acting. Training may have started since then. Do not launch another downloader, bootstrap or fit while an existing owner is active. Do not remove claims or reset transfer ledgers to force progress.

The full pinned release has 199,226 files and 96,041,292,441 logical bytes. The baseline needs 67,687 of these files and 32,607,815,901 bytes. Its current experiment deliberately uses a subset. The cache expansion includes test labels and the provider's `data/Not Used` archive for completeness. It does not add these files to fitting. No additional large download or higher bandwidth cap is part of this handoff. See [cache instructions](GTM_MARS_full_release_cache_20261030.md).

## First pass: finish and inspect the baseline

1. Read `handoff/skills/methane-server-research/SKILL.md`, [Linux setup](GTM_Model8_server_setup_20261030.md), and the current receipts. Check tmux sessions, process IDs, the bootstrap receipt and GPU use. Preserve the Windows run and any existing campus fit.
2. Let the existing acquisition and launch queue work. If it fails, identify the first real error. Use the private interpreter and the saved runtime lock. Do not weaken source, input-independence or data-quality checks. Stop repeated failing transfers and report the blocker instead of retrying indefinitely.
3. Verify that doctor and the frozen real-data preflight passed before calling the fit healthy. Confirm that training contains clear controls, measured fit-only plume patterns and correctly paired reference images. Confirm nonzero learning updates and finite losses. A CUDA smoke check does not prove learning.
4. Follow the existing fit. It uses native batch 96, validation batch 8, BF16, two workers, 65,536 draws per epoch, a 170-epoch cap and patience 30. The Windows native batch was 32; record this difference when comparing runs. Use status files and low-frequency checks; do not flood the chat with unchanged logs.
5. Publish only the predeclared first completed epoch and the final validation-selected checkpoint. The Linux controller normally publishes these after fitting completes. Its `publish` action can export `epoch_000.pt` earlier. Read its existing receipts and check resources before a manual export. Do not export many intermediate checkpoints or tune against the reserved scenes.
6. Use `tools/GTM_Model8_mars_review.py` through the existing server/controller commands and atomic publisher. Keep every evaluated scene, including misses and false alarms. Never replace measured outputs with made-up or expanded masks.

The fit directory is `~/gtm-model8/outputs/GTM_Model8_mars_baseline_20261030/fit`. New bundles go under `~/gtm-model8/outputs/GTM_viewer_bundles`. Read the launcher commands before invoking them; their roots and runtime lock are explicit arguments.

## Frozen research contract

Preserve these exact inputs for the current baseline:

- Fresh 13,587,585-parameter, 16-channel MARS-S2L U-Net. Input layout: six current bands, six reference bands, ratio, two wind channels and cloud mask. EMIT methane truth is supervision, not an extra inference input.
- Source revision `f7d264c2c845dfba1cb27f76ef6026275f8d8758`. Preserve the bundled source bytes, including recorded line endings. Do not normalize them to bypass parity checks.
- Dataset revision `c26b1d7e31a0c5241fa37c9140802622c215eb32`.
- Trainer SHA-256 `0e0f6adae01f89c7107f89ddd3e16338ff60896ffc3a30115c78c701695efe4a`.
- Plan SHA-256 `3b29c62bb250dd9155b1c2af90637b605f2709f7ec6c64b18662a0c9ed2a6345`, at `tool-diagnostics/mars-baseline-20261030/plan.json`.
- Manifest SHA-256 `59e9843ac329e6da81c5dbbebeb22cde5eaf170ee4a8a5a9c002e3afbfdc68d9`, at `tool-diagnostics/mars-baseline-20261030/train_manifest.json`.
- FIT 30,304 observations: 2,246 plume and 28,058 clear. Validation 1,102: 117 plume and 985 clear. Reserved evaluation 60: 20 plume and 40 clear. Geographic groups and both current/reference acquisition identities stay separate across roles.
- Checkpoint selection uses validation scene average precision. Pixel cutoff remains 0.50. Report pixel precision, recall and F1 separately from scene average precision.

Only six reserved observations are US scenes. Regions have earlier project exposure. Describe this as exploratory evaluation at model-unseen locations, not pristine independent national validation. The released reference overlaps some evaluation locations. Its scores are useful context and cannot establish a fair architecture ranking. The first-epoch fresh baseline had pixel F1 22.1%, so do not claim readiness from early validation scores.

## Visual review is a required deliverable

Use the current local viewer. It must show new Linux results, not only packaged Windows previews. Keep the old preview available under a clear label. Check original imagery, methane truth, probability maps and predictions on the same coordinates. Check valid-data support, orientation, native resolution, transparent backgrounds, model switching, and new-run discovery.

Review all reserved plume and clear scenes. Prepare a gallery with strong matches, partial matches, missed plumes and false alarms. Include an index of every prediction and its checkpoint. Report missed plume count and false-positive marked area on clear scenes. Measure outlines and false alarms without inflating the painted area to improve recall. Save the useful screenshots and a concise results report.

The setup session may already have started a viewer for the packaged preview. Inspect its process, output root and port before starting another. Keep access on localhost and use the existing approved private route or SSH tunnel. Do not install Tailscale on the shared server, expose a public listener, or change firewall rules. A successful HTTP response does not verify the images or remote access. If no browser is available, check bundle assets and state that visual browser inspection remains unverified.

## Next controlled experiment

After the baseline's validation result is fixed, prepare one wider U-Net comparison. Keep the same inputs, preprocessing, fit/validation split, labels, augmentation, loss, selection rule and training budget. Change model capacity only and measure parameter count, memory and throughput first. Use a separate output root and fingerprint. Do not silently resume a checkpoint with a changed architecture or runtime.

If the candidate passes the real-data checks and fits the available GPU without disturbing another job, train that single candidate after the baseline. Select it using validation evidence. Publish its final predictions and clearly disclose prior reserved-set exposure. Do not search repeatedly against the same reserved set. Before further tuning, establish a new geographically separated development/evaluation protocol. Stop for a serious data alignment failure, resource conflict, repeated download errors or a failed quality gate. Keep an honest result if the larger model does not help.

The expanded cache can support later experiments, but using more files requires a separate split review. It is not permission to merge provider train, validation and test roles. Training on MARS is not training over the whole US. Do not launch a national scan until the geographic test and visual quality justify it.

## Facility validation and external data

Read [facility validation](GTM_Model8_facility_validation_plan.md). The primary CAFO cohort is the intersection of the new CAFOSat inventory and the older Handan-Nader/Ho resource, published in 2019. Match the same physical facility with documented spatial tolerance and imagery checks. CAFO, oil facility and landfill coordinates are possible source associations, not pixel methane labels or proof of emission at the image time.

Report containment and distance from predicted plume edge, centroid and hottest pixel separately. Use acquisition timing and available wind carefully. The meeting's three-mile example is not a fixed universal acceptance rule, and a hottest pixel is not a proven source origin.

The Stanford controlled-release record `https://purl.stanford.edu/qh001qt3946/version/17` is a proposed external known-source test at two western US sites. Its event tables are not a verified set of paired rasters and plume masks. Keep it out of training and threshold selection initially. Additional downloads or airborne acquisition need a separate bounded plan. TESSERA and methane super-resolution remain follow-up research; do not claim them as completed work.

## Keep a usable handoff

Save live status, commands, run identity, validation decisions, data counts, measured metrics, gallery locations and unresolved limitations in account-local research notes. Keep credentials, private meeting material and user account details out of public Git. Do not edit the shared research journal or send collaborator messages without a new direct request.

Joshua prefers changes written and checked in the repo, then pushed on the research branch and fetched on the server. Review any server edits before publishing them. Never add data, checkpoints or unrelated drafts to a source commit. No Astra subagents; keep any requested delegated work bounded and use the current Sol/Luna setup.

Your first update should state which download and fit are actually active, whether real-data preflight passed, and the next visible result to expect. Continue the authorized work rather than stopping with a setup summary when the jobs are healthy. At completion, give actual prediction paths and evidence, the limits of the evaluation, and any exact blocker Joshua needs to resolve.
