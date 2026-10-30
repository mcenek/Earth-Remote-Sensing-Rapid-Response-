# Temporal model comparison: visual acceptance

Date: October 23, 2026

## Purpose

Compare option 1, a small temporal residual CNN, with option 2, a shared-encoder temporal U-Net. Use the same measured Sentinel image pairs, training schedule and held-out scenes. Publish actual saved predictions in the existing local viewer.

## Failure cases to check before publication

- The two models use different scene selections, support, thresholds or training budgets.
- A held-out location, connected geographic group, source acquisition or reference acquisition enters fitting or normalization.
- A no-plume label is inferred from a missing file rather than the producer's explicit observation label.
- Invalid pixels, clouds, missing reference pixels or unsigned nodata enter learning or scoring.
- A displayed prediction is a label, a fabricated plume or a resized placeholder.
- Training labels are coarsened without preserving the original annotation for evaluation.
- Reprojection changes the displayed grid but native reported scores are silently recomputed on that display.
- A model produces all background or full-box output and the report hides this behind accuracy or recall.
- The report omits a selected failure or chooses a threshold after inspecting held-out results.
- A checkpoint cannot reproduce its saved probabilities in a fresh process.
- An experiment is listed but its saved assets or score grid are missing.
- Switching models changes the selected scene, making the comparison misleading.
- Predictions are called measured methane concentrations or validated upscaling.

## Planned evidence

1. Seal a real-data manifest before either training run. Preserve official MARS test data. This new comparison uses only rows from the release's official training split. Prior internal development roles and earlier project exposure must be disclosed.
2. Preserve the 25 km geographic and shared-acquisition grouping. The available positives occupy only two resulting components. Fit the larger component and reserve the smaller one for the test. Development examples are negative-only. Use a fixed final checkpoint, not a checkpoint selected from negative-only validation or test results.
3. Run the same fixed crop and augmentation schedule for both models. Fit normalization on training imagery only. Save initial and final checkpoints, loss curves, configuration, source hashes and per-scene output arrays.
4. Recompute original-grid confusion counts independently for every test scene and both models. Show all-positive and all-negative controls. Include scene false alarms and predicted area.
5. Show current Sentinel, temporal reference, reviewed plume annotation, residual-CNN prediction and temporal-U-Net prediction together. Add score maps on a fixed zero-to-one scale and explicit errors. Every held-out scene remains accessible.
6. Publish two bundles with matching scene IDs through the existing atomic bundle publisher. Keep both models available from the viewer model selector. Add a shared comparison report.
7. Check actual browser rendering, model switching, positive and no-plume cases, image loading and a map overlay. Save screenshots and an HTTP/browser verification receipt. Do not call a small, previously examined development collection an untouched national benchmark.

## Visual and interaction direction

Use the existing viewer's dark research palette, clear typography and large satellite figures. The shared report leads with the paired comparison and its evidence limits, then the numerical results, then every scene and the repeat commands. Scene selection and overlay/score/error modes should update both models together. Motion is limited to short state transitions and an optional click-to-enlarge image view.

The report and bundles are research artifacts. They do not alter the shared journal, publish public data, change the viewer application, or promote a model to nationwide screening.
