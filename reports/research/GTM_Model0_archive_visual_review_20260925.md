# Original capstone output and Model6 visual comparison

Reviewed September 25, 2026. This is a local, read-only inspection of saved outputs. No inference, training, or imagery acquisition was performed.

## What is available

The original team's `ERSRR_Website/Predictions/testprediction.tif` is a genuine archived prediction rendering, first committed March 27, 2026 in `ffbd96f41f9fae454bf21bc8536a9c3e05711321` ("Red prediction overlay s2 tile"). Its Git blob is `ff471819d0c6240ee5343bc78d63b3bf07cdd882` and SHA-256 is `ed9edfdeb85046e240cdc60ee222d0a125651a9ee5ed2c7ec0ec7be12a7f18a2`.

The saved artifact is 2018 by 1450 pixels, EPSG:3857, four uint8 RGBA bands. The exporter preserves all four channels exactly; a decoded PNG round-trip matched every pixel. Output SHA-256 is `429294bc49aeea47ac77217abf6f541603ded55dc15061515425ac6a6112a877`.

It is a red visualization, not raw methane enhancement or probability data. The historical source revision references `checkpoint.weights.h5`, but the raster has no per-run receipt identifying its generating checkpoint. The April 2026 `checkpoint_new.weights.h5` must not be attributed to this March raster. Acquisition date and matched ground truth are unknown. Nearby `temp_s2.tif` has different bounds and unverified pairing, so it was not used as background. The archived KML has invalid longitude/latitude values; the viewer uses the raster georeference instead.

## Open and repeat the visual review

1. Start the existing server using `GTM_Model0_OpenViewer.ps1 -NoBrowser` from the repository root, if port 8766 is not already serving the viewer.
2. Open `http://127.0.0.1:8766/GTM_Model0_team_comparison.html`. The existing Tailscale route serves the same path at `https://desktop-v07a1pi.tail948ef9.ts.net:8766/`.
3. The left map is the unchanged original-team red overlay. The right map shows a selectable Model6 experiment and scene, with cached Sentinel RGB, orange prediction at its saved cutoff, and cyan reference boundary. The maps cover different places and different physical extents. Scale bars make the distinction visible. Optional online map context is off by default and is not historical source imagery.
4. Select R5 mean and `MARS_2c6ef011-0729-42e2-b64f-10ff8207778b` (November 1, 2023). Its native saved-scene IoU displays 0.388, recall 0.493, precision 0.644, over 7,744 evaluated pixels. Switch to R5 S128: the same scene remains selected, the overlay changes, and IoU becomes 0.580, recall 0.829, precision 0.659. This is a selected localized success, not representative run performance.
5. Select R5 mean and `MARS_dac5deb4-0c1c-4040-9c12-c059c692bd04` (December 16, 2022, reviewed no plume). Surface false positives remain visible in orange, with no positive reference outline. This demonstrates why the localized success does not validate the run.
6. Reload or copy the URL. Its experiment and scene query parameters retain the selection. The full-viewer links open the corresponding archive or current scene for more detailed inspection.

## Completed checks

- Exporter validated source identity, four-channel raster format, CRS, and pixel-exact PNG conversion. Re-running to an existing output directory is refused. To regenerate separately, run `python tools/GTM_Model0_export_capstone_archive.py --help` for source/output options in an environment with the existing raster dependencies.
- Local registry contains the archive and reported no registry warnings (49 experiments at review time).
- The main viewer displays the original rendering without probability thresholds or fabricated accuracy scores. Switching back to Model6 restores its normal controls and metrics.
- Browser inspection confirmed both maps, the localized plume, mean/S128 switching, and the negative control. Images and overlays changed with the selection; no stale scene remained.
- Reload retained the selected model and scene. Desktop layout was inspected at 1400 by 1000 and stacked layout at 430 by 900. Both maps, controls and scores were visible; DOM measurements showed no horizontal overflow at width 430. Temporary viewport overrides were reset after review.
- Browser error log was empty. Both JavaScript files passed `node --check`; `git diff --check` passed.
- The comparison page returned HTTP 200 through the existing private Tailscale route. No network or access settings changed.

## Interpretation and remaining gap

The recovered original rendering visibly repeats a tile-shaped activation pattern. The conversion preserves this existing pattern; it did not generate it. One archived output cannot characterize the original model's full performance.

This page is a visual comparison, not a matched accuracy benchmark. A same-scene replay of the original checkpoint against our data has not been run. The existing baseline harness has a 6 GiB free-memory guard, and this machine had about 2.47 GiB available during inspection. TensorFlow is absent from the selected Windows environment, although an older Linux environment contains it. Strict checkpoint loading and inference remain unverified.

Model6 R5 still fails its quality gate. Pooled IoU is about 2.15% on 65 reused development scenes, with missed plumes and substantial false positives. No superiority over the original team, quantitative methane upscaling, or nationwide validation is established here.

## Local artifacts

- Export script: `tools/GTM_Model0_export_capstone_archive.py`
- Comparison page: `GTM_Viewer/GTM_Model0_team_comparison.html`, with adjacent CSS and JavaScript
- Bundle: `outputs/GTM_viewer_bundles/GTM_Model0_capstone_archive_20260327/`
- Export receipt: `GTM_Model0_archive_provenance.json` and readable Markdown in that bundle
