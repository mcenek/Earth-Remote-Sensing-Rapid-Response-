# GTM experiment bundles

## Methane reconstruction scenes

Quantitative scenes use `scene.task: "methane_reconstruction"` and a continuous grid with explicit units. The additive grid fields are `observed`, `coarse`, `prediction`, `residual`, and optional `valid`, all row-major and aligned to `width` × `height`:

```json
{"task":"methane_reconstruction","width":2,"height":2,"units":"ppm*m","observed":[120,140,null,90],"coarse":[100,100,null,80],"prediction":[118,132,null,94],"residual":[-2,-8,null,4],"valid":[1,1,0,1]}
```

`observed` is the original/native target, `coarse` the degraded EMIT input, `prediction` the reconstructed surface, and `residual` prediction minus observed. Optional scene image fields `observedImage`, `coarseImage`, `predictionImage`, and `residualImage` provide fixed overlays. Continuous values never use the probability palette, segmentation threshold, or TP/FP/FN/TN metrics; the viewer reports valid-pixel MAE, RMSE, and bias. Missing units or all-missing continuous layers are rejected.

Use the Experiment selector to switch saved runs. Map explorer and Compare results share the selected experiment and scene. Compare results links pan/zoom across observed Sentinel, prediction and reference/disagreement views. The old experiment chart remains under Previous chart.

## Add a future experiment

Use Open experiment bundle to inspect a JSON file locally. Nothing is uploaded; the bundle lasts until reload. For portable local imports, embed PNG/JPEG/WebP image data URLs and inline the grid. For a persistent local experiment, create `outputs/GTM_viewer_bundles/<run>/experiment.json` with image/grid assets beside it. Use relative asset paths in the manifest, then click **Reload experiment folder**. Each run needs a unique experiment ID. A bundle may replace a built-in summary with the same ID; duplicate bundle IDs are rejected with a warning. No publishing, inference, or downloads occur.

Minimal structure (placeholders below describe the format, not measured research data):

```json
{
  "id": "GTM_Model5_run_id",
  "name": "Your checkpoint / run",
  "status": "Held-out evaluation or development",
  "notes": "Describe cohort, reference provenance and limitations",
  "scenes": [{
    "id": "stable_scene_id_shared_between_models",
    "name": "Scene label",
    "bounds": [[30.0, -100.0], [30.1, -99.9]],
    "rgb": "data:image/png;base64,...",
    "emit": "data:image/png;base64,...",
    "truthImage": "data:image/png;base64,...",
    "predictionAvailable": true,
    "grid": {
      "width": 2, "height": 2,
      "probability": [0.2, 0.8, 0.6, null],
      "truth": [0, 1, 0, null],
      "valid": [1, 1, 1, 0]
    },
    "date": "YYYY-MM-DD",
    "referenceDate": "YYYY-MM-DD or unknown",
    "split": "Validation; original site grouping",
    "source": "Source raster identity / hash",
    "checkpointHash": "Full checkpoint SHA-256",
    "resolution": "Native resolution and display grid resolution",
    "truthLabel": "EMIT-derived reference mask",
    "truthNote": "Exact target definition; independent truth only if demonstrated",
    "notes": "Known pairing, label or runtime limitations"
  }]
}
```

All arrays are row-major, top-to-bottom. For image-only crops, set `coordinateSystem: "pixel"` and `imageSize: [height, width]`, omit geographic bounds, and optionally set `locationPoint: [latitude, longitude]`. A point is not a verified footprint. For georeferenced exports, use one shared regular EPSG:3857 display grid. `bounds` is the WGS84 southwest/northeast bounding box of that grid. Reproject imagery and continuous predictions appropriately; use nearest-neighbor for masks. Do not stretch a geographic/UTM grid onto a Mercator rectangle and call that exact alignment. Keep quantitative source-resolution evaluation separate from this resampled display.

Probability is in [0,1]. Reference mask values are 0/1/null. Validity is 0/1; missing/no-data is never a negative. A threshold changes the display and display-pixel agreement, not the frozen benchmark. A methane regression surface is not a probability: export it with units and a dedicated continuous-layer contract before using this segmentation control.

The viewer accepts at most 1,000 scenes, 1,048,576 pixels per grid and a 50 MiB local bundle. Prefer a few representative held-out positive, negative, false-positive and false-negative cases. Only claim representativeness if the selection procedure supports it.

`predictionImage` and `disagreementImage` can supply fixed overlays without a probability grid, but do not enable threshold-dependent metrics. Reference-only scenes may supply `emit` and/or `truthImage` without predictions. Summary-only experiments use an empty `scenes` array and optional `metrics`, `reportUrl` and `description`.

Naming uses GTM_Model# artifact prefixes. These viewer IDs are organizational and do not rename historical architecture versions.

## Start locally

Double-click `GTM_Model0_OpenViewer.cmd` in the research checkout. The app runs on `http://127.0.0.1:8766/`, accessible only from this computer. The launcher reuses an existing viewer. All scripts, map controls and cached imagery are local; the optional online basemap makes tile requests only when enabled. Logs and the server PID are under `outputs/GTM_viewer_logs/`. Alternatively run `python tools/GTM_Model0_local_viewer.py --open` in a terminal, then Ctrl+C to stop it.

## v5.1 cached example export

`tools/GTM_Model51_export_viewer.py` joins the frozen NPZ probabilities/reference mask to the sealed H5 imagery by sample ID and checks labels. It writes the canonical v5.1 bundle. Repeat `--sample-id ID` to choose other cached test samples. The default eight examples cover two scene-level TP, FP, FN and TN cases; this is diagnostic selection, not a representative performance estimate.

The frozen pixel threshold is 0.40. The calibrated scene-score threshold is 0.7335791607366197, a separate decision; a scene-level TN can have a positive pixel mask. The viewer never gates pixel output by scene outcome. Summary metrics describe the full held-out test, while displayed pixel counts describe only the selected crop.
