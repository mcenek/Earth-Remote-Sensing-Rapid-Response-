# Three-CNN implementation and acceptance contract

Date: October 21, 2026. This contract is written before implementation.

## Intended result

CNN 1 learns from paired Sentinel-2 and airborne methane observations. CNN 2 learns independently from paired Sentinel-2 and EMIT observations. CNN 3 learns from the frozen spatial features of both networks. It uses EMIT supervision. Prediction takes only the five Sentinel bands. The EMIT branch alone remains an explicit comparison.

Use TensorFlow/Keras, which matches Martin's proposed VM environment. Use the existing local WSL runtime for verification. Download no imagery, weights, or packages for this work.

## Failures to prevent

1. EMIT, airborne measurements, target masks, or facility labels enter the satellite input.
2. Band order, radiometry, units, observation times, valid support, or source identities are missing or unresolved.
3. The input and target grids differ in position, CRS, footprint, or pixel order. A display resize hides that mismatch.
4. One site, parent image, source acquisition, overlapping footprint, or duplicate array enters different roles. Augmentation creates apparent independent samples.
5. Unknown labels become negatives, invalid corners enter a loss, or an empty support produces a finite-looking score.
6. Training losses mix airborne concentration with EMIT column enhancement.
7. CNN 3 changes either branch, including normalization state. A checkpoint from another manifest or model configuration is silently reused.
8. Evaluation uses training or validation scenes, or a threshold is selected on the test scenes.
9. Only central patches are exported, or tiling leaves unreported gaps or visible seams. Invalid input is shown as supported prediction.
10. The viewer shows a target, a fabricated heatmap, stale arrays, or a partially written bundle as a trained prediction.
11. Synthetic engineering outputs are described as a methane result or added to the default research presentation.
12. Training runs without a time bound, overwrites an earlier run, resumes incomplete state silently, or needs a large download just to start.

## Prepared data interface

The new versioned manifest is distinct from the retired four-context combiner contract. Version 1 uses `schema: GTM_Model6_three_cnn_manifest_v1`, `data_kind: synthetic | measured`, `bands: [B2,B3,B4,B11,B12]`, and an `observations` list. Every row has:

- `id`, `parent_id`, `site`, `group`, `source_ids`, and `role` (`branch_fit`, `fusion_fit`, `validation`, or `test`).
- `target_kind` (`airborne` or `emit`). Fusion fitting uses EMIT only.
- `input`: relative NPZ `path`, file `sha256`, `sensor: Sentinel-2`, `product_level: L1C | L2A`, `units: reflectance`, `radiometry_verified: true`, UTC `acquired_at`, and `grid`.
- `target`: `kind: plume_mask`, `units: binary`, UTC `acquired_at`, `support_reviewed: true`, `reviewer`, `source`, and `grid`. Initial training is segmentation only. Quantitative enhancement and its uncertainty head are deferred until units and QA are established.
- `review`: `status: approved`, `alignment_verified: true`, `timing_verified: true`, `independence_verified: true`, `reviewer`, and `notes`.

Each grid has `shape: [height,width]`, `crs`, and six affine coefficients `[x_step,0,x_origin,0,-y_step,y_origin]`. Input cells are 20 m. The target grid has the same footprint and CRS, with an integer number of input cells per target cell. Target grids may equal the input grid or be coarser. The model predicts on the 20 m input grid; training and scoring aggregate predictions to the target grid. A finer airborne raster requires a separately reviewed preparation step. This release does not silently resample it.

Each NPZ contains exactly `image` (H,W,5 floating-point reflectance), `input_valid` (H,W binary), `target` (h,w floating-point 0/1 or NaN), and `target_valid` (h,w binary). Invalid target cells must be NaN; valid target cells must be 0/1. A target cell may be used only if all corresponding input cells are valid. Unknown support is never negative. File hashes bind the arrays and their metadata. Duplicate input content and intersecting footprints cannot cross roles. All identity fields remain traceable after augmentation.

Synthetic fixtures have no geographic location and cannot populate the national map. Their grid uses `LOCAL_METRE_GRID`. Measured grids use a projected metre CRS and retain the native measurement grid. Unresolved measured data are rejected, not repaired by the trainer.

## Model and runner interface

New files are under `research/GTM_Model6/` with the `GTM_Model6_three_cnn_` prefix. The data module provides `load_manifest(path)` with validated rows, resolved array paths, and a manifest hash, plus `load_arrays(row)`. The fixture module provides a CLI to write a fresh synthetic manifest.

The runtime module provides `train_stage(manifest_path, run_dir, stage, config)`, `predict_run(manifest_path, run_dir, stage, roles)`, and `runtime_info()`. Stages are `airborne`, `emit`, and `fusion`. Configuration includes `seed`, `epochs`, `batch_size`, `base_channels`, `learning_rate`, `max_minutes`, `tile_size`, `halo`, and `threshold`.

Branch training uses only matching `branch_fit` rows. CNN 3 uses only EMIT `fusion_fit` rows. Early stopping uses matching `validation` rows. Predictions default to original `test` rows. Eight exact rotation/mirror transforms are generated only for fitting, without materializing eight copies. The transform set is the eight distinct right-angle/mirror symmetries; arbitrary 15 to 45 degree rotations are deferred because they need interpolation and corner-support checks.

Each stage writes a checkpoint and an atomic JSON receipt after successful completion. Receipts bind the source manifest, configuration, source code, training/validation groups, checkpoint hash, parents, normalization, runtime, and completed epochs. Fusion requires both matching branch receipts and verifies their weights are unchanged after fitting. A completed stage cannot be overwritten. No implicit resume.

Prediction writes actual model arrays and a results receipt under the run directory. Each prediction NPZ contains `probability` (H,W), `valid` (H,W), `native_probability` (h,w), `target` (h,w), and `target_valid` (h,w). The JSON receipt links each array to its row, stage, checkpoint, role, and native-grid counts. Replaying a saved checkpoint must reproduce arrays within a recorded numeric tolerance. Report probability range and predicted area, including all-positive and empty masks. A quality finding does not delete a failed output.

## E2E acceptance

Run the public workflow on a small synthetic dataset: fixture, preflight, both independent branches, frozen-feature fusion, checkpoint replay, full-scene prediction, export, atomic publication, HTTP load, and browser reload. Retain a JSON receipt, exact commands, checkpoint and array hashes, native metrics, and screenshots. Label every synthetic view as an engineering check. Do not change the default presentation.

Through the same CLI, reject swapped bands, shifted grids, cross-role identities and footprints, corrupt hashes, target leakage, unknown-as-negative arrays, wrong target kinds, changed checkpoint parents, and incomplete exports. Verify that failed publication preserves the last complete catalog. Native-grid metrics must reproduce directly from the saved prediction arrays. Viewer values must equal the saved model arrays before display reprojection. Browser checks must cover the original image, reference, probability, threshold mask, stage switching, and reload.

Real training readiness is a separate result. Local software acceptance does not identify the missing airborne dataset, resolve existing radiometry issues, establish independent methane controls, or provide VM access.

## Input-only inference extension

Deployment must accept reviewed Sentinel scenes without methane reference data. Requiring a target for inference would invite fabricated labels, while accepting a target field or hidden target array would let supervision leak into prediction. A scene with missing Sentinel source identity, band order, radiometry review, acquisition time, grid, valid support, or approval must fail before model execution. Mixed measured CRSs can evade footprint comparison without reprojection, and a synthetic fixture must never be described as measured. Converting paired test observations into deployment scenes must remove target bytes and target metadata entirely.

The distinct `GTM_Model6_three_cnn_scenes_v1` manifest has `data_kind: synthetic | measured`, the exact five `bands`, and nonempty `observations`. Each scene has exactly `id`, `source_ids`, `input`, and `review`; no target, target kind, group, or training role is needed. The input has the same path, SHA-256, Sentinel sensor, product level, reflectance units, verified radiometry, explicit UTC acquisition time, and 20 m projected grid as the paired contract. Scene review has `status: approved`, a reviewer, and notes. Each scene NPZ contains exactly `image` and `input_valid` with the paired contract's five-channel shape, binary support, and finite supported pixels. Empty support fails. Measured scenes use one common supported projected metre CRS per manifest; synthetic scenes use `LOCAL_METRE_GRID` and retain explicit synthetic provenance. Duplicate source identities, image content, or intersecting footprints are rejected across distinct scene IDs. The loader never invents a timestamp, target, or support mask.

Acceptance includes a generated input-only fixture from the four paired test scenes. Its JSON and NPZ files contain no target fields or arrays. The public inference path must load this fixture and reject an injected target field, an extra NPZ array, a bad hash, mixed measured CRSs, or synthetic provenance mislabeled as measured.
