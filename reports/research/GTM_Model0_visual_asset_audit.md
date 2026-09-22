# Visual asset audit (read-only, 2026-09-18)

> **Superseded inventory conclusion:** The original broad "no saved per-pixel outputs" finding below was incorrect. V5.1 dense caches, matching H5 imagery, and all three frozen checkpoints were subsequently found in the canonical sibling checkout. Eight checkpoint replays now reproduce the viewer masks exactly at the frozen threshold. See `GTM_Model51_checkpoint_replay_audit.json` and `GTM_Model0_honest_research_retrospective.md`. The original text is retained as a record of the inventory mistake, not current availability guidance.

## Finding

There are no saved per-pixel model prediction/probability masks, predicted GeoTIFFs, or aligned prediction-vs-ground-truth montage panels in the local `ersrr-ordinal-win`, `Earth-Remote-Sensing-Rapid-Response-`, or `ersrr-publish-2026-09-04` checkouts. The available NPZ artifacts are tabular scene/patch scores and metadata, not image tensors. A viewer that claims to show prediction overlays therefore cannot be populated from a real saved prediction layer without running inference or acquiring the missing outputs.

## Score artifacts (real, but not overlayable)

| Artifact | Contents / provenance | Viewer use |
|---|---|---|
| `ersrr-ordinal-win/outputs/mars_dofa_gaussian_champion_folds34_scores.npz` | `sample_ids`, `labels`, `sensors`, `groups`, `folds`, `released_primary_scores`, `spatial_prithvi_scores`, `champion_scores`, `gaussian_strength`, `protection_gate`, plus source hashes. The associated finalization report is `reports/acquisition/mars_dofa_v2_scene_features_folds34.json`; it identifies folds 3/4 development-only, 17,745 rows, labels NO_PLUME/PLUME, and Sentinel-2/Landsat counts. | Good for score tables/ranking and model provenance; no spatial raster or pixel mask. |
| `ersrr-ordinal-win/.research/mars_sensor_ordinal/candidate_folds34_predictions.npz` | `sample_ids`, `labels`, `sensors`, `groups`, `folds`, `scores`, `dense_counts`; this is an ordinal candidate score table, not a raster prediction. | Good for historical score comparison only. |
| `ersrr-ordinal-win/outputs/mars_gaussian_scene_aligned_replicate_folds34_states.pt` and `.research/mars_sensor_ordinal/**/endpoint*.pt` | Checkpoint/state artifacts for score pipelines; no direct image panels. Loading requires the matching research code and dependencies, and is not a ready CPU viewer asset. | Provenance/checkpoint reference only. |

## Real aligned source/ground-truth candidates

The strongest local source for an actual input/truth overlay is the original six-band training/validation GeoTIFF collection:

* `ersrr-ordinal-win/EarthRemoteSensingRapidResponse/Dataset/train_test/` — 99 TIFFs.
* `ersrr-ordinal-win/EarthRemoteSensingRapidResponse/Dataset/validation/` — 3 TIFFs.

These are the project’s EMIT/Sentinel paired tiles (the filenames contain the Sentinel acquisition, tile, EMIT `CH4PLM` product, and grid). They are the correct source for a future inference panel: Sentinel bands plus EMIT CH4 target in one aligned raster. They are not saved model predictions. The repository model contract documents a 256×256×6 convention (`B2,B3,B4,B11,B12,EMIT_CH4`); retain the band order and treat EMIT nodata as `-9999`.

For the user's US-only target, the following tile IDs are only coarse candidates. **UTM/MGRS zones 11–17 do not establish US membership**; they also include other countries. Transform each raster footprint into geographic coordinates and check it against US boundaries before labeling any example as US. The following files are unverified geographic candidates, not an approved US cohort:

* `Dataset/train_test/20230304T160149_20230304T160904_T17SNV_EMIT_L2B_CH4PLM_001_20230218T181054_000663grid_1.tif`
* `Dataset/train_test/20230525T181919_20230525T182343_T11SNR_EMIT_L2B_CH4PLM_001_20230426T182239_000868grid_1.tif`
* `Dataset/train_test/20240729T165851_20240729T171722_T14QMG_EMIT_L2B_CH4PLM_001_20240606T151017_003236grid_1.tif`
* `Dataset/train_test/20241014T165251_20241014T170213_T15RWP_EMIT_L2B_CH4PLM_001_20241014T201756_003690grid_1.tif`
* `Dataset/validation/20230203T171519_20230203T172113_T14RMU.tif`

Do not silently mix global historical examples into a US result selector. Also verify the Sentinel/EMIT date gap and actual band metadata before treating any stored alignment as contemporaneous ground truth.

## US input-only CAFO crops

Recent local Sentinel crops are useful for an input map but explicitly have no methane truth:

* `ersrr-ordinal-win/outputs/cafo_pilot_2026-09-18/site56217/` — `blue.tif`, `green.tif`, `red.tif`, `nir.tif`, `swir16.tif`, `swir22.tif`, `scl.tif`, `rgb_preview.png`, `report.json`.
* `ersrr-ordinal-win/outputs/cafo_pilot_2026-09-18/site56218/` — same complete band set and report.
* `ersrr-ordinal-win/outputs/cafo_pilot_2026-09-11/site56216/` — same complete band set and report.

The report for site 56217 records Sentinel-2 item `S2A_15TTF_20241219_0_L2A`, EPSG:32615, 1,280 m crop, 10/20 m band resolutions, and `methane_truth_available:false`; scope is “one-location engineering crop pilot; no training labels.” These are US input examples only and must not be presented as truth or prediction overlays.

## Published PNG fallback

`ersrr-ordinal-win/.research/external/MethaneUnion/` contains architecture and dataset plots (for example `Pictures/methaneunion_pipeline.png`, `Pictures/model_architecture.png`, and `data_preprocess/plot_output.png`). These are publication/research illustrations and are not tied to the local score NPZ rows with aligned input/prediction/truth identity. They may be used as labeled methodology figures, never as prediction overlays.

## Recommended viewer sample contract

Until a real prediction export is produced, classify candidate historical TIFFs geographically and verify their source/target contract before display. Expose 56216/56217/56218 as “US Sentinel input-only; methane truth unavailable.” For any future prediction export, require a shared `sample_id`/filename, identical raster shape/geotransform, explicit model name/checkpoint hash, cohort (`US historical` vs `non-US historical`), and separate layers for Sentinel RGB, EMIT reference, independently validated truth if present, predicted probability, and thresholded mask. Do not synthesize a mask from scores or imply that a score table is spatially aligned.

## Subsequent recovery on 2026-09-18

After this initial audit, fresh compact and physics ResUNet checkpoint inference was exported for two Texas validation-folder scenes. See GTM_Model1_prediction_export_runtime.md and GTM_Model0_visual_review.md. The newer foundation/spatial benchmark spatial-output gap remains unresolved.
