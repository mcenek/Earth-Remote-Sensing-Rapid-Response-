# GTM Model 7 temporal cohort protocol

This document records failure cases and the selection protocol before the cohort exporter is implemented. The source is the locally cached MARS-S2L publication v1 release. Only the release's official `train.csv` rows that are US Sentinel-2 L1C scenes enter the candidate pool. Publication v3 metadata must agree with those rows. The official test split stays sealed. No network acquisition is allowed.

## Failure cases to guard before implementation

1. A row in `train.csv` is absent from publication v3, or its label, product level, location, or split disagrees. Reject or fail; never infer a label from a missing plume file.
2. The official train pool contains only two 25 km positive groups (33 and 3 observations). A strict three-way geographic split cannot put positives into train, dev, and test. Do not silently relax this rule. Select a feasible arrangement and disclose its spatial independence limit.
3. A target or reference Sentinel acquisition, source asset hash, physical site, or 25 km geographic group occurs in two partitions. Detect connected components before partitioning; report if this merges positive groups.
4. Cached image, cloud mask, or annotation is missing, truncated, or differs from the release's pinned upstream identity. Exclude missing assets before assigning partitions; fail on changed assets.
5. Raster band names, 12-band layout, CRS, transform, 200 by 200 grid, dtypes, nodata, or 10 m resolution differ. Reject rather than guess a warp or channel order.
6. The target-date or background-date pixels are nonfinite or nodata; the target cloud mask is not clear (code 0; codes 1 through 4 are excluded). Require at least 90% of downsampled cells to have all four target pixels clear and all target and background bands finite/non-nodata. Background cloud state is not supplied by this release, so this condition does not assert background cloud clearance.
7. A producer positive mask is empty, nonbinary, mismatched to the scene grid, or has no valid positive cell. Reject. A producer `NO_PLUME` scene uses an all-zero target only because its reviewed release label says `NO_PLUME`.
8. Partitioning or negative sampling depends on model scores or generated predictions. Use fixed metadata, quality rules, and a recorded random seed before inference.
9. A test observation is reused for fitting or a fit statistic is computed across partitions. Publish split IDs and source checksums. Trainers compute normalization from the train partition only.
10. A 20 m prediction is scored against a 10 m annotation inconsistently. Retain the native 10 m annotation; the 20 m training target is positive if any source pixel in the 2 by 2 block is positive. For native scoring, repeat each 20 m probability into its 2 by 2 source block.

## Intended output contract

The exporter writes a new immutable data directory with `manifest.json` using schema `GTM_Model7_temporal_cohort_v1`, a preparation receipt, and one compressed NPZ per scene. Each scene records `id=MARS_<sample_id>`, `split`, `group_id`, `source_row`, point coordinates, grids, source hashes, and relative `array_file` and `array_sha256`. An NPZ contains `target_image` and `reference_image` as float32 `(100,100,6)` arrays in B02, B03, B04, B08, B11, B12 order; `valid` as bool `(100,100)`; `target` as binary float32 `(100,100)`; `label_native` as uint8 `(200,200)`; and `valid_native` as bool `(200,200)`.

The six target bands come from image channels 1 through 6. The six reference bands come from channels 7 through 12 of the same 12-band release asset. Each 20 m value is a plain 2 by 2 mean of its native 10 m pixels. Validity requires all four native target pixels to be cloud code 0 and all twelve spectral values to be finite and not nodata. `valid_native` repeats this strict 20 m validity onto the original grid. This checks radiometry, not background cloud state.

## Group and source audit

The official `train.csv` has 1,117 US Sentinel-2 L1C rows: 36 producer `PLUME` and 1,081 producer `NO_PLUME`. Publication v3 has provenance metadata for 1,000 of them: 36 positive and 964 negative. Its other 117 official-train negatives lack the needed target/reference acquisition identities and pinned asset hashes; they are excluded by ID in the manifest. Of the 1,000 provenance-backed rows, 539 have the prior `internal_training` role and 461 have `internal_validation`. Both roles are official training rows, and these prior roles do not determine the new partition.

There are 19 source `geo25` groups. Joining groups connected by a physical site, target or reference Sentinel acquisition, or a pair of sites closer than 25 km yields 15 components. Shared acquisitions merge four group pairs; proximity adds no other cross-group edge. The 36 positives occupy only two components: 33 in the larger one, 3 in the smaller one. All 33 in the larger component had prior `internal_validation` roles; all 3 in the smaller component had prior `internal_training` roles. The new split holds out the smaller positive component for test, uses the larger for fitting, and makes dev negative-only. The new test is held out from the two new models, while the wider project has previously used MARS data, so it is an exploratory comparison rather than a claim of untouched project test data.

Every producer positive was quality checked before negative selection. One large-component positive, sample `64fddfa7-bec6-4f8d-a77b-350096ede3d4`, had 88.47% valid 20 m cells and failed the stated 90% threshold. The other 35 passed. The fixed target cohort is 32 train positives, 3 test positives, 120 train negatives, 24 dev negatives, and 24 test negatives. Negatives are selected with the fixed seed in site-balanced rounds across connected geographic components, with at most 12 observations per physical site. This selection uses source metadata and image quality only. The final receipt records the actual groups, positive sites and pixels, temporal offsets, exclusions, and source hashes.

The preparation writes no model predictions, does not download data, and does not read the official test split. The output directory must be new and is never overwritten by the script.
