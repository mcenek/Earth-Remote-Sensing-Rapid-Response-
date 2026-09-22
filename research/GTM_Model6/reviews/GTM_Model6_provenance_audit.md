# GTM Model6 legacy paired-raster provenance audit

Date: 2026-09-18  
Scope: read-only audit of `EarthRemoteSensingRapidResponse/Dataset`; no files were downloaded, trained, or rewritten.

## Decision

The legacy directory contains 102 six-band TIFF candidates (100 `train_test`, two `validation`). It is not an approved Model6 cohort. The names prove a Sentinel-2 acquisition identifier and an EMIT product identifier, but they do not prove US eligibility, spatial co-registration to native EMIT, or a scientifically valid fine target. The Model6 catalog correctly marks this source `approved_for_model6: false` (`research/GTM_Model6/GTM_Model6_data_sources.json`).

## File and pairing evidence

The 100 regular names follow:

`<S2 start>_<S2 end>_<MGRS tile>_EMIT_L2B_CH4PLM_001_<EMIT timestamp>_<granule>grid_<n>.tif`.

Examples: `Dataset/train_test/20230225T170301_20230225T170953_T14QNG_EMIT_L2B_CH4PLM_001_20230224T181429_000726grid_1.tif` and `Dataset/train_test/20241014T165251_20241014T170213_T15RWP_EMIT_L2B_CH4PLM_001_20241014T201756_003690grid_1.tif`. The two exceptions are `Dataset/train_test/20230203T171519_20230203T172113_T14RMU.tif` and `Dataset/train_test/20230410T172859_20230410T174507_T13RFQ.tif`; they contain no EMIT identifier in the filename and cannot be paired from the filename alone.

The filename-derived S2-to-EMIT lags for the 100 regular rows range from -4.99 h to 1,273.81 h, with median 70.03 h. Several are multi-day or multi-week joins (for example `T17SNV` 333.85 h, `T12SWA` 695.83 h, and `T14QMG` 1,273.81 h). Negative values occur where the EMIT timestamp is later than the S2 start (for example `T39SWV` -4.46 h and `T15RWP` -3.42 h). Therefore “paired” means the historical collection join, not verified co-temporal observation.

## US eligibility

The MGRS token gives a useful screening field but is not a US boundary proof. The 24 regular rows with zones 11–17 are the CONUS-screening candidates:

`T11SNR` (6), `T11SMS` (1), `T12SWA` (1), `T12SVE` (1), `T12TVL` (1), `T13TEE` (1), `T14QMG` (3), `T14QNG` (1), `T14RMS` (2), `T14RNP` (1), `T15RWP` (1), `T16SDB` (2), `T16SDC` (1), `T16TEL` (1), and `T17SNV` (1). `T14QMG` is a southern zone-14 Q-band tile and is a Mexico/border false-positive risk; it must be checked against the actual footprint before inclusion. The remaining 21 zone-11–17 rows are only candidates, not approved US observations. All other regular rows (zones 30–49) are outside CONUS. The two exception files also require footprint metadata; their names alone do not establish US status.

No committed legacy manifest containing coordinates, polygon/footprint, CRS, or country test was found. Thus the exact US subset cannot be certified from the repository evidence currently available.

## Units, grid, nodata, and transformation evidence

`Data Collection/s2_image_exporter.py` selects `B2,B3,B4,B11,B12`, reprojects to the B12 projection at `scale=20`, and exports a `256x256` GeoTIFF in `EPSG:4326`. Its comment says Sentinel values are 0–10000, but the code uses `.divide(1)`, so no normalization is applied. `Data Collection/process_all.py` and `process_all_split_s2.py` resample the EMIT raster to the S2-derived grid; the latter explicitly sets `desired_resolution = 20` and uses `-9999` handling. These are pipeline implementation facts, not per-file metadata verification.

`ERSRR_Model.py` assumes band order `[B2,B3,B4,B11,B12,EMIT_CH4]` and treats EMIT `-9999` as nodata. The collection code also rejects tiles based on `emit_max_plume_val < 0` and a `MIN_EMIT_PPM_VAL = 300` heuristic. This creates a documented risk of inconsistent nodata/masking and selection bias. The source catalog records the same limitation: “EMIT -9999 nodata inconsistencies.”

The parent TIFF statistics audit reports 102×256×256 arrays and approximately 10 m EPSG:4326 export grids, with only about 1,000 unique EMIT values in most tiles. That pattern is consistent with a coarse EMIT field resampled/upsampled onto the Sentinel export grid. It is not native 20 m or 10 m methane truth. The native source files are present in the canonical sibling checkout below the repository's ignored/untracked data tree; they were missed by the first audit pass and are catalogued in the correction below.

## Correction: canonical ignored native source tree

Read-only path inventory found the canonical sibling checkout:

`../Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/`

The authenticated V002 batch contains 12 native `CH4PLM` TIFFs and corresponding `CH4ENH` products with `CH4SENS` and `CH4UNCERT` companions under `emit-v002-authenticated-2026-07`. The committed contract report `reports/acquisition/EMIT_V002_AUTHENTICATED_PLUME_AUDIT.md` / `.json` verifies the CH4PLM products as one-band float32, EPSG:4326, nominal 60 m, `-9999` nodata, and `ppm m`; it reports 12/12 contract-compliant granules and 9.6% aggregate valid fraction. The report also explicitly says CH4PLM is external positive evidence, not a Sentinel input channel, and that source enhancement, uncertainty, and sensitivity products must be retained for quantitative targets.

The authenticated CH4PLM footprints show three conservative inland US scenes: `EMIT_L2B_CH4PLM_002_20241020T170504_003677` (Georgia; tagged maximum about −84.3633, 34.2799; enhancement bounds about −85.1102…−83.9319, 33.6284…34.7123), `EMIT_L2B_CH4PLM_002_20241130T180310_003723` (Texas; bounds about −103.0955…−102.9269, 30.8435…31.0468), and `EMIT_L2B_CH4PLM_002_20250922T204933_003374` (Texas; about −102.1081…−101.9877, 32.1812…32.3010). The other nine authenticated pilot footprints are outside the US. These three native products are evidence candidates, not yet a Model6 approved paired cohort: Sentinel alignment, exact window transforms, enhancement/sensitivity validity, and independent split membership still require a manifest.

The existing `reports/acquisition/emit_v002_time_aligned_candidates.json` contains 70 prediction-blind catalog candidates and `emit_v002_l1c_pairs.json` contains 70 complete L1C target/reference pairs, but these are global catalog pairings and do not by themselves identify a US Model6 cohort. Their contracts document Sentinel bands `B02,B03,B04,B08,B11,B12`, official L1C asset authority, and time-alignment rules; Model6's intended five-band input still requires an explicit B08 exclusion/derivation decision.

## Concrete blockers for Model6 approval

1. Build a reviewed manifest with original native EMIT (`CH4ENH`, `CH4SENS`, `CH4UNCERT`) and Sentinel source IDs, coordinates/footprints, CRS/affine transforms, native resolutions, units/scales, nodata and valid-observation masks. The ignored sibling source tree must be referenced explicitly; it is not part of the tracked checkout.
2. Resolve the 21 CONUS-screening candidates against actual geometry; explicitly exclude or document `T14QMG` border/Mexico cases and classify the two filename exceptions.
3. Record actual acquisition times and lags per row; do not treat filename pairing as co-temporality.
4. Keep these rasters usable only for a labeled engineering/degradation demonstration. Interpolated or upsampled EMIT values cannot support a claim of validated 20 m fine methane detail.
5. Obtain independent fine methane truth and source/wind evidence before any Model6 upscaling or source-localization result is called scientifically validated.
