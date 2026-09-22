# Model6: first measured reconstruction diagnostic

Run: `GTM_Model6_legacy_reconstruction_01`. Disposition: **useful engineering result; not promoted as a methane upscaler or source detector**.

## Actual work

Audited all 102 existing paired TIFFs. Selected 13 observations entirely within conservative US interior regions, with EMIT acquisition IDs and nontrivial observed support. Grouped nearby observations within 20 km and repeated EMIT acquisitions before splitting: 11 groups. All source hashes, footprints and pairing lags are recorded in the run manifest. Original grids for these 13 legacy pairs were not recovered; targets are legacy resampled exports. A later ignored-file audit found a separate native V002 collection (three US acquisitions), now being tested independently. No imagery was acquired.

To avoid learning repeated 10 m export values as fine truth, aggregated each 256×256 export to 32×32 using masked 8×8 means, requiring at least 50% observed support per target cell. Deliberately degraded that aggregate 2× to make the model input. The 32×32 target never enters the model directly. Only coarse observations and their coarse support enter; target support is used for loss/evaluation. Missing measurements are not negatives.

Trained a compact residual U-Net with eight training views (0/15/30/45 degrees × mirrored/unmirrored), training-only normalization and fixed 600 optimizer updates. Five predeclared group partitions rotate through approximately 20% training / 80% held out. Unequal group sizes yield 2–5 training observations. Trained EMIT-only and Sentinel+EMIT variants: ten checkpoints, 104 held-location prediction instances. The entire run took about 36 seconds on the local RTX 5070. A separate one-observation memorization check established that the pipeline could learn, but is excluded from held-out results.

## Results

| Method | Mean held-out observation MAE | Location groups better than bilinear |
|---|---:|---:|
| Bilinear interpolation | 132.01 | — |
| Nearest interpolation | 135.46 | — |
| Constant crop | 327.87 | — |
| EMIT-only U-Net | **121.66** | **8/11** |
| Sentinel + EMIT U-Net | 135.18 | 2/11 |

Values are in legacy CH4 product units (expected ppm·m, not independently verified from original product metadata). EMIT-only improves MAE by 7.8%; Sentinel+EMIT is 2.4% worse than bilinear. Shuffling Sentinel at inference scarcely changes its mean error (135.19), so this run provides no useful Sentinel contribution evidence.

An exploratory bootstrap over the 11 location-group mean gains gives an EMIT-only gain interval of 1.56–20.01 units and a Sentinel+EMIT interval of −10.00–5.60. These are small-cohort conditional diagnostics, not confirmatory uncertainty: models share training partitions, observations were historically available, and repeated held-out predictions are correlated.

## Visual review

Reviewed the complete 13-observation input sheet and the complete prediction sheet. Each scene's displayed fold is fixed as `(site_partition + 1) mod 5`, so no visually best fold was selected. The viewer also exposes all ten model/fold combinations and every held-out case.

- Actual reconstructions vary spatially; the outputs are not the previous nearly constant whole-crop probabilities.
- The Arizona bright feature, Texas gradient and Louisiana enhancement structure remain visible. The EMIT-only output often sharpens the bilinear surface modestly; fine texture is still imperfect and should not be interpreted as new observed methane detail.
- Gains are not universal: Colorado, one Texas group and San Diego worsen on average. Residuals remain visible, including at high-contrast features.
- Sentinel+EMIT adds texture in some examples without an overall accuracy benefit. It is rejected as an improvement in this run.
- **The colored footprint is observed-data support, not a learned plume outline.** Outside that support is unknown; this dataset cannot test representative methane-free backgrounds. No apparent outline match is counted as successful source detection.
- Geographic viewer layers are explicitly reprojected to a common EPSG:3857 display grid; numerical evaluation uses the original 32×32 diagnostic grid. Small differences between viewer and report metrics are display resampling, disclosed in each scene.

Artifacts: `outputs/GTM_Model6/GTM_Model6_legacy_reconstruction_01/` contains source/split manifest, ten checkpoint files with training identities, native prediction arrays, metrics, group summaries, and `GTM_Model6_all_heldout_outputs.png`. Input audit: `outputs/GTM_Model6/GTM_Model6_legacy_audit_01/GTM_Model6_inputs.png`. Viewer: [local](http://127.0.0.1:8766/?experiment=GTM_Model6_emit_only_f0#compare) or [private Tailscale](https://desktop-v07a1pi.tail948ef9.ts.net:8766/?experiment=GTM_Model6_emit_only_f0#compare).

## Stop conditions and next evidence

The full 16/32/64/128 native-context branches, merger and Gaussian source head are **not run**. Original EMIT grids for this legacy cohort, independent fine methane truth, representative negative labels and verified source/wind labels are missing. Separate native V002 US rasters were subsequently found in the historical checkout; their full-scene reconstruction study is recorded separately. Sentinel lags extend to weeks and cloud/QA provenance is inadequate. Upsampling the current tiles to manufacture context would defeat the test.

Next useful input is original georeferenced EMIT enhancement data with validity/QA and product metadata, plus time-matched Sentinel bands and QA for the same US scenes. Independent finer-resolution methane measurements are additionally required to validate 60→20 m claims. Until then, this result supports a reconstruction pipeline and a small EMIT-only baseline improvement; it does not support national emissions mapping or accurate source points.
