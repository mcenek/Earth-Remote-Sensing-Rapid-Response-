# Model6 native-grid reconstruction: measured result and visual review

Run: `GTM_Model6_native_multiscale_01`. **Actual predictions available; modest exploratory result, not a validated deployment model.**

## What was run

The ignored historical checkout contains original authenticated V002 EMIT CH4ENH, CH4UNCERT, CH4SENS and CH4PLM products. Three acquisitions are in the US: Georgia on 2024-10-20, Texas on 2024-11-30 and Texas on 2025-09-22. Source grids, hashes, units, timestamps and window transforms are saved. Enhancement, uncertainty and sensitivity must share the exact grid. Validity uses their common finite/non-nodata support, with positive sensitivity and uncertainty. Negative enhancement retrievals remain signed observations, not missing pixels or negative-emission labels.

The task is real **controlled 2× reconstruction**: degrade native observations by masked 2×2 area averaging, then reconstruct the withheld original grid (nominally 120→60 m). It does not validate unmeasured 60→20 m methane detail.

Trained independent compact U-Nets at context windows **16, 32, 64 and 128**, each with the eight requested mirror/rotation views and fixed 300 updates. All models start from scratch. Three folds hold out one entire acquisition and train on the other two. This is an explicitly different engineering protocol from the proposed five 20%/80% rotations because only three native US acquisitions exist in this collection. Normalization and sampled training windows use training acquisitions only.

Each held acquisition contributes one predetermined catalog-peak window and four seeded random windows with >95% observed coverage. Every branch predicts the same 128×128 window using its own context size, a shared coarse input, and overlapping Hann-weighted tiling. The combination is a fixed arithmetic mean; it is **not a learned merger**. Twelve checkpoints and 15 common held-acquisition windows were evaluated in about 39 seconds on the local RTX 5070. No downloads were needed.

## Honest baseline comparison

MAE in ppm·m; first average windows within each acquisition, then average acquisitions. Plume support means the spatially aligned vetted CH4PLM footprint. The remaining observations are not assumed to be methane-free.

| Method | All observed pixels | Vetted plume support |
|---|---:|---:|
| Constant crop | 360.76 | 504.68 |
| Bilinear | 279.12 | 322.76 |
| **Nearest-neighbor** | **264.25** | **308.74** |
| S16 | 266.57 | 304.43 |
| S32 | 265.44 | 303.47 |
| S64 | 264.78 | 301.15 |
| S128 | 264.36 | 298.36 |
| Four-branch mean | 264.07 | 298.41 |

The mean improves by 5.4% against bilinear, but only **0.07% overall against nearest-neighbor**. Its plume-support improvement against nearest is **3.35%**, with improvement on the two Texas acquisitions and a slight worsening in Georgia. S128 alone is practically tied with the mean and marginally better on plume support. These results do not yet justify four-model complexity. No statistical significance is claimed from three acquisitions.

## Visual review and sanity checks

Reviewed all three catalog-peak windows across every branch and all 12 seeded random windows. Native enhancement, both simple interpolation baselines, all four outputs and their mean are rendered with the same −500…5000 ppm·m range. Values above the range are clipped only for display, never for scoring.

- The Georgian spatial enhancement structure and the two weaker Texas features remain visible. Actual outputs vary spatially and are not saturated whole-crop classification masks.
- The models restore some contrast relative to bilinear, but the strongest simple baseline already recovers much of this. Small differences among S64, S128 and the mean do not warrant a strong claim.
- Fine retrieval noise cannot reliably be recovered after averaging; the reconstruction is smoother than the measured target. A visually smoother output is not automatically more physically accurate.
- No obvious regular tiling grid appears in the reviewed maps. A zero-residual tiling check verifies that all four context sizes reproduce the same shared bilinear input without gaps or seam-induced changes. Wider independent mosaics still require review.
- Colored footprints indicate measured support, not learned plume outlines. Missing regions remain blank; no unobserved pixel is called negative.
- The local viewer provides georeferenced observed target, actual prediction, nearest baseline, residuals, pixel inspection and common-scale comparisons. Every held window is available; no best-case filter was used. Display reprojection is separate from native-grid scoring.

## What this establishes and what remains

This is a working multiscale **EMIT-only** reconstruction implementation with real native-grid predictions and a small plume-region improvement over interpolation. It is not yet the intended Sentinel+EMIT upscaler: aligned Sentinel context was not established across all three native acquisitions. The earlier legacy-cohort Sentinel ablation actually worsened results. Only three independent acquisitions are insufficient for credible geographic generalization or training a convincing learned merger.

Next evidence needed: a larger geographically independent US native EMIT/Sentinel cohort with pairing/QA, and independent finer methane measurements for any 20 m validation. Gaussian emission-origin evaluation additionally needs verified source coordinates and wind. CH4PLM peak coordinates are acquisition/patch anchors, not emission-source labels. Existing background retrievals can test reconstruction errors but cannot establish detector false-positive rates without independent labels.

Do not start another large architecture sweep from this result. Preserve nearest-neighbor and S128 as strong low-complexity baselines, expand the native cohort, then test whether Sentinel or a learned merger adds reproducible value.

## Artifacts

- `outputs/GTM_Model6/GTM_Model6_native_multiscale_01/`: source manifest, code snapshots, twelve hashed checkpoints, fifteen native prediction archives, per-window/per-support metrics and summary.
- `GTM_Model6_native_catalog_peaks.png`: complete three-acquisition branch comparison, including nearest baseline.
- `GTM_Model6_native_all_windows.png`: complete fixed visual cohort.
- [Local actual outputs](http://127.0.0.1:8766/?experiment=GTM_Model6_native_uniform_mean#compare) · [Private remote viewer](https://desktop-v07a1pi.tail948ef9.ts.net:8766/?experiment=GTM_Model6_native_uniform_mean#compare).

Historical audit correction: the first filesystem search missed ignored native data in the sibling checkout. The native-data absence claim was corrected before concluding the research pass. Original grids for the separate 13-scene legacy cohort were not recovered; that earlier diagnostic remains labeled accordingly.
