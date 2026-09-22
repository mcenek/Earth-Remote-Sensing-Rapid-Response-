# Model6 Sentinel-only plume experiment: GTM_Model6_plume_evidence_02

Decision: NOT PROMOTED. Actual held-location predictions exist; nationwide source detection is not demonstrated.

173 US examples: 36 reviewed MARS plume scenes, 128 reviewed no-plume controls, nine EMIT catalog-positive cases. Sixteen connected location/acquisition groups are kept intact. All 36 reviewed plume scenes occur in only two connected groups (33 and three scenes), so this is a small, imbalanced transfer diagnostic, not fresh confirmation.

Four independently trained compact U-Nets use real 16/32/64/128 contexts within 200x200 Sentinel target/reference crops. Bands B2/B3/B4/B11/B12 form 16 spectral/temporal contrasts. Eight training transforms: 0/15/30/45 degrees, with and without mirroring. No EMIT values, polygons, label masks, source coordinates, or CAFO locations enter model inputs.

EMIT trains positive-only BCE. Its exterior contributes no negative loss. Reviewed MARS data trains BCE plus Dice, including negative controls. Five outer held-group partitions have 82/67/9/8/7 examples. Remaining partitions train. Fixed 600 optimizer steps and cutoff 0.50; no outer-held checkpoint or threshold selection. This uses ordinary group-held folds, not Martin's proposed 20% train/80% stress rotation. No learned merger, quantitative enhancement head, or Gaussian emission-origin head was trained.

Evidence follow-up: Scene-wide robust temporal B12/B11 decrease supplies a fixed baseline logit (z minus 3). The U-Net can adjust it by at most +/-1.5. This was a targeted post-hoc response to first-run surface-pattern false positives; both runs use already inspected development data.

| Method | MARS pixel precision | Recall | IoU | Mean positive-scene IoU | Negative pixels flagged | EMIT positive-support covered |
|---|---:|---:|---:|---:|---:|---:|
| S16 | 2.41% | 32.86% | 2.29% | 8.22% | 7.98% | 9.06% |
| S32 | 2.41% | 32.91% | 2.30% | 8.22% | 7.96% | 8.94% |
| S64 | 2.60% | 32.82% | 2.47% | 8.28% | 7.25% | 8.49% |
| S128 | 3.18% | 14.60% | 2.68% | 7.71% | 2.46% | 4.69% |
| mean | 2.89% | 25.61% | 2.67% | 8.70% | 5.06% | 7.28% |
| spatial_prior | 2.53% | 0.01% | 0.01% | 0.00% | 0.00% | 0.00% |
| all_positive | 0.60% | 100.00% | 0.60% | 2.76% | 100.00% | 100.00% |
| all_negative | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| spectral_baseline | 3.48% | 10.20% | 2.66% | 6.38% | 1.72% | 1.51% |

The MARS aggregate above pools evaluated pixels. Negative-pixel coverage is averaged over the 128 reviewed no-plume scenes; it is not scene-level FPR or a national prevalence estimate. These controls are deliberately selected and not a representative US survey.

EMIT exterior remains unknown. The raw results retain explicitly labeled catalog-footprint agreement proxies, but these must not be quoted as verified methane precision, false-positive rates, or ground-truth IoU. The UI reports coverage of positive EMIT support and keeps exterior predictions visible. Large positive-support recall can still be achieved by drawing an entire box.

Source file sampling is 10m, Sentinel SWIR nominal resolution is 20m, and EMIT is nominally 60m. No claim of measured 10m methane detail is supported. Dates differ by up to several hours in the EMIT cohort; plume motion and intermittency limit exact cross-sensor boundary comparisons.

Candidate hotspot points come from connected predicted components of at least 100 source-grid pixels, placed at their highest predicted score. They are model-proposed plume peaks, not verified emission origins. They are not snapped to CAFO/oil locations and must not be treated as confirmed sources.

Visual review: full EMIT cohort and all reviewed positive cases are exported in paginated contact sheets. All 173 held cases, including every negative, are selectable in the viewer. No successful case is presented as representative. Some localized plumes appear; background false detections and EMIT misses are material failures.

Next useful evidence: more independently located US positive plumes with close Sentinel/EMIT timing and complete spatial coverage, representative reviewed negatives, and source/wind labels for origin localization. Restore and compare the exact capstone checkpoint on compatible common scenes. Do not expand into a nationwide scan or another architecture sweep from these results.

Artifacts: 20 hashed checkpoints, 173 held native prediction archives, input/source manifest, exact trainer source snapshot. Training/evaluation wall time 163.5 seconds. No imagery downloaded.

Verification note: These are short fixed-600-step development pilots, not exhaustive optimization or proof that the architecture cannot improve. The failure decision limits promotion and larger spending; it does not establish methane inference is impossible. Root reviewed every EMIT mean output; a Luna review covered every MARS-positive mean output plus eight chronological negative controls per run. See research/GTM_Model6/reviews/GTM_Model6_plume_visual_review.md and GTM_Model6_plume_MARS_visual_review.md.
