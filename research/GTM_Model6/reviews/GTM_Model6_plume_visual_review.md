# GTM_Model6 plume pilot visual and viewer review

Date: 2026-09-19. Decision: neither model promoted.

## Reviewed evidence

Root inspected EMIT_00.png and EMIT_01.png from each mean bundle, covering all nine eligible EMIT cases in both runs. A Luna reviewer inspected all five MARS positive pages and the first eight chronological negative controls for each run (see GTM_Model6_plume_MARS_visual_review.md). Every positive case is represented; the negative montage is a subset, with all 128 negatives available in the viewer.

The unconstrained model often responds to broad surface regions, field boundaries, roads and bright structures, with substantial missed EMIT support. The constrained follow-up reduces broad activation but leaves speckled edge/texture responses; it also loses much of the EMIT reference coverage. These are material failures, not a display-only problem. The catalog outline is supervision support; its unknown exterior cannot establish false-positive rates.

Root opened the actual viewer for MARS_2c6ef011-0729-42e2-b64f-10ff8207778b in the first mean run and emit25km-0046 in the constrained mean. The selected MARS case shows a localized elongated prediction overlapping the reviewed outline, with native IoU54.87% versus display-grid IoU55.9%. This deliberately selected success does not represent the cohort. The EMIT example shows substantial missed support and visible exterior predictions. Mean experiment switching retained the EMIT scene.

## Display verification

All13 bundles have173unique scenes with existing RGB, mask, truth and grid files and ordered geographic bounds. All117EMIT display entries have positive-only truth with unknown exterior, and nonzero visible predictions outside that truth. Input validity and reference validity are separate. Display reprojection uses the same EPSG:3857grid for RGB, prediction and reference. Native metrics remain frozen in the run report; display threshold changes are exploratory.

The browser renders actual georeferenced imagery and overlays, correct0.50 artifact threshold, model hotspot markers and shared comparison controls. EMIT summary now says support covered, exterior unknown, precision/IoU not measured, origin unverified. MARS summary explicitly says Display IoU. Candidate points are score peaks inside predicted components, not fitted/verified emission origins.

Five workspace tests and JavaScript syntax validation passed. Four plume behavior tests separately passed (unknown-label loss masking, multiplicative-scale feature invariance, shared evidence through multiscale tiling, rejection of fabricated context). See the separate replay verification for saved-weight and split checks.

## Remaining scientific limits

Neither run demonstrates national detection, quantitative methane enhancement upscaling, or source localization. Two independent connected groups carry all36reviewed positives; rotations add no new independent observations. More independent time-matched positives and representative negatives, quantitative EMIT target rasters for upscaling, and a compatible common-scene capstone baseline comparison are the next useful evidence. Training was stopped after two bounded failures. No imagery was downloaded.
