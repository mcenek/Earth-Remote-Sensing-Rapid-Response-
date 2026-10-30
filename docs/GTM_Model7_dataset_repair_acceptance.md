# Real dataset repair and visual acceptance, October 24, 2026

This failure list and end-to-end acceptance plan precede the new dataset exporter and result renderer. The user has authorized reuse of cached real imagery, reference-image checks, retraining both temporal architectures, and repeated visual assessment. No new imagery downloads are needed for this pass.

## Failures to prevent

1. More dates or crops at the same facility are counted as more independent sites. Report observations, sites, geographic components, states where known, and positive pixels separately.
2. An old test is silently reused as a new independent test. Record historical exposure. The October 23 confirmation is consumed and can only serve as development or regression evidence. An official release split repurposed for fitting invalidates official benchmark claims.
3. A physical site, nearby overlapping footprint, current acquisition, reference acquisition, or identical input crosses fitting, development, and final evaluation. Group before sample selection and augmentation; report unavoidable limits instead of weakening rules silently.
4. Development has no positive examples. Require both real positive and reviewed negative scenes in development and reserve geographic groups before fitting.
5. A facility coordinate, missing annotation, unknown exterior, or almost-full classification patch is presented as a verified methane plume boundary. Keep MARS, EMIT positive-only support, and MethaneS2CM label contracts distinct. Never invent a negative mask from a missing asset.
6. Sentinel L1C, Sentinel L2A, Landsat, 32-pixel crops, and 200-pixel scenes are pooled without a documented sensor and resolution contract. A separate matched experiment is preferable to an unjustified conversion.
7. A reference is called clear or plume-free because its spectral values are finite. Record the exact supporting acquisition, reviewed observation label, grid, cloud mask, and date when available. Flag unknown reference quality. Do not fabricate a cloud mask.
8. Reference selection uses model scores or held-out prediction quality. Freeze its metadata and image-quality rule before fitting. Reject poor inputs independently of predictions and preserve every exclusion.
9. Source arrays, normalization, checkpoint, cutoff, or support differ between scoring and the viewer. Hash sources; fit normalization on training only; preserve actual probability arrays; independently reconstruct native counts and original inputs.
10. A model cannot learn fixed real training examples but proceeds to expensive training. Require a small-crop learning check for both models in the same runtime before the main run.
11. A low false-alarm score hides an all-negative model; high recall hides an all-positive model. Include constant controls, precision, recall, overlap, per-site results, and no-plume false alarms.
12. Repeated tuning consumes the final evaluation set. Iterate on development only. Freeze both selected checkpoints and cutoffs before final inference. Preserve the consumed final set thereafter as regression evidence.
13. Nice examples conceal failures. Render all evaluation cases, with current and reference inputs, delivered labels, both saved predictions, fixed-scale scores, and errors. Keep missed plumes and no-plume false alarms selectable.
14. Success on a few methane monitoring sites is reported as nationwide capability. State geographic and source-type coverage and do not claim all-state detection without evidence.

## End-to-end artifact

Save a versioned dataset manifest, provenance and reference-quality receipts, split/exclusion summary, training histories and checkpoints, fresh-process inference replay, independent native-score recomputation, and a local HTML comparison. Publish both models' actual predictions in the existing viewer. Root personally inspects the visual results, checks positive and negative browser states and model switching, and saves screenshots with a repeatable verification record. Do not replace prior evidence or promote a failed run as a working detector.

## Larger real-data bundles, before viewer change

The repaired geographic splits exceed the viewer's former 1,000-scene cap. Raise the explicit frontend and backend cap to 5,000, while retaining the 50 MiB manifest limit, path checks, duplicate-ID checks, finite-grid validation and selected-scene asset loading. Failure cases are a split that silently truncates, a bundle accepted by only one validator, a scene switch that keeps another scene's prediction, or a slow refresh that prevents inspection. Verify a complete real split in the running browser, switch between positive and negative cases and between models, and save the loaded scene count and screenshots. Do not generate synthetic model results to fill the viewer.
### Result-status display check

Before the final viewer export, check that a model which passed development checks but failed geographic evaluation cannot look ready for statewide use. The evaluation page and experiment label must state that failure. Calculate this display status from the frozen prediction counts using the existing precision (0.50), recall (0.30), IoU (0.20), and no-plume pixel false-alarm (0.001) limits. This is a reporting check, not a new checkpoint or cutoff selection step. Keep every failed case visible.

### Overview artifact check

The overview must derive counts from the frozen manifests and read all numeric comparisons from saved calibration and verified inference receipts. Development and geographic evaluation must be separate tables. Include links to both models, real examples, false alarms, reference limitations, and the exact reproduction commands. Check all local report and image targets. An overview must not present a failed model as ready, confuse pixel false alarms with scene detections, or describe a historically consumed evaluation as new independent validation.

### Live registry responsiveness

The 77-run catalog now includes four 1,982-scene bundles. The browser showed a temporary backend-unavailable banner during full asset revalidation. Before changing the registry, measure the same running catalog with repeated API calls spanning refresh intervals. Preserve all asset/path validation and last-valid-publication fallback. Refresh a warm snapshot in the background so requests can use the last completed snapshot. Guard against overlapping refreshes and partial publications. Repeat the same HTTP workload after the change, check both large-model scene lists and actual imagery in the browser, and save latency receipts. First startup may still require a full validation.

