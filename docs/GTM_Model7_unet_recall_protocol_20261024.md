# Bounded U-Net recall diagnosis and improvement

Written before new training or final-cohort pixels are opened. User requests improved plume recall and genuinely new-site evaluation, with actual predictions in the viewer. This follows the completed centering pilot and explicitly authorizes a new bounded experiment rather than reopening its stopped search.

## Failure cases to prevent

- Calling the consumed 55-scene external cohort a new test; evaluating a new site already used in a previous project fit, prediction or visual review; accepting a different date at the same site as independent.
- Selecting final sites by how well their images or predictions look; replacing misses or hazy pairs after seeing results; using final labels for checkpoint, cutoff, architecture or feature choices.
- Training against facility coordinates as methane labels or representing binary MARS labels as quantitative EMIT upscaling.
- Counting resampled display pixels instead of original annotation pixels; hiding no-plume false detections; discarding invalid pixels differently across models.
- Altering old frozen source files, their normalizers, weights, datasets or previous output folders.
- Mistaking a high cutoff for the entire cause of misses; mistaking training recall or the six exposed DEV plumes for evidence of national generalization.
- Pooling away small plumes in loss, diagnostics or metrics; falsely attributing the joint effect of a changed loss to just one term.
- Mismatched training/inference preparation, threshold or model construction; invalid/all-background loss denominators; nonfinite gradients; shape mismatch for 100x100 scenes; source drift or silent dependence on unrecorded code.

## Diagnosis and cohort reservation

Audit the selected centered v5 U-Net checkpoint at step 3600 and cutoff 0.95. Analyze existing TRAIN and exposed DEV scores by plume size, per-scene score distribution, cutoff, reference age and temporal spectral contrast. The old external results may explain historical limitations but cannot select a new setting. Examine actual misses and false detections. Training labels are allowed for TRAIN-only diagnostics; no new final pixels or predictions may influence changes.

Reserve a deterministic, metadata-first cohort from cached real Sentinel-2 pairs before training. Exclude every known prior training, DEV, TEST, external inference and visual-review site, observation, acquisition, spectral hash and references, plus 25 km spatial proximity. Historical metadata-only inventory presence must be disclosed separately from actual model evaluation. Country/site round-robin selection, up to 20 sites and up to two plume/four explicit no-plume observations per site. Use the previous fixed same-grid, earlier-date reference and source-QA rules. Failed source checks stay recorded with their reason; no post-result replacement. Freeze IDs and selection order before source-pixel QA. Freeze its successful source manifest before final model inference. If genuinely new cached sites do not remain, state that and do not pretend that a consumed set is new.

## Training envelope

Keep the 958-observation TRAIN collection, the exposed 86-scene DEV collection, scene-centered log preprocessing, TRAIN-only normalizer, balanced domain/class draw sequence, D4 transforms, optimizer, learning rate, seed, batch size and 4800-update cap unchanged. Train at most two U-Net candidates, each with milestones 1200/2400/3600/4800. Preserve frozen v5 as the baseline. Do not retrain the residual control. Stop for nonfinite values, split/source failure or the existing 1200-second wall-time cap per fit.

Candidate details and the final selection objective will be recorded in an addendum after TRAIN/exposed-DEV diagnosis and before implementation/fitting. No candidate details can depend on new final imagery or results. The existing precision, recall, IoU and no-plume pixel false-alarm gates must remain. Any additional scene-balanced recall objective must be defined before fitting. Keep all candidates and failures visible. No further training or cutoff tuning is allowed after final inference.

## Pre-fit addendum: two candidates and selection

The selected v5 step-3600 training predictions contain 271 native connected plume components. Of these, 92 have no pixel at score 0.50 and 151 have no pixel at 0.95. In the 65-256-native-pixel bin only 5/17 reach 0.50. The exposed DEV set has six components and none below 257 native pixels. These observations motivate training changes and show why cutoff reduction alone is insufficient. Component counts are descriptive, not independent detection validation. The audit is `outputs/GTM_Model7_unet_recall_20261024/audit/audit_summary.json`.

Train two candidates only:

1. `scene_dice`: unchanged temporal U-Net; unchanged masked weighted BCE (positive weight 2). Replace pooled positive-batch Dice with the mean Dice loss computed separately for each positive scene, and increase its coefficient from 0.1 to 0.5. This jointly changes scene weighting and Dice strength; attribution to either alone is not established. Explicit no-plume scenes continue to contribute BCE over all valid pixels.
2. `delta_scene_dice`: identical loss and backbone, with a zero-initialized, bias-free 1x1 linear skip from the six signed, standardized current-minus-reference bands into the output logits. This bypasses encoder ReLU and pooling for weak temporal evidence. It adds no labels or external statistics. Zero initialization preserves the baseline backbone's initial predictions. The candidate factory must reconstruct the selected arm from frozen metadata; it must not infer it from the evaluation data.

No other architecture or data changes. Both use the exact saved v5 38400-draw schedule and normalizer. Both start from the same seed and initial U-Net backbone, not from a checkpoint selected using the old external cohort. Reuse the old residual CNN as an explicitly retained control without optimizing it again. Preserve its source checkpoints, hashes and counts when exporting paired viewer bundles.

For each candidate evaluate the four predeclared milestones on exposed DEV and the existing fixed cutoff grid (0.05 to 0.95 by 0.05, then 0.975/0.99/0.995). Keep the existing gate: native pooled precision >=0.50, recall >=0.30, IoU >=0.20, no-plume pixel FPR <=0.001 and at least half the positive scenes recovering >=10% of labeled pixels. Among eligible candidates including frozen v5, select by mean per-positive-scene native recall, then pooled IoU, then lower no-plume FPR, then earlier update, then larger cutoff. This objective targets misses without relaxing the false-alarm gate. Record baseline selection under its already frozen cutoff. If no new candidate improves this objective while passing the gate, retain v5 and label new runs failed development evidence. No cutoff selection on final data.

Freeze the selected candidate (or retained baseline), its source dependencies, cutoff and checkpoint before opening the reserved final pixels/predictions. Run frozen v5 baseline and the selected candidate on identical new-site plume/no-plume pairs only once after source-QA preparation. Keep the retained residual control too if useful for the existing viewer. Do not fit again based on final outcomes. Report baseline-versus-candidate improvement or regression at native resolution, with per-site and per-scene recall, IoU and false alarms. Final uncertainty remains high for a small convenience sample.

## Verification and publication

Independently reconstruct preprocessing and loss reductions from real arrays. Check valid/empty support, positive/negative scene handling and finite gradients before full fitting. Verify checkpoint/source hashes, training draw identity, untouched final arrays during fit and exact CPU inference replay. Compare baseline and selected candidate on identical final sites, same source validity and native annotation grid. Report pooled precision/recall/IoU, scene-balanced plume recall, fraction of plume scenes recovered, no-plume pixel false alarms and no-plume scene activation. Include per-site results.

Publish all final scenes and actual saved probabilities/masks, not hand-drawn examples. Check original/reference/annotation/prediction/error views and the georeferenced viewer personally. Keep missing plumes and false detections selectable. Save source reconstruction, HTTP asset/mask-count receipts, browser interaction receipts and screenshots. Report whether improvement transfers; if it does not, stop and report the failed experiment without a national-use claim.
