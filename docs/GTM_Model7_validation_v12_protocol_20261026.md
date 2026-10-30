# Missed-plume study with held-out validation

Written before new split arrays, normalization, implementation or fitting, October 26, 2026 UTC (October 25 local).

## Question and limits

Joshua asks for plume F1 above 70%, without simply marking more area, a validation set, and actual predictions in the viewer. V11 has exposed-development precision 71.48%, recall 42.19% and F1 53.06%. It misses 19 of 46 entire plumes. These historical sites have been used for development. No certifiably unused paired positive cached sites remain.

This study creates validation that is withheld from fitting in every new comparison model. All original observations have historical exposure, so this is a new-run held-out sanity check, not pristine independent validation or evidence of US-wide accuracy. Old checkpoints and the old normalizer cannot initialize or preprocess the new fits. Previously consumed final-test arrays remain untouched. No downloads are needed.

## Metadata-only split fixed before pixels

Use only the 958 original TRAIN observations from manifest SHA `1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a`. Keep the original 198 DEV observations as an exposed regression set, unavailable to checkpoint/cutoff selection in this study. Do not silently call those observations a new test set.

Build connected components across physical-site identities. Link sites if they are within 25 km, share any target/reference acquisition identifier, share target/reference observation identity, or share a spectral image source hash. Do not link by identical cloud-mask hashes because clear masks can legitimately be identical at distant sites. All dates and both classes at a site stay together. No acquisition, spectral hash, observation identity or 25 km cross-split neighbor is allowed.

Assign whole components to FIT, selection validation and locked assessment using a deterministic metadata-only algorithm, seed 20261026, with approximate observation budgets 70%/15%/15%. Large components can force uneven counts; save actual counts and country/site/component membership. The largest component stays in FIT. Each holdout must contain at least 20 plume and 40 clear observations from at least five components. If this cannot be met without leakage, stop and report it. Pin the exact assignment algorithm and source-bound membership receipt before reading arrays or fitting. Do not choose a partition based on old or new prediction scores. Never move poor cases out of validation later.

Resolve array paths without copying or mutating source arrays. Preserve source hashes and native support. Recompute all feature normalization on FIT only. Save normalization member IDs and hashes so validation pixels cannot enter its statistics.

Metadata-only assignment specification, fixed before implementation: sort components by descending observation count, tie by SHA256 of `20261026:` plus minimum physical-site ID. Force the largest component to FIT. Use random.Random(20261026) for 30,000 candidate assignments of the remaining sorted components: SELECT if a draw is below .23, ASSESS if below .46, otherwise FIT. Reject an assignment if either holdout has fewer than five components, 20 plume, 40 clear or one clear-only site, or is outside 100..210 observations. Minimize, across the two holdouts, `2*abs(observations-144) + 1.5*abs(plume-40) + 4*max(0,5-country_count) + 5*max(0,2-clear_only_sites)`. Keep the first assignment on a tie. This is metadata stratification, not prediction-based split selection. The read-only audit found trial index 29,450 gives FIT 670 (188 plume/482 clear), SELECT 144 (39/105), ASSESS 144 (41/103). The implementation must reproduce these counts or stop for review before pixels. Pin IDs and actual separation checks in the split receipt.

## Bounded matched experiment

First audit FIT labels and source arrays only. Compare native plume size to any-positive 20 m targets, quantify extra area implied by repeating those targets to 10 m, and compute the ideal 20 m block-constant representation ceiling. This diagnoses label-grid effects without promising they explain whole missed plumes.

At most three fresh fits, each 4,800 updates, checkpoints 1,200/2,400/3,600/4,800, maximum 30 minutes each. No seed, architecture or loss-weight sweep. Stop on numerical, split or provenance failures.

1. Control: v8 24-channel early-fusion U-Net, 121,937 parameters, any-positive 20 m target and original pooled BCE positive weight 2 plus 0.1 pooled positive Dice. Site-uniform sampling within each class, then observation-uniform. Four plume and four clear observations per batch, full 100x100 input, D4. Recompute FIT normalizer. No hard pool mined from old models that saw validation.
2. Native-label loss: exact same model, initial tensors, draw/augmentation schedule and preprocessing. Replace any-positive training targets by the fraction of the four native 10 m labels inside each supported 20 m block. BCE on these fractions equals BCE on repeated predictions against native binary labels. Apply the same positive weight 2 and pooled Dice coefficient 0.1, with the fractional target used consistently. This removes the implicit label expansion; it does not create finer input information. Independent NumPy and expanded-native calculations must agree before fitting.
3. Conditional ratio feature: only if FIT audit finds valid finite ratio features on all FIT inputs, reuse arm 2 and add one validity-aware, scene-normalized B12/B11 current/reference ratio feature, computed directly from raw bands with the existing MARS adapter's `compute_mbmp`. Use log ratio with lower clamp 1e-4, train-only mean/std, clip standardized features to +/-8, invalid zero. Same model widths and schedule; only first-layer input count changes. Report this as an additional feature test, not a new physical methane concentration measurement. Earlier spectral features failed; this arm has no promised benefit.

All fits use seed 20261026, AdamW learning rate 0.0005/weight decay 0.0001, gradient clip 1, float32, no AMP or TF32. Save all model/source/protocol/schedule/normalizer hashes. Never edit historical trainers or receipts. A fit must not open locked-assessment or old DEV arrays. Score only FIT and selection validation during fitting.

## Failure modes and E2E checks before fits

Reject cross-split sites, proximity links, acquisition/observation/hash overlap, absent classes, invalid source/array hashes, malformed shapes, unsupported label values, missing support, nonfinite inputs/loss/gradients and normalization that includes non-FIT IDs. Validate D4 alignment across inputs, fractional/native targets and support. Check architecture/initialization parity for arms 1 and 2, common initialization tensors for arm 3 except its input convolution, unchanged schedules and both-class participation. Check exact CPU save/load predictions and an actual finite optimization step with changed weights on a real balanced FIT batch. Write these controls and receipts before sustained training. Do not add unit tests after implementation.

## Selection, locked assessment and useful target

Use only selection validation for all choice. For each arm inspect exactly 22 cutoffs: 0.05 through 0.95 in steps of 0.05, then 0.975/0.99/0.995, at the four milestones. Compute pooled native TP/FP/FN/TN, precision, recall, F1, positive-site recall, whole-plume misses, half-recovered observations, clear pixel FPR and clear observation activation. Include every clear observation in pooled counts. No scene accuracy or all-background pixel accuracy headline.

Eligibility: precision >=0.70, recall >=0.30, clear pixel FPR <=0.001, clear observation activation <=15%, at least half plume observations recover >=10% of labels. Rank eligible candidates by F1, mean positive-site recall, mean positive-scene recall, negative clear FPR, earlier update, then higher cutoff. If no eligible candidate, freeze update1,200/cutoff0.70 as an explicitly failed diagnostic per arm. Save unconstrained best as a rejected diagnostic with failure reasons; no relaxed gates. Choose the cross-arm winner by the same eligible ranking, otherwise retain historical v11 and do not promote a failed fallback.

Freeze all per-arm choices, the cross-arm winner and source hashes BEFORE opening locked-assessment or old DEV arrays. Then evaluate each frozen arm once on all locked-assessment observations and all 198 exposed regression observations. Do not choose a different model/cutoff from their results. Save every prediction and native count. Independent fresh-process CPU replay must match scores within 1e-6 and exact support/native counts.

The useful target is F1 >=0.70, precision >=0.70 and recall >=0.65, with clear pixel FPR <=0.001 and clear activation <=15%, on locked assessment as well as selection validation. Report site and observation counts next to every score. A smaller marked area with fewer hits is not enough; a larger area with excessive false pixels is not enough. Do not substitute a high training score, country subgroup or precision alone. Validation history limits still apply even if thresholds pass.

## Visual acceptance and viewer

Root personally reviews all locked-assessment positive cases, its largest clear false alarms, representative improvements/regressions on the old 198-scene regression set, and all completely missed new validation cases. Actual saved scores, masks, annotation contours, false pixels and missed pixels must be available. Preserve every difficult observation. No dilation, filling, flooding, component deletion or label-shaped crop.

Export separate clearly named validation/locked-assessment and old exposed-regression views. Show whether a score selected weights or was measured after freezing. Keep source reconstruction, mask/score/native-count agreement, browser model/scene/display switching and private Tailscale links verifiable with repeatable artifacts. No external journal writes/messages or Git push are part of this study.
