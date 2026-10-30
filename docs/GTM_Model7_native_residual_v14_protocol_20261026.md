# Native residual correction, v14

October 26, 2026. Written before implementation, preparation or fitting. This is a separate bounded development study, not an additional arm or altered rule in v13.

## Question

Can a small native-grid corrector improve the retained v11 plume mask while keeping its useful detections? Does explicit pixel-class balancing help that same corrector learn rare plume pixels?

The v13 TRAIN-only audit found 0.83% plume pixels in pooled training support and 1.52% under balanced scene draws. Its first checkpoint matched a constant low-score predictor on a fixed TRAIN batch. Lower MBMP values have TRAIN rank AUC 0.737, so the input contains some discriminative signal. These observations support an optimization experiment; they do not establish causation or national performance. No new data or unseen-US success is claimed.

## Fixed sources and boundaries

Use the original 958 TRAIN / 198 historically exposed DEV scenes in `outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json`, SHA256 `1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a`. All 268 TRAIN plumes and 690 TRAIN clear scenes remain eligible. All 46 DEV plumes and 152 DEV clear scenes remain in selection and the viewer. Do not read protected final-test arrays or reopen v12 ASSESS for confirmation.

The fixed teacher is v11 update2400, checkpoint SHA256 `8a36587380ea14b86501b5bc245b3b26f82f52f380aa5e5e156900491ae39483`, selected cutoff 0.65. Use its already saved TRAIN and DEV native probabilities from the same checkpoint, verifying hashes against fit/inference receipts. Do not mix checkpoints. Preserve original labels and valid support exactly, as in v13. Reconstruct all raw native current/reference bands from the pinned original TIFFs; do not recover raw DN from clipped v13 features.

Reuse the v13 fixed balanced draw schedule, SHA256 `3efd3da71f6b1e80297903a145afdc329cde553b632d8e305a88d49fb36e5611`. No labels, site IDs, fluxes or source geometry enter the network. Source checks must confirm the teacher's fitting split matches this fitting split.

## Model and native features

Keep the teacher fixed. Its 20 m scores are repeated to the native 200 by 200 annotation grid. Convert them to logits after clipping scores to [0.000001,0.999999]. The final logit is teacher logit plus `6*tanh(raw_correction/6)`. The correction starts at zero, through a zero-initialized final convolution. Require initial masks to match the teacher on every scene at cutoff 0.65. The score difference may be at most 0.0000011 due to logit conversion and clipping.

The corrector is the v11 two-level U-Net structure: widths 16/32/64, paired bilinear decoder, two 3 by 3 convolutions with GroupNorm(4) and SiLU per block, one correction per native pixel. This avoids batch running-statistics dependence. Both arms start with identical seeded weights.

The 26 native channels are current six, reference six, current-minus-reference six, absolute difference six, a temporal SWIR contrast, and teacher logit divided by 6. For each date, use log1p of raw DN, subtract the valid per-band scene median, then apply the frozen v11 TRAIN center and scale, clipped to [-8,8]. The contrast is `-10*log(MBMP)` clipped to [-8,8], using valid-aware normalized B12/B11 ratios from raw DN, with a 0.0001 floor before log. Invalid feature pixels are zero. The teacher remains an input and a separate fixed output baseline. No DEV normalization is fitted.

## Two controlled fits

1. `residual_original_loss`: valid weighted BCE with positive weight 2, plus 0.1 pooled-positive Dice loss.
2. `residual_balanced_loss`: 0.25 times mean valid positive BCE plus 0.75 times mean valid negative BCE, plus the same 0.1 pooled-positive Dice loss. The means pool pixels within each batch; both classes must be present.

Both add `0.001*mean_valid(correction**2)`, so unnecessary departure from the teacher has a cost. No other difference between arms. Seed 20261026, full native scenes, batch 8 with four plume and four clear scenes, D4 on every spatial input including teacher scores, AdamW learning rate 0.0003 and weight decay 0.0001, gradient cap 1. Float32, no AMP or TF32, deterministic settings. Maximum 4,800 updates per arm, checkpoints 1,200/2,400/3,600/4,800, maximum 20 minutes per fit. Benchmark before fitting; stop if unsuitable. At most 8 GiB additional artifacts. Do not start either fit until both v13 arms and choices are frozen and root has reviewed this preflight.

Preflight must check original/cache/teacher identities, raw source reconstruction, exact labels and support, full-scene shape and D4, identical initial weights, all-scene initial teacher-mask identity, real positive/clear loss against independent NumPy for both arms, finite gradients and changed correction weights, checkpoint CPU serialization and actual warmed runtime. This is not a proof of generalization.

## Canonical inference, selection and replacement

Save canonical DEV probabilities using CPU inference from each checkpoint. Re-run them on CPU in a new process and require exact support, labels and scores before selection. Retain GPU training separately. Do not confuse cross-device float32 drift with a model-quality gain.

Use the same 22 cutoffs as v13 at all four checkpoints. The original selection guards remain P>=0.70, R>=0.30, clear-pixel FPR<=0.001, clear activation<=0.15 and half of positive observations recovering at least 10% of their labels. In addition, a replacement must have F1 at least 0.02 above v11, recall at least v11, no more than 18 activated clear observations, clear false pixels at most 1.10 times v11, and additional false-positive pixels no greater than additional true-positive pixels. Compute the v11 baseline from the exact same native labels and support.

Rank eligible replacements by F1, positive-site mean recall, positive-scene mean recall, lower clear FPR, earlier step, then higher cutoff. If no choice qualifies, retain the unchanged teacher at step0/cutoff0.65. Publish later failed diagnostics separately if needed to show learning, clearly marked as rejected. Never present a rejected mask as an improved model.

The useful target remains F1>=0.70, P>=0.70, R>=0.65 with the same clear-error guards. All figures and scores are development results; prospective paired US plume and clear imagery is still required for a claim about new states or national use. Freeze all choices before opening new candidate imagery. No extra fit, seed search, mask growth or case exclusions in this study.

## Visual and artifact checks

Publish actual predictions and original native imagery for every DEV scene, alongside v11. Inspect every plume, all US positives and the largest clear false alarms. Show counts of TP gained/lost and FP added/removed, not just marked area. Verify served assets, map/scene/model switching, error and score views, selected cutoff restoration and private Tailscale access. Preserve failed cases and source-bound verification artifacts.
