# Cached pretrained models: fixed comparison

Written October 26, before implementation or inference in this comparison. No model is trained, downloaded or selected here. This study does not change v13 or v14.

## Purpose and exposure

Compare the retained v11 detector with two publicly released methane detectors already on disk: MARS-S2L and CH4Net. Their larger training sets may provide useful inference candidates for future US imagery. Their predictions must first be visible and reproducible.

Use only the same 198 historically exposed development observations, 46 plume and 152 provider no-plume. Manifest SHA256: `1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a`. No protected final-test arrays or v12 ASSESS arrays may be opened. Released training-site exposure must be reported per model. These are descriptive comparisons of externally trained models, not independent validation, proof of transfer to new US states or a fair training-data-controlled architecture contest.

## Fixed checkpoints and inputs

- Retained v11: update 2400, cutoff 0.65, checkpoint SHA256 `8a36587380ea14b86501b5bc245b3b26f82f52f380aa5e5e156900491ae39483`. Use its verified saved scores unchanged.
- Released MARS-S2L: local `MARSS2L_20250326/best_epoch`, SHA256 `be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246`; config SHA256 `abeb92d01313fbb2939e6c5fc1c6281846b8102ea5edd7081668fe0db05bf79f`. Fixed pixel cutoff 0.5.
- Released CH4Net: local `CH4Net_20250329/best_epoch`, SHA256 `fbcdcad062fa199d7c66631f3607eb79d92f81650fbac21ce64332f2ad2b7a34`; config SHA256 `aa0d24567d650b6fbf2bb23c95ceebe4533bde2ac66416bb059294cd314cb1b5`. Fixed pixel cutoff 0.5.

Use the source-checked current and separately selected earlier-reference TIFFs from this manifest. Do not silently substitute the background bands embedded in the current TIFF. Both paired models use this same selected reference. This reference choice differs from the original release benchmark and must be stated. CH4Net uses the current six bands only.

Reproduce the released input builder and network in `tools/evaluate_released_marss2l.py`, without calling its cohort-evaluation entry point. Construct six current plus six selected-reference bands as float32 DN/5000 clipped to [0,2]. For MARS-S2L, prepend the released MBMP ratio with the release's nonzero-B11 normalization, append clipped raw wind divided by 8, and append the current CloudSEN12 mask greater than zero. Use the current cloud mask, not the union of the two dates. Do not use v13's validity-aware MBMP cache as if it were the released input. CH4Net receives the six scaled current bands. No labels or methane enhancement enter inference.

Both networks are inference-only, with BatchNorm in evaluation mode. Load the hash-pinned weights with weights_only and reject any missing parameters or surplus outside the documented unused out_mlp branch. Original source TIFFs, cloud masks, wind metadata and all helper files must be hash-bound. Evaluate on exactly the original common native labels and validity support. Do not score missing pixels as negatives.

## Verification and fixed reporting

Before inference, verify checkpoint/config/source identities, model shapes and finite outputs, and independently reconstruct the input on real source cases, including invalid edges. Confirm CH4Net input equals the current-band slice of the MARS-S2L spectral input. Compare the independent implementation with the released builder. Check expected active parameter counts and serialization. Benchmark warmed CPU inference before running the complete set.

Canonical probabilities use CPU float32, deterministic settings, no AMP or TF32. Run each full inference once and replay in a fresh process. Require exact probabilities, labels and validity. Maximum ten minutes per model per pass and two GiB of new artifacts before publication. Stop if unsuitable. No GPU fallback, checkpoint adjustment, fitting or repeated search is allowed.

Use pixel cutoff 0.5 for both releases with no tuning. The publisher uses >=; require that no saved valid score equals exactly 0.5 so the release's > convention gives the identical mask. Preserve raw scores and binary masks, with no smoothing, dilation or component removal. The separate published scene rule may be described as any 8-connected component with at least 100 pixels, but must not be confused with the unfiltered pixel metric.

Report native TP/FP/FN/TN, precision, recall, F1, clear-scene activation and clear-pixel error over every observation. Country groups are descriptive only. No model can be promoted as independently validated from this comparison, even if its descriptive F1 exceeds 70%.

Root must reconstruct source features independently, inspect actual predictions for all 46 plumes, the largest clear false alarms and all three US positives, and verify the served report and map. Show all models in the same viewer with clear external-training and reference-choice notes. Do not hide misses or label an already exposed observation a new test.
