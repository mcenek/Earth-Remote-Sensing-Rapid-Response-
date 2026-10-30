# Native v13 cross-device verification amendment

October 26, 2026. Written after the zero-wind diagnostic and before the wind diagnostic or scientific selection. The original training protocol and source stay unchanged. This document changes only the cross-device numerical check.

## Observed issue

The original verifier required an absolute CPU/GPU score difference at most 0.000001. The four zero-wind checkpoints differ by up to 0.00001448 in float32 inference. A repeated CPU forward is exact. Source labels and valid support are exact. Across 22 cutoffs, each checkpoint has one to five pixel disagreements in its entire 198-scene set. All these differences concern false-positive pixels; true-positive counts are unchanged. The fallback cutoff 0.70 has identical masks on CPU and GPU at every checkpoint.

This is a numerical verification failure, not evidence that a model passed its quality checks. Retain the original failure and diagnostic receipts. Do not change weights, inputs, scene membership, losses, cutoffs, ranking, performance gates, or saved GPU predictions to obtain a pass.

## Replacement verification rule

Use a separate verification module with its own source hash. Require all of the following before it writes an accepted replay receipt:

1. Verify the original source, protocol, cache index, all native caches, checkpoint and saved prediction hashes.
2. Recompute all 198 CPU score maps at each of the four checkpoints. Require finite scores in [0,1], the exact 200 by 200 shape, and pointwise absolute difference at most **0.00005**, with no relative tolerance. Repeat the first CPU case at each checkpoint and require exact scores.
3. Require exact source labels and valid support. Report mask disagreements and TP/FP changes for every one of the 22 fixed cutoffs, rather than hiding them in a mean error.
4. Apply the original selection procedure separately to CPU and saved GPU scores. Both must produce the same eligible checkpoint/cutoff or the same failed fallback. Their selected masks must be identical on every observation. Require matching eligibility counts, selected-arm ranking, and the frozen usefulness decision. Do not accept a different scientific choice as harmless roundoff.
5. Bind the diagnostic, this amendment, the separate verifier source, original trainer source, fit metadata and checkpoints in the accepted receipt. State that scores agree within tolerance, not that CPU and GPU arrays are bit-identical.

The absolute tolerance is 0.005 percentage points of a score. Acceptance also requires identical selected binary predictions and scientific choices, so this change cannot improve the published F1 or relax a false-alarm guard. Any failure stops acceptance and selection for review. No new training is authorized by this amendment.
