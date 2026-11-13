# Evaluation preparation before imagery

The remaining high-value software preparation is measuring the matched comparison consistently. `EarthRemoteSensingRapidResponse/temporal_attention_evaluation.py` now supplies score-level calibration and paired evaluation plus per-scene dense metrics. It does not read imagery, fit models or open historical outcomes.

## Operating-point contract

`calibrate(bundle, max_fpr=...)` accepts only a bundle explicitly marked `validation`. Both candidates share the same rows and inference QA. Each candidate maximizes recall within the specified **empirical scene-level** false-positive rate, with ties preferring fewer false positives and then a higher threshold. Predictions use `score >= threshold`; a tied score group cannot be partially admitted. An operating rule with no positive detections becomes explicit `reject_all`, not a JSON Infinity threshold.

Calibration records its input hash, sample IDs, geographic group IDs, negative count and false-positive-rate step per negative scene. This is neither probability calibration nor a population risk guarantee. Repeated crops at one location do not constitute independent evidence about rare false alerts. Group-balanced risk control already exists elsewhere in the repository and is a different estimand; this utility does not silently substitute it for scene-level FPR.

`evaluate_pair(bundle, calibration, ...)` accepts an explicitly marked `test` bundle and applies the supplied rules without tuning. It rejects sample-ID or geographic-group overlap with validation. The caller remains responsible for truthful split/group metadata, file/model provenance, final protocol freezing and controlling access to real held-out outcomes. These in-memory utilities are not a mechanism that enforces one-time test access or audits renamed observations across raw files.

## Metrics and uncertainty

Scene AP uses the existing scikit-learn dependency. AP weights precision by successive recall increments rather than using trapezoidal precision-recall area; this matches the [official average-precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Paired bootstrap draws resample whole supplied geographic groups with replacement and use exactly the same multiplicities for both models. This preserves pairing; [SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) likewise describes shared indices for paired resampling. The geographic grouping and percentile implementation here are project choices, not a claim that SciPy automatically establishes spatial independence.

The utility reports percentile intervals for attention-minus-CNN AP, fixed-rule recall and fixed-rule FPR. These are conditional on the fitted models, frozen operating rules and sampled locations; they omit training and calibration uncertainty. Both model seeds and geographic grouping must still be reported by the eventual experiment.

If the evaluation contains one class, AP is null. Recall, FPR and precision with zero denominators are also null. If any bootstrap draw loses a class, the interval is withheld and the number of degenerate draws is reported. This deliberately avoids quietly replacing undefined results with zero or reporting an interval conditional on only surviving draws. Sparse cohorts may therefore have point estimates without a usable interval.

Dense metrics require separate annotation availability and boolean sensor QA. Missing annotation or no valid pixels makes a scene ineligible. Nodata outside support is ignored. An all-background scene with no predicted plume has undefined IoU (null), rather than contributing a perfect score. Pixel confusion counts are retained for eventual aggregation. The dense threshold is supplied explicitly and must also be frozen before real test evaluation.

## Executable rehearsal

```powershell
$env:PYTHONPATH = (Resolve-Path '.venv/Lib/site-packages').Path
python tools/rehearse_temporal_evaluation.py --output-dir reports/experiments/temporal_evaluation_rehearsal_2026-09-07_validated
```

This completed using hand-built score fixtures only: separate synthetic validation/test identities, four geographic groups each containing both classes, one invalid positive scene, and identical candidate scores. The paired delta intervals were exactly zero; the invalid scene was counted as an abstention. A dense fixture containing one true positive and one false positive produced IoU 0.5. Source hashes and runtime versions are recorded beside the results. The output directory is exclusive and is not overwritten.

The complete focused suite passed **64 tests in 9.84 seconds**. The final rehearsal was repeated in the new `_validated` directory after adding malformed-calibration validation; the earlier run was preserved. No historical test outcomes were reopened.

## Preparation boundary

No real threshold, performance result, acceptance margin or scientific conclusion has been selected. The draft comparison protocol remains unfrozen. More architecture variants or tuning against synthetic fixtures would add little evidence now. The next substantive work requires verified imagery: product-specific preprocessing, adjudicated location/event grouping, training-only resource checks, and the frozen matched experiment.
