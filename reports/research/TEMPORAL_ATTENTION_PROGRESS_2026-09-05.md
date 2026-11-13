# Temporal architecture research progress — 2026-09-05

## Decision

The next defensible architecture experiment is a matched temporal residual CNN versus the same model with compact coarse temporal attention. Both are implemented and pass a synthetic fitting diagnostic. Neither has been evaluated on real methane imagery. The established spatial baseline remains the evidence-backed choice until this comparison succeeds on independently verified data.

Bulk MethaneUnion archive scanning remains paused. The current bottleneck is obtaining and validating complete examples, not GPU training or further architecture complexity.

## Implemented candidate

`EarthRemoteSensingRapidResponse/temporal_attention_candidate.py` implements sensor-specific stems, shared date encoders with widths 32/64/128/256, and a residual U-Net decoder. Raw target/reference inputs and explicitly named native-physics channels preserve direct spectral evidence. Resizing physics features does not create higher-resolution measurements.

The optional attention uses local target queries against a 3×3 reference neighborhood plus self, with two blocks at each of the two coarse scales. It is not a shifted-window spatial transformer. The matched-pair constructor copies all common initial weights; attention adds parameters, so this tests the practical incremental module rather than an equal-parameter control.

Per-pixel channel normalization avoids spatial normalization statistics changing with padding. It does not remove convolutional boundary effects. Invalid references make their attention contribution an identity operation. Callers must also invalidate reference-dependent physical features. Annotation availability is never an inference feature.

The scene classifier has an independent raw-input route plus detached dense features. Scene fitting therefore cannot damage the segmentation encoder through gradients. All-invalid inputs produce finite outputs with false support; downstream code must abstain or exclude them rather than interpret them as observed negatives.

## Training contracts and audit findings

`temporal_attention_training.py` separates annotation support from sensor QA. Missing pixel truth produces no dense loss or gradient. BCE averages equally across supervised scenes; Dice applies only to annotated positive scenes. The normalization helper accepts product-correct physical reflectance and preserves the distinction between 1.0 and 1.5 before clipping.

The historical ordinal training path conditionally accepts positive scenes without dense masks and represents absent masks as zeros; without an availability filter this fabricates background supervision. This is a code-path finding, not proof that the frozen experiment contained those cases or that it caused the failed result. Its historical radiometric preprocessing also clips before the later normalization can preserve values above physical reflectance 1.0. Historical experiment files and results were not changed.

The prospective full trainer still needs a verified product loader, geographic split enforcement, sampling counters, separately selected dense checkpoints, scene-head fitting and calibration, and immutable evaluation artifacts. The new helpers and synthetic runner are not that full trainer.

## Verification

The focused suite passed **31 tests** across model behavior, loss/normalization contracts, bounded TAR extraction, and local metadata mapping. Command: `python -m pytest tests/test_temporal_attention_candidate.py tests/test_temporal_attention_training.py tests/test_probe_methaneunion_training_member.py tests/test_methaneunion_original_mapping.py -q -p no:cacheprovider`.

The immutable diagnostic record is `reports/experiments/temporal_attention_synthetic_smoke.json`: four synthetic 32×32 examples, fixed seed 20260905, 24 dense steps followed by 24 scene steps, CPU only, no real data or held-out outcomes.

| Candidate | Parameters | Dense loss, initial → final | Scene loss, initial → final |
|---|---:|---:|---:|
| Temporal CNN | 4,095,682 | 0.699924 → 0.263710 | 0.657263 → 0.364307 |
| CNN + temporal attention | 5,414,850 | 0.692796 → 0.076601 | 0.652856 → 0.051972 |

Both learned this fitting diagnostic. These numbers establish functioning optimization, not generalization or superiority of attention. The diagnostic contains source hashes at execution time; the physical-reflectance helper was added afterward and covered by the final focused suite. The diagnostic was not overwritten.

## Data feasibility

The local mapping audit found that the frozen 1,632 rows comprise 108 coordinate-label groups but only **56 distinct coordinates**. Splitting coordinate-label groups independently would risk putting the same location in both partitions.

Matching by coordinate and label produces **1,633 original-scale rows**, 829 positive and 804 negative. Nine groups change sample count. Of 1,351 overlapping row IDs, 1,348 disagree in coordinates or label. Rewriting directory prefixes while retaining IDs is therefore invalid. Even coordinate-label matching defines a new cohort; it does not establish dates, plume events, crop offsets, or original sample identity.

The archive survey locates one proposed reference member at `data/original/geo/train/s2/254920/s2_pre.tif` in archive 139. A separate bounded probe downloaded exactly **1,048,576 bytes**, failed to obtain the complete member, retained no training raster, and stopped. See `reports/acquisition/methaneunion_one_member_probe.json`. No full archive hash was verified. No automatic extension or bulk scan is warranted by this result.

## Next experiment gates

1. Obtain a complete target/reference/mask example through a indexed or otherwise bounded acquisition route. Verify band ordering, physical units, native GSD, geotransforms, temporal dates, QA, mask meaning and co-registration before building a loader. A filename or a nonempty manifest cell is insufficient.
2. Freeze a new cohort and geographically grouped split using coordinates/site neighborhoods and event provenance. Treat the original-scale mapping as a new protocol; audit overlap with previously consumed evaluation sites. Do not reuse the old held-out outcomes for selection.
3. Run a small real-data pipeline diagnostic on training data only, including missing annotations, unavailable references, negatives and both sensors where available. Verify counts and gradients before allocating a training run.
4. Compare the matched pair under identical data, augmentation, initialization of common weights, optimization budget and checkpoint selection. Select dense checkpoints without scene AP replacing segmentation quality; then freeze dense weights for scene fitting. Freeze metrics, recall operating point, calibration policy, seeds and acceptance gates before evaluation.
5. Report scene AP and recall at the frozen false-positive constraint alongside dense segmentation metrics and paired uncertainty grouped by independent geography. Keep synthetic fitting results separate. Expand the architecture only if the matched comparison demonstrates a useful improvement.

No real-data training, new held-out evaluation, full-archive acquisition, or production promotion occurred in this work.
