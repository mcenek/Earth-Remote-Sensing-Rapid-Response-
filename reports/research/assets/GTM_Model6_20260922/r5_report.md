# GTM_Model6 multiscale mapper MARS pilot

Status: **completed engineering diagnostic; not promoted**.

This run trained the independent S16/S32/S64/S128 mask branches on the existing fully-labelled MARS masks and evaluated one held geographic partition. The EMIT quantitative enhancement head was disabled because the current EMIT cohort has positive-only support and no quantitative target. This is not evidence of EMIT upscaling, source-origin localization, CAFO recovery, or nationwide screening.

- Device: `cuda`; steps: `400`; held fold: `1`; train/held records: `99/65`
- Checkpoint SHA256: `39f2ca29688f90394cbe78c4521d158968f1901e3c1255e88900cac8457eccc4`
- Source bundle hash: `6ac13b5a4ccc68530b43b70f22567d94a101f4ced933d03c64d98093b0192b43`

## Reused development fold: pooled native aggregate

| IoU | Dice | Precision | Recall | Predicted fraction | Negative activation |
|---:|---:|---:|---:|---:|---:|
| 2.1485% | 4.2066% | 2.2487% | 32.5182% | 4.3493% | 4.2643% |

| Method | Pooled IoU | Precision | Recall | Background activation |
|---|---:|---:|---:|---:|
| mean | 2.148% | 2.249% | 32.518% | 4.264% |
| S16 | 0.517% | 0.551% | 7.733% | 4.214% |
| S32 | 1.206% | 1.238% | 31.923% | 7.683% |
| S64 | 1.622% | 1.673% | 34.765% | 6.165% |
| S128 | 1.549% | 1.575% | 47.852% | 9.019% |
| all_negative | 0.000% | 0.000% | 0.000% | 0.000% |
| all_positive | 0.301% | 0.301% | 100.000% | 100.000% |
| spectral_baseline | 1.697% | 1.941% | 11.897% | 1.813% |
| training_spatial_prior | 0.000% | 0.000% | 0.000% | 0.000% |

The local viewer bundle contains every held scene, fixed threshold 0.50, RGB, reviewed MARS mask, probability layer, prediction mask, georeferenced display bounds, and candidate hotspot markers. Exact S128 context with 16-pixel output tiles leaves the outer 56-pixel margin unsupported and blank in scoring/display. Inspect both positive and negative scenes before treating the aggregate as useful.

The next scientifically meaningful run requires time-aligned EMIT quantitative enhancement and representative controls. Until then the mapper's second head remains a tested contract, not a trained result.
