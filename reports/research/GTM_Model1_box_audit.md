# GTM Model 1 nearly-whole-scene mask audit

Audit date: 2026-09-18. Evidence is from the two exported validation scenes, the exporter, and the frozen `compact_resunet_v1` / `physics_resunet_v1` configs. No model training or download was performed.

## Finding

The broad masks are real model outputs, not a viewer thresholding or raster-rendering artifact. The exporter runs four native 128x128 tiles, stitches them to the 256x256 source grid, writes the probability raster, and creates the mask as `probability >= config.decision_threshold`. Re-reading all four native probability/mask pairs found zero mismatched pixels and all probabilities finite:

| scene | compact threshold 0.01 | physics threshold 0.05 |
|---|---:|---:|
| T14RMU 20230203 | 100.000% positive | 99.890% positive |
| T13RFQ 20230410 | 99.979% positive | 98.531% positive |

The probability distributions explain this directly. Compact medians are 0.607 and 0.480 (T14RMU/T13RFQ); physics medians are 0.511 and 0.476. Thus almost every pixel clears the configured thresholds before any display code runs.

## Reference/support semantics

The sixth band is an aligned EMIT CH4 reference, not a full-scene truth image. In the exporter, support is `isfinite(EMIT_CH4) & (EMIT_CH4 != -9999)` and truth is `EMIT_CH4 > 300 ppm*m` only on supported pixels; outside support the truth is unknown. Support is 9.305% for T14RMU and 10.890% for T13RFQ. Within supported pixels, the >300 fraction is 54.018% and 26.398%, respectively. Therefore a full-scene predicted mask cannot be judged against the white/unknown EMIT area; evaluation must be support-gated. The viewer's `valid` layer means finite Sentinel-2 model input, not EMIT reference support, so it is expected to be nearly/all valid here.

## Preprocessing and domain checks

The exporter’s compact path matches the shared contract: raw B2/B3/B4/B11/B12, clamp at zero, `log1p`, then frozen per-channel z-score. Physics adds the six exact engineered channels before its 11-channel z-score. Band order and tile size match the configs. Raw input magnitudes are plausible L1C-style integer DNs (roughly 137–7,693 across these scenes), and both configs explicitly declare `L1C_TOA`, scale 1, offset 0. There is no evidence here of a missing `/10000` or an L2A-vs-L1C conversion in the exporter.

There is distribution shift for T14RMU: compact raw log features have channel means approximately `[-1.09,-1.30,-1.37,-0.91,-0.94]` in training-z units, while T13RFQ is close to the fit distribution (`[0.04,-0.01,0.01,0.06,0.12]`). This can hurt T14RMU, but cannot explain T13RFQ's 99.979% / 98.531% positive masks because T13RFQ is already near the frozen fit normalization. The source arrays contain finite values throughout; there is no scene-wide input nodata being converted into a positive response.

## Historical/model evidence

The frozen compact calibration config reports full-pixel recall 0.99928 but specificity 0.000696, and full-scene-macro recall 0.99880 with specificity 0.000895. The physics config reports full-scene-macro recall 0.99830 and specificity 0.00479. These are the same failure mode seen in the exports: near-all-positive segmentation with very poor negative rejection. The model was trained with a validity-masked BCE + soft Dice loss and unit positive weight; valid EMIT pixels are sparse, and pixels outside EMIT support do not contribute to the loss. That makes the exported full-scene mask an extrapolation over mostly unscored pixels, while the selected thresholds (0.01 and 0.05) make broad probabilities operationally positive.

## Conclusions and concrete fixes

1. Keep the renderer/exporter unchanged pending a viewer check; native masks prove the threshold operation is correct. Display probability and mask from the same native grid and label pixels outside EMIT support as unknown in validation views.
2. Do not present 0.01/0.05 as useful alert thresholds. Recalibrate on an independent, negative-containing validation cohort, report precision/specificity and scene-level false-positive area, and choose a threshold against that objective. The current frozen calibration is not evidence of useful calibration.
3. For any claimed validation metric, gate both truth and predictions by `EMIT_CH4` support. Keep full-scene predictions available only as model extrapolation, clearly distinct from supported reference evaluation.
4. Add an inference contract check that records product level, raw-DN scale/offset, feature means/stds, and the fraction of finite Sentinel-2 pixels; reject silent L2A/SR inputs. This is a preventive fix, not the demonstrated cause of the current T13RFQ result.
5. Investigate training design: sparse, positive-centered EMIT support plus no verified plume-free negatives and unit positive weighting is consistent with the measured specificity collapse. Retrain/evaluate with explicit negative scenes or valid negative pixels and a loss/threshold protocol that penalizes false positives.

Evidence paths: `ersrr-ordinal-win/tools/GTM_Model1_export_predictions.py`, `ersrr-ordinal-win/outputs/GTM_Model1_viewer_export/`, `Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/artifacts/{compact_resunet_v1,physics_resunet_v1}/config.json`, `Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/ersrr_core.py`, and `Earth-Remote-Sensing-Rapid-Response-/tools/run_unet_experiment.py`.
