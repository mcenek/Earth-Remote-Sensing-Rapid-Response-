# GTM_Model6 bounded research gates

- Scope: bounded read-only research readiness gate; no training, download, or held-out label opening
- Decision: **Do not launch another Model6 sweep on the current nine EMIT-positive cohort. Resolve quantitative EMIT target/provenance and CAFO controls first; retain the existing pilots as failed engineering evidence.**
- Available RAM at gate: 4.05 GiB

## Current local cohort

- Records: 77 ({'MARS': 68, 'EMIT': 9}); negative cap: 32
- Connected groups: 17; positive groups: 11
- Excluded during bounded inventory: 3
- EMIT labels positive-only: True

## Native quantitative EMIT availability

- Native CH4ENH/CH4SENS/CH4UNCERT/CH4PLM sets are available for 20241020T170504, 20241130T180310 and 20250922T204933.
- They are currently an EMIT-only diagnostic resource. A paired Sentinel/native-EMIT mapper manifest is not ready: the six cached L2A crops include missed or truncated plume peaks and do not yet prove complete support, timing, grid alignment and independent splits.
- The data-readiness gates below are not permission gates; bounded implementation/testing is authorized.

## Gates

- `legacy_artifact_identity`: **True**
- `legacy_runtime_inference_available`: **False**
- `quantitative_emit_target_available_in_current_model6_cohort`: **False**
- `cafo_supervised_methane_rotation_ready`: **False**
- `user_authorized_for_bounded_implementation`: **True**
- `native_quantitative_emit_products_available`: **True**
- `native_paired_sentinel_mapper_ready`: **False**
- `new_real_pilot_data_ready`: **False**
- `promotion_candidate_available`: **False**

## Completed pilot summaries

| Run | MARS pixel precision | MARS IoU | Negative activation | EMIT positive support recall |
|---|---:|---:|---:|---:|
| GTM_Model6_plume_01 | 1.3772% | 1.3285% | 8.8985% | 28.5227% |
| GTM_Model6_plume_evidence_02 | 2.8892% | 2.6655% | 5.0602% | 7.2809% |

## Limitations

- The current Model6 EMIT records use positive-only footprint support; exterior pixels are unknown.
- The 36 reviewed MARS positives are concentrated in two connected groups.
- The local CAFO inventory has no date-specific plume labels or reviewed facility negatives.
- Native quantitative EMIT products exist for three US acquisitions, but the paired Sentinel mapper contract is not yet met.
- Legacy checkpoint inference is not runtime-verified in this environment.
