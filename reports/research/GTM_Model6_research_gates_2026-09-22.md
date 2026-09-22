# GTM_Model6 bounded research gates

- Scope: bounded read-only research readiness gate; no training, download, or held-out label opening
- Decision: **Do not launch another Model6 sweep on the current nine EMIT-positive cohort. Resolve quantitative EMIT target/provenance and CAFO controls first; retain the existing pilots as failed engineering evidence.**
- Available RAM at gate: 4.14 GiB

## Current local cohort

- Records: 77 ({'MARS': 68, 'EMIT': 9}); negative cap: 32. This is a capped readiness inventory; the completed pilots used their frozen 173-record manifest.
- Connected groups: 17; positive groups: 11
- Excluded during bounded inventory: 3
- EMIT labels positive-only: True

## Gates

- `legacy_artifact_identity`: **True**
- `legacy_runtime_inference_available`: **False**
- `quantitative_emit_target_available_in_current_model6_cohort`: **False**
- `cafo_supervised_methane_rotation_ready`: **False**
- `new_real_pilot_authorized`: **False**
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
- Legacy checkpoint inference is not runtime-verified in this environment.
