# GTM Model6 native pair readiness v3

Strict timing threshold: **6.0 h**

| Acquisition | Scene | Lag h | Grid | Common support | PLM positive-valued pixels | Peak in footprint | Status |
|---|---|---:|---|---:|---:|---|---|
| 2024-10-20T17:05:04+00:00 | S2B_16SGC_20241020_0_L2A | -0.517 | True | 99.874% | 5822 | False | native_peak_outside_s2_footprint |
| 2024-10-20T17:05:04+00:00 | S2A_16SGD_20241025_0_L2A | 119.479 | True | 99.874% | 5822 | False | timing_exceeds_threshold; native_peak_outside_s2_footprint |
| 2024-11-30T18:03:10+00:00 | S2B_13RFQ_20241120_0_L2A | -240.297 | True | 100.000% | 3100 | False | timing_exceeds_threshold; native_peak_outside_s2_footprint |
| 2024-11-30T18:03:10+00:00 | S2B_13RFQ_20241210_0_L2A | 239.703 | True | 100.000% | 3100 | False | timing_exceeds_threshold; native_peak_outside_s2_footprint |
| 2025-09-22T20:49:33+00:00 | S2C_13SGR_20250921_0_L2A | -27.071 | True | 99.624% | 220 | True | timing_exceeds_threshold |
| 2025-09-22T20:49:33+00:00 | S2B_14SKA_20250923_0_L2A | 20.759 | True | 99.624% | 220 | True | timing_exceeds_threshold |

Native ENH/SENS/UNCERT support is ANDed only after CRS, shape, affine, and bounded-window transform equality checks. PLM is evaluated in its own native grid through transformed geometry.

These PLM counts are continuous positive enhancement values, not a verified binary plume mask or target. See [PLM semantics](native_pair_plm_semantics.md).
