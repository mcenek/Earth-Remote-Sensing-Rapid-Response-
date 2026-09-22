# GTM Model6 native pair readiness

Strict timing threshold: **6.0 h**

| Acquisition | Scene | Lag h | Common native support | Native plume positive pixels | Peak in footprint | Status |
|---|---|---:|---:|---:|---|---|
| 2024-10-20T17:05:04+00:00 | S2B_16SGC_20241020_0_L2A | -0.517 | 99.878% | 6521 | False | native_peak_outside_s2_footprint |
| 2024-10-20T17:05:04+00:00 | S2A_16SGD_20241025_0_L2A | 119.479 | 99.878% | 6521 | False | timing_exceeds_strict_6h; native_peak_outside_s2_footprint |
| 2024-11-30T18:03:10+00:00 | S2B_13RFQ_20241120_0_L2A | -240.297 | 100.000% | 3305 | False | timing_exceeds_strict_6h; native_peak_outside_s2_footprint |
| 2024-11-30T18:03:10+00:00 | S2B_13RFQ_20241210_0_L2A | 239.703 | 100.000% | 3305 | False | timing_exceeds_strict_6h; native_peak_outside_s2_footprint |
| 2025-09-22T20:49:33+00:00 | S2C_13SGR_20250921_0_L2A | -27.071 | 99.172% | 220 | True | timing_exceeds_strict_6h |
| 2025-09-22T20:49:33+00:00 | S2B_14SKA_20250923_0_L2A | 20.759 | 99.172% | 220 | True | timing_exceeds_strict_6h |

The manifest records native raster headers, source hashes, bounded support windows, and strict timing only. It does not establish a national/data-readiness claim from one co-temporal case.
