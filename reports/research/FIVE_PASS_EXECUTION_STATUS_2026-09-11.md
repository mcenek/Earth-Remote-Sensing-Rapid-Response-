# Five-pass execution status — 2026-09-11

Expansion stopped at a scientific-label roadblock, under the user's instruction to stop before wasting bandwidth. No model training or TESSERA acquisition was launched in this execution.

| Pass | Result | Next gate |
|---|---|---|
| 1. Cohort and evaluation | Audited provisional Iowa inventory; 12,703 active facilities organized into five geographically grouped 20% train / 80% evaluation rotations. Label-readiness audit completed. | Confirm inventory provenance/datum and task labels. No reviewed controls or acquisition-date plume labels aligned to this cohort were found. |
| 2. Sentinel-2 pilot | One real 1,280 m square crop acquired and verified: B02/B03/B04/B08 at 10 m, B11/B12 and SCL at 20 m. Stopped before the planned 25–50-location expansion. | Resolve supervision and sampling plan before spending on a larger cohort. |
| 3. Baseline rotations | Not run. | A defensible labeled task and locked evaluation protocol. |
| 4. Sentinel-2 architecture comparison | Not run. Existing candidate implementation remains preparation, not experimental evidence. | Valid baseline and matched evaluation data. |
| 5. TESSERA and fusion | Not run. | Sentinel-2 baseline first, then aligned cohort and representation comparison. |

## Verified pilot

- Facility inventory ID: 56216; image: S2C_15TWF_20241206_1_L2A, 2024-12-06.
- Actual imagery response payload: **11,010,048 bytes (10.5 MiB)**, against a persistent 100 MiB cap. This excludes prior catalog queries and network protocol overhead.
- SCL classes 4/5 covered 100% of this crop. This is a product quality flag, not an independent cloud validation.
- All seven output rasters passed native-grid, bounds, CRS and stored radiometric metadata checks. Spectral scale is 0.0001 with offset -0.1; these L2A values must not be silently treated as older L1C inputs.
- Red-band cache replay reproduced exact pixels with no additional charged payload. Cached bytes remain available for reuse.
- Seven focused acquisition tests passed, covering bounded reads/cache behavior and tiled crop decoding.
- Visual review shows agricultural fields and buildings near the coordinate. Neither facility identity nor methane emissions have been independently established.
- The single successful crop cannot establish a regional acquisition success rate.

The reader requires bounded HTTP 206 responses, rejects full-file fallback, reserves budget before requests, and performs no automatic retries. The budget/cache is intended for a single acquisition process, not concurrent workers. No automatic expansion is scheduled.

## Roadblocks and the smallest useful next input

1. Confirm whether the available Iowa inventory is Huang's dataset or obtain Huang's coordinates and coordinate reference information.
2. Establish the scientific target with Martin: facility detection/retrieval can use facility presence plus reviewed controls (or a separately justified positive-unlabeled protocol); date-specific methane detection requires independent plume/emission evidence. Inventory membership alone cannot label methane activity.
3. Review geographic balance before training: one connected 2 km group contains 1,770 facilities, around 70% of one training fold. Preserve group separation and report effective geographic diversity; do not split it randomly merely to improve apparent balance.

These are grounds to pause acquisition expansion, not proof that the project is infeasible. Further architecture sweeps cannot resolve missing ground truth.

## Local evidence

- `reports/research/CAFO_PASS1_LABEL_READINESS_2026-09-11.md`
- `reports/acquisition/cafo_rotations_2026-09-11/summary.json`
- `reports/acquisition/cafo_pilot_verified_2026-09-11.json` (source URLs, hashes, bounds, QA and budget receipt)
- `outputs/cafo_pilot_2026-09-11/site56216/rgb_preview.png`
- `outputs/cafo_pilot_2026-09-11/site56216/` (seven small local GeoTIFF crops)
- `outputs/cafo_pilot_2026-09-11/range_cache/` (retained bounded source blocks)
