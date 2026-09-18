# CAFO facility-recovery pilot handoff

This packet continues the advisor-directed facility experiment using the provisional Iowa DNR inventory. Inventory presence is a weak facility label, not an emission label. The source is not confirmed as Huang's dataset.

## Actual execution

Six sites were planned; four were attempted, three produced complete crops, one failed crop quality, and two were left unattempted after that stop. The first three were the existing ID-ordered catalog feasibility sites; three more were predeclared to cover missing folds. This is a convenience engineering sample, not an estimate of Iowa-wide performance.

| Inventory ID | Fold | Crop result | SCL clear fraction |
|---|---:|---|---:|
| 56216 | 2 | Complete, seven rasters | See receipt |
| 56217 | 2 | Complete, seven rasters | See receipt |
| 56218 | 1 | Complete, seven rasters | See receipt |
| 56222 | 0 | Quality rejection; only SCL downloaded | 36.47% |
| 56225 | 3 | Not attempted after stop | — |
| 56220 | 4 | Not attempted after stop | — |

Total imagery response payload is **31,809,998 bytes (30.34 MiB)**, including the earlier 10.5 MiB crop. This turn added 20,799,950 bytes (19.84 MiB). The conservative budget charged 31,981,568 bytes against the same persistent 104,857,600-byte cap. Additional catalog payload this turn was 130,034 bytes. Protocol overhead and small provenance web lookups are not included in these counters.

All 22 saved rasters passed file hash, bounds, native resolution, CRS and stored radiometric checks. All 22 were reconstructed exactly from cached source blocks with network requests disabled. The three RGB previews were inspected: fields and structures are visible, but their agricultural use and legal CAFO classification are not independently adjudicated. Site 56217's marker is offset from the most visible structures; crop-center proximity alone is not sufficient identity evidence.

The failed site demonstrates that scene-level cloud filtering cannot replace crop-level QA. No alternate date or replacement site was fetched after failure. No model was trained, and no facility accuracy, recall or methane outcome is claimed. The three successful crops cover only two of the five folds.

## Review materials

- [Review table](review.csv): inventory labels are explicit; reviewed facility and methane labels are blank, not zero.
- [56216 preview](56216.png), [56217 preview](56217.png), [56218 preview](56218.png). Yellow circles mark assumed WGS84 inventory coordinates. Brightness/gamma are display transformations only.
- Per-site JSON receipts contain acquisition timestamps, source asset URLs, scales/offsets, hashes and verification counts.
- [Summary](summary.json) preserves attempted, rejected and unattempted counts separately.
- Native GeoTIFFs and compressed source blocks remain in ignored local `outputs/`; this small packet is suitable for Git. A clone does not contain the raw crops.

## Next scientific experiment

1. Establish the endpoint as **recovery of inventory-listed animal-feeding facilities**. Confirm narrower regulatory CAFO identity separately. Keep methane validation as a distinct endpoint.
2. Review coordinate placement, facility identity and whether these winter acquisitions show enough useful structure. Record reviewer and evidence in the table; an RGB guess is not an independent methane annotation.
3. Before another acquisition batch, freeze a seasonal/date selection rule and a bounded fallback allowance. The present run stops on the first quality rejection; do not expand the cap or silently substitute favorable scenes.
4. Construct a geographically balanced inventory-positive and background-unlabeled cohort. Unlisted farmland is not a verified negative. A provisional presence-versus-background classifier can rank review candidates, but its background labels must be described as weak labels and it cannot support a claim of facility precision.
5. Use a small spectral/context baseline before attention: native-band summaries or a small CNN, with background examples assigned by the same geographic groups. Fit preprocessing and choose thresholds only within each 20% training pool. Keep all dates and nearby background samples with their site group.
6. Measure held-out inventory recovery at a fixed surveyed area/review budget and report every fold, observable-site denominator, abstentions and reviewed top-ranked candidates. Do not call acquisition coverage model recall. Reviewed controls are required before reporting conventional precision/FPR. Tiny per-fold samples are pipeline checks only.
7. After a credible Sentinel-2 result, compare TESSERA on exactly the same cohort and splits, as Martin requested.

The original five-rotation manifest remains a draft. Its largest connected group dominates one fold; resolve or explicitly analyze that dependence before freezing a scientific benchmark. Training three crops now would not provide a useful answer.

## Reproduction

Run `tools/acquire_cafo_cog_pilot.py` with `--catalog`, `--facility-id`, a new `--output-dir`, and the **existing shared** `--cache-dir outputs/cafo_pilot_2026-09-11/range_cache`. A failed crop returns nonzero so a batch can stop. Do not run concurrent processes against this cache. Default cap remains 100 MiB; do not create a fresh cache merely to reset it.

The new catalog `--facility-ids` option accepts up to five explicit active inventory IDs and stops at the first request failure. It does not paginate. No further acquisition is running or scheduled.
