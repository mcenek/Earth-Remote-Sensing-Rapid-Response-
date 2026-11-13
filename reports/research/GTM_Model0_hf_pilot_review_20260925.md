# Hugging Face pilot: downloads and read-only suitability review

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

## Result

The bounded download is complete. CAFOSat adds 17 actual aerial image patches for inspection. The newer EMIT training manifest lists 1,793 US observations that may be useful for a targeted methane-data expansion. Neither source has been added to model training, relabeled, or combined with existing train/test data.

The plan was communicated before reading the new samples. Original downloaded bytes remain intact. The external research journal was not edited.

## Acquired files

| Source | Pinned revision | Acquired |
|---|---|---|
| [CAFOSat](https://huggingface.co/datasets/oishee3003/CAFOSat) | `152f4c5885c16dd63eddf295a461890d3c761343` | Complete CA, ID, LA, WA archives; README; repository catalogs |
| [MARS-Hyperspectral-EMIT-v2025](https://huggingface.co/datasets/UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025) | `4e6c50999eb430fa33abc9488b71cd1a0dcd7991` | `csvs/train_t_v2025A.csv`; README; root and CSV file catalogs |

The four CAFOSat archives total 34,711,337 bytes. The EMIT training CSV is 4,612,166 bytes. Each complete file was checked against its upstream LFS SHA-256 or Git blob identity. An offline second pass rechecked all seven original files without downloading anything.

This pass received **39,339,325 bytes (37.52 MiB)** of metadata and files through the bounded downloader. Conservative reservations charged **40,967,917 bytes (39.07 MiB)**. The persistent 100 MiB budget now has **21,422,355 bytes (20.43 MiB)** remaining. Search/web page inspection is separate from this dataset-transfer ledger. No failed transfer or retry occurred.

All originals and receipts are under `outputs/GTM_Model0_hf_pilot_20260925/`. The 1.25 GB CAFOSat master CSV, large state archives, synthetic augmentation archives and 33.9 GB negative archive were not downloaded. No EMIT raster, validation manifest or test manifest was downloaded in this pass.

## CAFOSat findings

The four complete small archives contain CA: 2 images, ID: 4, LA: 1, WA: 10. All inspected rasters are 833 by 833 uint8 images, with three or four bands and 0.3 m projected pixel spacing in their headers. This is file-grid spacing, not an independently established effective sensor resolution. Display previews use the first three RGB bands only, without making assumptions about the fourth band's physical meaning.

I inspected every patch in the contact sheet. They contain visibly different rural structures, pens, fields and a pond, alongside repeated imagery. This inspection did not create facility-class annotations or establish methane emissions.

There are material intake issues:

- **Eight of 17 files fail a coarse archive-state/latitude check.** Both `CA` archive images have header centers near latitude 45.2, outside California. Six `WA` archive images have centers below latitude 45.5. This establishes a conflict between archive naming and file metadata, not which source is correct. Unflagged files have not passed a full state-boundary or independent location validation.
- **Two TIFF files are byte-for-byte duplicates.** Contact-sheet entries 10 and 13 have identical source hashes. Entries 10, 13 and 15 have identical RGB pixels despite a three/four-band packaging difference. There are 16 unique file hashes and only 15 unique RGB arrays. Related nearby crops also remain possible. Nothing was deleted or deduplicated.
- Dates, verification flags, authoritative facility IDs, bounding boxes and split membership were not established from these archives. The master CSV was deliberately not acquired. Filename descriptions are not verified labels.
- This small size-selected sample contains no separately validated negative-control cohort or matched Sentinel/EMIT acquisitions. It cannot establish dataset-wide quality or training accuracy.

**Decision:** retain the intact pilot for inspection, but do not use its header coordinates for satellite acquisition, transfer its filename classes into training labels, or claim 17 independent sites. Resolve the original/refined metadata and location conflicts first. CAFOSat remains a facility-context resource, not methane plume truth.

## EMIT training-manifest findings

The newer manifest contains 12,855 unique row IDs across 94 literal country values. The exact country value `United States of America` identifies **1,793 rows**, with **615 distinct location names/coordinate pairs** and **497 tiles**. The literal `United States` matches zero rows, so a naive filter would miss the US cohort.

Those US rows declare **504 `isplume=True`** and **1,289 `False`**. These are provider metadata labels. No corresponding pixel masks, retrieval quality, methane units or input/target alignment were checked. Distinct names/coordinates are not yet independent sites, and the cohort has not been checked against our existing or protected evaluation locations. The full training manifest dates span August 9, 2022 through December 31, 2025; this is not a separate claim about the US-only date range.

CSV SHA-256: `cbbee1489168630a4966c17117e65b831e580718339a0ba1c09e697f9f5a62ea`.

**Decision:** this training manifest is the more directly relevant next lead for methane work. Its availability does not yet justify downloading all 23.2 GB of EMIT imagery or starting training.

## Existing-data checks

The prior MARS-S2L acquisition receipts show substantial existing local material. The earlier MethaneSET/MARS audit found extensive Sentinel overlap, including evaluation identities. MethaneUnion's earlier audit identified a bounded candidate pool amid approximately 145 GiB of archives. The earlier MARS-Hyperspectral transfer protocol failed its negative-pair gate. These historical reports were read to avoid duplicate or inappropriate acquisition; protected held-out outcomes were not revisited.

These prior audits concern their pinned versions. They do not prove that every future release is duplicate or unusable.

## Recommended next pass, not performed

1. Compare the new EMIT training manifest's scene IDs and geographic groups against our existing data and protected evaluation inventories, using identities only. Save exclusions and reasons; do not relax a frozen evaluation boundary.
2. Select a small US development-only set with documented positive and reviewed negative masks. Check the exact file types, units, georeferencing, acquisition timing and cost before downloading it.
3. Establish sufficiently close Sentinel matches and native EMIT support. Download only a bounded positive/control pilot, preserving the remaining budget unless Joshua explicitly changes it.
4. Display the actual imagery, provider masks/retrievals and alignment before defining a training manifest. Keep any CAFO inventory association separate from methane accuracy.
5. Revisit CAFOSat only after obtaining a bounded, source-traceable metadata subset that explains the observed state/coordinate conflicts and identifies original sites and derivatives.

No new architecture sweep is warranted from these downloads alone.

## Repeatable artifacts and verification

- Acquisition plan and failure cases: `reports/research/GTM_Model0_hf_download_plan_20260925.md`
- Downloader: `tools/GTM_Model0_hf_bounded_stage.py`; `--catalog` and explicit `--fetch repo:path` only. It refuses large/unlisted files, uses pinned commits and the existing ledger, and reuses originals only after upstream-hash verification.
- Original file receipts: adjacent `*.receipt.json` files under the staging directory. Catalogs preserve published sizes/hashes. The final offline check is in `latest_stage_receipt.json` and `runs/`.
- Raster review: `tools/GTM_Model0_review_cafosat_pilot.py`. It performs bounded member reads without blind archive extraction, preserves originals and refuses to overwrite its review output.
- Visual artifacts: `review/CAFOSat/GTM_Model0_CAFOSat_contact_sheet.png`, `review/CAFOSat/index.html`, and 17 full-size RGB PNGs.
- Intake metadata: `review/CAFOSat/intake_manifest.json`; duplicate RGB evidence: `review/CAFOSat/rgb_duplicate_review.json`.
- EMIT metadata summaries: `review/EMIT_train_manifest_summary.json` and `.md`.

Verification used the actual downloads and visual outputs. No unit tests were added. All source checksums passed; archive/raster reads completed; the final offline verification added zero budget charge. No model result or synthetic prediction is presented as new evidence.
