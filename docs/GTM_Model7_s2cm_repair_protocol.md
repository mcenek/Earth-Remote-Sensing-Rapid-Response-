# Model 7 cached MethaneS2CM repair protocol

Fixed October 24, 2026, before writing the exporter or fitting either model. This is a separate real Sentinel-2 L2A crop experiment. It does not revise the L1C MARS experiment or reuse the MethaneS2CM official location test for fitting.

## Failure cases to reject

1. An official MethaneS2CM `test.csv` row or packed location-test image is read. Input must be the pinned official `train.csv`, its pinned train development manifest, and `v5_train_packed.h5` only.
2. A point is called US because it lies in a broad latitude/longitude box. Check each coordinate against the locally cached US state TopoJSON; record the state and exclude points outside polygons.
3. A point within 25 km of any scene point in the October 23 sealed MARS comparison enters the cohort. Use great-circle distance and strictly greater than 25 km; verify the MARS manifest SHA-256. This spatial screen does not prove cross-dataset acquisition separation because MethaneS2CM lacks acquisition IDs.
4. A physical site or pre-existing 25 km source group crosses fit, development, and final evaluation. Retain internal-fitting groups for fit; divide internal-development groups between dev and test by a frozen seed hash. Verify every cross-split site distance is at least 25 km. If source groups merge under 25 km, reject the conflicting partition.
5. A split lacks positive or negative source annotations, or development/test contains only one geographic group. Fail before writing files.
6. Date repeats from prolific sites dominate scene counts. Select at most four positive and four negative observations per exact site, sorted by a fixed seed hash of source sample ID. Selection cannot inspect mask area, model score, or plume appearance.
7. A label is inferred from a missing mask or an almost-full mask is silently dropped. Check binary mask and label agreement for every selected observation. Preserve full-box and broad masks and count them in the receipt.
8. An L2A source is silently treated as MARS L1C. Preserve native 32 by 32 20 m crops, uint16 source radiometry converted to float32 without rescaling in the prepared array, and the exact source product/band metadata. Select B02/B03/B04/B08/B11/B12 from source page indices 1/2/3/7/10/11. Model input is target plus `reference90`; `reference365` remains source metadata, not an input.
9. Zero-filled or nonfinite values enter learning or scoring. A pixel is supported only when all six selected bands on both dates are nonzero and finite. Reject selected samples with less than 90% supported pixels. The release has no pixel cloud mask, so supported must never be described as cloud clear.
10. Identical target/reference source pairs cross splits. Hash the prepared inputs. Quarantine all records of a cross-split duplicate instead of assigning a favorite split. Record duplicate exclusions.
11. An HDF5 point coordinate is turned into an invented raster affine. Save a pixel-only 32 by 32 grid with nominal 20 m scale and explicit unknown georeferencing. Scene latitude/longitude locate the source site only.
12. Native scoring silently repeats or interpolates a 32 by 32 mask. Here `label_native`, `target`, and predictions are all 32 by 32. Use `reference_mapping=identity` and repeat factor 1.
13. A source file, released label, or exported array changes unnoticed. Record source and output SHA-256, split counts, source sample IDs, quality failures, site/group/state counts, and a bounded replay receipt. Output directory must be new and must not be overwritten.
14. The prior project's MethaneS2CM validation/test exposure is misrepresented as untouched research evidence. This is a new comparison for the two new model fits, but the wider project has used the source. The official location test remains outside this new fit/dev/test partition.

## Frozen selection and output

Input source is official MethaneS2CM location-split **train** metadata and local packed train HDF5 at pinned revision `ee9a96d4994ca6bc45725c1e92d7a06258131eaf`. The spatial gate uses `GTM_Viewer/assets/us-states-10m.json` and the October 23 MARS cohort manifest SHA-256 `93ebc93f87183e63e09e6611fdec1b3b8a9be0acaa2fe4f2b24bfe67e1611bd8`. The selection seed is `GTM_Model7_s2cm_repair_20261024_seed_1`.

Save `manifest.json`, `receipt.json`, and one compressed NPZ per selected sample under a new `outputs/GTM_Model7_dataset_repair_20261024/s2cm_data` directory. NPZ keys are `target_image`, `reference_image` as float32 `(32,32,6)` in B02/B03/B04/B08/B11/B12 order; `valid`, `valid_native` as bool `(32,32)`; `target` as binary float32 `(32,32)`; and `label_native` as binary uint8 `(32,32)`. The shared temporal cohort schema may be reused with explicit `reference_mapping=identity`, `native_repeat_factor=1`, and pixel-only grid metadata. A validation/replay pass reads the saved files, recomputes source-selected arrays for a deterministic sample from every split, and records a SHA-bound receipt.

The final experiment must report per-source domain, independent sites and components, broad-mask frequency, any remaining unknown cloud/reference quality, and all positive and negative evaluation cases. No US national detection claim follows from these crops.
