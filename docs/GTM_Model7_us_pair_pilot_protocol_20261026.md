# Small US transfer pilot

Written on October 26, before opening the selected 2025 image crops or native EMIT annotations. This is an acquisition and transfer pilot. It is not a national accuracy test.

## Fixed question and model

Can the retained model find plume evidence at geographically separate US sites, with a properly matched earlier reference, without marking large areas in control observations?

Use retained v11, update 2400, cutoff 0.65. Checkpoint SHA256: `8a36587380ea14b86501b5bc245b3b26f82f52f380aa5e5e156900491ae39483`. Do not fit, choose a threshold, fill gaps, enlarge masks or select a more favorable checkpoint from these pilot labels. Preserve the original native predictions and every accepted case, including misses.

The model predicts binary plume candidates. It does not estimate methane emission rates. Its output scores are not calibrated probabilities. Repeating its 20 m scores onto a 10 m grid is not demonstrated methane upscaling.

## Acquisition candidates

Use the frozen metadata in `outputs/GTM_Model7_us_transfer_v13_20261026/readiness/asset_preflight_v2.json`, SHA256 `82e2114f0b1440b0fdc796e0b358499b6212a9e2b476deac8ea2ddf2daf905ac`.

Process in this order:

1. July 31, 2025, `USA_w7z4jseci`: Sentinel `S2B_17RLP_20250731_0_L1C`, with the selected July 28 reference. Sentinel precedes EMIT by 12.094 minutes.
2. July 29, 2025, `WM_Security_Landfill`: the selected July 29 Sentinel observation and July 24 reference. Sentinel precedes EMIT by 46.837 minutes.
3. July 28, 2025, `Okeechobee`: the selected July 28 Sentinel observation and July 26 reference. Sentinel precedes EMIT by 59.897 minutes.

These are possible pairs, not accepted examples. A source coordinate inside a Sentinel tile does not establish full plume coverage. Time differences can change the plume. Record them beside every prediction.

Start with the first candidate. Check actual transferred bytes before acquiring the next band or event. A small JP2 window can require a large transfer. Do not acquire a whole tile as an unplanned fallback. The existing ledger is nearly exhausted. No transfer increase is authorized by this document.

## Required checks before inference

1. Verify source object identity, size and hash. Retain the original annotations and metadata.
2. Inspect the exact L1C product quantification, offsets and NoData rules. Convert once to the established cached-DN convention. Do not assume that a 2025 raw JP2 already matches a harmonized training TIFF.
3. Check both dates' six bands, transforms, projected metre CRS and shared footprint. Produce aligned 200 by 200 source crops at 10 m. Preserve the original 20 m SWIR information and record any resampling.
4. Obtain the local CloudSEN12 masks. The audited released producer needs all 13 Sentinel L1C bands, in order B01/B02/B03/B04/B05/B06/B07/B08/B8A/B09/B10/B11/B12, scaled by 1/10,000. The six-band asset shortlist is insufficient: B01/B05/B06/B07/B8A/B09/B10 are also needed for each date. Add their exact products to a separately frozen acquisition allowlist before any transfer. Reconcile their 10/20/60 m sampling with the producer before claiming matching masks. A tile-wide cloud percentage or QA60 bit test is not a substitute. Record valid support and show excluded pixels.
5. Inspect the earlier reference without looking at model output. Check for cloud, shadow, shifted pixels, missing coverage and evidence of an existing plume. Keep the declared UTC acquisition times. Reject unusable pairs for a documented data-quality reason, not because the model misses a plume.
6. Compare each site's coordinates and acquisition IDs with all prior fitting and assessment inventories. Record unknown historical site names. Do not describe the pilot as pristine while those links remain unresolved.

Save a dated acceptance or rejection receipt for every candidate. Do not silently replace rejected candidates with easier cases.

## Controls and annotation limits

Choose control observations from the already inventoried provider no-plume records using site and timing metadata before inference. Check their labels, images and references independently. A location outside a positive plume outline is not automatically a verified no-plume scene. A CAFO, oil facility or landfill coordinate is not a dated methane label.

Freeze the accepted control IDs and their quality reasons before viewing their model scores. If control labels cannot be verified, show the predictions as unverified and do not report a false-positive rate.

Inspect the native EMIT annotation's units and support semantics before defining its evaluation mask. If it provides positive support only, report coverage of that support. Do not label the exterior negative or calculate precision, IoU or F1 there. If it supplies a defensible positive and negative evaluation domain, preserve that domain unchanged and report all four pixel counts.

## Evidence to retain

For each accepted pair, save the source and conversion receipts, current and earlier-reference RGB images, original EMIT annotation, validity layer, raw model scores, thresholded mask and georeferenced outputs. Publish the actual saved prediction in the local viewer. Show all cases, including controls and empty predictions.

Report per-scene counts and pooled counts only where annotation support permits them. Include marked area, missed labeled area and extra predicted area. Source coordinates, predicted components and plume extent are different objects. Do not call a crop center or a plume centroid the methane origin.

This small, selected pilot can reveal transfer failures and help plan a broader US validation set. It cannot establish detection of all methane sources, nationwide F1 or quantitative emission accuracy.
