# Martin's CAFO and TESSERA research direction

The next research cycle uses known animal-facility locations as acquisition anchors, evaluates five approximately 20%-training/80%-validation rotations, establishes the strongest defensible Sentinel-2 result, and then repeats the comparison using TESSERA and fusion. Nationwide screening is the eventual objective; the available Iowa inventory is the pilot, not a national ground-truth set.

This direction provides a practical acquisition path that does not depend on searching the MethaneUnion archives. The earlier temporal candidates and evaluation tools remain available; their completed results are unchanged.

## Advisor instructions preserved

Martin proposes using Huang's GPS/UTM facility coordinates, measuring recovery of known CAFOs when eventually screening the US, allowing additional detections that may correspond to other methane sources, training on 20% and validating on the remaining 80% with repeated repartitioning, and comparing the best baseline against TESSERA and data fusion afterward.

Meeting exceptions are October 2, November 13 and December 4, with his 10:20–11:10 conflict requiring an earlier meeting. The year is recorded as 2026 from the current conversation. No calendar change was made because the earlier start time and timezone were not supplied.

## What the labels mean

Keep three separate evidence types:

1. **Facility inventory:** a listed animal-operation location. It is useful for facility-retrieval evaluation and targeted image collection. Inventory inclusion does not establish a plume at a particular overpass, and may not establish regulatory CAFO status without checking the source definition.
2. **Date-specific methane observation:** independently supported plume presence/absence or a spatial mask. This supports methane detection/segmentation evaluation when sensor timing and observability are compatible.
3. **Unlisted detections:** review candidates. Some could be oil/gas, industrial or unlisted agricultural sources. Their absence from the CAFO list must not automatically count as a false methane detection, but it also cannot establish that they are true emissions.

Known-facility coverage alone can be gamed by marking every location positive. Report coverage together with surveyed area, candidate density, cloud/observation availability and a reviewed sample of detected sites. A facility classifier also needs independently reviewed controls or an explicitly positive-unlabeled method; random locations and cloud-free tiles are not automatically methane negatives. Location IDs/coordinates are sampling and evaluation metadata, not predictive features.

## Five rotations, matching the requested 20/80 direction

Build five geographic groups of roughly equal facility count. In rotation k, train on fold k and evaluate on the other four. This reverses the usual 80/20 five-fold convention intentionally. Each facility is used for training once and evaluation four times. Keep repeated dates and nearby facilities together; never split individual pixels or overlapping crops independently.

The initial planning buffer is 2 km, for a proposed maximum 1,280 m square crop width (diagonal about 1.81 km). This is an overlap-control assumption, not proof that nearby landscapes are statistically independent. Larger crops or a stronger geographic-transfer claim require revisiting the buffer before finalizing the splits. Whole-component grouping can make fractions approximate; report the counts instead of breaking components to force exactly 20%.

Tune architecture/hyperparameters and alert thresholds inside the 20% training pool, using internal grouping. Freeze each fitted model and rule before scoring its outer 80%. If outer scores are repeatedly used to pick the best architecture, call those development/model-selection results and reserve a new geographic confirmation set for the final claim. Four predictions for one held-out facility are correlated observations; do not multiply the effective sample size by four.

## Acquisition and first model steps

The local source is `EarthRemoteSensingRapidResponse/Polygon_CSV_Files/iowa_cafos_2024_arcgis_api.csv`. Its association with Huang is pending confirmation. Start with its listed-active records as a provisional cohort, preserving other statuses for audit; current/snapshot activity does not prove activity in 2024.

A bounded public catalog probe was performed at three active facility coordinates, with at most two items per query. The first results were mostly cloudy; a second metadata probe applied a 20% scene-cloud filter. These are discovery records, not selected training observations. Cloud percentage for a whole satellite tile does not certify that the facility crop is clear. Before retaining any image, inspect crop-level SCL/QA, coordinate accuracy, facility visibility, radiometric scale/offset, acquisition date and complete spatial coverage.

Use a small, geographically varied pilot to verify the acquisition adapter before expanding. Sentinel-2 L2A catalog assets can support the new facility study, but cannot silently replace the L1C products expected by frozen methane benchmarks. Preserve raw SWIR resolution. For the facility task, compare a simple regularized baseline to a compact image model before adding complexity. For plume inference, run a frozen methane detector only after verifying its product/input contract, and report matches as candidates requiring methane evidence.

## TESSERA phase

TESSERA produces annual per-pixel representations from Sentinel-1 and Sentinel-2 time series. The original model releases annual 10 m embeddings; the access library documents 128 channels. This makes it a plausible representation for facility/land-use classification. It is not a supplied methane-label dataset. [Paper](https://arxiv.org/abs/2506.20380), [official access library](https://github.com/ucam-eo/geotessera).

After the Sentinel-2 baseline is selected and frozen, compare: (A) baseline alone; (B) frozen TESSERA features with a small regularized classifier; (C) baseline plus TESSERA fusion. Use the same labels, geographic rotations, temporal window and evaluation coverage, and record any facilities excluded because embeddings are missing. Fit normalization and any dimensionality reduction inside training folds only.

Annual features may contain observations acquired after a particular target date. They are suitable for a retrospective annual facility comparison, but would create future-information leakage in a purported real-time plume detector unless the feature window precedes the target. This is a project inference from the annual representation, not a performance result.

The official library currently requires Python 3.12+, while the local research runtime is Python 3.11. Use a separate environment for a later pinned adapter. The library lists v1.1 as available while v2 regional embeddings may need a request with uncertain turnaround; verify the Iowa/year coverage before transferring data. No package installation, embedding download or request to authors has been made. [Library documentation](https://github.com/ucam-eo/geotessera).

## Decisions still needed before a scientific run

- Confirm whether this Iowa file is Huang's collection and whether he has a broader inventory, footprints or coordinate-accuracy documentation.
- Choose the temporal study window and verify historical activity; 2024 is presently a filename-based planning assumption.
- Obtain facility control labels for classification, and independent plume evidence wherever methane accuracy is claimed.
- Adjudicate crop size, geographic separation and any new geographic confirmation region before freezing the protocol.
- Establish regional runtime, storage and candidate-review workload before scaling to a US survey.

The runnable planning configuration is `configs/cafo_tessera_research_plan.json`. It records this sequence and its assumptions without inventing a completed experiment.

## Completed implementation and checks

- Audited 14,155 rows, including 12,703 listed Active and 1,452 Inactive. Both `progid` and `uniqueID` are unique in this snapshot. All lat/lon values parse, but the source provides no explicit CRS/datum for either geographic or projected coordinate columns. The source includes a small number of non-IA state values; the initial cohort is the active inventory, not a strict state-boundary filter.
- Built `tools/build_cafo_rotation_plan.py` and executed it with seed 20260911. The draft output at `reports/acquisition/cafo_rotations_2026-09-11/` has 4,244 connected components. Fold counts are 2,541 / 2,541 / 2,541 / 2,540 / 2,540 facilities. Each rotation trains on one fold and validates the remainder. Its source CSV hash is bound in `summary.json`.
- The largest component contains **1,770 facilities**. This is a material heterogeneity warning: balanced facility counts do not imply balanced independent evidence. Preserve the component and inspect its geographical extent before adopting this grouping for science; do not break it merely to improve a model score.
- Seven focused tests passed, including deterministic reordering, nearby-site grouping, active-status/coordinate conflicts, small-cohort rejection, source hashing, exclusive outputs, and antimeridian/polar grouping.
- Executed `tools/probe_cafo_sentinel_catalog.py`. The restricted initial attempt failed before network access; the explicit public-network retry succeeded. All three locations returned two candidate catalog items in both the unfiltered and cloud-filtered queries. The clear-query receipt is `reports/acquisition/cafo_sentinel_catalog_clear_probe_2026-09-11.json`. Local facility visibility and image QA remain unverified, and no raster assets were downloaded.

The next concrete acquisition step is a small crop-level Sentinel-2 pilot using these catalog assets, after checking source-coordinate quality and choosing representative sites from the draft folds. Nothing in these planning files supplies the missing facility-control labels or date-specific plume labels.
