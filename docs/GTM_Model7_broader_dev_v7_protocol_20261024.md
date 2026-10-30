# Broader DEV and fresh TEST protocol (2026-10-24)

This protocol was recorded before opening any new source raster, annotation, prediction, or QA result. Selection is deterministic from cached MARS CSV and pinned JSONL metadata. No download is allowed. Existing output artifacts are immutable.

## Failure modes to prevent

1. Calling a previously fitted, evaluated, or viewer-exposed location "fresh TEST".
2. Reusing a target or reference acquisition, spectral SHA-256, observation ID, or any site within 25 km across frozen TRAIN and new DEV/TEST.
3. Selecting a clear reference from a different grid, site, release split, or later date; silently substituting a reference after seeing QA or model output.
4. Treating implicit non-plume or weak/missing annotation as explicit `NO_PLUME` or `PLUME` truth.
5. Allowing cached-file availability or quality rejection to trigger outcome-guided replacement.
6. Relabeling previously evaluated data as pristine. DEV reuse must be disclosed with original provenance and every source identity.
7. Changing training data, TEST identity, scoring thresholds, or model selection after TEST pixels or predictions are opened.
8. Mistaking metadata catalog presence alone for actual model exposure, or overlooking saved viewer output as actual exposure.

## Fixed selection and preparation

Audit all actual local fitted/evaluated manifests and viewer scene IDs, including the latest 92-row fresh-data manifest, against official MARS IDs/sites/coordinates. First freeze TEST from genuinely project-unexposed, cached, clear >=90%, Sentinel-2 L1C sites with a preceding explicit `NO_PLUME` same-grid reference 24 to 1440 hours earlier. A TEST site must be >=25 km from every actual prior fitted/evaluated/viewer site and from TRAIN and DEV. TEST is metadata only until the model and decision rules are frozen; no pixel or prediction read now. If no such positive site exists, record zero availability and do not invent a substitute.

Then choose DEV from previously evaluated but never TRAIN-used sites, preferring diversity of countries, physical sites, and positive observations. Reused DEV is a development set with a disclosed look history. Ensure DEV is disjoint from the existing TRAIN by site/25 km, target/reference observation, acquisition, and spectral hash. Choose at most four positive and four negative targets per site; pair each with the nearest preceding explicit clear `NO_PLUME` reference on the same grid and release split. Fixed raster QA follows the existing MARS preparation contract. Record all attempts and failures; no outcome-driven target replacement. Reuse of a previously evaluated DEV observation is permitted only with explicit provenance. Cross-site acquisition/hash overlap between DEV records is prohibited.

After preparation, run source reconstruction E2E verification: manifest, arrays, source hashes, labels, references, grid, disjointness, and no network receipt. The output must be repeatable from a frozen selection manifest and provide file SHA-256 receipts.

