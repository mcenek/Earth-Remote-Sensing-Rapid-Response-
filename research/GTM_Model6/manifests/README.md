# Model6 data manifests

No approved Model6 training manifest exists yet. The local source catalog is an inventory of candidates, not a dataset approval.

The first manifest is `GTM_Model6_US_pairs.jsonl`. One row represents an original parent observation; do not duplicate rows to manufacture an eightfold independent sample count. Keep source assets in their existing directories and reference them.

Each row must record:
- Stable sample ID, physical site/group ID and parent acquisition IDs.
- Geographic footprint/CRS/affine, verified US eligibility and independent split role.
- Sentinel/EMIT timestamps and actual pairing lag.
- Band order, physical units/scales, nodata and valid-observation masks.
- Original source paths and hashes, native grid resolution, and real context available at each window size.
- Explicit input, degradation operator and withheld-target identities. Do not label interpolated EMIT as independent fine-resolution truth.
- Plume/source/wind label availability and provenance. Missing labels remain missing.
- Data-quality/visual-review status and exclusion reasons.

Fit transforms and normalization only inside the training partition. Keep all geometric siblings, overlapping crops and repeated site observations together. Preserve an untouched primary evaluation; transformed validation is a separately reported robustness analysis.

The current default may be replaced by an approved manifest only after the data review. No existing historical test is silently adopted as untouched Model6 confirmation.
