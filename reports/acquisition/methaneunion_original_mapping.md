# MethaneUnion original-scale mapping audit

Local metadata only; no downloads, imagery, test manifests, or model outcomes opened.

The frozen 1632 rows represent 108 coordinate-label groups but only 56 distinct coordinates.

Exact coordinate+label mapping gives **1633 original-scale S2 rows**, 829 positive and 804 negative, covering 108 groups. All 1633 have four nonempty manifest paths. This is a new cohort requiring a separate protocol.

Reusing IDs is unsafe: 1351 overlap, but 1348 disagree in coordinates or label. 9 coordinate-label groups change sample count.

The observed first member of **dataset_part_139.tar.gz** matches original training ID 254920 (label 0) after the explicit path alias hypothesis:

`data/original/geo/train/s2/254920/s2_pre.tif`

This proves one path exists, not that a complete training example or usable mask exists. Proposed expected members:

- `data/original/geo/train/s2/254920/s2.tif`
- `data/original/geo/train/s2/254920/s2_pre.tif`
- `data/original/geo/train/s2/254920/s2_pre_pre.tif`
- `data/original/geo/train/s2/254920/s2_plume.tif`

A possible next step is one first-member pilot capped at 16 MiB network / 64 MiB decompressed, starting at 256 KiB. Member size is absent from the saved survey, so success within those caps is unknown. Do not download the full archive automatically.

An independent existing probe already stopped at its 1,048,576-byte cap with status `bounded_probe_failed`. It did not obtain a complete member. The larger cap above is only a hypothetical planning bound, not an instruction to extend that stopped probe.

## Limits

- This defines a new original-scale cohort, not recovery of the frozen cross-scale sample identities.
- Coordinate and label do not establish acquisition date, plume event identity, crop offset, or one-to-one sample correspondence.
- A first TAR member confirms one path only. Other temporal members and masks may be anywhere or absent, including beyond the release tail.
- Manifests contain no shape, dtype, band order, radiometric units, geotransform, or payload size; those remain unverified.
- Original-scale name does not prove native pixel scale. Local legacy crop source uses S2 32 px crops resized to 224 and claims 10 m GSD, while raw acquisition uses 20 m products; neither proves these release TIFFs.
- Binary scene labels and plume paths are supplied; mask semantics, co-registration, temporal observability, and independent geographic grouping require separate adjudication.
