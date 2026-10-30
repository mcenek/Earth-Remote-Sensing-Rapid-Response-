# MethaneS2CM reference-image repair rule

Frozen October 24, 2026 **before accessing any repaired-cohort TEST image arrays**. This rule uses source imagery only. Labels, model outputs, and viewer impressions of test scenes cannot choose a reference.

## Training/development evidence used to choose the rule

The cached MethaneS2CM source README calls the three frames T, T-90, and T-365. It says Sentinel-2 data were reprojected, resized, and cropped, but neither the released CSV nor packed HDF5 records exact acquisition IDs or timestamps, per-pixel cloud masks, or a georeferenced affine. The candidate frames are therefore named `reference90` and `reference365`, without a claim of exact lag or cloud clearance.

Only the repaired cohort's 6,056 TRAIN and 1,982 DEV images were inspected. For each target and reference, visible brightness is the median of raw L2A DN values across B02/B03/B04 and all 32 by 32 pixels. A **strict bright-shift gate** flags `reference90` when its visible median is above 4,500 DN and more than twice the target visible median. It flags 52 TRAIN and 56 DEV scenes. All 108 have `reference365` below the same gate. For those flagged, the median reference90/target brightness ratio is 6.73 TRAIN and 6.84 DEV; reference365/target is 1.13 and 1.18. After removing each band's median from log1p visible+NIR pixels, reference365 has a smaller median absolute spatial residual to the target in 42/52 TRAIN and 48/56 DEV flagged cases. These are image similarities, not methane or cloud truth. A fixed-scale contact sheet shows saturated white reference90 cases and typically more recognizable reference365 context.

## Frozen selection

For every scene, including the sealed TEST scenes, keep `reference90` by default. Switch to `reference365` **only if** all of these image-only conditions hold:

1. `median_RGB(reference90) > 4500` raw DN and `median_RGB(reference90) > 2 * median_RGB(target)`.
2. `reference365` does **not** satisfy that same strict bright-shift gate relative to target.
3. At least 90% of native 32 by 32 pixels have nonzero values in all six selected target bands and all six selected `reference365` bands.

If any condition fails, retain `reference90` and record the unresolved quality flag. Do not search a wider date window, adjust thresholds, reject a scene by label or model score, or change the split. Applying the frozen rule may inspect TEST **imagery** once; the TEST labels remain unused for reference choice, and no TEST model prediction is inspected. The final comparison must disclose that input quality selection was applied to TEST images, so those images are no longer image-unseen, while label/score selection remains sealed.

## Output contract

Produce a new immutable `s2cm_reference_repair_data` directory. Preserve all baseline scene IDs, split/group assignments, source rows, source HDF5 hashes, target imagery, native labels, and mask geometry. Add scene `reference_dataset` equal to `reference90` or `reference365`; update `reference_image`, valid support, valid native support, array SHA-256, input-pair SHA-256, and quality metadata for switched scenes. The source has no verified cloud classification, acquisition time, or georeferencing. Save an exact source-array replay for each changed scene and a receipt with counts by split, unresolved flags, TEST image-array accesses, and source/manifests hashes.
