# GTM Model 7 MARS clear-date variant

This is a new, versioned variant of the sealed repaired MARS temporal cohort. Its base rules, source hashes, connected-component partition, closest earlier reviewed `NO_PLUME` reference, bidate cloud validity, and 90% valid-cell threshold are defined in `GTM_Model7_mars_repair_protocol.md`. The base `mars_data` directory remains unchanged.

## Frozen image-quality rule, set before opening test imagery

For each target and selected reference date separately, use only native 10 m pixels supported by the base cohort's strict *both-date* cloud/radiometry mask. A pixel is snow-like when B03 (green) is greater than 3000 DN, B08 (NIR) is greater than 1000 DN, and `(B03 - B11) / (B03 + B11)` is greater than 0.4. Exclude the scene if more than 10% of supported native pixels meet all three conditions on either date. This is an image-quality heuristic, not a certified snow class; a cloud-mask code of 0 does not prove a surface is snow-free.

The rule was first inspected on train/dev source rasters only. It flagged two negative scenes among 483 selected train/dev scenes, one train and one dev, and no positive scenes. Both visually extreme dark-reference examples were caught. The full train/dev-only fractions are saved in `outputs/GTM_Model7_dataset_repair_20261024/inventory/snowlike_train_dev_gate.json`. No test image, prediction, or score was used to choose the thresholds or decide the rule.

## Variant selection and verification

The variant replays the original fixed split assignment and seeded site/component ordering. It applies the frozen rule to every candidate during source QA *before* negative quotas are filled, then continues to the next candidate in the same assigned split and site-cap policy. All source-supported positives are evaluated; the variant aborts if it would drop any positive ID from the sealed base cohort. It retains the base limits of 300 train, 80 dev, and 80 test negatives and at most eight per physical site. If an assigned component cannot fill its allocation under the rule, the exporter records the smaller count instead of relaxing the rule.

The new directory is `outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/`, with the same scene/NPZ contract and an additional frozen-rule audit in the manifest and receipt. The output is created only when new; reruns must use another versioned directory. Test labels are used only as source annotations for the predeclared evaluation split, and test model outputs are not read during preparation.
