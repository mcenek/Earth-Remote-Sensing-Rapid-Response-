# GTM Model6 MARS visual review

This is a read-only visual inspection of the twelve fixed-threshold contact sheets listed below. Each tile shows Sentinel RGB, the reference overlay, prediction mask, and probability heatmap. No quantitative metrics were inferred from the images.

## `GTM_Model6_plume_01_mean`

The unconstrained predictions frequently form large contiguous regions covering fields, crop edges, or the right and bottom portions of a tile. The reviewed reference regions are usually compact and elongated, while the predicted regions are commonly displaced or much larger. Heatmaps often follow broad land-cover structure, roads, field boundaries, bright structures, and textured surfaces. A few tiles contain localized hits near a reviewed region, but the overall pattern is not consistent plume-shaped localization.

The negative-control page contains substantial false-positive area in several tiles, including connected regions around roads, buildings, field edges, and terrain texture. Other tiles have sparse responses, so the behavior is variable but does not show reliable abstention on negatives.

## `GTM_Model6_plume_evidence_02_mean`

The evidence-constrained predictions are generally sparser and show fewer whole-tile or whole-box fills than the first run. However, the compact reviewed regions are often not matched. Orange responses more commonly track roads, crop boundaries, bright buildings, and high-frequency surface texture. A small number of tiles show localized responses near the central source area, but there is no stable plume geometry or direction across the positive pages.

False alarms remain on the negative-control page. Some negatives contain only scattered speckles, while others contain connected or rectangular patches along scene structure. Visually, this run is less dominated by whole-box behavior than the first run, but it still looks more like a surface/edge response than plume-specific segmentation.

## Assessment

Neither run visually demonstrates reliable localized methane-plume segmentation. The second run reduces broad scene-wide activation, but positive-mask overlap remains weak and the same road, edge, building, and texture responses persist in negative controls. These contact sheets support treating both outputs as exploratory segmentation results rather than evidence of plume-specific localization or verified background performance.

## Exactly viewed

First run:

- `outputs/GTM_viewer_bundles/GTM_Model6_plume_01_mean/montages/MARS_00.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_01_mean/montages/MARS_01.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_01_mean/montages/MARS_02.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_01_mean/montages/MARS_03.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_01_mean/montages/MARS_04.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_01_mean/montages/negative_controls_00.png`

Evidence-constrained run:

- `outputs/GTM_viewer_bundles/GTM_Model6_plume_evidence_02_mean/montages/MARS_00.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_evidence_02_mean/montages/MARS_01.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_evidence_02_mean/montages/MARS_02.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_evidence_02_mean/montages/MARS_03.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_evidence_02_mean/montages/MARS_04.png`
- `outputs/GTM_viewer_bundles/GTM_Model6_plume_evidence_02_mean/montages/negative_controls_00.png`
