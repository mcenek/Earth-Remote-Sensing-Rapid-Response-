# GTM_Model6: A new plan for mapping methane plumes across the United States

This is the design I want to test next. The goal is to use Sentinel-2 images to flag possible methane plumes anywhere in the United States. EMIT's measured methane plumes will teach and check the model where the two satellites have suitable observations of the same place and time. At national scale, the model must work from Sentinel alone because EMIT does not cover every place and date. Known CAFOs and oil facilities are useful checks on the resulting candidates, but a facility location by itself is not proof of a plume.

## What the output should look like

For each scene, I want a map showing where the model believes a plume is, how confident it is, and, when the data supports it, an estimate of the methane enhancement within the plume. The map should retain its geographic coordinates so I can place it over the original Sentinel image and compare it with the actual EMIT observation. A source-location marker may be added later, but only when we can test it against credible source and wind information.

This is also the sense in which I mean *upscaling*. I want to infer a more useful, finer map of methane evidence from the higher-resolution Sentinel image, using EMIT's coarser methane measurement as supervision. Merely stretching EMIT pixels to a finer grid would not recover new detail. I will treat fine-scale reconstruction as an experimental claim that needs its own held-out comparison, not as something the model has already achieved.

## Martin's multi-scale idea

One model should not have to guess the right viewing scale. I plan to train four separate models around each location, using 16 x 16, 32 x 32, 64 x 64, and 128 x 128 Sentinel-pixel windows. The small window may help trace narrow plume structure; the larger windows may reveal the surrounding surface, facility, and wind-shaped context. Each model will make its own prediction for the same central area. I can then see which scale is useful and which is being fooled by roads, buildings, or other bright features.

First I will compare those four independent predictions and use a simple average as a baseline. Once there are enough independent observations, **a fifth, separately trained model will learn how to combine the four outputs**. That second-stage model should learn when to trust each scale rather than treating all scales as equally reliable. It must train on predictions from sites that the four underlying models did not train on; otherwise it could look good by memorizing their training examples. The current pilot has four separate scales and an average. **The learned combining model has not been trained yet.**

The intended flow is:

Sentinel target and nearby reference image -> four independent window-size models -> a separately trained combining model -> plume outline and methane-enhancement map -> reviewed source candidates on the US map.

EMIT is the reference for training and review, not a required input to the final national scan. I will add a quantitative methane-enhancement output only after the native EMIT values, units, quality information, and Sentinel alignment are verified. A plume-outline-only example cannot teach the model how much methane is present.

## Data and training

The model needs many more *independent* examples than we have now: dated EMIT methane observations, Sentinel target and comparison images close enough in time, usable plume outlines or quantitative methane values, and reviewed no-plume scenes from varied US landscapes. I will keep a record of where each observation came from, its acquisition time, its valid area, and any uncertainty or missing pixels. Unknown areas will not be labeled as clean background.

Following Martin's suggestion, each training example can be shown in eight ways: 0, 15, 30, and 45 degrees, with and without a mirror. The image and every label must be transformed together. These views may help the model learn orientation-independent features, but they remain eight views of one observation, not eight new plumes. All views of a site stay in the same training split. I can apply the same transforms to held-out scenes as a *robustness check*, then count the result once per original scene.

I will use grouped cross-validation so overlapping images and the same physical site cannot appear on both sides of a comparison. I also want to run Martin's data-efficiency test: train on about 20% of the independent CAFO-site groups, check the remaining 80%, and rotate which groups are in training. This is harder than a conventional 80%-training split and will show whether the method can cope with our limited data. Known CAFO and oil locations will be used to evaluate whether a national candidate map recovers plausible sources, not as exact plume labels.

## Source identification and later extensions

An apparent plume and its source are different questions. Once plume outlines are credible, I would fit a smooth, Gaussian-like plume pattern and use wind direction, when available, to work back toward a possible origin. The proposed source point should be shown with uncertainty and checked against independent source locations. Without source and wind labels, a bright pixel or a model peak is only a place to review, not a verified emission point.

TESSERA is a useful later comparison. I would test whether its time-aware Sentinel representation improves the same held-out cases and whether it helps the four-scale merger. I do not want to add it until the Sentinel/EMIT data and visual evaluation are trustworthy, because otherwise a better score could hide the same errors.

## What we have measured so far

I have a working local viewer that places predictions over the source imagery and available labels. The most recent bounded mask pilot used 65 reused development scenes, including three labeled plume scenes. Its four-scale *average* detected part of some visible plumes, but it missed another plume and made many detections on surface features. Across those scenes, its overlap with the labeled plume pixels was only about 2.1%, with about 2.2% pixel precision. This is **not a successful methane detector**, an upscaling result, or a national validation. The selected good-looking example does not represent the overall result.

The visual review is why I am resetting the model plan. I have also located one near-time Georgia Sentinel crop around a native EMIT observation and checked that the two images overlap. That is useful preparation, but I still need to resolve the correct Sentinel reflectance conversion and verify the EMIT plume outline before training from that pair. The current viewer also shows older experiments for comparison; their scores use different data and should not be ranked as if they were one test.

## What will count as progress

For every experiment I will look at the actual map, not just a score. The viewer should show the original Sentinel scene, the EMIT observation or reviewed label, each scale's prediction, and the combined prediction in connected views. I will inspect all available positives, the largest false positives, and misses, with the same fixed settings used for reported results. I will track plume overlap, misses, extra detections in reviewed negative scenes, and the practical number of US-wide candidates that would need human review. Any source-origin or fine-resolution methane claim will require separate evidence.

The next gate is a trustworthy set of independent, time-matched positive and negative observations. Then I can train each scale, compare the simple average with the learned combining model on held-out sites, add the quantitative enhancement output if the EMIT labels are valid, and only then attempt a US-wide screening run. If the maps still light up roads or facilities rather than plumes, I will stop and fix the data or model before expanding the scan.

## Review materials

The local viewer and portable sample instructions are in `GTM_Viewer/` on the research branch. The sample is for inspection of current work, not evidence that the new architecture is finished. The visual review packet is at:

https://github.com/mcenek/Earth-Remote-Sensing-Rapid-Response-/tree/research/gtm-model6-restart-2026-09-22/reports/research/assets/GTM_Model6_20260922

The detailed experiment and data notes remain in `docs/GTM_Model6_technical_appendix.md` and `reports/research/GTM_Model6_followup_verification_2026-09-22.md` for anyone who wants to inspect the methods and numbers.
