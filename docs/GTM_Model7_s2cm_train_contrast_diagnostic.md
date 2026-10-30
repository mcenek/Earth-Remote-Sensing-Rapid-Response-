# MethaneS2CM TRAIN-only temporal contrast diagnostic

Frozen before computing any contrast results on 2026-10-24. This is a descriptive image and released-mask check, not classifier fitting or physical methane validation. The diagnostic reads only the sealed `s2cm_reference_repair_data` TRAIN NPZs. It does not read DEV or TEST arrays, images, scores, or model predictions.

## Ways this check can fail or mislead

1. Released positive masks may fill most or all of a 32 x 32 crop. Without enough supported unlabelled pixels, inside-versus-background contrast is not identifiable. These scenes are reported and excluded from contrast summaries, never relabelled.
2. The nominal earlier reference can differ because of cloud, snow, land change, acquisition geometry, illumination, atmosphere, season, or radiometric processing. The source lacks exact acquisition times, cloud masks, and footprints. A contrast association cannot establish methane causation or reference plume absence.
3. Log band ratios can vary for surface and sensor reasons; the sign expected from MARS L1C cannot be presumed for MethaneS2CM L2A. Report both signs and robust distributions without choosing a decision threshold.
4. A distant background pixel may differ systematically from plume pixels because of site structure. Compare both all supported outside pixels and a 1–5 pixel ring immediately outside the mask.
5. Repeated observations at one physical location are not independent. Report scene counts and equal-site median summaries; do not infer a population error bar from crop count.
6. Tiny positive masks or incomplete valid support can make a median unstable. Record the supported positive, outside, and ring counts, and exclude zero-support comparisons explicitly.
7. The data were selected for US geography, valid support, and a frozen image-only reference choice. This is conditional on that cohort and cannot be generalized to the source's official location test or national plume prevalence.

## Fixed computation

For each supported TRAIN pixel, using stored native float32 DN and band order `B02,B03,B04,B08,B11,B12`, compute

`D = log1p(target B12) - log1p(target B11) - log1p(reference B12) + log1p(reference B11)`.

For positive scenes, compute the median D in annotated pixels minus (a) the median D in all supported outside pixels, and (b) the median D in a supported ring outside the annotation at Chebyshev distance 1–5 pixels. Compute outside/ring median absolute deviation around their own median. Report all supported sample counts, unanalysable full-mask cases, per-scene sign and quantiles, and equal-site median sign and quantiles. The ring is an image-pixel neighborhood, not a metric distance. For negative scenes, compute the median across supported pixels of the per-pixel mean of six absolute `log1p(target) - log1p(reference)` band differences, plus individual B11/B12 absolute differences. This is a photometric mismatch descriptor, not a cloud classifier. Use no DEV/TEST arrays and fit no thresholds.

The earlier MARS L1C result is contextual only: local-ring D contrast was negative for 95/97 TRAIN positive scenes and 25/26 equal-site medians, using a different product and a reviewed earlier no-plume reference. Do not pool the two datasets.
