# Training-only spectral signal diagnostic

Written before implementation on October 23, 2026. This isolates input representation from CNN optimization. It reads only the existing TRAIN scenes and negative-only DEV scenes. Both the old exposed diagnostic test and the new confirmation pool remain excluded.

Failure cases: a test raster is opened; labels enter the anomaly calculation; a per-scene fit excludes known plume pixels; unsupported or nonpositive SWIR values contaminate division; current/reference order reverses; the result is presented as measured concentration; source arrays change; visualization stretches each score map independently and hides background artifacts.

Use a label-blind fractional SWIR anomaly inspired by Varon et al. (2021), sections 3.2 and 3.3, https://amt.copernicus.org/articles/14/2771/2021/. For each date, fit c = sum(B12*B11)/sum(B12^2) on all supported pixels, then f = c*B12/B11 - 1. Define absorption-positive proxy f_reference - f_current. This is a difference of fractional reflectance residuals, NOT the paper's radiative-transfer inversion to methane columns. Do not call it a quantitative MBMP retrieval.

Subtract each scene's supported median and divide by 1.4826 times its median absolute deviation, floored at 1e-5. A fixed score >=3 is an anomaly diagnostic, not a fitted detector cutoff. Save all scene masks/scores, train positive-pixel ranks and negative activation, and actual figures with a common [-5,5] score scale. Do not select a model or threshold from either test pool.

E2E evidence: load all eligible train/dev scenes from the sealed manifest with hash checks; rehash after; assert no other split was processed; verify identity current/reference inputs produce zero proxy on each actual scene; verify finite scores/support and native repeat mapping; score native labels for TRAIN diagnostics only; keep DEV no-plume false alarms separate. Save an immutable receipt and a contact sheet for direct visual inspection.
