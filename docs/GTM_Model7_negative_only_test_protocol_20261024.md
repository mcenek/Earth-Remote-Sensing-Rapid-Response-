# Reserved no-plume sites after broader development selection

Written before final pixels or inference. This adds a false-alarm check to the broader development protocol. It cannot measure plume recall, plume overlap, source accuracy or national accuracy.

Failure cases and controls:

- Reserve at most ten previously unused cached sites with at most two explicit no-plume observations per site. Exclude all catalog sites with a plume observation. Freeze metadata first. The data audit must exclude all known fit, evaluation and viewer locations, acquisitions and spectral content, the broader DEV sites, and sites within 25 km of those locations. Include both target and reference identities.
- Before source decoding, require completion of both bounded fits, fresh-process DEV replay, an eligible improved DEV selection, and a root gate that pins the selection, selected ordinary inference freeze, original baseline freeze and negative-only reservation hashes. A different cutoff or checkpoint invalidates the gate.
- Fixed QA matches the existing MARS reference procedure: earlier explicit no-plume reference within 60 days, same grid, both-date cloud and finite/positive DN support, and the original fixed support rules. Record exclusions without replacing cases after predictions. No downloads and no training on reserved sites.
- Evaluate original v5 U-Net and selected U-Net on identical accepted observations. Verify original TIFF reconstruction and fresh-process prediction replay, independently count native false-positive pixels, and preserve every scene in the viewer. No recall/IoU success gate applies to all-negative data. Report false-alarm pixels, affected observations and affected physical sites.
- Check actual outputs, including the largest false activation and a clear scene. Gray invalid pixels must not count. Static map masks and interactive scores must use each frozen cutoff. Publish all accepted cases with explicit no-plume-only labels and test limitations.
- No model or cutoff selection, replacement or retraining follows these outcomes. These sites become consumed after this evaluation. Upstream no-plume annotation does not prove zero emissions. Global clear scenes do not establish a US deployment false-alarm rate.
