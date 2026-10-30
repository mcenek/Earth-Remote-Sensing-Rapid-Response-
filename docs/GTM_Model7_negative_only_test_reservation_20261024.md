# Separate negative-only TEST reservation

This protocol was recorded before the negative-only metadata search and before any new TEST raster or prediction was opened.

Prevent these failures: calling negative-only evidence a plume recall test; reusing a site, target/reference ID, acquisition, or spectral hash already fitted, evaluated, or shown by the viewer; pairing a target with a later, plume-labeled, different-grid, or different-release-split reference; silently replacing a failed site after pixel QA; and opening TEST pixels before model/threshold freeze.

Search all cached official MARS `NO_PLUME` observations with fixed clear >=90%, Sentinel-2 L1C, preceding same-grid explicit `NO_PLUME` reference 24 to 1440 hours earlier. Exclude every actual saved fit/evaluation site and viewer site plus every broader DEV site, with 25 km clearance. Freeze up to ten physical sites and at most two target dates each using metadata and cache file sizes only. Do not prepare arrays or assess pixel QA yet. This cohort can measure false alarms only; it cannot validate plume recall.
