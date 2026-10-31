# Evaluate predicted plumes against known facilities

This is a proposed next evaluation stage. It does not replace real plume labels or change the frozen Model8 test.

## CAFO validation cohort

Use facilities common to both inventories as the primary CAFO validation list: the recent Hoque et al. [CAFOSat resource (2026)](https://arxiv.org/abs/2606.00548) and the older Handan-Nader and Ho study, [Deep learning to map concentrated animal feeding operations (2019)](https://www.nature.com/articles/s41893-019-0246-x). Martin proposed this intersection in the October 30 meeting, and Joshua clarified it afterward. The older paper was published on April 8, 2019; the dates of its underlying inventory records are separate.

Match the same physical facility using coordinates, available facility information and imagery checks. Do not require identical coordinate decimals. Record and justify the matching tolerance before evaluating predictions, resolve ambiguous matches, and remove duplicate facilities. Retain both original coordinates, source versions and observation dates. Report the resulting count and geographic coverage; the intersection may cover only part of the United States.

Keep locations appearing in only one inventory in a separate group, outside this primary cohort. Agreement supports facility persistence. It does not establish methane emission at the satellite acquisition time or supply a pixel-level plume label.

## Required observations

Retain UTC acquisition time, footprint, valid-pixel support, current and reference dates, CRS, pixel transform, raw probabilities, fixed-cutoff mask, model version and checkpoint hash. Store ground-truth identity, time and mask where available. Keep each facility's coordinate, inventory source, observation year and verification status.

A scene without a measured plume is not automatically a verified negative everywhere. Report reviewed clear controls separately from unknown areas. A facility outside valid coverage is not a missed detection. Include facilities in the denominator only when coverage and evaluation rules are satisfied.

## Measurements

1. Apply the threshold chosen on development data. Preserve the unmodified probability map and binary mask.
2. Form connected candidates. If filling or merging is tested, save a second layer and its parameters. Keep raw pixel evaluation unchanged. Do not expand masks to improve facility containment.
3. For every covered facility, report whether its coordinate is inside a candidate.
4. Report nearest distance to a candidate boundary, with zero for points inside. Also report distances to its probability-weighted center of gravity and highest-score pixel. Store the chosen candidate ID and both point locations.
5. Calculate distances in metres using a suitable local projected CRS or a geodesic method, not degrees or screen pixels. Report kilometres and, where useful, miles.
6. Retain observations with no candidate, missing timing, cloud obstruction or unverified facilities. Record explicit states instead of silently removing them.

The center of gravity and highest-score pixel describe the enhancement. Neither is automatically the emission origin. Wind-aware source estimation needs separate evidence.

## Time and wind

Record the difference between satellite and independent plume acquisition times. Report simultaneous, near-time and poorly matched observations separately, with actual time differences. Use measured wind direction, speed and uncertainty where available.

The example of 20 mph for ten minutes corresponds to about 3.33 miles. Three miles was a plausibility example, not a confirmed acceptance threshold. Wind displacement is directional. It does not justify accepting every facility within that radius.

Choose thresholds and distance cutoffs with development data and physical justification. Freeze them before reserved evaluation. Show distance distributions and sensitivity curves, with sample counts and uncertainty. Do not select the best-looking cutoff from the final test.

## Report and inspect

- Raw pixel precision, recall, F1 and IoU against measured plume labels.
- False alarms on reviewed no-plume observations and false-positive area.
- Covered facilities, coordinate matches and unmatched facilities.
- Distances to the boundary, center of gravity and highest-score pixel.
- Time offsets, wind availability, cloud support and coordinate uncertainty.
- Raw and filled/merged maps side by side, with fixed rules and counts.
- Control locations or a spatial null comparison to estimate chance proximity.

CAFO association is not methane-detection precision. Inventories are incomplete, emissions can be diffuse, and oil facilities or landfills can be real sources. Historical coordinates need validation. Agreement between old and recent inventories supports persistence, not a date-specific plume.

Before claiming US transfer, reserve new US sites and acquisitions that were not used to design the model or evaluation. Confirm matched satellite observations and ground truth. The current six US evaluation observations are insufficient.
