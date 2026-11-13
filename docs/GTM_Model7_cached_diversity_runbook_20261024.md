# Cached data ready for another comparison

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

Prepared October 24, 2026. This is data preparation, not a new model result. No new imagery was downloaded and no model was trained in this pass.

## What is ready

The new Sentinel-2 L1C supplement contains **171 released plume observations and 390 explicit no-plume observations at 100 international sites**. It covers 14 named countries plus an Offshore category. These are MARS source annotations, not 561 new native EMIT measurements. These sites are new to this repaired fitting cohort, but global source sites have historical project exposure.

| Split | Plume observations | No-plume observations |
| --- | ---: | ---: |
| Training | 268 | 690 |
| Development | 6 | 80 |
| Existing evaluation | 2 | 80 |

All previous 565 observations retain their original assignments and identical arrays. Added data is training only. The combined manifest has 1,126 observations and 308,482,437 bytes of prepared arrays, about 294 MiB. It is compatible with the existing normalized residual CNN and temporal U-Net comparison trainer.

Each reference is a real, earlier, same-site observation with an explicit source NO_PLUME class. It has the same delivered grid, its own cloud mask and six-band image. References are 24 hours to 60 days earlier. Source hashes, band order, finite positive inputs, label support, fixed snow-like checks and at least 90% two-date valid support were checked. Of 568 raster reference attempts, 561 passed, five failed the snow-like check and two failed valid support. The exporter tries the next eligible reference when the closest fails. It never ranks references by plume label shape or model performance.

Independent verification reconstructed all 561 added arrays from TIFFs, checked all 565 old arrays and preserved the split exclusions. Shared grids do not prove perfect registration. Haze and brightness differences remain in some pairs. Root inspected one positive example from every new site and representative no-plume inputs. This is source and input QA, not proof of successful learning or national transfer.

The metadata audit found a larger cached non-US pool of 1,847 source-complete, earlier-reference Sentinel plume observations at 143 sites before the selected raster QA. The first export is deliberately bounded to 100 positive sites and at most two positives and four negatives per site. There is no need to download that pool again.

## Inspect the data

Open the local review through the existing private viewer:

http://127.0.0.1:8766/bundle-assets/GTM_Model7_cached_diversity_inputs_v3/index.html

Use Country and Class to inspect 199 samples: one positive per site, and one no-plume sample at the 99 sites with an accepted negative pair. Cyan shows the released annotation. These panels do not contain model predictions. Both dates share a per-band display scale; training arrays are unchanged. The report links to the previous actual predictions, including failures.

## US sites that are still missing

The one cached New Mexico Sentinel positive at PB_K_0011 has no same-site, same-grid NO_PLUME observation in the source table at any date. Expanding the date window cannot fix that missing reference.

Two New Mexico Landsat pairs at PB_30 pass source and image QA. Their site is 3.20 km from an existing development scene, inside the frozen 25 km exclusion. They remain diagnostic-only with six previews and no exported training arrays. Landsat also needs its own sensor contract; it must not silently enter the Sentinel branch. See `outputs/GTM_Model7_nontexas_readiness_20261024/nm_landsat_candidates/manifest.json` and its independent E2E receipt.

The cached Georgia continuous EMIT and Florida EMIT-mask pilots remain useful separate resources. Their label types, resolutions and Sentinel L2A radiometry differ from this MARS L1C comparison. They were not silently appended or described as new independent plume sites.

## CAFO data from the journal

The October 23 journal snapshot identifies the 2019 Handan-Nader/Ho resource and the 2026 CAFOSat resource. The live journal was not changed or freshly fetched in this pass. Source and local-cache findings are in `outputs/GTM_Model7_nontexas_readiness_20261024/cafo_audit/cafo_audit_20261024.md`.

Use these data for facility mapping, candidate image acquisition and later comparison of frozen plume detections with facility coordinates. A facility label is not a measured methane plume. A residential image or absence of a CAFO is not a verified no-plume label.

The cached Iowa inventory has 14,155 rows and 14,085 unique facility IDs. The metadata-only comparison found zero existing MARS/S2CM dataset-site entries within 5 km of these points, across 1,498 entries. That result concerns this Iowa inventory and existing scenes only; it does not say all national CAFO resources have no overlap. Its coordinate datum is undocumented, so the matching assumption is explicit.

The 17 cached CAFOSat TIFFs contain 15 distinct RGB arrays. Twelve are RGBA and five RGB, all 833 square pixels with 0.3 m pixel size in their headers. Eight have unresolved state-coordinate conflicts; acquisition dates and a verified annotation-table join are missing locally. Resolve those before treating them as a supervised CAFO dataset. No methane labels were created from them.

## Reproduce the preparation

Run from the repository root with the existing environment. Export commands refuse an existing output directory; use a new output path to reproduce the full export.

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_prepare_cached_diversity.py --help
.venv/Scripts/python.exe tools/GTM_Model7_verify_cached_diversity_e2e.py
.venv/Scripts/python.exe tools/GTM_Model7_verify_nm_landsat_candidates_e2e.py
.venv/Scripts/python.exe tools/GTM_Model7_train_repaired_mars_log_v3.py validate --branch mars_repaired --manifest outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json
```

Combined manifest SHA-256: `b7d44b43e113baa98d61353750a8721d8297ab6a9d94fd6e74ca0ae50da75836`.

Supplement SHA-256: `2747941bdf76b19e86ab1e451232b87f7989cd4404e059dbcf98a9175905039f`.

## Next fit, not executed in this pass

```powershell
.venv/Scripts/python.exe tools/GTM_Model7_train_repaired_mars_log_v3.py train --branch mars_repaired --manifest outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json --output outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training_log_v3_fit
```

This runs both models with the same 2,400-step schedule, class balancing and uniform physical-site sampling within each class. Keep the current log-radiometry and optimizer settings so the first comparison measures the dataset change. Normalization is fit on training data only. Preserve the original baseline outputs.

The comparison protocol and output IDs are fixed in `docs/GTM_Model7_diversity_comparison_protocol_20261024.md`. Select checkpoints and decision cutoffs on development data only. Report native-pixel plume overlap, recall, precision, negative pixel false alarms and negative scenes activated. Inspect every development prediction beside its current image, reference and annotation, including empty predictions and false alarms. Publish actual saved probability arrays through the existing prediction export and independent replay, with clear experiment names.

Do not choose settings using the existing Kansas/Mississippi evaluation results. Those scenes have already been inspected in earlier work, so they are historical comparisons, not a fresh final test. A national claim requires newly acquired or genuinely untouched sites with verified labels. If the enlarged fit still fails to trace plumes while rejecting clear scenes, stop and investigate rather than applying it over more states. This preparation does not establish emission rates, native EMIT reconstruction, learned super-resolution or CAFO detection accuracy.
