# Viewer update, October 2, 2026

The local viewer now connects a US overview, acquisition timeline, scene comparison, saved imagery and experiment list. It follows the supplied dark UI concepts using the existing Leaflet renderer and result catalog. No training, relabeling or scientific result changes were made.

## What changed

- The map groups saved observations across experiments, keeps portable copies from creating duplicate observations, and links the selected scene to its saved model outputs. Unknown locations remain unmapped; markers identify saved observations, not verified emission origins.
- Satellite mode dims the area outside US land. US imagery stays clear, with state borders and compact labels. Geographic context tiles are separate from the dated research imagery.
- The acquisition bar has a time-scaled axis, year ticks, actual saved dates, a highlighted selection, tooltips and previous/next controls. Keyboard arrows, Home and End navigate the saved acquisitions. Date filters narrow crowded periods. Missing acquisition dates are not invented.
- Compare shows the input, reference and prediction in linked panels, with adjustable opacity, disagreement and a shared pixel cursor. The branch buttons load independently saved 16/32/64/128/Mean outputs. Missing branches are disabled, and Mean prefers the complete local export when available.
- Native scene scores, display scores and the full run's failed quality gate remain distinct. Hatching is derived from missing model support. Input-only and original archive views cannot imply reference scores they do not have.
- The viewer automatically discovers safely published exports. It keeps the selected scene, zoom, threshold and layer controls during an update. Connection loss, invalid grids and broken imagery are visible; malformed replacement manifests retain the last valid publication.

## Verified result

The [HTTP/CLI receipt](assets/GTM_Model0_viewer_polish_20261002/http-e2e.json) passes all 18 checks. The [browser record](assets/GTM_Model0_viewer_polish_20261002/browser-trace.json) contains 18 final passing checks and the intermediate observations, including a navigation issue found and repaired during review. Branch identity is retained even when several outputs share a cache-file hash; only matching portable copies are deduplicated.

Browser verification used real saved R5 outputs, replayed under a clearly labeled QA experiment on a separate local server. Queued, first-scene and second-scene publications arrived without reloading. The test then interrupted the manifest, grid, image and backend in turn, restored each, and checked the displayed result. The replay was not a new model run.

The complete local catalog had 50 experiment entries and 215 grouped US observations. Its selected November 1 R5 example still reported native IoU 38.8%, precision 64.4%, recall 49.3% and 19.36% evaluated area. The full development run remained at 2.15% pooled IoU across 65 scenes. Switching the same scene to S16 displayed that branch's different saved output and 5.4% native IoU.

Desktop checks covered the satellite mask, timeline navigation, filtering, linked image alignment, pixel inspection, exact-scene new-tab links, model switching, Back navigation and the original-team archive. A 390 by 844 viewport kept filters, scene selection and input-only comparison accessible without horizontal page overflow. Final browser error logs were empty. JavaScript syntax and Git whitespace checks passed.

## Inspect the actual output

- [Satellite overview and acquisition bar](assets/GTM_Model0_viewer_polish_20261002/national-satellite-timeline.jpg)
- [Comparison using real saved R5 imagery](assets/GTM_Model0_viewer_polish_20261002/comparison-desktop.jpg)
- [Narrow map layout](assets/GTM_Model0_viewer_polish_20261002/national-mobile.jpg)
- [Input-only comparison on a narrow screen](assets/GTM_Model0_viewer_polish_20261002/input-only-mobile.jpg)
- [Live publication replay](assets/GTM_Model0_viewer_polish_20261002/live-publication.jpg)
- [Backend outage with the current evidence retained](assets/GTM_Model0_viewer_polish_20261002/backend-offline.jpg)

## Repeat the checks

Run these from the repository root using Python 3.11 or newer:

```text
python tools/GTM_Model0_viewer_e2e.py --artifact output/viewer-http-e2e.json
python tools/GTM_Model0_local_viewer.py --open
```

Open the Map view, choose the saved R5 November 1 example, select Satellite and inspect the acquisition bar. Use a date marker and the arrow keys, then open the same scene in Compare. Change a branch when complete local exports are present. Fresh clones have only the portable sample, so unavailable branches should remain disabled. Inspect an input-only example and the original archive as well. Repeat at a narrow viewport. The screenshots and browser record show the expected states and selected artifact identities.

For a live update rehearsal, use a separate `--bundles-dir`, prepare a copy of the portable R5 bundle under a distinct QA experiment ID, and publish cumulative manifests containing zero, one and then two saved scenes. Include explicit `run` metadata for queued/running/completed status. Follow the [publisher and remote setup guide](../../docs/GTM_Model0_live_viewer.md). Observe the same open tab while publishing; do not reload to simulate discovery. Interrupt only the QA copies when checking recovery.

## Handoff limit

The viewer and publisher are ready to run on the backend machine. They have not been deployed there in this change. The producer must call the publisher after a complete export; the viewer does not attach itself to a training loop. Complete local runs and raw data are not included in Git. No national inference coverage, methane upscaling accuracy or source-detection improvement is claimed by this UI update.
