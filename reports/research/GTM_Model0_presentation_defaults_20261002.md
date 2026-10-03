# Prediction-first viewer handoff

## Failure cases and acceptance recorded before implementation

1. A fresh GitHub checkout must open a saved prediction and reference in Compare. It must not select an empty registry entry or depend on the local training cache.
2. The default experiment selectors and map must contain only the two portable prediction collections. The six R5 cases must retain both overlaps and failures. The three EMIT support cases must remain labeled as positive-only references.
3. A full local cache must not replace this presentation with dozens of folds and diagnostics. A visible archive control must restore the complete catalog. Existing direct links must still select the requested historical experiment.
4. Each presentation scene must serve its original input, prediction, reference and numeric grid. The starting scene must contain nonzero predicted pixels at its saved threshold. Prediction pixels, support, thresholds and original scores must not change.
5. Map and Compare must remain linked. Selecting Map must show the selected scene at a useful scale. Zooming out must explain why scene overlays are not visible.
6. Archived continuous reconstruction grids must render in the map and inspector, with their units and task identified. They must not be presented as plume masks.
7. Refresh and live publication must preserve the selected scene and catalog mode. A missing preferred experiment must produce an available fallback.
8. Browser checks must cover first load, both presentation collections, overlay controls, a missed plume, a false alarm, archive switching, a historical reconstruction and fresh-clone behavior. Save screenshots and the HTTP E2E receipt.
9. Visual inspection found that the offline state fill dimmed the saved imagery and prediction. Move contextual state fills into a lower pane, then verify both Map and Satellite after boundaries have loaded. Saved scene pixels must retain their display colors.
10. The opening comparison must give the prediction the large overlay panel. The original image and reference must remain available beside it. Start presentation must restore this layout after a user selects another layout.

## Scope

This change presents existing inference outputs. The R5 run failed its quality gate; the selected overlap is not evidence of overall success. Training, thresholds, raw data and model weights are unchanged. New inference is necessary only if a usable saved checkpoint lacks the required outputs.

## Verification

Completed on Windows on October 2, 2026.

- HTTP E2E: 68 checks passed, zero skipped. This includes all four required assets for each of nine scenes, numeric probabilities, nonzero opening predictions, publication discovery and invalid-bundle rejection. Receipt: [http-e2e.json](assets/GTM_Model0_presentation_20261002/http-e2e.json).
- Portable startup: copied Git-tracked viewer runtime files and portable bundles into a separate folder with no training outputs. Started the normal server on port 8770. The root URL opened Compare, selected R5 scene `MARS_2c6ef011-0729-42e2-b64f-10ff8207778b`, and displayed the large prediction overlay with two model options. [Screenshot](assets/GTM_Model0_presentation_20261002/portable-default.jpg) and [DOM receipt](assets/GTM_Model0_presentation_20261002/portable-default-dom.txt).
- Both collections: selected the EMIT collection and its default scene `emit25km-0049`. Map showed the same selected scene at 32.08332, -104.47770. Native support recall was 8.4%; precision and IoU were labeled not measured. [Map](assets/GTM_Model0_presentation_20261002/emit-map.jpg), [Satellite](assets/GTM_Model0_presentation_20261002/emit-satellite.jpg).
- Failures retained: June 4 R5 case had native IoU 0 and 257 false-positive pixels. October 27 control had 717 native false-positive pixels. Checked Disagreement and Prediction off/on. [Miss](assets/GTM_Model0_presentation_20261002/missed-plume.jpg), [false alarm](assets/GTM_Model0_presentation_20261002/false-alarm.jpg), [prediction off](assets/GTM_Model0_presentation_20261002/prediction-off.jpg).
- Archive: the full local cache exposed all 50 experiments. A direct Fold 5 reconstruction link selected the archive automatically and displayed the saved continuous grid with unverified legacy units. [Screenshot](assets/GTM_Model0_presentation_20261002/archive-reconstruction.jpg). Start presentation restored the two collections, selected case, enabled layers and large overlay layout.
- Map display: offline state fill no longer covers saved imagery. Both Map and Satellite showed the actual orange and cyan layers at scene scale. Selecting US hid scene overlays and displayed the Fit scene instruction. Fit scene restored them. Geographic links now retain the fitted location.
- JavaScript syntax checks and `git diff --check` passed. Browser error log was empty after the portable interaction checks.

The viewer update does not improve model accuracy. The opening scene has 38.8% native IoU, but the full R5 run still has 2.15% IoU and fails its quality gate. The EMIT cases remain positive-support development evidence, with unknown background labels. All saved prediction files and benchmark scores are unchanged. No new training, inference, model downloads or imagery acquisition occurred. Optional online satellite context was loaded during the UI check. macOS execution has not been tested.

## Repeat the check

1. Run `python tools/GTM_Model0_viewer_e2e.py --artifact output/viewer-e2e.json` from the repository root.
2. Start `python tools/GTM_Model0_local_viewer.py --open` with no private training folders available.
3. Open the root URL. Confirm Compare, Overlay, two model choices and the November 1 R5 case.
4. Select the EMIT collection. Confirm the cyan outline and orange prediction. Confirm unknown background labels.
5. Select Map, then US, then Fit scene. Toggle Satellite. Confirm imagery is not dimmed by the state fill.
6. Select Full research archive and open an older run. Select Start presentation. Confirm the default collection and overlay return.
7. Select the missed-plume and no-plume cases. Check their overlays and unchanged failure scores.

Future published bundles appear in Full research archive. The New results control opens that collection. To add a reviewed presentation example, set its bundle ID, label and opening scene in the registry's `presentation.experiments` list.
