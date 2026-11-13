# EMIT v2025 pilot: a visible plume annotation over Sentinel imagery

> Public source publication note: Referenced generated viewer pages, images, local outputs, and runtime receipts are retained as research pointers but are omitted from this Git publication. Those links require the original local artifacts.

We now have a small, inspectable example of the supervision this project needs: a provider-annotated EMIT plume over a real Sentinel image of the same landfill, acquired 12.7 minutes later. This is a data-readiness result. It is not a new model prediction, an accuracy improvement, or proof that Sentinel alone can recover this plume.

Open the local review at `http://127.0.0.1:8766/GTM_Model6_emit_2025_intake.html`. Switch between Sentinel RGB and the EMIT matched-filter retrieval, toggle the cyan annotation boundary, and zoom. The full native positive and control images are available below the map. The map uses the actual projected raster coordinates; no online basemap or extra imagery downloads are needed.

## What was acquired and checked

The approved pass first compared the new release's US training records with existing sample identities and known research/evaluation locations. Of 1,793 US rows, 240 rows across 74 location names remained after exclusions. The audit conservatively excluded locations within 25 km of known MARS-S2L, MethaneS2CM and protected coordinates, old exact sample IDs, historical EMIT holdout identities, and rows below 90% declared clear. These are candidate observations, not 240 new independent sites. Thirty-nine historical US holdout names could not be resolved to coordinates, and the new release's validation/test inventories were not acquired. This cohort cannot be called a certified independent benchmark.

We froze 12 separated training events before requesting Sentinel metadata. Only the Volusia event had a qualifying match within one hour and below 30% scene cloud. Ten queries returned no scenes; the MAIR query returned two scenes above 76% cloud. Returned counts matched catalog matched counts in all saved replies, so this pass did not need extra result pages. The first matched provider-positive and the first preselected provider-control were frozen before opening masks. No replacement was selected based on how attractive the masks looked.

| Observation | Date | Native mask | Sentinel availability |
|---|---|---|---|
| Volusia landfill, Florida | 2025-06-01 | 1,503 positive cells of 65,536; full boundary retained | S2B_17RMN_20250601_0_L2A, 12.742 minutes later |
| Black Oak landfill, Missouri | 2025-02-02 | All zero; no annotated positive cells | No match in the queried one-hour window |

For each observation, five pinned files were downloaded: `plumemask.tif`, `rgb.tif`, `mf.tif`, `wmf.tif`, and `mag1c.tif`. All ten matched their published Git-blob or LFS hashes. The newer sample folders contain no `info.json`; metadata comes from the pinned training CSV. The files share a 256 by 256 native grid at 60 m, with matching CRS, transform and bounds within each observation. Both listed facility coordinates fall inside their rasters.

The Volusia Sentinel crop contains the complete positive-mask bounding box, the listed location, and a 300 m margin. It spans 3.68 by 3.32 km. SCL indicates 98.66% strict-clear crop area and 98.78% strict-clear annotated area using classes 4/5/6. Acquired products are RGB at native 10 m and SCL at 20 m. The displayed mask has exactly 54,108 positive 10 m display cells, retaining the area of all 1,503 native 60 m cells. Nearest-neighbor display resampling does not create new measured detail or a 10 m target.

## Problems caught before training

- Both provider `rgb.tif` files contain only ones in all four bands. They cannot supply photographic context or independently establish quality. The visible background is separately acquired Sentinel imagery.
- Both plume masks declare zero as nodata. We inspect raw codes and show zero as unannotated; we do not treat an all-zero mask as verified methane-free ground truth. Background validity and negative-label policy remain unresolved.
- Retrieval files do not declare physical units. Previews use stored values and are not labeled as methane concentration or emission rate.
- Earth Search marks the radiometric offset as applied while its band metadata still lists an offset. We preserve raw Sentinel DN and record that conflict. RGB uses a documented 2nd-to-98th-percentile display stretch, not an approved quantitative conversion.
- The native annotation is broad and contains a smaller bright retrieval feature near the landfill. Its polygon boundary is not a measured fine-resolution concentration boundary. The facility coordinate is not a verified emission origin.
- Close timing and plausible visual co-location do not establish exact cross-sensor alignment or a stationary plume. No source-origin accuracy is claimed.
- NIR, SWIR and a Sentinel reference observation were not acquired. The all-spectral preflight estimate was 15.25 MiB after header/SCL work, exceeding the 12 MiB local attempt ceiling. RGB fit; we did not shrink the plume crop to make a complete training pair appear available.

## Bandwidth and reproducibility

This pass received **13,135,957 bytes (12.53 MiB)** including bounded metadata, ten native files, range headers, SCL and RGB. The persistent ledger charged 14,770,585 bytes because metadata caps are reserved conservatively. Cumulative charged use is **98,205,830 / 104,857,600 bytes**, leaving **6,651,770 bytes (6.34 MiB)**. The ledger was not reset or raised. No full EMIT scenes, hyperspectral cubes or large auxiliary archives were downloaded.

Pinned source: `UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025` at `4e6c50999eb430fa33abc9488b71cd1a0dcd7991`. Training CSV SHA-256: `cbbee1489168630a4966c17117e65b831e580718339a0ba1c09e697f9f5a62ea`.

Artifacts are in `outputs/GTM_Model0_hf_pilot_20260925/emit_followup/`: source-overlap audit, frozen query and native selection, exact catalog responses, original TIFFs and hash receipts, native reviews, Sentinel preflight and acquisition receipts. Viewer assets and their hashes are under `outputs/GTM_viewer_bundles/GTM_Model6_emit_2025_intake/`. This data-review folder deliberately has no experiment manifest, so it cannot appear as a trained model in the experiment selector.

For repeatable offline rendering from this checkout in PowerShell:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) '.venv\Lib\site-packages')
python tools/GTM_Model6_review_emit_2025.py
```

The rendering path makes no network requests and does not alter original TIFFs or label values. The separate acquisition tools refuse unbounded responses, enforce the persistent ledger and preserve receipts. Do not reset the ledger or rerun acquisition as a way to obtain a new allowance.

## What this changes

This is a more concrete lead for plume supervision than a scene-wide filled mask: we can see a real provider annotation, its retrieval, and the corresponding land context. It justifies resolving label validity/units and completing a small, independently grouped paired cohort. It does not justify restarting training with the new files immediately. A complete paired positive and control set, required spectral/reference inputs, radiometric checks, and visible QA must come first. No model was trained or promoted, no labels were consolidated, and no Google research-journal entry, commit or push was made in this pass.

## Actual-output verification

The scientific previews were opened and visually inspected, including both full 15.36 km native scenes and the complete 3.68 by 3.32 km paired crop. Offline verification checked all ten native files against upstream hashes, all four Sentinel TIFFs against their acquisition receipts, and all six PNG assets against the viewer receipt. The verification artifact is `emit_followup/offline_verification.json`; it required no additional network transfer.

The local page was exercised in the actual in-app browser. An initial asset-URL error was fixed, and the final page displayed real imagery. Verified Sentinel/retrieval switching, mask hiding/restoration, opacity reduction from 90% to 15%, zoom and reset to the full extent, a metric scale, and expansion of both native figures. The figure captions correctly describe locally generated retrieval/mask previews, not the unusable provider RGB. No mobile-device verification was performed.

The existing private Tailscale route was also opened and visually verified with the Sentinel background and complete cyan boundary: `http://127.0.0.1:8766/GTM_Model6_emit_2025_intake.html`. It uses the already-running local viewer; no hosting or access configuration changed.

Repeat the browser check by opening that page, switching to **EMIT retrieval**, toggling **Provider mask boundary**, adjusting **Mask opacity**, zooming, and pressing **Reset extent**. Expand **Full native training examples** to compare Volusia's annotated mask with Black Oak's all-zero mask. The paired PNG and native figures are the repeatable visual artifacts; none depict a new model prediction.
