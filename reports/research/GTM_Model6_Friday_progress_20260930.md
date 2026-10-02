# Friday research preparation

Prepared September 30 for the October 2, 2026 meeting with Martin.

The deliverable is a local visual demo with a short PDF handout. It shows the new Volusia input/reference pair, all three saved R5 plume cases, the two saved R5 no-plume scenes with the most false-positive pixels, and Martin's proposed four-scale architecture with a later learned combiner. No new model output is invented for Volusia.

## Data progress

The separately approved September 30 pass completed B11 and B12 for the previously selected Volusia crop. It did not reset or increase the persistent 100 MiB ledger or alter the September 25 receipts. The pass had a 3 MiB ceiling and received exactly 2,883,584 bytes (2.75 MiB). The two 166 by 184 rasters are native 20 m, match the SCL grid exactly, contain no nodata pixels, and were opened for visual inspection. RGB, B11 and B12 now form the five target bands. Missing NIR is not a universal blocker for this five-band baseline.

The remaining allowance is 3,768,186 bytes (3.59 MiB). Cumulative charge is 101,089,414 of 104,857,600 bytes; actual received bytes are 96,627,506. Local rendering and the offline demo add no satellite-data transfers.

The new files and checks are under `outputs/GTM_Model0_hf_pilot_20260925/emit_followup/swir_20260930/`. `receipt.json` binds them to the prior preflight, unchanged scene/grid and exact source URLs; `input_readiness.json` records why training is still blocked. The reproducible acquisition entry point is `tools/GTM_Model6_complete_volusia_swir.py`; it preserves an existing attempt rather than silently reusing an allowance.

## What the source review resolved

The exact [v2025 provider card](https://huggingface.co/datasets/UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025) does not document mask-zero meaning, retrieval calibration or physical units. The pinned card was also sparse. Older releases can suggest questions to ask but do not establish exact-version semantics. The sampled TIFFs contain band names, but those names do not supply units. We retain provider-annotated positive cells and leave background validity unresolved.

[Element 84's documentation](https://github.com/Element84/earth-search/blob/main/README.md#gainoffset-in-items-after-jan-25-2022) describes applying a nonzero per-asset offset after scaling. The exact saved item also marks `earthsearch:boa_offset_applied=true`. A [reported legacy-collection offset issue](https://github.com/Element84/earth-search/issues/66) reinforces the need for an exact-source check; it is not proof about our particular scene. Neither the raw values' appearance nor a generic flag resolves the conflict. Raw DN remains unconverted pending verification against the source product or another demonstrably equivalent representation.

Training therefore remains blocked by target semantics, radiometric verification, a missing reference-date image and paired reviewed negative, and the absence of an independent multi-site cohort. A one-pair memorization run would not address those failures and was not run.

## Meeting walkthrough

1. Open the Volusia case. Show the Sentinel image and EMIT retrieval/annotation. State that this is the input and reference we want to learn from; the prediction panel is intentionally empty.
2. Select the localized R5 result, followed by the missed plume and largest false-positive control. Point out that all three positive cases and both highest-false-positive controls are available. The pooled development IoU is 2.15%, not the selected example's 38.8% overlap. Existing R5 failed its quality gate.
3. Show the architecture diagram. Four models see 16, 32, 64 and 128 pixel contexts and predict the same central area. The learned second-stage combiner is still proposed; the saved run used a fixed average. Quantitative upscaling and source-origin estimates require separate validation.
4. Agree on the first acceptance criterion, request exact label/calibration documentation or a provider contact, and identify sources of reviewed negatives plus independent source/wind information.

The project objective is unchanged: screen the US using Sentinel imagery, use EMIT for supervision and validation, and use CAFO/oil matches as corroborating evidence. The first milestone question concerns sequencing and acceptance, not replacing that objective.

## Completed deliverables and verification

Completed overnight September 30 to October 1. The demo is `GTM_Viewer/friday_20261002/index.html`, linked from the main experiment viewer. Its six cases show 17 saved image panels, linked zoom/pan/reset, keyboard navigation, run-wide failure scores, a full-size architecture diagram, and meeting decisions. The local and existing private Tailscale routes both loaded the actual images. The new Volusia prediction is explicitly absent. All five selected native confusion counts were recomputed from their saved NPZ arrays and matched the original scene metrics.

The four-page handout is `output/pdf/GTM_Model6_Friday_brief_20261002.pdf`. All four pages were rendered and visually checked, including the final revised image shading. The PDF downloaded through the portable UI is byte-identical to the generated file. Poppler emitted local fallback-font diagnostics, but the rendered pages contain no clipped text or missing glyphs.

The portable package is `output/friday_20261002/GTM_Model6_Friday_review_20261002.zip` (2,481,039 bytes). All 26 payload files matched their hashes after extraction. A separate loopback server served only the extracted folder for browser verification: its images, cases, full-size diagram and PDF work without the original project assets; links requiring the full viewer are hidden. The temporary verification server was stopped after checking. Direct file-URL launch and phone-sized screens were not exercised. The package uses local scripts/images without a build step, API call, remote font or map tile.

Repeatable evidence is in `GTM_Viewer/friday_20261002/verification.json`, `SHA256SUMS.json`, and `output/friday_20261002/package_verification.json`. Actual browser screenshots and case checks are under `output/friday_20261002/qa/`. The six cases loaded without broken image panels or horizontal page overflow at the observed 999-pixel-wide viewport. Linked images shared the same zoom and pan transform, reset returned all images to 100%, and left-arrow navigation moved to the preceding case. A missing optional favicon on the standalone test server did not affect required assets.

To reproduce from the repository: run `tools/GTM_Model6_export_friday_evidence.py` with the existing NumPy/Pillow environment, `tools/GTM_Model6_build_friday_brief.py` with the bundled ReportLab runtime, then `tools/GTM_Model6_package_friday_review.py`. Open the demo, inspect every case, zoom and drag, reset, use the left/right arrows, open the diagram and download the PDF. The export uses existing data and saved predictions; it does not retrain or transfer satellite data.

## No external publication

The Google research journal was not changed. No email was sent, no repository commit or push was made, and no new public hosting or access was configured. The existing private viewer route can serve the demo; the portable archive is intended for sharing by Joshua.
