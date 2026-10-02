GTM Model6 | Visual research review for Martin | October 2, 2026
Prepared September 30, 2026

OPEN THE REVIEW
Extract the entire ZIP, then open index.html in a browser. Keep the assets
folder beside it. No installation, login, internet or GPU is needed.
GTM_Model6_Friday_brief_20261002.pdf is the four-page meeting handout.
Optional links back to the complete research viewer appear only when this
review is served from its folder in the original project.

FIVE-MINUTE WALKTHROUGH
1. Start with Volusia. Left is Sentinel imagery. Middle shows the EMIT
   retrieval with its provider annotation. The third panel is intentionally
   empty: this scene has no new model prediction.
2. Open the localized R5 example. Compare the image, reference and actual
   saved prediction. Cyan outlines the reviewed mask. Warm colors show the
   prediction. Dimmed areas are outside the valid comparison support.
3. Open the missed plume and the two false-positive controls. All three
   positive cases are included, along with the two highest false-positive
   controls. This is a diagnostic selection, not a representative sample.
4. Use + / - or the mouse wheel to zoom all image panels together. Drag a
   zoomed image to pan. Reset restores the view. Left/right arrows switch cases.
5. Scroll to Martin's architecture and the meeting decisions. The dashed
   components are proposed work, not completed model capabilities.

WHAT WE CAN HONESTLY SAY
- The new Volusia pair has RGB plus both shortwave-infrared target bands.
  Sentinel observed it 12.7 minutes after EMIT. The full annotation is retained.
- The native EMIT reference is 60 m. A sharper display is not finer measured truth.
- The saved R5 run failed: 2.15% pooled native IoU and 2.25% precision over
  65 reused development scenes. One better example does not override that.
- R5 averaged four independently trained context models. The learned second
  combining model, quantitative methane upscaling and source-origin fit
  still need trustworthy training targets and their own evaluation.
- No new model was trained for this handout. Zero-valued background labels,
  retrieval units and the exact Sentinel radiometric conversion remain
  unresolved. We also need a reference-date image and reviewed negative pairs.

MEETING QUESTIONS
- Agree on the first visual and numerical acceptance criteria for plume
  recovery and false alarms before expanding to a US-wide scan.
- Ask for exact provider definitions of masks, retrieval units and quality
  flags, or a contact who can explain them.
- Identify reviewed negative examples and independent source/wind records.
  A known facility coordinate alone does not prove methane or source origin.

PROVENANCE AND REUSE
The evidence export records input hashes, counts and image files in
verification.json. SHA256SUMS.json verifies the complete portable package.
R5 checkpoint: 39f2ca29688f90394cbe78c4521d158968f1901e3c1255e88900cac8457eccc4
R5 frozen decision threshold: 0.50. Display resampling is not used for native scores.

New EMIT source: UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025
https://huggingface.co/datasets/UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025
Pinned revision: 4e6c50999eb430fa33abc9488b71cd1a0dcd7991
Provider card license: CC-BY-NC-SA-4.0. Source-derived review images preserve
that attribution; this bundle does not grant new rights in source datasets.
Sentinel scene: S2B_17RMN_20250601_0_L2A, via Earth Search.
Radiometric documentation:
https://github.com/Element84/earth-search#gainoffset-in-items-after-jan-25-2022
An issue report raises a question, not a finding about our exact scene:
https://github.com/Element84/earth-search/issues/66
