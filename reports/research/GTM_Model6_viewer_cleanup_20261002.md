# Viewer cleanup, October 2, 2026

## Acceptance recorded before editing

- A checkout without local `outputs/` must open the Friday review, the original-team comparison, and the Volusia intake. All declared images and receipts must resolve.
- Review links must select the intended saved R5 scene on that checkout. The original-team image remains an unscored historical rendering.
- Keep saved prediction pixels, native counts, model source, checkpoints, raw imagery, and earlier evidence unchanged.
- Ignore generated `output/` files. Package only declared review assets, never unrelated screenshots or stale extraction contents.
- Keep current setup instructions short and remove stale claims about the active study. Preserve historical reports.
- Exercise the public packaging command with missing, unsafe, duplicate, and unrelated files. Check the resulting ZIP and a fresh extraction, then inspect the viewer in a browser, including case switching and linked zoom/reset.

## Scope

This is a viewer and repository cleanup, not a new experiment. Historical model files are retained because saved manifests and receipts depend on their paths and hashes. Unrelated temporal-attention work is outside this change.

## Verification

The public packaging command passed 11 checks: two successful exports and nine deliberately invalid inputs. Unrelated files were excluded, the previous extraction was preserved, and all 26 declared payload files matched their SHA-256 hashes. The package is 2,481,059 bytes; SHA-256 `2cd2ea3c2b611e5cf1696a0975df32ee2c49ec7bd9aa0770255f47b74b9ddfb4`.

A separate local server served a copy containing only viewer files, the server script and design documents, with zero local run bundles. Its registry reported no warnings. All 47 registered portable asset URLs, five intake images and 26 Friday package hashes passed HTTP checks. The PDF matched the saved handout byte for byte.

Browser review covered all six cases, linked zoom/reset, the original-team and Model6 maps, switching the intake background, and the formerly missing false-alarm scene link. That link selected the intended R5 scene and displayed its original 717 native false-positive pixels. The figures and architecture rendered without broken images. Five model cases showed no horizontal page overflow at the observed 1,432-pixel viewport. No new phone-layout or model-quality claim is made.

The baseline hash check preserved all 52 checked model-source, image and existing R5 asset files. Three metadata files changed: the portable scene list, the Friday viewer links, and their verification receipt. An additional existing R5 scene and saved archive/intake exports were copied into the portable sample. Text line endings are fixed at LF so Windows checkouts retain the package hashes. Raw imagery, weights and native scores were not edited.

## Replay

1. Start `python tools/GTM_Model0_local_viewer.py --port 8768` from a fresh clone without local `outputs/` data.
2. Open `http://127.0.0.1:8768/friday_20261002/index.html`. Select every case, zoom, reset, and follow the final false-alarm case into the full viewer. Check the native FP count is 717.
3. Open the original-team comparison and Volusia intake from the viewer. Check both maps and toggle Sentinel/EMIT backgrounds. The intake remains explicitly unscored.
4. Run `python tools/GTM_Model6_package_friday_review.py --output-dir output/cleanup_replay`. Its receipt names a fresh extracted directory and verifies every declared asset. Repeat the command; the extraction path must change while earlier files remain.
5. Open the extracted `index.html` and its PDF. The research review must work without the complete viewer's local cache.

The [verification receipt](assets/GTM_Model6_viewer_cleanup_20261002/verification.json) and [browser screenshot](assets/GTM_Model6_viewer_cleanup_20261002/review.jpg) retain the checks from this pass. Detailed local fault-case fixtures and screenshots are in ignored `output/viewer_cleanup_20261002/`.
