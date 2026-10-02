# Local experiment viewer

Run from the repository root with Python 3.11 or newer:

```text
python tools/GTM_Model0_local_viewer.py --open
```

On Windows, `GTM_Model0_OpenViewer.cmd` starts or reuses the viewer. It reads saved files; opening a page does not train a model or acquire imagery. Python's standard library is sufficient. No Node build, TensorFlow, Torch or GPU is needed.

## Pages

| Page | Address after starting the viewer |
|---|---|
| Methane research review | http://127.0.0.1:8766/friday_20261002/index.html |
| Maps and experiments | http://127.0.0.1:8766/ |
| Original team vs Model6 | http://127.0.0.1:8766/GTM_Model0_team_comparison.html |
| Volusia EMIT/Sentinel input | http://127.0.0.1:8766/GTM_Model6_emit_2025_intake.html |

Start with the research review. Select each case, zoom or drag an image to move all three panels together, then reset. Cyan marks the reference; warm colors mark the saved prediction. Gray stripes mean no model output. Volusia has no prediction. Scroll down for the architecture and open the PDF for the meeting handout.

The saved R5 model failed its quality gate. A selected good-looking example is not the overall result. The original-team archive is a rendered overlay without matched input/truth or a recorded run checkpoint, so it cannot support a performance ranking.

## Saved evidence

`portable_bundles/` contains selected real outputs and their provenance. The comparison and intake pages work with these files on a fresh clone. The Volusia receipt preserves the September 25 acquisition snapshot; B11/B12 were added on September 30, as described in the Friday review. The provider's native EMIT annotation is 60 m even when displayed on a finer grid.

Complete local experiments are read from `outputs/GTM_viewer_bundles/*/experiment.json`. Keep each experiment ID unique and put referenced assets beside its manifest. Restarting the server is not needed after adding a valid export. The main viewer lists the available experiments; the team-comparison page lists georeferenced Model6 plume results.

All review assets and Leaflet scripts are local. Optional online basemaps contact OpenStreetMap only when enabled. Raw training imagery and checkpoints are not required to view the checked-in sample.

## Share an offline review

```text
python tools/GTM_Model6_package_friday_review.py
```

This creates `output/friday_20261002/GTM_Model6_Friday_review_20261002.zip` and a hash-verification receipt. Extract the whole ZIP and open `index.html`; keep its assets folder beside it. The six cases, diagram and PDF are included. Links requiring the complete viewer are hidden outside the project's served page.

Packaging uses existing saved files. Regenerating the evidence images or PDF is a separate step and requires the original local research artifacts plus the dependencies used by `GTM_Model6_export_friday_evidence.py` and `GTM_Model6_build_friday_brief.py`.
