# Offline Tessera research scaffold

This additive prototype accepts stable point and site IDs, longitude then latitude, observation date, geometry type, uncertainty, and provenance. It checks one explicit annual embedding pin set. `verified_source` means the input coordinate was verified by a future source review; the scaffold itself does not verify it. `centroid`, `interior`, and `background` remain distinct. A later classification workflow can use neighborhood vectors with their site, spatial group, year, and geometry type. There is no classifier, fit, or prediction here.

Run from this directory with the existing Python and NumPy:

```powershell
python -X utf8 -m unittest discover -s tests -v
python -X utf8 cli.py demo
python -X utf8 cli.py run --manifest demo_output/manifest.json --region demo_output/synthetic_region.npz --output demo_output/reloaded
```

The demo writes a tiny synthetic `demo_output/` fixture and invokes the same saved-file loader as `run`. The manifest has `pins` and `points`; the NPZ has vectors, coverage, affine transform, CRS, year, variant, SDK version, and dataset version. The loader uses `allow_pickle=False`, verifies the independent pins, then writes JSON, CSV, and GeoJSON. The demo reads these exports back and checks known rank, pixel-center location, and provenance. A no-match run still writes a CSV header and an empty GeoJSON collection. `result.json` preserves the grid CRS, affine, and shape. No network, GeoTessera SDK import, or SDK construction occurs on this path. The injected `GeoTesseraAdapter` is a future interface only. Its factory requires an explicit `allow_live=True`, and the injected client must confirm pinned SDK, dataset, and variant metadata. The fake client test checks the documented call shapes, not live compatibility.

The fixture provider stores an H,W,128 array with per-pixel coverage (`valid`, `missing`, `water`, `out_of_bounds`). Invalid, zero, or nonfinite vectors abstain. Ranking uses bounded cosine similarity and deterministic pixel row/column tie order. It excludes all seed pixels. Haversine meters are reported separately from similarity. Only north-up EPSG:4326 affine pixel centers are supported. The synthetic polygon and nearest-point helpers are geometry checks, not an inventory join.

Annual embeddings describe surrounding context. Similarity does not establish methane, a physical source, or an SVM margin. The annual feature may include observations later than a point's date, so no prospective date-specific claim follows. Future work needs Sean's reviewed coordinates, coverage and temporal review, controlled labels/splits, and explicit permission before any live Tessera access. This scaffold does not compute facility performance metrics.

The future adapter's call shapes were checked against the [official GeoTessera API documentation](https://geotessera.readthedocs.io/en/stable/geotessera.html) and [upstream SDK](https://github.com/ucam-eo/geotessera). Connecting the actual SDK, handling its coverage metadata, and checking a real region remain future integration work.
