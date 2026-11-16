# Offline Tessera research scaffold

## Preparing reviewed pixel points

`tools/GTM_plume_points_to_wgs84.py` converts reviewed plume pixel positions from their **own georeferenced source rasters** to WGS84. It is local geometry preparation only. It does not locate methane sources, fetch Tessera data, or run a model. From the repository root, using an environment with `rasterio` and `pyproj`:

```powershell
python tools/GTM_plume_points_to_wgs84.py --manifest path/to/plume_pixels.json --output path/to/converted_points
```

Example input (paths are relative to the manifest):

```json
{
  "pins": {"sdk_version": "explicit-version", "dataset_version": "explicit-version", "variant": "explicit-variant", "year": 2024},
  "points": [{
    "point_id": "p-001", "scene_id": "scene-001", "site_id": "site-001",
    "source_raster": "scene-001.tif", "observation_date": "2024-06-01",
    "pixel_row": 12, "pixel_col": 8, "coordinate_convention": "pixel_index",
    "geometry_type": "centroid", "provenance": "review record or file reference",
    "uncertainty_status": "unknown", "uncertainty_m": null
  }]
}
```

`pixel_index` requires integer zero-based row and column. `fractional_index` permits a subpixel center-of-mass position in the same index space. Both conventions add 0.5 to row and column before applying the source raster's full affine transform. Thus `(0, 0)` is the center of the top-left pixel, while `(0.25, 0.75)` is displaced by those fractions from that center. A center-of-mass already expressed in corner coordinates must be converted to this index convention before input. Each shifted position must lie within the raster footprint; its containing band-1 pixel must be readable, finite, and unmasked. A raster without a declared nodata/mask cannot prove data validity beyond a finite sample. No image size, guessed bounding box, or assumed resolution supplies georeferencing.

Outputs are `points.json`, `points.csv`, and `points.geojson`. Use a fresh output directory: existing exports are preserved. Every export is read back and checked after writing. They include scene and point identity, geometry type, uncertainty status, provenance, the exact source raster path, source raster SHA-256, CRS, affine, and input manifest SHA-256. `points.json` also has `scaffold_points`, containing only records with known numeric uncertainty when all explicit scaffold pins are present. To use the existing offline scaffold, place those records under its `points` key and use the explicit `pins` object. Unknown uncertainty stays unknown rather than receiving an invented accuracy. `centroid` marks plume geometry, and may even fall outside a concave plume. `interior` labels a point the reviewer intends to be within plume geometry. Only `verified_source` may describe a source that was independently verified, and this converter does not perform that verification. No segmentation mask check is made here; review containment separately when a reviewed mask is supplied.

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
