# Plume pixel coordinates to geographic coordinates

This handoff converts reviewed plume pixel positions to WGS84 longitude and latitude using the matching source raster. It requires no model checkpoint or Tessera access.

## Get the tool

After this change is merged, update your normal checkout. Before merging, fetch and check out `research/plume-coordinate-handoff-20261116` in a separate checkout. The tool is `tools/GTM_plume_points_to_wgs84.py`.

Use Python with NumPy, rasterio and pyproj. If those are not already installed, create a project-local environment:

```sh
python -m venv .venv-plume-points
# Windows: .venv-plume-points\Scripts\activate
# Linux/macOS: source .venv-plume-points/bin/activate
python -m pip install numpy rasterio pyproj
python -m unittest discover -s tests -p test_GTM_plume_points_to_wgs84.py -v
```

## Supply the reviewed points

Keep each source GeoTIFF and its point manifest together. `source_raster` is relative to the manifest, or an absolute path on the machine running the tool. The raster must retain its own CRS and affine transform. A PNG screenshot or an array without its original georeference is insufficient. The paths below are placeholders, not a published dataset location.

```json
{
  "points": [{
    "point_id": "reviewed-point-001",
    "scene_id": "scene-001",
    "site_id": "site-001",
    "source_raster": "imagery/scene-001.tif",
    "observation_date": "2024-06-01",
    "pixel_row": 12.25,
    "pixel_col": 8.75,
    "coordinate_convention": "fractional_index",
    "geometry_type": "centroid",
    "provenance": "Reference to the reviewed mask and extraction method",
    "uncertainty_status": "unknown",
    "uncertainty_m": null
  }]
}
```

Row means Y and column means X. Both are zero-based pixel-center indices: `(0, 0)` means the center of the top-left pixel. Integer positions use `pixel_index`; subpixel center-of-mass positions use `fractional_index`. Coordinates already measured from pixel corners must be converted to this convention first. Do not resize or crop the raster independently of the point coordinates.

The converter reads each raster's resolution, rotation and CRS separately. It rejects missing georeferencing, points outside the raster, and masked/nonfinite band-1 samples. This sample check does not validate the plume annotation or all spectral bands.

## Run on the machine holding the imagery

```sh
python tools/GTM_plume_points_to_wgs84.py --manifest /path/to/plume_pixels.json --output /path/to/new_point_exports
```

Use a fresh output directory. The tool preserves existing exports. It writes and reads back `points.json`, `points.csv` and `points.geojson`. These record longitude, latitude, source raster SHA-256, CRS, affine, input manifest hash and point provenance. Outputs include local source paths, so review them before sharing.

A centroid can fall outside a concave plume. If an interior point is required, the extraction method must explicitly check containment against the reviewed mask or polygon. Independent X/Y medians do not guarantee containment. `interior` records the intended geometry type; this converter does not perform that containment check. Neither a centroid nor an interior point establishes the physical emission source. Reserve `verified_source` for independently verified sources.

Unknown uncertainty is valid for coordinate export and remains unknown. It must not be replaced by an invented accuracy. Optional annual embedding pins and scaffold integration are described in the existing research branch; they are not required for this standalone conversion.

## Ready for real processing

The next input is the reviewed point file, its matching georeferenced rasters, and the extraction-method description. The imagery may remain on its current machine; run the converter there rather than duplicating the dataset. No dataset download, embedding acquisition, classifier training or methane prediction is triggered by this tool.
