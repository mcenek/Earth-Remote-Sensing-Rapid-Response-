from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import rasterio
    from rasterio.transform import from_origin
    from GTM_Model6_native_pair_readiness import _read_masked, grid_signature, peak_in_s2_footprint, product_meta, timing_pass
    RASTERIO_OK = True
except (ImportError, PermissionError):
    RASTERIO_OK = False


@unittest.skipUnless(RASTERIO_OK, "rasterio/native DLL unavailable in this interpreter")
class NativePairReadinessTests(unittest.TestCase):
    def raster(self, directory: str, name: str, values, transform=None, nodata=-9999.0):
        path = Path(directory) / name
        with rasterio.open(path, "w", driver="GTiff", height=len(values), width=len(values[0]), count=1, dtype="float32", crs="EPSG:4326", transform=transform or from_origin(-1, 1, 0.25, 0.25), nodata=nodata) as dst:
            dst.write(np.asarray(values, dtype="float32"), 1)
        return path

    def test_geometry_mask_excludes_off_footprint_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.raster(directory, "native.tif", [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
            shape = {"type": "Polygon", "coordinates": [[[-0.95, 0.05], [-0.95, 0.95], [-0.50, 0.95], [-0.50, 0.05], [-0.95, 0.05]]]}
            _, values, valid, inside = _read_masked(path, [shape], "EPSG:4326", (-1, 0, 0, 1))
            self.assertEqual(int(valid.sum()), 8)
            self.assertEqual(int(inside.sum()), 8)
            self.assertEqual(values.shape, (4, 4))

    def test_peak_transform_distinguishes_inside_and_outside(self):
        bounds = (735480.0, 3786140.0, 740600.0, 3791260.0)
        self.assertTrue(peak_in_s2_footprint((34.21, -84.42), "EPSG:32616", bounds))
        self.assertFalse(peak_in_s2_footprint((34.28, -84.36), "EPSG:32616", bounds))

    def test_fractional_bounds_preserve_boundary_pixel_centers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.raster(directory, "fractional.tif", [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
            shape = {"type": "Polygon", "coordinates": [[[-0.76, 0.24], [-0.76, 0.99], [-0.01, 0.99], [-0.01, 0.24], [-0.76, 0.24]]]}
            _, _, valid, inside = _read_masked(path, [shape], "EPSG:4326", (-0.76, 0.24, -0.01, 0.99))
            self.assertEqual(valid.shape, (4, 4))
            self.assertEqual(int(inside.sum()), 9)

    def test_grid_signature_detects_affine_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.raster(directory, "a.tif", [[1, 2], [3, 4]])
            second = self.raster(directory, "b.tif", [[1, 2], [3, 4]], transform=from_origin(-1, 1, 0.2, 0.2))
            self.assertNotEqual(grid_signature(product_meta(first)), grid_signature(product_meta(second)))

    def test_timing_boundary_is_inclusive(self):
        self.assertTrue(timing_pass(6.0))
        self.assertTrue(timing_pass(-6.0))
        self.assertFalse(timing_pass(6.001))


if __name__ == "__main__":
    unittest.main()
