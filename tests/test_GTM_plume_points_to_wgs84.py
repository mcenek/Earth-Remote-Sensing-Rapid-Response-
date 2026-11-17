import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "GTM_plume_points_to_wgs84.py"
spec = importlib.util.spec_from_file_location("plume_converter", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class PlumeConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transform = Affine(30, 4, 500000, -3, -20, 5100000)
        self.raster = self.root / "source.tif"
        with rasterio.open(self.raster, "w", driver="GTiff", width=4, height=3,
                           count=1, dtype="float32", crs="EPSG:32610", transform=self.transform,
                           nodata=-9999) as dst:
            data = np.ones((3, 4), dtype=np.float32)
            data[2, 3] = -9999
            dst.write(data, 1)
        self.point = {"point_id": "p1", "scene_id": "s1", "site_id": "site1",
                      "source_raster": "source.tif", "observation_date": "2024-06-01",
                      "pixel_row": 1, "pixel_col": 2, "coordinate_convention": "pixel_index",
                      "geometry_type": "centroid", "provenance": "synthetic fixture",
                      "uncertainty_status": "unknown", "uncertainty_m": None}
        self.manifest = self.root / "manifest.json"

    def save(self, point=None, pins=None):
        self.manifest.write_text(json.dumps({"points": [point or self.point], **({"pins": pins} if pins else {})}), encoding="utf-8")

    def test_full_rotated_affine_crs_and_cli_exports_read_back(self):
        point = {**self.point, "pixel_row": 0.25, "pixel_col": 1.75,
                 "coordinate_convention": "fractional_index", "uncertainty_status": "known", "uncertainty_m": 12.5}
        pins = {"sdk_version": "test", "dataset_version": "test", "variant": "test", "year": 2024}
        self.save(point, pins)
        output = self.root / "export"
        subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(self.manifest), "--output", str(output)], check=True, capture_output=True, text=True)
        saved = json.loads((output / "points.json").read_text(encoding="utf-8"))
        record = saved["points"][0]
        x, y = self.transform @ (2.25, 0.75)
        expected = Transformer.from_crs(32610, 4326, always_xy=True).transform(x, y)
        self.assertAlmostEqual(record["lon"], expected[0], places=9)
        self.assertAlmostEqual(record["lat"], expected[1], places=9)
        self.assertEqual(record["source_transform"], list(tuple(self.transform)[:6]))
        self.assertEqual(len(record["source_raster_sha256"]), 64)
        self.assertEqual(saved["scaffold_points"][0]["uncertainty_m"], 12.5)
        with (output / "points.csv").open(encoding="utf-8", newline="") as handle:
            csv_point = next(csv.DictReader(handle))
        geo = json.loads((output / "points.geojson").read_text(encoding="utf-8"))
        self.assertEqual(float(csv_point["lon"]), record["lon"])
        self.assertEqual(geo["features"][0]["geometry"]["coordinates"], [record["lon"], record["lat"]])
        self.assertEqual(geo["manifest_sha256"], saved["manifest_sha256"])

    def test_unknown_uncertainty_not_scaffold_ready(self):
        self.save(self.point, {"sdk_version": "test", "dataset_version": "test", "variant": "test", "year": 2024})
        result = converter.convert(self.manifest)
        self.assertIsNone(result["points"][0]["uncertainty_m"])
        self.assertEqual(result["scaffold_points"], [])

    def test_each_raster_uses_its_own_affine(self):
        other = self.root / "other.tif"
        other_transform = Affine(0.02, 0, -120, 0, -0.03, 40)
        with rasterio.open(other, "w", driver="GTiff", width=4, height=3, count=1,
                           dtype="uint8", crs="EPSG:4326", transform=other_transform) as dst:
            dst.write(np.ones((3, 4), dtype="uint8"), 1)
        second = {**self.point, "point_id": "p2", "source_raster": "other.tif"}
        self.manifest.write_text(json.dumps({"points": [self.point, second]}), encoding="utf-8")
        first, latter = converter.convert(self.manifest)["points"]
        self.assertNotEqual((first["lon"], first["lat"]), (latter["lon"], latter["lat"]))
        self.assertAlmostEqual(latter["lon"], -119.95)
        self.assertAlmostEqual(latter["lat"], 39.955)

    def test_rejects_invalid_input_and_nodata(self):
        variants = [
            ({**self.point, "pixel_col": 4}, "outside"),
            ({**self.point, "pixel_row": float("nan")}, "finite"),
            ({**self.point, "pixel_col": 1.2}, "integer"),
            ({**self.point, "pixel_row": 2, "pixel_col": 3}, "nodata"),
            ({**self.point, "uncertainty_status": "known", "uncertainty_m": None}, "uncertainty"),
        ]
        for point, phrase in variants:
            with self.subTest(phrase=phrase):
                self.save(point)
                with self.assertRaisesRegex(ValueError, phrase):
                    converter.convert(self.manifest)

    def test_rejects_raster_without_georeference(self):
        plain = self.root / "plain.tif"
        with rasterio.open(plain, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint8") as dst:
            dst.write(np.ones((2, 2), dtype="uint8"), 1)
        self.save({**self.point, "source_raster": "plain.tif", "pixel_row": 0, "pixel_col": 0})
        with self.assertRaisesRegex(ValueError, "lacks CRS"):
            converter.convert(self.manifest)

    def test_fractional_point_checks_actual_containing_pixel_and_footprint(self):
        # Index (1.8, 2.8) becomes corner (2.3, 3.3), in nodata cell (2, 3).
        self.save({**self.point, "pixel_row": 1.8, "pixel_col": 2.8,
                   "coordinate_convention": "fractional_index"})
        with self.assertRaisesRegex(ValueError, "nodata"):
            converter.convert(self.manifest)
        self.save({**self.point, "pixel_col": 3.8, "coordinate_convention": "fractional_index"})
        with self.assertRaisesRegex(ValueError, "outside"):
            converter.convert(self.manifest)

    def test_existing_exports_preserved(self):
        self.save()
        output = self.root / "export"
        result = converter.convert(self.manifest)
        converter.write_exports(result, output)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            converter.write_exports(result, output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
