"""Synthetic contract tests for the isolated CloudSEN12 runtime."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cloud_runtime_v16", ROOT / "tools/GTM_Model7_cloud_runtime_v16.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def synthetic_tiff(path: Path, bands=runtime.BANDS):
    rng = np.random.default_rng(1601)
    data = rng.integers(100, 12000, size=(13, 200, 200), dtype=np.uint16)
    data[0, 10, 20] = 0
    data[12, 30, 40] = 0
    with rasterio.open(path, "w", driver="GTiff", height=200, width=200, count=13,
                       dtype="uint16", crs="EPSG:32613", transform=from_origin(500000, 4500000, 10, 10), nodata=0) as dst:
        dst.write(data)
        for index, band in enumerate(bands, 1):
            dst.set_band_description(index, band)
    return data


class CloudRuntimeTests(unittest.TestCase):
    def test_registry_contract_and_no_module_leak(self):
        site = runtime.RUNTIME / "site"
        self.assertEqual(runtime.sha256(site / "cloudsen12_models/cloudsen12.py"), runtime.PACKAGE_SHA256)
        self.assertEqual(runtime.published_bands(site / "georeader/readers/S2_SAFE_reader.py"), runtime.BANDS)
        before = {name: sys.modules.get(name) for name in ("georeader.plot", "georeader.readers.S2_SAFE_reader")}
        model = runtime.load_model()
        self.assertEqual(tuple(model.bands), runtime.BANDS)
        for name, previous in before.items():
            self.assertIs(sys.modules.get(name), previous)

    def test_package_source_prediction_and_geotiff_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, output, receipt_path = (folder / name for name in ("synthetic_input.tif", "synthetic_cloud.tif", "receipt.json"))
            data = synthetic_tiff(source)
            original = data.copy()
            loaded, invalid, profile = runtime.read_input(source)
            self.assertTrue(np.array_equal(loaded, original))
            self.assertEqual(int(invalid.sum()), 2)
            model = runtime.load_model()
            canonical = loaded.astype(np.float32) / 10000.0
            canonical[:, invalid] = 0.0
            package_output = np.asarray(model.predict(canonical), dtype=np.uint8)
            self.assertEqual(package_output.shape, (200, 200))
            expected = package_output.copy()
            expected[invalid] = 4
            class Spy:
                def __init__(self, target):
                    self.target = target
                    self.seen = None
                def predict(self, array):
                    self.seen = array.copy()
                    return self.target.predict(array)
            spy = Spy(model)
            self.assertTrue(np.array_equal(runtime.predict_mask(spy, loaded, invalid), expected))
            self.assertTrue(np.array_equal(spy.seen, canonical))
            self.assertTrue(np.all(spy.seen[:, invalid] == 0.0))
            receipt = runtime.infer(source, output, receipt_path, "l1c_dn_div10000")
            with rasterio.open(output) as dst:
                self.assertEqual(dst.crs, profile["crs"])
                self.assertEqual(dst.transform, profile["transform"])
                self.assertEqual(dst.nodata, 4)
                self.assertEqual(dst.descriptions, ("CloudSEN12_UNetMobV2_V2",))
                self.assertTrue(np.array_equal(dst.read(1), expected))
            self.assertEqual(receipt["output"]["invalid_pixels"], 2)
            self.assertEqual(json.loads(receipt_path.read_text())["output"]["sha256"], runtime.sha256(output))
            with self.assertRaises(FileExistsError):
                runtime.infer(source, output, receipt_path, "l1c_dn_div10000")

    def test_rejects_wrong_band_and_radiometry(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "wrong.tif"
            synthetic_tiff(source, bands=("B02",) + runtime.BANDS[1:])
            with self.assertRaisesRegex(ValueError, "13 bands"):
                runtime.read_input(source)
            with self.assertRaisesRegex(ValueError, "radiometry"):
                runtime.infer(source, folder / "out.tif", folder / "out.json", "unknown")

    def test_rejects_empty_support_and_shifted_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "synthetic.tif"
            synthetic_tiff(source)
            with rasterio.open(source, "r+") as dst:
                dst.write(np.zeros((13, 200, 200), dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "no valid"):
                runtime.read_input(source)
            synthetic_tiff(source)
            with rasterio.open(source, "r+") as dst:
                dst.transform = from_origin(500001, 4500000, 10, 10)
            with self.assertRaisesRegex(ValueError, "origin"):
                runtime.read_input(source)

    def test_rejects_scaled_offset_and_l2a_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "synthetic.tif"
            synthetic_tiff(source)
            with rasterio.open(source, "r+") as dst:
                dst.scales = (2.0,) * 13
            with self.assertRaisesRegex(ValueError, "scales"):
                runtime.read_input(source)
            synthetic_tiff(source)
            with rasterio.open(source, "r+") as dst:
                dst.offsets = (1000.0,) * 13
            with self.assertRaisesRegex(ValueError, "offsets"):
                runtime.read_input(source)
            synthetic_tiff(source)
            with rasterio.open(source, "r+") as dst:
                dst.update_tags(PROCESSING_LEVEL="L2A")
            with self.assertRaisesRegex(ValueError, "L2A"):
                runtime.read_input(source)
            synthetic_tiff(source)
            with rasterio.open(source, "r+") as dst:
                dst.update_tags(1, RADIO_ADD_OFFSET="1000")
            with self.assertRaisesRegex(ValueError, "offset"):
                runtime.read_input(source)


if __name__ == "__main__":
    unittest.main()
