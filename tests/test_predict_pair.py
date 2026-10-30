"""Actual retained-v11 pair flow on historically exposed DEV imagery only."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import GTM_Model7_predict_pair as pair  # noqa: E402

MANIFEST = ROOT / "outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json"
OLD = ROOT / "outputs/GTM_Model7_hard_negative_v11_20261025/inference/siamese"


class PairInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = json.loads(MANIFEST.read_text(encoding="utf-8"))["scenes"]
        cls.examples = {kind: next(row for row in rows if row["split"] == "dev" and row["scene_kind"] == kind)
                        for kind in ("PLUME", "NO_PLUME")}

    def sources(self, row: dict) -> dict[str, Path]:
        return {source["role"]: Path(source["path"]) for source in row["sources"]}

    def test_real_exposed_positive_and_clear_user_flow(self) -> None:
        for kind, row in self.examples.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                paths = self.sources(row)
                output = Path(folder) / "prediction"
                receipt = pair.run(paths["target_image"], paths["reference_image"],
                                   paths["target_cloud"], paths["reference_cloud"],
                                   pair.RADIOMETRY, output, current_utc=row["satellite_time"],
                                   reference_utc=row["reference_time"])
                self.assertTrue(receipt["cpu_repeat_exact"])
                self.assertTrue(receipt["acquisition_times"]["reference_before_current"])
                self.assertEqual(receipt["model"]["cutoff"], .65)
                source_array = MANIFEST.parent / row["array_file"]
                with np.load(source_array, allow_pickle=False) as original, \
                        np.load(output / "prediction.npz", allow_pickle=False) as prediction, \
                        np.load(OLD / f"{row['id']}.npz", allow_pickle=False) as prior:
                    prepared, _ = pair.load_pair(paths["target_image"], paths["reference_image"],
                                                  paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)
                    np.testing.assert_array_equal(prepared["target_image"], original["target_image"])
                    np.testing.assert_array_equal(prepared["reference_image"], original["reference_image"])
                    np.testing.assert_array_equal(prepared["valid"], original["valid"])
                    np.testing.assert_array_equal(prediction["valid_native"], original["valid_native"])
                    np.testing.assert_array_equal(prediction["probability20"], prior["probability20"])
                    np.testing.assert_array_equal(prediction["probability_native"], prior["probability_native"])
                    expected_mask = ((prior["probability_native"] >= .65) & prior["valid_native"]).astype(np.uint8)
                    np.testing.assert_array_equal(prediction["mask_native"], expected_mask)
                with rasterio.open(output / "probability.tif") as score, rasterio.open(output / "mask.tif") as mask, rasterio.open(output / "valid.tif") as valid:
                    self.assertEqual(score.shape, (200, 200))
                    self.assertEqual(score.transform, mask.transform)
                    self.assertEqual(score.transform, valid.transform)
                    self.assertEqual(score.crs, mask.crs)
                    np.testing.assert_array_equal(mask.read(1), expected_mask)
                with self.assertRaises(FileExistsError):
                    pair.run(paths["target_image"], paths["reference_image"],
                             paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY, output,
                             current_utc=row["satellite_time"], reference_utc=row["reference_time"])

    def test_wrong_radiometry_and_shifted_reference_fail(self) -> None:
        row = self.examples["PLUME"]
        paths = self.sources(row)
        with self.assertRaises(ValueError):
            pair.load_pair(paths["target_image"], paths["reference_image"],
                           paths["target_cloud"], paths["reference_cloud"], "sentinel_l2a_reflectance")
        with tempfile.TemporaryDirectory() as folder:
            shifted = Path(folder) / "shifted.tif"
            with rasterio.open(paths["reference_image"]) as source:
                profile = source.profile.copy()
                profile["transform"] = source.transform @ Affine.translation(1, 0)
                with rasterio.open(shifted, "w", **profile) as target:
                    target.write(source.read())
                    for i, description in enumerate(source.descriptions, 1):
                        target.set_band_description(i, description)
            with self.assertRaisesRegex(ValueError, "grids/CRS differ"):
                pair.load_pair(paths["target_image"], shifted,
                               paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)

    def test_cloud_class4_nodata_integrates_through_inference_and_publisher(self) -> None:
        import GTM_Model7_publish_pair as publisher
        row = self.examples["PLUME"]
        paths = self.sources(row)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cloud = folder / "canonical_class4_cloud.tif"
            with rasterio.open(paths["target_cloud"]) as source:
                profile = source.profile.copy()
                profile["nodata"] = 4
                values = source.read(1)
                values[0, 0] = 4
                with rasterio.open(cloud, "w", **profile) as target:
                    target.write(values, 1)
            loaded, _ = pair.load_pair(paths["target_image"], paths["reference_image"],
                                       cloud, paths["reference_cloud"], pair.RADIOMETRY)
            self.assertFalse(loaded["valid_native"][:2, :2].any())
            self.assertTrue(loaded["valid_native"][2:, 2:].any())
            output = folder / "prediction"
            pair.run(paths["target_image"], paths["reference_image"], cloud,
                     paths["reference_cloud"], pair.RADIOMETRY, output,
                     current_utc=row["satellite_time"], reference_utc=row["reference_time"])
            receipt, verified = publisher.verify_receipt(output / "receipt.json")
            self.assertEqual(receipt["input"]["inputs"]["current_cloud"]["grid"]["nodata"], 4)
            self.assertFalse(verified["valid"][:2, :2].any())

    def test_unprojected_or_nonmetre_grid_and_invalid_times_fail(self) -> None:
        row = self.examples["PLUME"]
        paths = self.sources(row)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for crs in ("EPSG:4326", "EPSG:2277"):
                modified = folder / f"{crs.replace(':', '_')}.tif"
                with rasterio.open(paths["target_image"]) as source:
                    profile = source.profile.copy()
                    profile["crs"] = crs
                    with rasterio.open(modified, "w", **profile) as target:
                        target.write(source.read())
                        for i, description in enumerate(source.descriptions, 1):
                            target.set_band_description(i, description)
                with self.assertRaisesRegex(ValueError, "projected CRS with metre units"):
                    pair.load_pair(modified, paths["reference_image"],
                                   paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)
            output = folder / "prediction"
            for current, reference in (("2024-11-05", "2024-10-23T17:24:01Z"),
                                       ("2024-10-23T17:24:01Z", "2024-11-05T17:35:21Z"),
                                       ("2024-11-05T17:35:21-07:00", "2024-10-23T17:24:01Z")):
                with self.subTest(current=current, reference=reference), self.assertRaises(ValueError):
                    pair.run(paths["target_image"], paths["reference_image"],
                             paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY,
                             output, current_utc=current, reference_utc=reference)
                self.assertFalse(output.exists())

    def test_wrong_band_and_empty_support_fail(self) -> None:
        row = self.examples["NO_PLUME"]
        paths = self.sources(row)
        with tempfile.TemporaryDirectory() as folder:
            copied_current = Path(folder) / "copied_current_as_reference.tif"
            shutil.copyfile(paths["target_image"], copied_current)
            with self.assertRaisesRegex(ValueError, "identical to current"):
                pair.load_pair(paths["target_image"], copied_current,
                               paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)
            offset_image = Path(folder) / "offset_image.tif"
            shutil.copyfile(paths["target_image"], offset_image)
            with rasterio.open(offset_image, "r+") as changed:
                changed.update_tags(RADIO_ADD_OFFSET="1000")
            with self.assertRaisesRegex(ValueError, "band, dtype"):
                pair.load_pair(offset_image, paths["reference_image"],
                               paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)
            no_nodata = Path(folder) / "missing_nodata.tif"
            with rasterio.open(paths["target_image"]) as source:
                profile = source.profile.copy()
                profile["nodata"] = None
                with rasterio.open(no_nodata, "w", **profile) as target:
                    target.write(source.read())
                    for i, description in enumerate(source.descriptions, 1):
                        target.set_band_description(i, description)
            with self.assertRaisesRegex(ValueError, "band, dtype"):
                pair.load_pair(no_nodata, paths["reference_image"],
                               paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)
            wrong = Path(folder) / "wrong_band.tif"
            with rasterio.open(paths["target_image"]) as source:
                with rasterio.open(wrong, "w", **source.profile) as target:
                    target.write(source.read())
                    for i, description in enumerate(source.descriptions, 1):
                        target.set_band_description(i, "B01" if i == 1 else description)
            with self.assertRaisesRegex(ValueError, "band, dtype"):
                pair.load_pair(wrong, paths["reference_image"],
                               paths["target_cloud"], paths["reference_cloud"], pair.RADIOMETRY)
            cloudy = Path(folder) / "all_cloud.tif"
            with rasterio.open(paths["target_cloud"]) as source:
                with rasterio.open(cloudy, "w", **source.profile) as target:
                    target.write(np.ones((200, 200), dtype=np.uint8), 1)
            with self.assertRaisesRegex(ValueError, "Empty or nonfinite"):
                pair.load_pair(paths["target_image"], paths["reference_image"],
                               cloudy, paths["reference_cloud"], pair.RADIOMETRY)

    def test_stale_checkpoint_and_normalizer_fail(self) -> None:
        frozen = json.loads(pair.FREEZE.read_text(encoding="utf-8"))
        selected = json.loads(pair.SELECTION_RECEIPT.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "stale.pt"
            checkpoint.write_bytes(b"not the selected checkpoint")
            frozen["models"]["siamese"]["checkpoint"] = str(checkpoint)
            altered_freeze = folder / "freeze.json"
            altered_freeze.write_text(json.dumps(frozen), encoding="utf-8")
            selected["freeze_sha256"] = pair.sha(altered_freeze)
            altered_receipt = folder / "receipt.json"
            altered_receipt.write_text(json.dumps(selected), encoding="utf-8")
            original_freeze, original_receipt = pair.FREEZE, pair.SELECTION_RECEIPT
            try:
                pair.FREEZE, pair.SELECTION_RECEIPT = altered_freeze, altered_receipt
                with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                    pair.load_retained_v11()
            finally:
                pair.FREEZE, pair.SELECTION_RECEIPT = original_freeze, original_receipt
        model, norm, _ = pair.load_retained_v11()
        bad = dict(norm, scale=[0] * 6)
        data = {"target_image": np.ones((100, 100, 6), np.float32),
                "reference_image": np.ones((100, 100, 6), np.float32),
                "valid": np.ones((100, 100), bool)}
        with self.assertRaisesRegex(ValueError, "Unsupported frozen normalizer"):
            pair.features(data, bad)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
