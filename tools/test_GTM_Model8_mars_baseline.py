"""Source-only controls for the frozen MARS baseline runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

SCRIPT = Path(__file__).with_name("GTM_Model8_mars_baseline.py")
SPEC = importlib.util.spec_from_file_location("mars_baseline", SCRIPT)
mars = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mars)


class ManifestTests(unittest.TestCase):
    def test_rejects_overlap_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"fit_ids": ["a", "a"], "val_ids": ["b"]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                mars.frozen_manifest(path)
            path.write_text(json.dumps({"fit_ids": ["a"], "val_ids": ["a"]}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                mars.frozen_manifest(path)

    def test_raster_audit_enforces_site_separation_and_missing_paths(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as directory:
            root = Path(directory)
            fields = ["id_loc_image", "location_name", "isplume", "s2path",
                      "cloudmaskpath", "plumepath", "ch4path", "plume", "background_image_tile"]
            rows = [
                {"id_loc_image": "a", "location_name": "fit", "isplume": "True",
                 "s2path": "a_s2.tif", "cloudmaskpath": "a_cloud.tif",
                 "plumepath": "", "ch4path": "a_ch4.tif",
                 "plume": "POLYGON ((0 0, 1 0, 1 1, 0 0))", "background_image_tile": ""},
                {"id_loc_image": "b", "location_name": "val", "isplume": "True",
                 "s2path": "b_s2.tif", "cloudmaskpath": "b_cloud.tif",
                 "plumepath": "", "ch4path": "",
                 "plume": "POLYGON ((0 0, 1 0, 1 1, 0 0))", "background_image_tile": ""},
                {"id_loc_image": "c", "location_name": "val", "isplume": "False",
                 "s2path": "c_s2.tif", "cloudmaskpath": "c_cloud.tif",
                 "plumepath": "", "ch4path": "", "plume": "", "background_image_tile": ""},
            ]
            with (root / "validated_images_all.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with (root / "validated_images_plumes.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id_loc_image", "geometry"])
                writer.writeheader()
                writer.writerow({"id_loc_image": "a", "geometry": "POLYGON ((0 0, 1 0, 1 1, 0 0))"})
            for row in rows:
                for key in ("s2path", "cloudmaskpath", "ch4path"):
                    if row[key]:
                        (root / row[key]).touch()
            result = mars.raw_audit(root, ["a"], ["b", "c"])
            self.assertEqual((result["missing_rasters"], result["fit_plume_source_rows"]), (0, 1))
            (root / "a_ch4.tif").unlink()
            self.assertEqual(mars.raw_audit(root, ["a"], ["b", "c"])["missing_rasters"], 1)
            rows[1]["location_name"] = "fit"
            with (root / "validated_images_all.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "share sites"):
                mars.raw_audit(root, ["a"], ["b", "c"])

    def test_upstream_subset_csv_excludes_held_out_rows(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as directory:
            root = Path(directory)
            original = root / "all.csv"
            selected = root / "fit.csv"
            with original.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id_loc_image", "label"])
                writer.writeheader()
                writer.writerows([
                    {"id_loc_image": "fit", "label": "1"},
                    {"id_loc_image": "held_out", "label": "secret"},
                ])
            self.assertEqual(mars.write_selected_csv(original, selected, {"fit"}), 1)
            with selected.open(newline="") as file:
                self.assertEqual(list(csv.DictReader(file)),
                                 [{"id_loc_image": "fit", "label": "1"}])


class LossTests(unittest.TestCase):
    def test_numpy_loss_matches_independent_torch_expression(self):
        logits = np.array([[[0.2, -0.7], [1.1, -1.4]]], dtype=np.float32)
        target = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
        ch4 = np.array([[[0, 50], [2500, 0]]], dtype=np.float32)
        positive = F.softplus(-torch.tensor(logits)) * 10 * torch.clamp(torch.tensor(ch4), 100, 2000) / 1000
        negative = F.softplus(torch.tensor(logits))
        reference = torch.where(torch.tensor(target).bool(), positive, negative).sum((-2, -1)).mean().item()
        self.assertAlmostEqual(mars.independent_loss(logits, target, ch4), reference, places=5)

    def test_source_crop_bridge_preserves_square_window(self):
        self.assertEqual(mars.square_window_size((200, 200)), 200)
        self.assertEqual(mars.square_window_size(200), 200)
        with self.assertRaisesRegex(ValueError, "square"):
            mars.square_window_size((200, 192))

    def test_source_validation_cutoff_tracks_hundredth_pixel(self):
        scores = torch.linspace(0, 1, 400).reshape(20, 20)
        result = mars.source_threshold_cutoff(scores, threshold_pixels=100, tol=1e-4)
        hundredth = scores.flatten().topk(100).values[-1].item()
        self.assertLessEqual(abs(result - hundredth), 0.003)
        self.assertEqual(mars.source_threshold_cutoff(torch.ones(20, 20)), 1.0)

    def test_indexed_source_draws_repeat_across_call_order(self):
        class RandomSource:
            def __getitem__(self, index):
                return float(np.random.random()), int(torch.randint(0, 100000, ()).item())

        draws = mars.SeededDraws(RandomSource(), epoch=3, count=5)
        first = [draws[index] for index in range(5)]
        second = [draws[index] for index in reversed(range(5))]
        self.assertEqual(first, list(reversed(second)))
        self.assertNotEqual(first, [mars.SeededDraws(RandomSource(), epoch=4, count=5)[index]
                                    for index in range(5)])


if __name__ == "__main__":
    unittest.main()
