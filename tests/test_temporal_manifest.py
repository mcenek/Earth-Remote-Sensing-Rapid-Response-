from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from validate_temporal_manifest import validate_manifest  # noqa: E402


def sample(sample_id: str = "a", split: str = "train", **changes):
    value = {
        "sample_id": sample_id, "split": split, "site_id": "site-a", "event_id": "event-a",
        "latitude": 45.0, "longitude": -120.0,
        "target": {"path": f"{sample_id}-target.tif", "provenance": f"observation:{sample_id}:target", "annotation_available": True, "mask_path": f"{sample_id}-mask.tif"},
        "reference": {"available": True, "path": f"{sample_id}-reference.tif", "provenance": f"observation:{sample_id}:reference"},
    }
    value.update(changes)
    return value


class TemporalManifestTests(unittest.TestCase):
    def valid(self, rows):
        return validate_manifest({"schema_version": "temporal-manifest/v1", "samples": rows})

    def test_valid_manifest_and_split_summary(self):
        report = self.valid([sample("a", "train"), sample("b", "validation", site_id="site-b", event_id="event-b", latitude=46.0)])
        self.assertTrue(report["valid"])
        self.assertEqual(report["summary"]["by_split"]["validation"], 1)

    def test_rejects_duplicate_ids_and_site_event_coordinate_leakage(self):
        rows = [sample("a", "train"), sample("a", "test"), sample("c", "test", site_id="site-a", event_id="event-c")]
        report = self.valid(rows)
        codes = {error["code"] for error in report["errors"]}
        self.assertTrue({"duplicate_sample_id", "site_id_split_leakage", "coordinate_split_leakage"} <= codes)

    def test_rejects_missing_annotation_when_claimed_available(self):
        row = sample()
        del row["target"]["mask_path"]
        report = self.valid([row])
        self.assertIn("missing_annotation", {error["code"] for error in report["errors"]})

    def test_rejects_invalid_coordinates_and_missing_provenance(self):
        row = sample(latitude=float("nan"), longitude=181.0)
        row["reference"]["provenance"] = ""
        report = self.valid([row])
        codes = {error["code"] for error in report["errors"]}
        self.assertTrue({"coordinates", "provenance"} <= codes)

    def test_missing_mask_is_allowed_when_unavailable(self):
        row = sample()
        row["target"]["annotation_available"] = False
        row["target"].pop("mask_path")
        self.assertTrue(self.valid([row])["valid"])

    def test_rejects_empty_manifest_and_malformed_split(self):
        self.assertIn("empty_samples", {e["code"] for e in self.valid([])["errors"]})
        report = self.valid([sample(split=[])])
        self.assertIn("split", {e["code"] for e in report["errors"]})

    def test_unavailable_reference_is_explicit_and_valid(self):
        row = sample()
        row["reference"] = {"available": False, "path": None, "provenance": None}
        self.assertTrue(self.valid([row])["valid"])

    def test_event_and_coordinate_leakage_are_independent(self):
        first = sample("a", "train")
        second = sample("b", "test", site_id="different-site", event_id="event-a")
        report = self.valid([first, second])
        codes = {e["code"] for e in report["errors"]}
        self.assertIn("event_id_split_leakage", codes)
        self.assertIn("coordinate_split_leakage", codes)

    def test_check_files_checks_only_available_asset_paths(self):
        row = sample()
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a-target.tif", "a-mask.tif", "a-reference.tif"):
                (Path(directory) / name).touch()
            self.assertTrue(self.valid([row])["valid"])
            report = validate_manifest(
                {"schema_version": "temporal-manifest/v1", "samples": [row]},
                base_dir=directory,
                check_files=True,
            )
            self.assertTrue(report["valid"])

    def test_rejects_reused_asset_path_across_splits(self):
        first = sample("a", "train")
        second = sample("b", "test", site_id="site-b", event_id="event-b", latitude=46.0)
        second["target"]["path"] = first["target"]["path"]
        report = self.valid([first, second])
        self.assertIn("observation_split_leakage", {e["code"] for e in report["errors"]})

    def test_rejects_reused_observation_provenance_across_splits(self):
        first = sample("a", "train")
        second = sample("b", "test", site_id="site-b", event_id="event-b", latitude=46.0)
        second["reference"]["provenance"] = first["target"]["provenance"]
        report = self.valid([first, second])
        self.assertIn("observation_split_leakage", {e["code"] for e in report["errors"]})


if __name__ == "__main__":
    unittest.main()
