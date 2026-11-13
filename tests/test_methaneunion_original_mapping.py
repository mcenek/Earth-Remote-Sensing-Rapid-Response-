from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_methaneunion_original_mapping import PATH_COLUMNS, audit, coordinate_label, proposed_member_path


def row(identifier, lat=10.25, lon=20.5, label=1):
    result = {"id": str(identifier), "latitude": str(lat), "longitude": str(lon),
              "label": str(label), "available_sensor": "S2"}
    for column, filename in zip(PATH_COLUMNS, ("s2.tif", "s2_pre.tif", "s2_pre_pre.tif", "s2_plume.tif")):
        result[column] = f"data/original_scale/train/s2/{identifier}/{filename}"
    return result


def candidates():
    return {"candidate_ids": [1], "coordinate_groups": [
        {"latitude": 10.25, "longitude": 20.5, "label": 1, "row_ids": [1]}]}


class OriginalMappingTests(unittest.TestCase):
    def test_id_collision_does_not_select_wrong_site_and_new_ids_map(self):
        original = [row(1, lat=-5), row(8), row(9)]
        result = audit(original, [row(1)], candidates(), {"results": []})
        self.assertEqual(result["original"]["mapped_rows"], 2)
        self.assertEqual(result["unsafe_id_join"]["coordinate_or_label_disagreements"], 1)
        self.assertEqual(result["changed_group_counts"][0]["original_rows"], 2)

    def test_label_is_part_of_mapping_and_only_observed_member_is_proven(self):
        member = "data/original/geo/train/s2/8/s2_pre.tif"
        survey = {"results": [{"archive": "dataset_part_139.tar.gz", "archive_bytes": 5000, "first_member": member}]}
        result = audit([row(8), row(9, label=0)], [row(1)], candidates(), survey)
        self.assertEqual(result["original"]["mapped_rows"], 1)
        pilot = result["deterministic_pilot"]
        self.assertEqual(pilot["proven_members"], [member])
        self.assertEqual(len(pilot["expected_members_under_alias_hypothesis"]), 4)
        self.assertFalse(pilot["payload_or_quality_verified"])

    def test_does_not_round_nearby_coordinate_into_cohort(self):
        result = audit([row(8, lat=10.25000001)], [row(1)], candidates(), {"results": []})
        self.assertEqual(result["original"]["mapped_rows"], 0)
        self.assertEqual(len(result["missing_groups"]), 1)

    def test_alias_rejects_test_or_traversal_paths(self):
        for path in ("data/original_scale/test/s2/8/s2.tif", "data/original_scale/train/s2/../8/s2.tif"):
            with self.assertRaises(ValueError):
                proposed_member_path(path)

    def test_frozen_manifest_mismatch_and_invalid_coordinates_fail(self):
        with self.assertRaises(ValueError):
            audit([row(8)], [row(1, label=0)], candidates(), {"results": []})
        with self.assertRaises(ValueError):
            coordinate_label(row(8, lat=float("nan")))


if __name__ == "__main__":
    unittest.main()
