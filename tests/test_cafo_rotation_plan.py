from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cafo_plan", ROOT / "tools/build_cafo_rotation_plan.py")
cafo = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(cafo)


class CafoPlanTests(unittest.TestCase):
    def records(self):
        return [
            {"facility_id": "a", "latitude": 41.0, "longitude": -93.0, "operation_status": "Active", "state": "IA"},
            {"facility_id": "b", "latitude": 41.005, "longitude": -93.0, "operation_status": "Active", "state": "IA"},
            {"facility_id": "c", "latitude": 42.0, "longitude": -94.0, "operation_status": "Active", "state": "IA"},
            {"facility_id": "d", "latitude": 42.5, "longitude": -94.0, "operation_status": "Active", "state": "IA"},
            {"facility_id": "e", "latitude": 43.0, "longitude": -95.0, "operation_status": "Active", "state": "IA"},
            {"facility_id": "f", "latitude": 44.0, "longitude": -95.0, "operation_status": "Active", "state": "IA"},
        ]

    def test_nearby_sites_share_group_and_each_group_one_fold(self):
        rows, summary = cafo.build_plan(self.records())
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        folds = {r["group_id"]: r["fold"] for r in rows}
        self.assertEqual(len(folds), summary["group_count"])
        for row in rows:
            self.assertEqual(row["train_rotations"], [row["fold"]])
            self.assertEqual(row["validate_rotations"], [k for k in range(5) if k != row["fold"]])

    def test_deterministic_under_input_reordering(self):
        first, _ = cafo.build_plan(self.records(), seed=11)
        second, _ = cafo.build_plan(list(reversed(self.records())), seed=11)
        self.assertEqual(first, second)

    def test_csv_filters_active_and_rejects_conflicts(self):
        fields = ["progid", "Latitude", "Longitude", "opStatus", "State"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.csv"
            with path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
                writer.writerow({"progid": "x", "Latitude": 41, "Longitude": -93, "opStatus": "Active", "State": "IA"})
                writer.writerow({"progid": "y", "Latitude": 41, "Longitude": -93, "opStatus": "Inactive", "State": "IA"})
            rows, stats = cafo.load_facilities(path)
            self.assertEqual([r["facility_id"] for r in rows], ["x"])
            self.assertEqual(stats["status_counts"]["Inactive"], 1)
            with path.open("a") as fh:
                fh.write("x,41.1,-93,Active,IA\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                cafo.load_facilities(path)

            path.write_text("progid,Latitude,Longitude,opStatus,State\nx,41,-93,Active,IA\nx,41,-93,Inactive,IA\n")
            with self.assertRaisesRegex(ValueError, "opStatus"):
                cafo.load_facilities(path)

    def test_invalid_bounds_and_source_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.csv"; p.write_text("progid,Latitude,Longitude,opStatus\nx,91,0,Active\n")
            with self.assertRaisesRegex(ValueError, "bounds"):
                cafo.load_facilities(p)
            self.assertEqual(cafo.source_sha256(p), hashlib.sha256(p.read_bytes()).hexdigest())

    def test_outputs_have_compact_rotation_summary(self):
        rows, summary = cafo.build_plan(self.records())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            cafo.write_outputs(rows, summary, out)
            with (out / "facilities.csv").open(newline="") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), len(rows))
            loaded = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(loaded["rotation_counts"]), 5)
            with self.assertRaises(FileExistsError):
                cafo.write_outputs(rows, summary, out)

    def test_antimeridian_and_polar_neighbors(self):
        records = [dict(facility_id=str(i), latitude=lat, longitude=lon)
                   for i, (lat, lon) in enumerate([(70, 179.99), (70, -179.99),
                                                  (89.999, 0), (89.999, 180)])]
        self.assertEqual(cafo.geographic_components(records, 2), [[0, 1], [2, 3]])

    def test_fewer_than_five_components_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "five"):
            cafo.build_plan(self.records()[:4], buffer_km=2.0)


if __name__ == "__main__":
    unittest.main()
