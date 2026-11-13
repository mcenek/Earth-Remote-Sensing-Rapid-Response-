#### Small offline behavioral checks for Tessera research scaffold ####

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cli import fixture, run_files, save_fixture
from scaffold import (FixtureProvider, GeoTesseraAdapter, Region, haversine_m,
                      point_in_polygon, provenance_key, run_offline, synthetic_nearest,
                      validate_points)


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.points, self.region = fixture()

    def provider(self):
        return FixtureProvider(self.region, self.manifest)

    def test_point_validation_and_pins(self):
        validate_points(self.points, self.manifest)
        for change in ({"point_id": "source-A"}, {"observation_date": "2023-06-01"}, {"variant": "other"}):
            bad = [dict(self.points[0]), dict(self.points[1], **change)]
            with self.assertRaises(ValueError):
                validate_points(bad, self.manifest)
        bad = [dict(self.points[0])]
        del bad[0]["site_id"]
        with self.assertRaises(ValueError):
            validate_points(bad, self.manifest)
        with self.assertRaises(ValueError):
            FixtureProvider(self.region, dict(self.manifest, year=2023))
        for key in ("sdk_version", "dataset_version", "variant", "year"):
            changed = dict(self.manifest, **{key: "different"})
            if key == "year":
                changed[key] = 2023
            self.assertNotEqual(provenance_key(changed), provenance_key(self.manifest))

    def test_centers_lon_lat_and_bounds(self):
        self.assertEqual(self.region.center(0, 1), (-92.985, 41.995))
        self.assertEqual(self.region.index(-92.985, 41.995), (0, 1))
        sample = self.provider().sample_points([(-92.995, 41.995), (-94, 42)], 2024)
        self.assertEqual(sample[0][1], "valid")
        self.assertEqual(sample[1][1], "out_of_bounds")

    def test_invalid_vectors_and_coverage_abstain(self):
        for vector, status, expected in ((np.zeros(128), "valid", "invalid_vector"), (np.full(128, np.nan), "valid", "invalid_vector"), (np.ones(128), "water", "water"), (np.ones(128), "missing", "missing")):
            vectors = self.region.vectors.copy()
            coverage = self.region.coverage.copy()
            vectors[0, 0] = vector
            coverage[0, 0] = status
            region = Region(vectors, coverage, self.region.transform, self.region.crs, 2024, "synthetic128d", "synthetic-1", "synthetic-1")
            result = run_offline(self.points, self.manifest, FixtureProvider(region, self.manifest), max_pixels=6)
            self.assertEqual(result["seeds"][0]["coverage_status"], expected)
            self.assertFalse(any(m["point_id"] == "source-A" for m in result["matches"]))

    def test_shape_crs_year_and_limits(self):
        for changes in ({"vectors": np.zeros((2, 3, 127))}, {"crs": "EPSG:3857"}, {"variant": "other"}):
            params = dict(vectors=self.region.vectors, coverage=self.region.coverage, transform=self.region.transform, crs=self.region.crs, year=2024, variant="synthetic128d", sdk_version="synthetic-1", dataset_version="synthetic-1")
            params.update(changes)
            with self.assertRaises(ValueError):
                FixtureProvider(Region(**params), self.manifest)
        with self.assertRaises(ValueError):
            run_offline(self.points, self.manifest, self.provider(), max_pixels=5)
        with self.assertRaises(ValueError):
            self.provider().sample_points([(-92.995, 41.995)], 2023)
        for key in ("sdk_version", "dataset_version"):
            params = dict(vectors=self.region.vectors, coverage=self.region.coverage, transform=self.region.transform, crs=self.region.crs, year=2024, variant="synthetic128d", sdk_version="synthetic-1", dataset_version="synthetic-1")
            params[key] = "wrong"
            with self.assertRaises(ValueError):
                FixtureProvider(Region(**params), self.manifest)
        with self.assertRaises(ValueError):
            run_offline(self.points, dict(self.manifest, dataset_version="wrong"), self.provider(), max_pixels=6)

    def test_large_float32_cosine_is_finite(self):
        vectors = self.region.vectors.copy()
        vectors[0, 0, 0] = np.float32(1e30)
        vectors[0, 1, 0] = np.float32(1e30)
        region = Region(vectors, self.region.coverage, self.region.transform, self.region.crs, 2024, "synthetic128d", "synthetic-1", "synthetic-1")
        result = run_offline(self.points, self.manifest, FixtureProvider(region, self.manifest), max_pixels=6)
        self.assertTrue(np.isfinite(result["matches"][0]["cosine_similarity"]))
        self.assertAlmostEqual(result["matches"][0]["cosine_similarity"], 1.0)

    def test_rank_excludes_all_seed_pixels_and_ties(self):
        result = run_offline(self.points, self.manifest, self.provider(), top_k=3, max_pixels=6)
        self.assertEqual(result["matches"][0]["pixel_id"], "r0c1")
        seed_ids = {"r0c0", "r1c1"}
        self.assertTrue(all(m["pixel_id"] not in seed_ids for m in result["matches"]))
        self.assertEqual([m["pixel_id"] for m in result["matches"] if m["point_id"] == "interior-B"][:2], ["r0c2", "r1c2"])
        self.assertGreater(result["matches"][0]["distance_m"], 0)
        self.assertGreater(result["matches"][0]["cosine_similarity"], 0.9)
        self.assertEqual(result["neighborhoods"][0]["spatial_group"], "group-A")
        self.assertEqual(result["neighborhoods"][0]["geometry_type"], "verified_source")

    def test_synthetic_facility_geometry(self):
        square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        self.assertTrue(point_in_polygon(0, 0, square))
        self.assertTrue(point_in_polygon(1, 0, square))
        self.assertFalse(point_in_polygon(2, 0, square))
        nearest = synthetic_nearest((0, 0), [{"facility_id": "far", "lon": 0.1, "lat": 0}, {"facility_id": "near", "lon": 0.001, "lat": 0}])
        self.assertEqual(nearest["facility_id"], "near")
        self.assertAlmostEqual(nearest["distance_m"], 111.195, delta=0.2)
        self.assertAlmostEqual(haversine_m((0, 0), (0, 0)), 0)

    def test_lazy_client_contract_and_offline_guard(self):
        created = []
        def factory(pin):
            created.append(pin)
            class Fake:
                sdk_version = "synthetic-1"
                dataset_version = "synthetic-1"
                variant = "synthetic128d"
                def sample_points(self, coordinates, *, year):
                    self.sample_args = (coordinates, year)
                    return np.ones((len(coordinates), 128))
                def read_region(self, bbox, *, year):
                    self.region_args = (bbox, year)
                    return np.ones((1, 1, 128)), (0.1, 0, 0, 0, -0.1, 1), "EPSG:4326"
            return Fake()
        with self.assertRaises(PermissionError):
            GeoTesseraAdapter(self.manifest, client_factory=factory)
        self.assertEqual(created, [])
        adapter = GeoTesseraAdapter(self.manifest, allow_live=True, client_factory=factory)
        self.assertEqual(adapter.sample_points([(1, 2)], 2024).shape, (1, 128))
        self.assertEqual(adapter.client.sample_args, ([(1, 2)], 2024))
        self.assertEqual(adapter.read_region((0, 0, 1, 1), 2024)[0].shape, (1, 1, 128))
        self.assertEqual(adapter.client.region_args, ((0, 0, 1, 1), 2024))
        with self.assertRaises(ValueError):
            adapter.sample_points([(1, 2)], 2023)

    def test_demo_subprocess_no_network_and_readback(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            script = "import socket,builtins; socket.socket=lambda *a, **k: (_ for _ in ()).throw(AssertionError('network')); original=builtins.__import__; builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(AssertionError('SDK import')) if name.startswith('geotessera') else original(name,*a,**k); import cli; cli.demo(__import__('pathlib').Path(" + repr(temp) + "))"
            completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=ROOT, text=True, capture_output=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((Path(temp) / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["pins"]["variant"], "synthetic128d")
            self.assertEqual(result["matches"][0]["pixel_id"], "r0c1")
            self.assertEqual(result["seeds"][0]["provenance"], "synthetic-demo")

    def test_saved_inputs_and_no_match_exports(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp)
            save_fixture(path)
            result = run_files(path / "manifest.json", path / "synthetic_region.npz", path / "exports", max_pixels=6)
            completed = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "cli.py"), "run", "--manifest", str(path / "manifest.json"), "--region", str(path / "synthetic_region.npz"), "--output", str(path / "cli_exports"), "--max-pixels", "6"], cwd=ROOT, text=True, capture_output=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            cli_result = json.loads((path / "cli_exports" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(cli_result["matches"], result["matches"])
            self.assertEqual(result["matches"][0]["pixel_id"], "r0c1")
            self.assertEqual(result["matches"][0]["seed_lon"], -92.995)
            self.assertEqual(result["grid"]["crs"], "EPSG:4326")
            self.assertEqual(result["grid"]["transform"], list(self.region.transform))
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            manifest["pins"]["dataset_version"] = "wrong"
            (path / "bad_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_files(path / "bad_manifest.json", path / "synthetic_region.npz", path / "bad_exports", max_pixels=6)
            with np.load(path / "synthetic_region.npz", allow_pickle=False) as saved:
                fields = {key: saved[key] for key in saved.files}
            fields["coverage"] = np.full((2, 3), "water", dtype="<U16")
            np.savez_compressed(path / "water.npz", **fields)
            empty = run_files(path / "manifest.json", path / "water.npz", path / "empty", max_pixels=6)
            self.assertEqual(empty["matches"], [])
            self.assertEqual((path / "empty" / "matches.csv").read_text(encoding="utf-8").count("\n"), 1)
            self.assertEqual(json.loads((path / "empty" / "matches.geojson").read_text(encoding="utf-8"))["features"], [])


if __name__ == "__main__":
    unittest.main()
