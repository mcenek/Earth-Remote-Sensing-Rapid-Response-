"""Small behavioral tests; no checkpoint loads, GPU, downloads or full evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from GTM_Model0_compare_plumes import compare, digest, load_case, score, write_json
from GTM_Model0_legacy_baseline import normalize_and_pad
from GTM_Model0_legacy_baseline import predict as legacy_predict


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.grid = dict(shape=[2, 3], crs='pixel', affine=[1, 0, 0, 0, -1, 2])
        np.savez(self.base / 'evidence.npz', truth=np.array([[1, 1, 0], [0, 0, 0]], dtype='float32'),
                 valid=np.ones((2, 3)), rgb=np.arange(18).reshape(2, 3, 3))
        self.case = dict(id='fixture', source_sha256='synthetic-source', grid=self.grid,
                         evidence='evidence.npz', reference_semantics='reviewed_mask')
        for role, p in [('legacy', [[.9, .1, .8], [.1, .1, .1]]), ('current', [[.9, .9, .1], [.1, .1, .1]])]:
            np.savez(self.base / (role + '.npz'), probability=p, valid=np.ones((2, 3)))
            self.case[role] = dict(path=role + '.npz', metadata=role + '.json')
            self.metadata(role)
        self.thresholds = dict(legacy=.5, current=.5)

    def metadata(self, role, **updates):
        meta = dict(scene_id='fixture', source_sha256='synthetic-source', grid=self.grid,
                    model_id=role + '_synthetic', score_kind='plume_probability',
                    prediction_sha256=digest(self.base / (role + '.npz')), provenance={'fixture': True})
        meta.update(updates); write_json(self.base / (role + '.json'), meta)

    def test_end_to_end_known_geometry_and_no_overwrite(self):
        write_json(self.base / 'manifest.json', dict(schema_version=1, selection_reason='Synthetic engineering fixture, not model results',
                   thresholds=self.thresholds, cases=[self.case]))
        r = compare(self.base / 'manifest.json', self.base / 'out')
        m = r['cases'][0]['metrics']
        self.assertAlmostEqual(m['legacy']['iou'], 1/3)
        self.assertEqual(m['current']['iou'], 1)
        self.assertEqual(m['legacy']['area_error_pixels'], 0)  # equal area can hide wrong location
        self.assertEqual(m['legacy']['pixel_false_positive_rate'], .25)
        self.assertTrue((self.base / 'out/00_current.png').is_file())
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            compare(self.base / 'manifest.json', self.base / 'out')

    def test_positive_only_exterior_never_counts_as_false_positive(self):
        truth = np.array([[1, 1, np.nan], [np.nan, np.nan, np.nan]])
        with np.load(self.base / 'evidence.npz') as e:
            rgb = e['rgb']
        np.savez(self.base / 'evidence.npz', truth=truth, valid=np.ones((2, 3)), rgb=rgb)
        self.case['reference_semantics'] = 'positive_only'
        row, _ = load_case(self.case, self.base, self.thresholds, 1)
        self.assertIsNone(row['metrics']['legacy']['fp'])
        self.assertIsNone(row['metrics']['legacy']['iou'])
        self.assertEqual(row['metrics']['legacy']['recall'], .5)
        self.assertEqual(row['metrics']['legacy']['unknown_pixels_flagged'], 1)

    def test_common_support_guard_and_explicit_coverage(self):
        np.savez(self.base / 'current.npz', probability=np.ones((2, 3)), valid=[[1, 0, 1], [1, 1, 1]])
        self.metadata('current')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            load_case(self.case, self.base, self.thresholds, .95)
        row, _ = load_case(self.case, self.base, self.thresholds, .8)
        self.assertEqual(row['metrics']['legacy']['fn'], 0)
        self.assertAlmostEqual(row['common_reference_coverage'], 5/6)

    def test_equal_shape_different_geography_rejected(self):
        self.metadata('legacy', grid={**self.grid, 'affine': [1, 0, 50, 0, -1, 2]})
        with self.assertRaisesRegex(ValueError, 'grid mismatch'):
            load_case(self.case, self.base, self.thresholds, 1)

    def test_wrong_scene_or_changed_cache_rejected(self):
        self.metadata('legacy', scene_id='different-scene')
        with self.assertRaisesRegex(ValueError, 'scene_id mismatch'):
            load_case(self.case, self.base, self.thresholds, 1)
        self.metadata('legacy', prediction_sha256='wrong')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_case(self.case, self.base, self.thresholds, 1)

    def test_empty_reference_has_no_perfect_iou(self):
        r = score(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2), bool), .5)
        self.assertIsNone(r['iou']); self.assertIsNone(r['recall'])
        self.assertIsNone(r['relative_area_error']); self.assertEqual(r['pixel_false_positive_rate'], 0)

    def test_missing_prediction_does_not_create_empty_mask(self):
        self.case['legacy']['path'] = 'missing.npz'
        with self.assertRaises(FileNotFoundError):
            load_case(self.case, self.base, self.thresholds, 1)

    def test_legacy_normalization_padding_and_band_guard(self):
        x = np.arange(5*2*3, dtype='float32').reshape(5, 2, 3)
        y = normalize_and_pad(x, np.ones((2, 3)))
        np.testing.assert_allclose(y[:2, :3], np.moveaxis(x, 0, -1) / (29 + 1e-6))
        self.assertEqual(y.shape, (256, 256, 5)); self.assertEqual(float(y[2:].sum()), 0)
        with self.assertRaisesRegex(ValueError, 'five ordered'):
            normalize_and_pad(np.zeros((6, 2, 3)), np.ones((2, 3)))

    def test_low_memory_stops_before_scene_or_model_load(self):
        with patch('GTM_Model0_legacy_baseline.check', return_value={'available_ram_gib': 1}):
            with self.assertRaisesRegex(ValueError, '6 GiB'):
                legacy_predict('nonexistent_scene.json', self.base/'unused.npz', self.base/'unused.json')
        self.assertFalse((self.base/'unused.npz').exists())

    def test_missing_tensorflow_stops_before_scene_or_model_load(self):
        with patch('GTM_Model0_legacy_baseline.check', return_value={'available_ram_gib': 10, 'tensorflow_installed': False}):
            with self.assertRaisesRegex(ValueError, 'TensorFlow is absent'):
                legacy_predict('nonexistent_scene.json', self.base/'unused.npz', self.base/'unused.json')


if __name__ == '__main__':
    unittest.main()
