"""Small saved-pair verification for the unlabeled Model7 viewer publisher."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
from rasterio import Affine

import GTM_Model7_publish_pair as pair
from GTM_Model0_local_viewer import load_bundle


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('exposed_dev_positive_utc', 'exposed_dev_positive_bundle_v2',
     'GTM_Model7_v11_exposed_dev_sample_a_engineering_20261026', 1076),
    ('exposed_dev_clear_utc', 'exposed_dev_clear_bundle_v2',
     'GTM_Model7_v11_exposed_dev_sample_b_engineering_20261026', 0),
)
PREDICTIONS = ROOT / 'outputs/GTM_Model7_pair_inference_verification_20261026'
RESULTS = ROOT / 'outputs/GTM_Model7_pair_viewer_verification_20261026'


class PublishPairTests(unittest.TestCase):
    def test_two_complete_receipts_and_rendered_unlabeled_bundles(self):
        for source_name, bundle_name, identifier, marked in CASES:
            with self.subTest(case=source_name):
                receipt_path = PREDICTIONS / source_name / 'receipt.json'
                receipt, verified = pair.verify_receipt(receipt_path)
                self.assertEqual(receipt['marked_native_pixels'], marked)
                bundle = RESULTS / bundle_name
                manifest = json.loads((bundle / 'experiment.json').read_text(encoding='utf-8'))
                scene = manifest['scenes'][0]
                self.assertEqual(manifest['id'], identifier)
                self.assertNotIn('metrics', manifest)
                self.assertNotIn('nativeMetrics', scene)
                self.assertNotIn('label', scene)
                self.assertNotIn('truthImage', scene)
                self.assertNotIn('emit', scene)
                grid = json.loads((bundle / 'grid.json').read_text(encoding='utf-8'))
                self.assertNotIn('truth', grid)
                published = ROOT / 'outputs/GTM_viewer_bundles' / identifier / 'experiment.json'
                loaded, _ = load_bundle(published, 'bundle-assets/')
                self.assertEqual(loaded['scenes'][0]['id'], 'pair')
                metadata = json.loads((bundle / 'provenance.json').read_text(encoding='utf-8'))
                self.assertFalse(metadata['labelUsed'])
                self.assertEqual(metadata['savedPredictionSha256'], receipt['outputs'])

                shape = (grid['height'], grid['width'])
                transform = Affine(*grid['displayTransform'])
                valid = pair._warp(verified['valid'].astype(np.uint8), verified, transform, shape).astype(bool)
                probability = pair._warp(verified['probability'], verified, transform, shape, nodata=None)
                mask = pair._warp(verified['mask'], verified, transform, shape)
                with Image.open(bundle / 'score.png') as im:
                    score_png = np.asarray(im)
                with Image.open(bundle / 'mask.png') as im:
                    mask_png = np.asarray(im)
                self.assertTrue(np.array_equal(score_png[..., 3], np.where(valid, 240, 0)))
                self.assertTrue(np.array_equal(mask_png[..., 3], np.where(mask == 1, 230, 0)))
                self.assertTrue(np.array_equal(mask_png[mask == 1, :3],
                                               np.tile([255, 135, 35], (int(mask.sum()), 1))))
                stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                                  [94, 201, 98], [253, 231, 37]], dtype=float)
                scaled = np.minimum(3.99999, np.maximum(0, probability.astype(float)) * 4)
                index = np.floor(scaled).astype(int)
                expected = np.floor(stops[index] +
                                    (stops[index + 1] - stops[index]) * (scaled - index)[..., None] + .5).astype(np.uint8)
                self.assertTrue(np.array_equal(score_png[valid, :3], expected[valid]))

                for role, image_name in (('current', 'current_rgb.png'), ('reference', 'reference_rgb.png')):
                    with Image.open(bundle / image_name) as im:
                        rgb = np.asarray(im)
                    for channel, band in enumerate((2, 1, 0)):
                        stretch = metadata['rgbStretch']['channels'][pair.BANDS[band]]
                        raw = verified['source'][role][band].astype(np.float32)
                        stretched = np.rint(np.clip((raw - stretch['low']) /
                                                       max(stretch['high'] - stretch['low'], 1), 0, 1) * 255).astype(np.uint8)
                        expected_channel = pair._warp(stretched, verified, transform, shape, nodata=None)
                        self.assertTrue(np.array_equal(rgb[..., channel], expected_channel))

    def test_rejects_receipt_and_saved_mask_tampering(self):
        original = PREDICTIONS / CASES[0][0]
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory)
            for name in (*pair.OUTPUTS, 'receipt.json'):
                shutil.copy2(original / name, staged / name)
            receipt_file = staged / 'receipt.json'
            document = json.loads(receipt_file.read_text(encoding='utf-8'))
            document['model']['cutoff'] = 0.5
            receipt_file.write_text(json.dumps(document), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'model identity or cutoff'):
                pair.verify_receipt(receipt_file)
            shutil.copy2(original / 'receipt.json', receipt_file)
            mask = staged / 'mask.tif'
            with mask.open('r+b') as stream:
                stream.seek(-1, 2)
                last = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes([last ^ 1]))
            with self.assertRaisesRegex(ValueError, 'mask.tif hash mismatch'):
                pair.verify_receipt(receipt_file)

    def test_refuses_existing_published_id(self):
        source_name, _, identifier, _ = CASES[0]
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            pair.run(PREDICTIONS / source_name / 'receipt.json', experiment_id=identifier,
                     output=RESULTS / 'would_not_write', publish_bundle=True)


if __name__ == '__main__':
    unittest.main()
