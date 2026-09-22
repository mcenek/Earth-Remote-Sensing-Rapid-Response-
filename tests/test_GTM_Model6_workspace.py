"""Preparation contract checks; these do not test or claim a trained model."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gtm6', ROOT / 'research/GTM_Model6/GTM_Model6.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(MODULE.DEFAULT_CONFIG.read_text())

    def test_pilot_cannot_be_mistaken_for_scientific_validation(self):
        status = MODULE.status(self.config)
        self.assertTrue(status['configuration_valid'])
        self.assertFalse(status['full_design_implemented'])
        self.assertFalse(status['native_multiscale_diagnostic']['fine_resolution_validated'])
        self.assertEqual(status['scientific_readiness'], 'not_established')
        self.assertIsNone(status['approved_manifest'])
        self.assertFalse(status['plume_pilot']['nationally_validated'])

    def test_pilot_split_and_cutoff_contract(self):
        evaluation = self.config['evaluation']
        self.assertEqual(evaluation['outer_folds'], 5)
        self.assertEqual(evaluation['fixed_cutoff'], 0.5)
        self.assertFalse(evaluation['inner_threshold_calibration'])
        self.assertEqual(evaluation['fold_sizes_records'], [82, 67, 9, 8, 7])

    def test_reject_augmentation_leakage_or_unreviewed_promotion(self):
        for key, value in [('primary_validation_augmented', True), ('parent_groups_split_before_augmentation', False), ('visual_review_required', False)]:
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config['evaluation'][key] = value
                self.assertTrue(MODULE.validate_config(config))

    def test_reject_resuming_historical_model_or_automatic_work(self):
        config = copy.deepcopy(self.config)
        config['initial_checkpoint'] = 'old.pt'
        self.assertTrue(MODULE.validate_config(config))
        for key in ['automatic_training', 'downloads_enabled']:
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config['execution'][key] = True
                self.assertTrue(MODULE.validate_config(config))

    def test_reject_false_resolution_claim(self):
        self.config['output']['fine_resolution_validated'] = True
        self.assertTrue(MODULE.validate_config(self.config))


if __name__ == '__main__':
    unittest.main()
