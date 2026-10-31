"""Focused contract tests for the explicit Model8 checkpoint migration."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "tools/GTM_Model8_checkpoint_migration.py"
spec = importlib.util.spec_from_file_location("model8_migration", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPT.parents[1] / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {**migration.FROZEN_SCIENCE,
                       "torch_version": "windows-torch", "numpy_version": "windows-numpy",
                       "runtime_lock_sha256": "windows-lock"}
        self.target = {**self.source, "torch_version": "linux-torch",
                       "numpy_version": "linux-numpy", "runtime_lock_sha256": "linux-lock"}
        self.full = {"epoch": 4, "model_state_dict": {"weight": torch.tensor([1.5])},
                     "optimizer_state_dict": {"state": {0: {"exp_avg": torch.tensor([0.25]),
                                                            "step": torch.tensor(5.)}},
                                              "param_groups": [{"lr": 5e-4}]},
                     "torch_rng_state": torch.get_rng_state(),
                     "numpy_rng_state": np.random.get_state(),
                     "python_rng_state": (3, (1, 2, 3), None),
                     "cuda_rng_state": torch.tensor([1, 2], dtype=torch.uint8),
                     "best_ap": 0.8, "best_epoch": 3, "fingerprint": self.source}
        self.best = {"epoch": 3, "model_state_dict": {"weight": torch.tensor([1.25])},
                     "fingerprint": self.source, "validation": {"average_precision": 0.8}}

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def args(self):
        source_full = self.root / "source_full.pt"
        source_best = self.root / "source_best.pt"
        torch.save(self.full, source_full)
        torch.save(self.best, source_best)
        target_fit = self.root / "target_fit"
        target_fit.mkdir()
        source_plan = SCRIPT.parents[1] / "tool-diagnostics/mars-baseline-20261030/plan.json"
        target_plan = source_plan
        from argparse import Namespace
        return Namespace(source_full=source_full, source_best=source_best,
                         source_full_sha256=migration.sha256(source_full),
                         source_best_sha256=migration.sha256(source_best),
                         source_plan=source_plan, target_plan=target_plan,
                         plan_sha256=migration.sha256(source_plan),
                         source_settings=self.write_json("source_settings.json", migration.SETTINGS),
                         target_settings=self.write_json("target_settings.json", migration.SETTINGS),
                         source_preflight=self.write_json("source_preflight.json",
                                                          {"status": "passed", "fingerprint": self.source}),
                         target_preflight=self.write_json("target_fit/preflight.json",
                                                          {"status": "passed", "fingerprint": self.target}),
                         target_fit=target_fit)

    def test_allowed_rebase_preserves_all_training_state(self):
        args = self.args()
        original_full = migration.sha256(args.source_full)
        original_best = migration.sha256(args.source_best)
        receipt = migration.migrate(args)
        derived = torch.load(args.target_fit / "last_epoch.pt", weights_only=False)
        derived_best = torch.load(args.target_fit / "best_epoch.pt", weights_only=False)
        self.assertEqual(derived["fingerprint"], self.target)
        self.assertEqual(derived_best["fingerprint"], self.target)
        for source, result in ((self.full, derived), (self.best, derived_best)):
            self.assertTrue(migration.equal_state(
                {key: value for key, value in source.items() if key != "fingerprint"},
                {key: value for key, value in result.items() if key != "fingerprint"}))
        self.assertEqual(set(receipt["permitted_environment_differences"]),
                         migration.ENVIRONMENT_FIELDS)
        self.assertEqual(migration.sha256(args.source_full), original_full)
        self.assertEqual(migration.sha256(args.source_best), original_best)
        self.assertTrue((args.target_fit / "migration_receipt.json").is_file())
        with self.assertRaises(FileExistsError):
            migration.migrate(args)

    def test_rejects_scientific_source_and_unknown_differences(self):
        for field, value in (("batch_size", 96), ("runner_sha256", "changed"),
                             ("source_file_sha256", {"trainer.py": "changed"}),
                             ("unexpected", "changed")):
            with self.subTest(field=field):
                target = {**self.target, field: value}
                with self.assertRaises(ValueError):
                    migration.fingerprint_diff(self.source, target)

    def test_rejects_missing_or_wrong_hash_before_loading(self):
        args = self.args()
        args.source_full_sha256 = ""
        with self.assertRaises(ValueError):
            migration.migrate(args)
        args.source_full_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            migration.migrate(args)
        self.assertFalse((args.target_fit / "last_epoch.pt").exists())

    def test_rejects_best_mismatch_nonfinite_and_settings_change(self):
        args = self.args()
        self.best["epoch"] = 2
        with self.assertRaises(ValueError):
            migration.validate_payload(self.full, self.best, self.source)
        self.best["epoch"] = 3
        self.full["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.tensor([float("nan")])
        with self.assertRaises(ValueError):
            migration.validate_payload(self.full, self.best, self.source)
        args.target_settings.write_text(json.dumps({**migration.SETTINGS, "patience": 29}))
        with self.assertRaises(ValueError):
            migration.migrate(args)


if __name__ == "__main__":
    unittest.main()
