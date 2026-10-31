"""State and command contract checks without Torch or server access."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import GTM_Model8_linux_job as job


class LinuxJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.args = argparse.Namespace(code_root=root, data_root=root / "data",
            source_root=root / "source", output_root=root / "outputs",
            runtime_lock=root / "lock", batch_size=32, validation_batch_size=8,
            precision="bf16", num_workers=2, draws=65536, epochs=170, patience=30)

    def test_atomic_status_and_exclusive_claim(self):
        path = job.job_dir(self.args) / "status.json"
        job.atomic_json(path, {"fit": "running"})
        self.assertEqual(job.read_json(path), {"fit": "running"})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])
        claim = path.parent / "run.claim"
        job.claim(claim)
        with self.assertRaisesRegex(RuntimeError, "Existing controller claim"):
            job.claim(claim)

    def test_duplicate_fit_is_rejected_before_commands(self):
        fit = job.fit_dir(self.args)
        fit.mkdir(parents=True)
        (fit / "epoch_000.pt").write_bytes(b"checkpoint")
        existing = job.job_dir(self.args) / "status.json"
        job.atomic_json(existing, {"state": "training", "fit": "running"})
        with patch.object(job.server, "doctor") as doctor:
            self.assertEqual(job.run(self.args), 1)
            doctor.assert_not_called()
        self.assertEqual(job.read_json(existing)["fit"], "running")
        self.assertFalse((job.job_dir(self.args) / "run.claim").exists())

    def test_final_review_requires_completed_fit(self):
        (job.fit_dir(self.args) / "best_epoch.pt").parent.mkdir(parents=True)
        (job.fit_dir(self.args) / "best_epoch.pt").write_bytes(b"checkpoint")
        with self.assertRaisesRegex(RuntimeError, "completed fit"):
            job.review(self.args, "final", "holdout", {})

    def test_first_pass_uses_exact_checkpoint_and_tracks_publication(self):
        fit = job.fit_dir(self.args)
        fit.mkdir(parents=True)
        (fit / "epoch_000.pt").write_bytes(b"checkpoint")
        def fake_child(command, cwd, log, receipt, phase, fingerprint):
            self.assertIn("epoch_000.pt", " ".join(command))
            self.assertNotIn("--completed", command)
            destination = job.server.paths(self.args)["outputs"] / job.server.RUN_NAME / "review" / "first_pass" / "validation"
            job.atomic_json(destination / "publication.json", {"baseline": {}, "released_reference": {}})
            job.atomic_json(receipt, {"state": "completed", "exit_code": 0})
            return 0
        with patch.object(job, "run_child", side_effect=fake_child):
            self.assertTrue(job.review(self.args, "first_pass", "validation", {}))
        receipt = job.job_dir(self.args) / "reviews" / "first_pass_validation.json"
        self.assertEqual(set(job.read_json(receipt)["publication"]), {"baseline", "released_reference"})
        self.assertFalse(receipt.with_suffix(".claim").exists())

    def test_run_orders_doctor_preflight_fit_then_reviews(self):
        phases = []
        reviews = []
        def fake_child(command, cwd, log, receipt, phase, fingerprint):
            phases.append(phase)
            if phase == "preflight":
                job.atomic_json(job.fit_dir(self.args) / "preflight.json", {"status": "passed"})
            else:
                job.atomic_json(job.fit_dir(self.args) / "run_status.json", {"state": "completed"})
            job.atomic_json(receipt, {"state": "completed", "exit_code": 0})
            return 0
        def fake_review(args, stage, role, fingerprint):
            reviews.append((stage, role))
            return True
        with patch.object(job.server, "doctor", return_value={"status": "preflight_required"}), \
             patch.object(job, "package_fingerprint", return_value={"files": {}}), \
             patch.object(job.server, "command", return_value=["python", "mock.py"]), \
             patch.object(job, "run_child", side_effect=fake_child), \
             patch.object(job, "review", side_effect=fake_review):
            self.assertEqual(job.run(self.args), 0)
        self.assertEqual(phases, ["preflight", "train"])
        self.assertEqual(reviews, [("first_pass", "validation"), ("first_pass", "holdout"),
                                   ("final", "validation"), ("final", "holdout")])
        self.assertEqual(job.read_json(job.job_dir(self.args) / "status.json")["state"], "completed")


if __name__ == "__main__":
    unittest.main()
