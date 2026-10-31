"""Source-only bootstrap gates; no environment, GPU, or tmux commands run."""
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import GTM_Model8_server_bootstrap as bootstrap


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.args = argparse.Namespace(code_root=root / "code", data_root=root / "data",
            source_root=root / "source", source_zip=root / "source.zip",
            output_root=root / "outputs", venv=root / "venv", python3_12=None,
            install_runtime=False, start_training=False, execute=True,
            tmux_session="gtm-model8-linux", batch_size=32, validation_batch_size=8,
            precision="bf16", num_workers=2, draws=65536, epochs=170, patience=30)

    def test_data_report_lists_required_missing_files(self):
        report = bootstrap.data_report(self.args.data_root)
        self.assertFalse(report["exists"])
        self.assertIn("validated_images_all.csv", report["missing"])
        self.assertIn("trained_models/MARSS2L_20250326/best_epoch", report["missing"])

    def test_missing_data_prevents_any_prepare(self):
        with patch.object(bootstrap.prepare, "prepare") as prepare:
            with self.assertRaisesRegex(RuntimeError, "Data root is missing"):
                bootstrap.execute(self.args, {"data": bootstrap.data_report(self.args.data_root)})
            prepare.assert_not_called()

    def test_fresh_fit_allows_only_preflight_receipt(self):
        fit = self.args.output_root / bootstrap.server.RUN_NAME / "fit"
        fit.mkdir(parents=True)
        (fit / "preflight.json").write_text("{}", encoding="utf-8")
        bootstrap.fresh_fit(self.args)
        (fit / "epoch_000.pt").write_bytes(b"x")
        with self.assertRaisesRegex(RuntimeError, "Existing fit"):
            bootstrap.fresh_fit(self.args)

    def test_gpu_compute_process_blocks_launch(self):
        with patch.object(bootstrap.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
             patch.object(bootstrap, "run_command", return_value={"exit_code": 0,
                  "stdout": "1234, 800", "stderr": ""}):
            with self.assertRaisesRegex(RuntimeError, "active compute processes"):
                bootstrap.idle_gpu()

    def test_existing_nonprivate_venv_is_rejected(self):
        self.args.venv.mkdir()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            bootstrap.ensure_venv(self.args.venv, self.args.output_root / "venv.log")

    def test_training_launch_follows_doctor_preflight_and_idle_checks(self):
        self.args.start_training = True
        self.args.source_root.joinpath("environment").mkdir(parents=True)
        (self.args.source_root / "environment/requirements-test.lock").write_bytes(b"pinned\n")
        events = []
        session_checks = 0
        def fake_run(argv, **kwargs):
            nonlocal session_checks
            if argv[:2] == ["tmux", "has-session"]:
                session_checks += 1
                return {"exit_code": 1 if session_checks == 1 else 0,
                        "stdout": "", "stderr": ""}
            if "doctor" in argv:
                events.append("doctor")
            elif "preflight" in argv:
                events.append("preflight")
                fit = self.args.output_root / bootstrap.server.RUN_NAME / "fit"
                fit.mkdir(parents=True, exist_ok=True)
                (fit / "preflight.json").write_text('{"status":"passed"}', encoding="utf-8")
            elif argv[:2] == ["tmux", "new-session"]:
                events.append("launch")
            return {"exit_code": 0, "stdout": "", "stderr": "", "log": str(kwargs.get("log"))}
        def fake_idle():
            events.append("idle")
            return {"compute_processes": []}
        with patch.object(bootstrap.prepare, "prepare", return_value={"status": "ready"}), \
             patch.object(bootstrap, "ensure_venv", return_value={"version": "3.12.0"}), \
             patch.object(bootstrap.shutil, "which", return_value="/usr/bin/tmux"), \
             patch.object(bootstrap, "run_command", side_effect=fake_run), \
             patch.object(bootstrap, "idle_gpu", side_effect=fake_idle):
            result = bootstrap.execute(self.args, {"data": {"missing": []}})
        self.assertEqual(events, ["doctor", "idle", "preflight", "idle", "launch"])
        self.assertEqual(result["status"], "training_started")
        self.assertTrue(Path(result["retained_preflight"]["path"]).is_file())
        self.assertFalse((self.args.output_root / bootstrap.server.RUN_NAME / "fit/preflight.json").exists())


if __name__ == "__main__":
    unittest.main()
