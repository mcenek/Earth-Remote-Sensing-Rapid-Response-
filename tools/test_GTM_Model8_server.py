"""Source-only checks for the Linux handoff; no Torch import or GPU allocation."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("GTM_Model8_server.py")
SPEC = importlib.util.spec_from_file_location("gtm_model8_server", SCRIPT)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class ServerTests(unittest.TestCase):
    def args(self, folder: Path, action: str, *extra: str):
        return [action, "--code-root", str(SCRIPT.parents[1]),
                "--data-root", str(folder / "MARS-S2L"),
                "--source-root", str(folder / "marss2l-code"),
                "--output-root", str(folder / "outputs"),
                "--runtime-lock", str(folder / "requirements.lock"), *extra]

    def test_missing_server_requirements_fail_closed_without_model_import(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = server.main(self.args(Path("/server-handoff-test"), "doctor"))
        report = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual(report["status"], "missing_requirements")
        self.assertIn("data_link", report["missing"])
        self.assertIn("runtime_lock", report["missing"])
        self.assertIn("preflight", report["note"].lower())

    def test_commands_keep_frozen_manifest_and_role_separation(self):
        folder = Path("/server-handoff-test")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = server.main(self.args(folder, "commands", "--stage", "final", "--role", "holdout"))
        commands = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        train = commands["train"]
        self.assertEqual(train[train.index("--mode") + 1], "train")
        self.assertTrue(train[train.index("--manifest") + 1].endswith("train_manifest.json"))
        self.assertNotIn("final_candidate_ids.json", train)
        review = commands["review"]
        self.assertEqual(review[review.index("--role") + 1], "holdout")
        self.assertIn("--completed", review)
        self.assertIn("--publish", review)
        self.assertTrue(review[review.index("--checkpoint") + 1].endswith("best_epoch.pt"))
        self.assertEqual(Path(review[review.index("--output") + 1]).parts[-3:], ("review", "final", "holdout"))
        self.assertTrue(commands["viewer"][-1].endswith("GTM_viewer_bundles"))

    def test_review_first_pass_uses_only_predeclared_checkpoint(self):
        folder = Path("/server-handoff-test")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            server.main(self.args(folder, "commands", "--stage", "first_pass",
                                  "--role", "validation"))
        review = json.loads(output.getvalue())["review"]
        self.assertNotIn("--completed", review)
        self.assertTrue(review[review.index("--checkpoint") + 1].endswith("epoch_000.pt"))
        self.assertEqual(review[review.index("--role") + 1], "validation")

    def test_final_review_refuses_missing_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temporary:
            with patch.object(server, "doctor", return_value={"status": "preflight_required"}), \
                 patch.object(server.subprocess, "run") as run, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    server.main(self.args(Path(temporary), "review", "--stage", "final"))
            self.assertEqual(exit_status.exception.code, 2)
            run.assert_not_called()

    def test_final_review_refuses_incomplete_fit(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temporary:
            fit = Path(temporary) / "outputs" / server.RUN_NAME / "fit"
            fit.mkdir(parents=True)
            (fit / "best_epoch.pt").write_bytes(b"synthetic checkpoint marker")
            (fit / "run_status.json").write_text('{"state":"running"}', encoding="utf-8")
            with patch.object(server, "doctor", return_value={"status": "preflight_required"}), \
                 patch.object(server.subprocess, "run") as run, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    server.main(self.args(Path(temporary), "review", "--stage", "final"))
            self.assertEqual(exit_status.exception.code, 2)
            run.assert_not_called()

    def test_final_review_invokes_reviewer_after_completed_fit(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temporary:
            fit = Path(temporary) / "outputs" / server.RUN_NAME / "fit"
            fit.mkdir(parents=True)
            (fit / "best_epoch.pt").write_bytes(b"synthetic checkpoint marker")
            (fit / "run_status.json").write_text('{"state":"completed"}', encoding="utf-8")
            with patch.object(server, "doctor", return_value={"status": "preflight_required"}), \
                 patch.object(server.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                status = server.main(self.args(Path(temporary), "review", "--stage", "final", "--role", "holdout"))
            self.assertEqual(status, 0)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertTrue(command[2].endswith("GTM_Model8_mars_review.py"))
            self.assertIn("--completed", command)
            self.assertIn("--publish", command)
            self.assertEqual(command[command.index("--role") + 1], "holdout")
            self.assertEqual(command[command.index("--checkpoint") + 1], str(fit / "best_epoch.pt"))


if __name__ == "__main__":
    unittest.main()
