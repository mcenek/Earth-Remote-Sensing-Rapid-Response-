"""Acquisition wait gates without network, tmux, or training processes."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import GTM_Model8_server_bootstrap as bootstrap


class AssetWaitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.args = argparse.Namespace(code_root=root / "code", data_root=root / "data",
            output_root=root / "outputs", acquisition_tmux_session="gtm-model8-data",
            asset_wait_timeout=120, asset_poll_seconds=30)
        self.args.data_root.mkdir()
        self.destination = self.args.output_root / "bootstrap.json"
        self.identity = {"data_root": str(self.args.data_root.resolve()),
                         "catalog_sha256": bootstrap.assets.DEFAULT_CATALOG_SHA256}
        self.source_receipt = self.args.data_root / ".gtm_model8_fetch_receipt.json"
        self.catalog_patch = patch.object(bootstrap.assets, "load_catalog", return_value=(
            [{"path": "one"}, {"path": "two"}], bootstrap.assets.DEFAULT_CATALOG_SHA256))
        self.count_patch = patch.object(bootstrap, "EXPECTED_ASSET_OBJECTS", 2)
        self.catalog_patch.start()
        self.count_patch.start()
        self.addCleanup(self.catalog_patch.stop)
        self.addCleanup(self.count_patch.stop)

    def put(self, state, **overrides):
        body = {"identity": self.identity, "state": state, "verified_existing": 1,
                "missing": 1, "downloaded": 0, "remaining": 1,
                "failed": [], "corrupt": []}
        body.update(overrides)
        self.source_receipt.write_text(json.dumps(body), encoding="utf-8")

    def wait(self):
        return bootstrap.wait_for_assets(self.args, {"status": "starting"}, self.destination)

    def test_waits_then_accepts_exact_complete_receipt(self):
        self.put("downloading")
        def complete(_seconds):
            self.put("complete", downloaded=1, remaining=0)
        with patch.object(bootstrap, "run_command", return_value={"exit_code": 0}) as tmux, \
             patch.object(bootstrap.time, "sleep", side_effect=complete):
            result = self.wait()
        self.assertEqual(result["asset_wait"]["state"], "complete")
        self.assertEqual(tmux.call_count, 1)
        self.assertEqual(json.loads(self.destination.read_text())["status"], "waiting_for_assets")

    def test_failed_and_incomplete_receipts_stop(self):
        for state in ("failed", "incomplete", "incomplete_budget", "unknown"):
            with self.subTest(state=state):
                self.put(state)
                with patch.object(bootstrap, "run_command") as tmux:
                    with self.assertRaisesRegex(RuntimeError, "Acquisition stopped"):
                        self.wait()
                    tmux.assert_not_called()

    def test_wrong_identity_stops(self):
        self.put("complete", identity={"data_root": "wrong", "catalog_sha256": self.identity["catalog_sha256"]})
        with self.assertRaisesRegex(RuntimeError, "different data root"):
            self.wait()

    def test_tmux_disappearance_stops_missing_receipt(self):
        with patch.object(bootstrap, "run_command", return_value={"exit_code": 1}):
            with self.assertRaisesRegex(RuntimeError, "tmux session disappeared"):
                self.wait()

    def test_completion_during_tmux_exit_is_accepted(self):
        self.put("downloading")
        def exit_after_completion(_argv):
            self.put("complete", downloaded=1, remaining=0)
            return {"exit_code": 1}
        with patch.object(bootstrap, "run_command", side_effect=exit_after_completion), \
             patch.object(bootstrap.time, "sleep") as sleep:
            result = self.wait()
        self.assertEqual(result["asset_wait"]["state"], "complete")
        sleep.assert_not_called()

    def test_timeout_stops_waiting(self):
        self.put("inventory")
        self.args.asset_wait_timeout = 2
        with patch.object(bootstrap.time, "monotonic", side_effect=[0, 3]), \
             patch.object(bootstrap, "run_command") as tmux:
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                self.wait()
            tmux.assert_not_called()

    def test_complete_count_mismatch_rejected(self):
        self.put("complete", downloaded=0, remaining=0)
        with self.assertRaisesRegex(ValueError, "inconsistent counts"):
            self.wait()

    def test_complete_existing_cache_needs_no_live_acquisition(self):
        self.put("complete", verified_existing=2, missing=0, remaining=0)
        with patch.object(bootstrap, "run_command") as tmux:
            result = self.wait()
        self.assertEqual(result["asset_wait"]["state"], "complete")
        tmux.assert_not_called()

    def test_different_catalog_is_rejected(self):
        with patch.object(bootstrap.assets, "load_catalog", return_value=([{}, {}], "different")):
            with self.assertRaisesRegex(RuntimeError, "catalog hash or object count"):
                self.wait()

    def test_cli_requires_execute_for_wait(self):
        with self.assertRaises(SystemExit) as error:
            bootstrap.main(["--data-root", str(self.args.data_root), "--wait-for-assets"])
        self.assertEqual(error.exception.code, 2)

    def test_cli_refreshes_data_and_requires_explicit_training_flag(self):
        for start in (False, True):
            with self.subTest(start_training=start):
                def finish(args, receipt):
                    self.assertEqual(args.start_training, start)
                    self.assertEqual(receipt["data"]["missing"], [])
                    return {"status": "training_started" if start else "ready"}
                with patch.object(bootstrap, "plan", return_value={"data": {"missing": ["pending"]}}), \
                     patch.object(bootstrap, "wait_for_assets"), \
                     patch.object(bootstrap, "data_report", return_value={"missing": []}), \
                     patch.object(bootstrap, "execute", side_effect=finish), \
                     patch.object(bootstrap, "write_receipt"):
                    flags = ["--start-training"] if start else []
                    self.assertEqual(bootstrap.main(["--data-root", str(self.args.data_root),
                            "--execute", "--wait-for-assets"] + flags), 0)


if __name__ == "__main__":
    unittest.main()
