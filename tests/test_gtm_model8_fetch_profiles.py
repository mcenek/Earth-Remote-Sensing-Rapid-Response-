"""Supplementary cache profile and shared-claim checks without network."""
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import GTM_Model8_fetch_assets as fetch


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "full.jsonl.gz"
        payload = b"asset"
        item = {"path": "data/train/a.bin", "size": len(payload),
                "source_url": "https://huggingface.co" + fetch.URL_PREFIX +
                              "data/train/a.bin?download=true",
                "remote_oid_type": "sha256_lfs",
                "remote_oid": hashlib.sha256(payload).hexdigest()}
        with gzip.open(self.catalog, "wt", encoding="utf-8") as output:
            output.write(json.dumps(item) + "\n")

    def args(self, *extra):
        return ["--data-root", str(self.root), "--catalog", str(self.catalog), *extra]

    def test_namespaces_preserve_original_paths(self):
        default = fetch.profile_paths(self.root, None)
        full = fetch.profile_paths(self.root, "full_release")
        self.assertEqual(default["receipt"].name, ".gtm_model8_fetch_receipt.json")
        self.assertEqual(default["transfer"].name, ".gtm_model8_fetch_transfer.jsonl")
        self.assertTrue(set(default.values()).isdisjoint(full.values()))
        self.assertIn("profile-full_release", full["receipt"].name)

    def test_invalid_profiles_are_rejected(self):
        for value in ("", "Full", "../full", "full.release", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fetch.profile_paths(self.root, value)

    def test_full_release_path_with_space_uses_pinned_encoded_url(self):
        with gzip.open(self.catalog, "rt", encoding="utf-8") as source:
            item = json.loads(source.readline())
        item["path"] = "data/Not Used/example.tif"
        item["source_url"] = "https://huggingface.co" + fetch.URL_PREFIX + "data/Not%20Used/example.tif?download=true"
        self.assertEqual(fetch.validate_item(item), item)
        item["source_url"] = item["source_url"].replace("Not%20Used", "Not Used")
        with self.assertRaisesRegex(ValueError, "Unpinned or mismatched"):
            fetch.validate_item(item)

    def test_profile_inventory_leaves_baseline_receipt_untouched(self):
        baseline = fetch.profile_paths(self.root, None)["receipt"]
        baseline.write_text('{"sentinel":true}', encoding="utf-8")
        with patch.object(fetch, "urlopen") as network:
            self.assertEqual(fetch.main(self.args("--profile", "full_release")), 1)
            network.assert_not_called()
        self.assertEqual(json.loads(baseline.read_text()), {"sentinel": True})
        profile = fetch.profile_paths(self.root, "full_release")["receipt"]
        self.assertEqual(json.loads(profile.read_text())["identity"]["profile"], "full_release")

    def test_universal_claim_excludes_profile_and_preserves_owner(self):
        claim = self.root / ".gtm_model8_fetch.claim"
        claim.write_text('{"pid":12345}', encoding="utf-8")
        with patch.object(fetch, "urlopen") as network:
            self.assertEqual(fetch.main(self.args("--profile", "full_release")), 1)
            network.assert_not_called()
        self.assertEqual(json.loads(claim.read_text())["pid"], 12345)
        self.assertFalse(fetch.profile_paths(self.root, "full_release")["receipt"].exists())

    def test_wait_acquires_after_other_owner_releases(self):
        claim = self.root / ".gtm_model8_fetch.claim"
        claim.write_text('{"pid":12345}', encoding="utf-8")
        with patch.object(fetch.time, "sleep", side_effect=lambda _seconds: claim.unlink()) as sleeping:
            fetch.acquire_claim(claim, 60)
        self.assertEqual(sleeping.call_count, 1)
        self.assertEqual(json.loads(claim.read_text())["pid"], os.getpid())

    def test_wait_timeout_keeps_other_owner_claim(self):
        claim = self.root / ".gtm_model8_fetch.claim"
        claim.write_text('{"pid":12345}', encoding="utf-8")
        with patch.object(fetch.time, "monotonic", side_effect=[0, 2]), \
             patch.object(fetch.time, "sleep") as sleeping:
            with self.assertRaisesRegex(RuntimeError, "Existing cache operation"):
                fetch.acquire_claim(claim, 1)
            sleeping.assert_not_called()
        self.assertEqual(json.loads(claim.read_text())["pid"], 12345)

    def test_profile_budget_resumes_only_its_own_ledger(self):
        profile = fetch.profile_paths(self.root, "full_release")
        baseline = fetch.profile_paths(self.root, None)
        identity = {"data_root": str(self.root), "catalog_sha256": "a" * 64,
                    "profile": "full_release"}
        budget = fetch.TransferBudget(profile["transfer"], identity, 10)
        self.assertEqual(budget.read(io.BytesIO(b"abc"), 5), b"abc")
        budget.close()
        resumed = fetch.TransferBudget(profile["transfer"], identity, 10, previous=3)
        self.assertEqual(resumed.used, 3)
        resumed.close()
        self.assertFalse(baseline["transfer"].exists())


if __name__ == "__main__":
    unittest.main()
