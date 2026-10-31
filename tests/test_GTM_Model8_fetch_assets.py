"""Pinned catalog, byte budget, and atomic download tests without network."""
import gzip
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from collections import namedtuple
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import GTM_Model8_fetch_assets as fetch


class Response(io.BytesIO):
    def geturl(self):
        return "https://cdn.example/immutable"


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.payload = b"pinned bytes\n"
        self.item = {"path": "data/train/a.bin", "size": len(self.payload),
            "source_url": "https://huggingface.co" + fetch.URL_PREFIX +
                          "data/train/a.bin?download=true",
            "remote_oid_type": "sha256_lfs",
            "remote_oid": hashlib.sha256(self.payload).hexdigest()}
        self.identity = {"data_root": str(self.data), "catalog_sha256": "a" * 64}

    def budget(self, cap):
        budget = fetch.TransferBudget(self.data / "ledger.jsonl", self.identity, cap)
        self.addCleanup(budget.close)
        return budget

    def test_catalog_rejects_unpinned_and_traversal(self):
        fetch.validate_item(self.item)
        for bad in ({**self.item, "path": "../outside.bin"},
                    {**self.item, "source_url": self.item["source_url"].replace(fetch.REVISION, "main")}):
            with self.assertRaises(ValueError):
                fetch.validate_item(bad)

    def test_conflicting_duplicate_catalog_rejected(self):
        path = self.root / "catalog.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as output:
            output.write(json.dumps(self.item) + "\n")
            output.write(json.dumps({**self.item, "size": self.item["size"] + 1}) + "\n")
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            fetch.load_catalog(path)

    def test_budget_persists_and_refunds_short_reads(self):
        budget = self.budget(10)
        self.assertEqual(budget.read(io.BytesIO(b"abc"), 5), b"abc")
        self.assertEqual(budget.used, 3)
        budget.close()
        resumed = fetch.TransferBudget(self.data / "ledger.jsonl", self.identity, 10, previous=3)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.used, 3)
        self.assertEqual(resumed.read(io.BytesIO(b"1234567"), 7), b"1234567")
        with self.assertRaises(fetch.BudgetExceeded):
            resumed.read(io.BytesIO(b"x"), 1)

    def test_download_counts_retry_body_and_verifies_exact_bytes(self):
        budget = self.budget(100)
        error = HTTPError(self.item["source_url"], 429, "busy", {}, io.BytesIO(b"wait"))
        with patch.object(fetch, "urlopen", side_effect=[error, Response(self.payload)]), \
             patch.object(fetch.time, "sleep"), \
             patch.object(fetch.os, "link", side_effect=lambda source, target:
                          Path(target).write_bytes(Path(source).read_bytes())):
            result = fetch.download_one(self.data, self.item, budget)
        self.assertEqual(result["state"], "verified_download")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(budget.used, len(self.payload) + 4)
        self.assertEqual((self.data / self.item["path"]).read_bytes(), self.payload)
        self.assertEqual(list(self.data.rglob("*.part.*")), [])

    def test_existing_corrupt_file_is_never_overwritten(self):
        target = self.data / self.item["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"bad")
        self.assertEqual(fetch.audit_one(self.data, self.item)["state"], "corrupt")
        budget = self.budget(100)
        with patch.object(fetch, "urlopen") as urlopen:
            result = fetch.download_one(self.data, self.item, budget)
            urlopen.assert_not_called()
        self.assertEqual(result["state"], "failed")
        self.assertEqual(target.read_bytes(), b"bad")

    def test_exact_budget_and_git_blob_hash(self):
        blob = hashlib.sha1(f"blob {len(self.payload)}\0".encode() + self.payload).hexdigest()
        item = {**self.item, "remote_oid_type": "git_blob_sha1", "remote_oid": blob}
        budget = self.budget(len(self.payload))
        with patch.object(fetch, "urlopen", return_value=Response(self.payload)), \
             patch.object(fetch.os, "link", side_effect=lambda source, target:
                          Path(target).write_bytes(Path(source).read_bytes())):
            result = fetch.download_one(self.data, item, budget)
        self.assertEqual(result["state"], "verified_download")
        self.assertEqual(budget.used, len(self.payload))
        self.assertEqual(fetch.audit_one(self.data, item)["state"], "verified")

    def test_existing_claim_blocks_duplicate_and_preserves_receipt(self):
        claim = self.data / ".gtm_model8_fetch.claim"
        claim.write_text('{"pid":12345}', encoding="utf-8")
        receipt = self.data / ".gtm_model8_fetch_receipt.json"
        receipt.write_text('{"sentinel":true}', encoding="utf-8")
        with patch.object(fetch, "urlopen") as urlopen:
            result = fetch.main(["--data-root", str(self.data), "--download",
                                 "--max-network-bytes", "100"])
        self.assertEqual(result, 1)
        urlopen.assert_not_called()
        self.assertEqual(json.loads(receipt.read_text()), {"sentinel": True})
        self.assertEqual(json.loads(claim.read_text()), {"pid": 12345})

    def test_cli_dry_then_missing_only_download(self):
        catalog = self.root / "catalog.jsonl.gz"
        with gzip.open(catalog, "wt", encoding="utf-8") as output:
            output.write(json.dumps(self.item) + "\n")
        common = ["--data-root", str(self.data), "--catalog", str(catalog)]
        with patch.object(fetch, "urlopen") as urlopen:
            self.assertEqual(fetch.main(common), 1)
            urlopen.assert_not_called()
        usage = namedtuple("usage", "total used free")(100 * 1024**3, 0, 100 * 1024**3)
        with patch.object(fetch, "urlopen", return_value=Response(self.payload)), \
             patch.object(fetch.shutil, "disk_usage", return_value=usage), \
             patch.object(fetch.os, "link", side_effect=lambda source, target:
                          Path(target).write_bytes(Path(source).read_bytes())):
            self.assertEqual(fetch.main(common + ["--download", "--max-network-bytes",
                                                  str(len(self.payload))]), 0)
        receipt = json.loads((self.data / ".gtm_model8_fetch_receipt.json").read_text())
        self.assertEqual(receipt["state"], "complete")
        self.assertEqual(receipt["network_response_bytes"], len(self.payload))


if __name__ == "__main__":
    unittest.main()
