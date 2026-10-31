"""Focused archive safety, link conflict, and bounded probe tests."""
import argparse
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import GTM_Model8_server_prepare as prepare


class ServerPrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_conflicting_existing_path_is_rejected(self):
        link = self.root / "code" / "outputs"
        target = self.root / "new_outputs"
        link.mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "Conflicting existing path"):
            prepare.validate_links([(link, target)])

    def test_traversal_archive_is_rejected_without_writing_destination(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("marss2l/../escape.py", "bad")
        destination = self.root / "source"
        with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
            prepare.extract_source(archive, destination)
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / "escape.py").exists())

    def test_archive_symlink_is_rejected(self):
        archive = self.root / "bad_link.zip"
        entry = zipfile.ZipInfo("marss2l/link.py")
        entry.create_system = 3
        entry.external_attr = 0o120777 << 16
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr(entry, "../../outside")
        with self.assertRaisesRegex(ValueError, "Unsupported archive entry type"):
            prepare.extract_source(archive, self.root / "source")

    def test_safe_archive_extracts_bytes_once(self):
        archive = self.root / "safe.zip"
        payload = b"line one\r\nline two\r\n"
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("marss2l/loaders.py", payload)
            target.writestr("environment/requirements-test.lock", b"torch==2.12.0\n")
        destination = self.root / "source"
        receipt = prepare.extract_source(archive, destination)
        self.assertEqual((destination / "marss2l/loaders.py").read_bytes(), payload)
        self.assertEqual(receipt["entries"], 2)
        with self.assertRaises(FileExistsError):
            prepare.extract_source(archive, destination)

    def test_probe_search_is_bounded_to_explicit_root(self):
        data = self.root / "data" / "MARS-S2L"
        data.mkdir(parents=True)
        (data / "validated_images_all.csv").write_text("id\n", encoding="utf-8")
        (data / "validated_images_plumes.csv").write_text("id\n", encoding="utf-8")
        found = prepare.candidates([self.root], max_depth=3, max_dirs=20)
        self.assertEqual(found["candidate_data_roots"], [str(data)])
        self.assertLessEqual(found["directories_seen"], 20)


if __name__ == "__main__":
    unittest.main()
