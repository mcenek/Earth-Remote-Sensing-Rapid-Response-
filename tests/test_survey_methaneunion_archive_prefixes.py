from __future__ import annotations

import gzip
import io
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from survey_methaneunion_archive_prefixes import first_tar_member_from_gzip_prefix  # noqa: E402


def test_first_tar_member_from_partial_gzip_payload() -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        payload = b"example"
        info = tarfile.TarInfo("data/480m_GSD/s2/7/s2.tif")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    compressed = gzip.compress(raw.getvalue())
    assert first_tar_member_from_gzip_prefix(compressed[:65536]) == (
        "data/480m_GSD/s2/7/s2.tif"
    )
