import gzip
import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from probe_methaneunion_training_member import extract_first_expected


def test_exact_training_member_and_wrong_path():
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo('data/original/geo/train/s2/254920/s2_pre.tif')
        info.pax_headers = {'mtime': '123.4'}
        info.size = 5
        archive.addfile(info, io.BytesIO(b'image'))
    packed = gzip.compress(raw.getvalue())
    assert extract_first_expected(packed, info.name) == b'image'
    with pytest.raises(ValueError, match='exact selected'):
        extract_first_expected(packed, 'data/480m_GSD/s2/254920/s2_pre.tif')


def test_incomplete_member_is_not_retained():
    raw = io.BytesIO()
    info = tarfile.TarInfo('train.tif')
    info.size = 20000
    # A valid header that declares more file bytes than the prefix holds.
    packed = gzip.compress(info.tobuf() + b'partial')
    with pytest.raises(ValueError, match='Complete member'):
        extract_first_expected(packed, 'train.tif')
