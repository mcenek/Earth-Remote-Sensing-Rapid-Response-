import io
import json
from pathlib import Path

import pytest

from tools.bounded_http_range import Budget, BudgetExceeded, HTTPRangeReader


class Response:
    def __init__(self, status, body=b"", headers=None):
        self.status, self._body, self.headers = status, body, headers or {}
        self.read_called = False
        self.closed = False

    def read(self, n=-1):
        self.read_called = True
        return self._body[:n]

    def close(self):
        self.closed = True


def test_budget_persists_charge_on_failed_request(tmp_path, monkeypatch):
    b = Budget(tmp_path, 10)
    def fail(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(OSError):
        HTTPRangeReader("https://sentinel-cogs.s3.us-west-2.amazonaws.com/a", b, 8).read(1)
    assert Budget(tmp_path, 10).snapshot()["charged_bytes"] == 8
    with pytest.raises(BudgetExceeded):
        HTTPRangeReader("https://sentinel-cogs.s3.us-west-2.amazonaws.com/a", Budget(tmp_path, 10), 8).read(1)


def test_http_200_is_rejected_before_body_read(tmp_path, monkeypatch):
    response = Response(200, b"must not consume")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: response)
    with pytest.raises(OSError):
        HTTPRangeReader("https://sentinel-cogs.s3.us-west-2.amazonaws.com/a", Budget(tmp_path, 100), 8).read(1)
    assert not response.read_called


def test_cache_hit_seek_eof_and_corrupt_cache(tmp_path, monkeypatch):
    calls = []
    payload = b"abcdefghij"
    def open_range(request, timeout):
        calls.append(request.headers)
        start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
        end = min(end, len(payload) - 1)
        return Response(206, payload[start:end + 1],
                        {"Content-Range": f"bytes {start}-{end}/{len(payload)}", "ETag": '"x"'})
    monkeypatch.setattr("urllib.request.urlopen", open_range)
    b = Budget(tmp_path, 100)
    r = HTTPRangeReader("https://sentinel-cogs.s3.us-west-2.amazonaws.com/a", b, 4)
    assert r.read(10) == payload
    assert r.seek(100) == 100 and r.read(2) == b""
    assert len(calls) == 3
    r2 = HTTPRangeReader(r.url, Budget(tmp_path, 100), 4)
    r2.seek(0)
    assert r2.read(10) == payload
    assert len(calls) == 3
    meta = json.loads(next(tmp_path.glob("*/meta.json")).read_text())
    block = tmp_path / next(p for p in tmp_path.iterdir() if p.is_dir() and (p / "0.bin").exists()).name / "0.bin"
    block.write_bytes(b"bad")
    with pytest.raises(ValueError, match='Corrupt cached'):
        HTTPRangeReader(r.url, Budget(tmp_path, 12), 4).read(1)


def test_corrupt_budget_cannot_reset_and_limit_cannot_increase(tmp_path):
    b = Budget(tmp_path, 10)
    b.reserve(8)
    assert Budget(tmp_path, 1000).snapshot()['limit_bytes'] == 10
    (tmp_path/'budget.json').write_text('{broken')
    with pytest.raises(ValueError):
        Budget(tmp_path, 10)


def test_url_allowlist(tmp_path):
    with pytest.raises(ValueError):
        HTTPRangeReader("http://sentinel-cogs.s3.us-west-2.amazonaws.com/a", Budget(tmp_path, 1))
