"""Offline transport and real JP2 opener tests; no network or shared ledger."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bounded_http_range import Budget, BudgetExceeded
from GTM_Model7_bounded_jp2_v13 import JP2RangeReader, FrozenJP2Opener, validate_frozen_url, load_frozen_urls, _NoRedirect

URL = "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/17/R/LP/2025/7/31/0/B02.jp2"


class Response:
    def __init__(self, body, start, end, total, etag='"test-etag"', status=206,
                 final_url=URL, encoding=None):
        self.body = BytesIO(body)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}", "ETag": etag,
                        "Content-Length": str(len(body))}
        if encoding:
            self.headers["Content-Encoding"] = encoding
        self.final_url = final_url

    def read(self, size=-1):
        return self.body.read(size)

    def geturl(self):
        return self.final_url

    def close(self):
        self.body.close()


class BytesTransport:
    def __init__(self, data):
        self.data = data
        self.calls = []
        self.change = None

    def __call__(self, request):
        self.calls.append(request)
        assert request.full_url == URL
        assert request.headers["Range"].startswith("bytes=")
        start, end = map(int, request.headers["Range"][6:].split("-"))
        end = min(end, len(self.data)-1)
        body = self.data[start:end+1]
        response = Response(body, start, end, len(self.data))
        if self.change:
            self.change(response)
        return response


class RangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.budget = Budget(Path(self.tmp.name) / "ledger", 10000)
        self.transport = BytesTransport(bytes(range(256))*8)

    def reader(self, **kw):
        return JP2RangeReader(URL, self.budget, allowed_urls={URL}, block_size=256,
                              transport=self.transport, **kw)

    def test_reserves_before_transport_and_reuses_cache(self):
        def check(request):
            self.assertEqual(self.budget.charged_bytes, 257)
            return BytesTransport.__call__(self.transport, request)
        original = self.transport
        reader = JP2RangeReader(URL, self.budget, allowed_urls={URL}, block_size=256,
                                transport=check)
        self.assertEqual(reader.read(12), original.data[:12])
        self.assertEqual(self.budget.charged_bytes, 257)
        self.assertEqual(self.budget.received_bytes, 256)
        self.assertEqual(reader.seek(3), 3)
        self.assertEqual(reader.read(5), original.data[3:8])
        self.assertEqual(len(original.calls), 1)

    def test_rejects_bad_range_etag_size_status_redirect_and_encoding(self):
        cases = [
            lambda r: setattr(r, 'status', 200),
            lambda r: r.headers.update({"Content-Range": "bytes 1-256/2048"}),
            lambda r: r.headers.pop("ETag"),
            lambda r: r.headers.update({"Content-Length": "999"}),
            lambda r: setattr(r, 'final_url', 'https://elsewhere.invalid/file.jp2'),
            lambda r: r.headers.update({"Content-Encoding": "gzip"}),
        ]
        for change in cases:
            with self.subTest(change=change):
                with TemporaryDirectory() as folder:
                    budget = Budget(Path(folder), 10000)
                    transport = BytesTransport(bytes(range(256))*8)
                    transport.change = change
                    reader = JP2RangeReader(URL, budget, allowed_urls={URL}, block_size=256,
                                            transport=transport)
                    with self.assertRaises((OSError, ValueError)):
                        reader.read(1)
                    self.assertEqual(budget.charged_bytes, 257)

    def test_rejects_changed_etag_or_object_size(self):
        reader = self.reader()
        reader.read(1)
        self.transport.change = lambda r: r.headers.update({"ETag": '"changed"'})
        reader.seek(300)
        with self.assertRaises(OSError):
            reader.read(1)
        self.transport.change = lambda r: r.headers.update({"Content-Range": "bytes 512-767/9999"})
        reader.seek(512)
        with self.assertRaises(OSError):
            reader.read(1)

    def test_cache_hash_corruption_and_budget_exhaustion(self):
        reader = self.reader()
        reader.read(1)
        cache_file = next((Path(self.tmp.name) / "ledger").glob("jp2-*/*/0.bin"))
        cache_file.write_bytes(b"bad")
        with self.assertRaises(ValueError):
            self.reader().read(1)
        with TemporaryDirectory() as folder:
            budget = Budget(Path(folder), 256)
            transport = BytesTransport(b"x" * 1024)
            reader = JP2RangeReader(URL, budget, allowed_urls={URL}, block_size=256,
                                    transport=transport)
            with self.assertRaises(BudgetExceeded):
                reader.read(1)
            self.assertEqual(len(transport.calls), 0)

    def test_rejects_unfrozen_host_path_and_sidecars(self):
        bad = [URL.replace("https:", "http:"), URL.replace("sentinel-s2-l1c", "evil"),
               URL.replace("/B02.jp2", "/../../B02.jp2"), URL+"?download=1",
               URL.replace("/B02.jp2", "/B02.tif")]
        for value in bad:
            with self.subTest(url=value), self.assertRaises(ValueError):
                validate_frozen_url(value, {URL})
        opener = FrozenJP2Opener({URL}, self.budget, transport=self.transport, block_size=256)
        with self.assertRaises(FileNotFoundError):
            opener(URL + ".aux.xml", "rb")
        with self.assertRaises(ValueError):
            opener(URL, "wb")
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "Moved", {},
                                                          "https://elsewhere.invalid/file.jp2"))
        with self.assertRaises(ValueError):
            self.reader().read(-1)

    def test_loads_only_exact_hashed_three_event_urls(self):
        names = ("blue", "green", "red", "nir", "swir16", "swir22")
        bands = ("B02", "B03", "B04", "B08", "B11", "B12")
        events = []
        for index in range(3):
            scenes = {}
            for role, day in (("same_domain_current", 31-index*2),
                              ("same_domain_reference", 30-index*2)):
                scenes[role] = {"collection": "sentinel-2-l1c",
                                "spectral_assets": {name: {"normalized_https_url":
                                    f"https://sentinel-s2-l1c.s3.amazonaws.com/tiles/17/R/LP/2025/7/{day}/0/{band}.jp2"}
                                    for name, band in zip(names, bands)}}
            events.append(scenes)
        document = {"status": "offline_preflight_only_no_transfer", "events": events}
        source = Path(self.tmp.name) / "frozen.json"

        def write_and_load(payload):
            source.write_text(json.dumps(payload), encoding="utf-8")
            return load_frozen_urls(source, hashlib.sha256(source.read_bytes()).hexdigest())

        allowed = write_and_load(document)
        self.assertEqual(len(allowed), 36)
        self.assertIn(URL, allowed)
        with self.assertRaises(ValueError):
            load_frozen_urls(source, "0" * 64)
        malformed = json.loads(json.dumps(document))
        del malformed["events"][0]["same_domain_current"]["spectral_assets"]["blue"]
        with self.assertRaises(ValueError):
            write_and_load(malformed)
        duplicate = json.loads(json.dumps(document))
        duplicate["events"][1]["same_domain_reference"]["spectral_assets"]["blue"] = (
            duplicate["events"][0]["same_domain_reference"]["spectral_assets"]["blue"])
        with self.assertRaises(ValueError):
            write_and_load(duplicate)

    def test_real_local_jp2_window_via_opener(self):
        path = Path(self.tmp.name) / "fixture.jp2"
        array = np.random.default_rng(32).integers(0, 65535, (1024, 1024), dtype=np.uint16)
        transform = from_origin(500000, 4500000, 10, 10)
        with rasterio.open(path, "w", driver="JP2OpenJPEG", width=1024, height=1024,
                           count=1, dtype="uint16", crs="EPSG:32617", transform=transform,
                           REVERSIBLE="YES", QUALITY=100) as dst:
            dst.write(array, 1)
        source = path.read_bytes()
        transport = BytesTransport(source)
        budget = Budget(Path(self.tmp.name) / "fixture_ledger", max(10_000_000, len(source)*2))
        opener = FrozenJP2Opener({URL}, budget, transport=transport, block_size=65536)
        window = Window(400, 400, 64, 64)
        with rasterio.open(path) as local, rasterio.open(URL, driver="JP2OpenJPEG", opener=opener) as remote:
            np.testing.assert_array_equal(remote.read(1, window=window), local.read(1, window=window))
            self.assertEqual(remote.transform, local.transform)
            self.assertEqual(remote.crs, local.crs)
            self.assertEqual(remote.width, local.width)
        self.assertEqual(budget.received_bytes, sum(len(transport.data[int(c.headers['Range'][6:].split('-')[0]):
                                                                    min(int(c.headers['Range'][6:].split('-')[1])+1,len(source))])
                                                    for c in transport.calls))
        self.assertGreater(budget.charged_bytes, 0)
        self.assertLessEqual(budget.received_bytes, budget.charged_bytes)
        print(json.dumps({"fixture_total_bytes": len(source), "fixture_received_bytes": budget.received_bytes,
                          "fixture_charged_bytes": budget.charged_bytes, "range_requests": len(transport.calls)}))


if __name__ == "__main__":
    unittest.main()
