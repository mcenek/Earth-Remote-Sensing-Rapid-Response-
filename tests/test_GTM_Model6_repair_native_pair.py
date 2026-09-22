import json
import sys
from pathlib import Path

import pytest
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import GTM_Model6_repair_native_pair as repair
from bounded_http_range import BudgetExceeded


@pytest.mark.parametrize('pixels', [128, 256])
def test_peak_bbox_is_native_grid_and_contains_peak(pixels):
    bbox, xy = repair.snap_peak_bbox(affine=Affine(20, 0, 735480, 0, -20, 3791260), pixels=pixels)
    left, bottom, right, top = bbox
    assert all(abs(v / 20 - round(v / 20)) < 1e-9 for v in bbox)
    assert right - left == pixels * 20
    assert top - bottom == pixels * 20
    assert left <= xy[0] < right
    assert bottom <= xy[1] < top


def test_frozen_manifest_uses_scenes_and_five_band_urls():
    urls = repair._manifest_assets(repair.DEFAULT_MANIFEST)
    assert set(urls) == set(repair.BANDS)
    assert urls["blue"].endswith("/B02.tif")


def test_attempt_budget_forwards_global_and_enforces_local_cap(tmp_path):
    shared = repair.Budget(tmp_path / "cache", limit_bytes=1000)
    budget = repair.AttemptBudget(shared, limit_bytes=600)
    budget.reserve(400)
    budget.received(399)
    assert budget.snapshot()["attempt_charged_bytes"] == 400
    assert budget.snapshot()["shared"]["charged_bytes"] == 400
    with pytest.raises(BudgetExceeded):
        budget.reserve(201)
    assert budget.snapshot()["shared"]["charged_bytes"] == 400


def test_metadata_reader_reserves_full_mib_and_reads_bounded(monkeypatch, tmp_path):
    class Response:
        headers = {"Content-Length": "17"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def geturl(self): return repair.DEFAULT_STAC
        def read(self, amount):
            assert amount == repair.METADATA_LIMIT
            return json.dumps({"id": "scene"}).encode()

    monkeypatch.setattr(repair.urllib.request, "urlopen", lambda request, timeout: Response())
    budget = repair.AttemptBudget(repair.Budget(tmp_path / "cache", 2 * repair.METADATA_LIMIT))
    item, body = repair._read_metadata(repair.DEFAULT_STAC, budget)
    assert item["id"] == "scene"
    assert body.startswith(b"{")
    assert budget.charged_bytes == repair.METADATA_LIMIT


def test_reader_adapter_exposes_cache_dir(tmp_path):
    shared = repair.Budget(tmp_path / "cache", 1024)
    assert repair.AttemptBudget(shared).cache_dir == shared.cache_dir


def test_global_limit_still_wins_and_is_not_reset(tmp_path):
    shared = repair.Budget(tmp_path / 'cache', 1000)
    shared.reserve(900)
    attempt = repair.AttemptBudget(shared, 600)
    with pytest.raises(BudgetExceeded):
        attempt.reserve(101)
    persisted = json.loads((tmp_path / 'cache/budget.json').read_text())
    assert persisted['charged_bytes'] == 900 and persisted['limit_bytes'] == 1000
    assert attempt.charged_bytes == 0


def test_metadata_rejects_other_endpoints_before_spending_budget(tmp_path):
    attempt = repair.AttemptBudget(repair.Budget(tmp_path / 'cache', repair.METADATA_LIMIT))
    with pytest.raises(ValueError, match='frozen EarthSearch'):
        repair._read_metadata('https://example.com/other', attempt)
    assert attempt.charged_bytes == 0


def test_canonical_earthsearch_item_completes_header_validation_without_network(tmp_path, monkeypatch):
    from argparse import Namespace
    urls = {name: 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/scene/' + aliases[1] + '.tif'
            for name, aliases in repair.BANDS.items()}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'scenes': [{'scene_id': repair.SCENE_ID, 'band_urls': {
        'B2': urls['blue'], 'B3': urls['green'], 'B4': urls['red'],
        'B11': urls['swir16'], 'B12': urls['swir22']}}]}))
    assets = {}
    for name in repair.ASSET_NAMES:
        resolution = 10 if name in ('blue', 'green', 'red') else 20
        meta = {'nodata': 0, 'spatial_resolution': resolution}
        if name != 'scl':
            meta.update(scale=.0001, offset=-.1)
        assets[name] = {'href': urls.get(name, 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/scene/SCL.tif'), 'raster:bands': [meta]}
    item = {'id': repair.SCENE_ID, 'properties': {'datetime': repair.SCENE_DATETIME.replace('+00:00', 'Z')}, 'assets': assets}
    monkeypatch.setattr(repair, '_read_metadata', lambda url, budget: (item, json.dumps(item).encode()))
    def probe(name, asset, budget, bbox, epsg):
        res = asset['raster:bands'][0]['spatial_resolution']
        return {'resolution_m': res, 'epsg': epsg, 'transform': [res, 0, bbox[0], 0, -res, bbox[3]], 'crop_range_block_upper_bound': 1}
    monkeypatch.setattr(repair, '_probe_asset', probe)
    report = repair.run(Namespace(output_dir=tmp_path/'out', cache_dir=tmp_path/'cache', manifest=manifest, stac_url=repair.DEFAULT_STAC, acquire=False))
    assert report['status'] == 'header_probe_complete', report.get('error')
    assert len(report['assets']) == 6
    assert report['budget']['attempt_charged_bytes'] == 0


def test_probe_rejects_partial_coverage(monkeypatch, tmp_path):
    class FakePage:
        imagewidth = imagelength = 10
        tilewidth = tilelength = 10
        samplesperpixel = imagedepth = 1
        is_tiled = True
        dtype = "uint16"
        dataoffsets = [0]
        databytecounts = [10]

    class FakeTif:
        pages = [FakePage()]
        def __enter__(self): return self
        def __exit__(self, *args): return False

    class FakeStream:
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(repair, "HTTPRangeReader", lambda href, budget: FakeStream())
    monkeypatch.setattr(repair.tifffile, "TiffFile", lambda stream: FakeTif())
    monkeypatch.setattr(repair, "georeference", lambda page: (32616, Affine(20, 0, 0, 0, -20, 200)))
    budget = repair.AttemptBudget(repair.Budget(tmp_path / "cache", 1024 * 1024))
    with pytest.raises(ValueError, match="fully contain"):
        repair._probe_asset("blue", {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/x.tif"},
                            budget, (0, 0, 400, 400), 32616)
