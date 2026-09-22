"""Prepare, probe, and optionally acquire one bounded Georgia Sentinel crop.

The default mode performs a small STAC metadata request and bounded COG header
probes.  ``--acquire`` is required before any raster crop is written.  The
shared CAFO range-cache budget is never reset; this helper adds a 12 MiB local
per-attempt ceiling on top of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import tifffile
from rasterio.transform import Affine
from rasterio.warp import transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acquire_cafo_cog_pilot import crop_asset, georeference  # noqa: E402
from bounded_http_range import Budget, BudgetExceeded, HTTPRangeReader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "outputs/cafo_pilot_2026-09-11/range_cache"
DEFAULT_MANIFEST = ROOT.parent / (
    "Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/"
    "Data Collection/s2_emit_pairs/emit-v002-2026-07/"
    "EMIT_L2B_CH4PLM_002_20241020T170504_003677/manifest.json"
)
DEFAULT_STAC = (
    "https://earth-search.aws.element84.com/v1/collections/"
    "sentinel-2-l2a/items/S2B_16SGC_20241020_0_L2A"
)
SCENE_ID = "S2B_16SGC_20241020_0_L2A"
SCENE_DATETIME = "2024-10-20T16:34:04.191000+00:00"
PEAK_LONLAT = (-84.36325, 34.27989)
BANDS = {"blue": ("blue", "B02"), "green": ("green", "B03"), "red": ("red", "B04"),
         "swir16": ("swir16", "B11"), "swir22": ("swir22", "B12")}
ASSET_NAMES = ("scl", *BANDS)
LOCAL_ATTEMPT_LIMIT = 12 * 1024 * 1024
METADATA_LIMIT = 1 * 1024 * 1024


class AttemptBudget:
    """Forward every charge to the persistent budget with a local hard cap."""

    def __init__(self, shared: Budget, limit_bytes: int = LOCAL_ATTEMPT_LIMIT):
        self.shared = shared
        # HTTPRangeReader requires the persistent cache root on its budget adapter.
        self.cache_dir = shared.cache_dir
        self.limit_bytes = int(limit_bytes)
        self.charged_bytes = 0
        self.received_bytes = 0

    def reserve(self, amount: int) -> None:
        amount = int(amount)
        if amount < 0 or self.charged_bytes + amount > self.limit_bytes:
            raise BudgetExceeded("repair attempt budget exhausted")
        self.shared.reserve(amount)
        self.charged_bytes += amount

    def received(self, amount: int) -> None:
        amount = int(amount)
        self.shared.received(amount)
        self.received_bytes += amount

    def snapshot(self) -> dict:
        return {"attempt_limit_bytes": self.limit_bytes,
                "attempt_charged_bytes": self.charged_bytes,
                "attempt_received_bytes": self.received_bytes,
                "attempt_remaining_bytes": self.limit_bytes - self.charged_bytes,
                "shared": self.shared.snapshot()}


def snap_peak_bbox(lon: float = PEAK_LONLAT[0], lat: float = PEAK_LONLAT[1],
                   epsg: int = 32616, affine: Affine = Affine(20, 0, 735480, 0, -20, 3791260),
                   pixels: int = 256) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
    """Return a native-grid bbox and peak coordinate in that grid's CRS."""
    x, y = transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    col, row = (~affine) * (x[0], y[0])
    center_col, center_row = math.floor(col), math.floor(row)
    left = affine.c + (center_col - pixels // 2) * affine.a
    top = affine.f + (center_row - pixels // 2) * affine.e
    right, bottom = left + pixels * affine.a, top + pixels * affine.e
    return (left, bottom, right, top), (x[0], y[0])


def _manifest_assets(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pair = next(p for p in data["scenes"] if p.get("scene_id") == SCENE_ID)
    urls = pair["band_urls"]
    out = {"blue": urls["B2"], "green": urls["B3"], "red": urls["B4"],
           "swir16": urls["B11"], "swir22": urls["B12"]}
    return out


def _read_metadata(url: str, budget: AttemptBudget) -> tuple[dict, bytes]:
    """Read at most one reserved MiB from one STAC item response."""
    if url != DEFAULT_STAC:
        raise ValueError('Only the frozen EarthSearch item endpoint is supported')
    budget.reserve(METADATA_LIMIT)
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{METADATA_LIMIT - 1}"})
    with urllib.request.urlopen(request, timeout=20) as response:
        body = response.read(METADATA_LIMIT)
        final_url = response.geturl()
    budget.received(len(body))
    parsed = urllib.parse.urlsplit(final_url)
    expected_path = "/v1/collections/sentinel-2-l2a/items/" + SCENE_ID
    if parsed.scheme != "https" or parsed.hostname != "earth-search.aws.element84.com" or parsed.path != expected_path:
        raise ValueError("STAC redirected outside the exact EarthSearch item endpoint")
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) > METADATA_LIMIT:
        raise ValueError("STAC metadata exceeds 1 MiB reservation")
    return json.loads(body.decode("utf-8")), body


def _asset_href(asset: dict) -> str:
    href = asset.get("href")
    if not isinstance(href, str) or not href:
        raise ValueError("asset missing href")
    return href


def _probe_asset(name: str, asset: dict, budget: AttemptBudget, bbox: tuple[float, ...], epsg: int) -> dict:
    href = _asset_href(asset)
    with HTTPRangeReader(href, budget) as stream:
        with tifffile.TiffFile(stream) as tif:
            page = tif.pages[0]
            found_epsg, affine = georeference(page)
            if found_epsg != epsg:
                raise ValueError(f"{name} CRS {found_epsg} != {epsg}")
            bounds = (affine.c, affine.f + affine.e * page.imagelength,
                      affine.c + affine.a * page.imagewidth, affine.f)
            if bbox[0] < bounds[0] or bbox[1] < bounds[1] or bbox[2] > bounds[2] or bbox[3] > bounds[3]:
                raise ValueError(f"{name} does not fully contain crop bbox")
            if not page.is_tiled or page.samplesperpixel != 1 or page.imagedepth != 1:
                raise ValueError(f"{name} is not a tiled single-band raster")
            c0, r0 = (~affine) * (bbox[0], bbox[3])
            width = (bbox[2] - bbox[0]) / affine.a
            height = (bbox[3] - bbox[1]) / (-affine.e)
            if any(abs(v - round(v)) > 1e-6 for v in (c0, r0, width, height)):
                raise ValueError(f"{name} crop is not aligned to native grid")
            tile_cols = math.ceil(page.imagewidth / page.tilewidth)
            tile_bytes = 0
            range_blocks = set()
            for row in range(int(r0) // page.tilelength, (int(r0) + int(height) - 1) // page.tilelength + 1):
                for col in range(int(c0) // page.tilewidth, (int(c0) + int(width) - 1) // page.tilewidth + 1):
                    index = row * tile_cols + col
                    offset, length = int(page.dataoffsets[index]), int(page.databytecounts[index])
                    tile_bytes += length
                    range_blocks.update(range(offset // 262144, (offset + length - 1) // 262144 + 1))
            return {"href": href, "epsg": found_epsg, "shape": [page.imagelength, page.imagewidth],
                    "resolution_m": affine.a, "bounds": list(bounds),
                    "transform": list(affine), "dtype": str(page.dtype), "tiled": bool(page.is_tiled),
                    "crop_compressed_tile_bytes": tile_bytes,
                    "crop_range_block_upper_bound": len(range_blocks)}


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shared = Budget(args.cache_dir, limit_bytes=100 * 1024 * 1024)
    budget = AttemptBudget(shared)
    pixels = getattr(args, 'pixels', 256)
    if pixels not in (128, 256):
        raise ValueError('Only 128- or 256-pixel 20m crops are supported')
    bbox, peak_xy = snap_peak_bbox(pixels=pixels)
    report = {"tool": "GTM_Model6_repair_native_pair.py", "scene_id": SCENE_ID,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "stac_url": args.stac_url, "peak_lonlat": list(PEAK_LONLAT),
              "peak_projected": list(peak_xy), "bbox_projected": list(bbox),
              "pixels_at_20m": pixels, "crop_width_m": pixels * 20,
              "scope": "peak-centered paired-data diagnostic; full plume containment not assumed",
              "training_approved": False,
              "radiometry_status": "raw DN acquisition only; metadata conversion requires independent QA",
              "status": "started", "mode": "acquire" if args.acquire else "probe",
              "assets": {}}
    try:
        manifest_urls = _manifest_assets(args.manifest)
        if getattr(args, 'stac_file', None):
            with args.stac_file.open('rb') as source:
                stac_body = source.read(METADATA_LIMIT + 1)
            if len(stac_body) > METADATA_LIMIT:
                raise ValueError('Cached STAC item exceeds metadata limit')
            item = json.loads(stac_body)
            report['metadata_source'] = str(args.stac_file)
        else:
            item, stac_body = _read_metadata(args.stac_url, budget)
            report['metadata_source'] = args.stac_url
        if item.get("id") != SCENE_ID:
            raise ValueError("STAC id or datetime does not match frozen manifest")
        properties = item.get("properties", {})
        try:
            if datetime.fromisoformat(str(properties.get("datetime", "")).replace("Z", "+00:00")) != datetime.fromisoformat(SCENE_DATETIME):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("STAC id or datetime does not match frozen manifest")
        report["stac"] = {"id": item.get("id"), "datetime": properties.get("datetime")}
        stac_path = args.output_dir / "GTM_Model6_stac_item.json"
        stac_path.write_bytes(stac_body)
        report["stac"]["saved_path"] = str(stac_path)
        report["stac"]["sha256"] = hashlib.sha256(stac_body).hexdigest()
        assets = item.get("assets", {})
        expected = {"scl": assets.get("scl") or assets.get("SCL")}
        for name, alternatives in BANDS.items():
            expected[name] = next((assets[key] for key in alternatives if key in assets), None)
        if any(v is None for v in expected.values()):
            raise ValueError("STAC missing SCL or required five-band asset")
        for name, asset in expected.items():
            href = _asset_href(asset)
            if name != "scl" and href != manifest_urls[name]:
                raise ValueError(f"{name} URL differs from frozen manifest")
            metadata = asset.get("raster:bands") or asset.get("raster_bands") or []
            if len(metadata) != 1:
                raise ValueError(f"{name} needs one explicit raster metadata entry")
            meta = metadata[0]
            required = {"nodata", "spatial_resolution"}
            if name != "scl":
                required |= {"scale", "offset"}
            if not required.issubset(meta):
                raise ValueError(f"{name} missing required raster metadata: {sorted(required - set(meta))}")
            if any(meta[key] is None or not math.isfinite(float(meta[key])) for key in required):
                raise ValueError(f'{name} raster metadata must be finite and explicit')
            expected_resolution = 10 if name in ('blue', 'green', 'red') else 20
            if float(meta['spatial_resolution']) != expected_resolution:
                raise ValueError(f'{name} unexpected native resolution')
            if name != 'scl' and float(meta['scale']) <= 0:
                raise ValueError(f'{name} reflectance scale must be positive')
            if name == "scl" and not href.lower().endswith("/scl.tif"):
                raise ValueError("SCL asset is not the actual SCL raster")
            report["assets"][name] = {"href": href, "metadata": metadata}
            if name != "scl":
                report["assets"][name]["offset_metadata_conflict"] = (
                    item["properties"].get("earthsearch:boa_offset_applied") is True
                    and float(meta["offset"]) != 0.0)
        epsg = 32616
        for name in ASSET_NAMES:
            report["assets"][name]["probe"] = _probe_asset(name, expected[name], budget, bbox, epsg)
        reference = report["assets"]["scl"]["probe"]
        for name in ASSET_NAMES:
            probe = report["assets"][name]["probe"]
            meta = report["assets"][name]["metadata"][0]
            if abs(float(meta["spatial_resolution"]) - float(probe["resolution_m"])) > 1e-6:
                raise ValueError(f"{name} metadata/native resolution mismatch")
            # Every native grid must have the same CRS and a pixel-aligned bbox;
            # 10 m grids are allowed, but their origins must still align to 20 m.
            if probe["epsg"] != reference["epsg"]:
                raise ValueError(f"{name} CRS differs from SCL")
            origin_x, origin_y = probe["transform"][2], probe["transform"][5]
            res = probe["resolution_m"]
            aligned = ((bbox[0] - origin_x) / res, (bbox[2] - origin_x) / res,
                       (origin_y - bbox[3]) / res, (origin_y - bbox[1]) / res)
            if any(abs(value - round(value)) > 1e-5 for value in aligned):
                raise ValueError(f"{name} x grid does not align with crop")
        report["status"] = "header_probe_complete"
        if args.acquire:
            upper_bound = sum(report['assets'][name]['probe']['crop_range_block_upper_bound']
                              for name in ASSET_NAMES) * 262144
            report['crop_transfer_upper_bound_bytes'] = upper_bound
            if upper_bound > min(budget.limit_bytes - budget.charged_bytes,
                                 shared.limit_bytes - shared.charged_bytes):
                raise BudgetExceeded('Conservative crop block bound exceeds remaining budget; stopping before raster transfer')
            arrays = {}
            for name in ASSET_NAMES:
                meta = report["assets"][name]["metadata"]
                data, affine = crop_asset(expected[name], budget, bbox, epsg)
                if float(meta[0]["spatial_resolution"]) != affine.a:
                    raise ValueError(f"{name} native resolution mismatch")
                out = args.output_dir / f"GTM_Model6_{name}.tif"
                nodata = meta[0].get("nodata")
                with rasterio.open(out, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                                   count=1, dtype=data.dtype, crs=f"EPSG:{epsg}", transform=affine,
                                   nodata=nodata, compress="deflate") as dst:
                    dst.write(data, 1)
                    if name != "scl":
                        dst.scales = (float(meta[0]["scale"]),)
                        dst.offsets = (float(meta[0]["offset"]),)
                report["assets"][name]["output"] = str(out)
                report["assets"][name]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
                arrays[name] = data
                if name == "scl":
                    clear = np.isin(data, (4, 5, 6))
                    report["strict_clear_fraction"] = float(clear.mean())
                    if report["strict_clear_fraction"] < 0.9:
                        raise ValueError("Crop clear fraction below 90%; stop before spectral transfer")
            report["status"] = "crop_complete"
    except Exception as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
    report["budget"] = budget.snapshot()
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stac-url", default=DEFAULT_STAC)
    parser.add_argument('--stac-file', type=Path, help='Reuse the saved bounded STAC response, still validating item/time/assets')
    parser.add_argument('--pixels', type=int, choices=(128, 256), default=256, help='Crop width in native 20m pixels')
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "budget": report["budget"]}, indent=2))
    return 0 if report["status"] in {"header_probe_complete", "crop_complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
