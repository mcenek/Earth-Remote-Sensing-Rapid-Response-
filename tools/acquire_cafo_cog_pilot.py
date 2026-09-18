"""One-location COG pilot with persistent bounded HTTP range reads.

Selects the lowest catalog scene-cloud candidate for facility 56216; this is
engineering selection only. Reads crop SCL before six observed spectral bands.
No methane labels, model fitting, archive downloads or automatic retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform
import tifffile

from bounded_http_range import Budget, HTTPRangeReader

BANDS = ('blue', 'green', 'red', 'nir', 'swir16', 'swir22')


def georeference(page):
    tags = page.tags
    scale, tie = tags['ModelPixelScaleTag'].value, tags['ModelTiepointTag'].value
    keys = tags['GeoKeyDirectoryTag'].value
    entries = {keys[i]: tuple(keys[i+1:i+4]) for i in range(4, len(keys), 4)}
    crs = entries.get(3072)
    if not crs or crs[:2] != (0, 1) or not (32601 <= crs[2] <= 32660 or 32701 <= crs[2] <= 32760):
        raise ValueError('Pilot requires explicit WGS84 UTM projected CRS')
    if entries.get(1025) != (0, 1, 1):
        raise ValueError('Pilot requires PixelIsArea georeferencing')
    if len(tie) != 6 or scale[0] <= 0 or scale[1] <= 0 or scale[0] != scale[1]:
        raise ValueError('Unsupported scale/tiepoint contract')
    return crs[2], Affine(scale[0], 0, tie[3]-tie[0]*scale[0],
                          0, -scale[1], tie[4]+tie[1]*scale[1])


def read_window(page, stream, col, row, width, height):
    """Decode only intersecting full-resolution tiles, retaining native pixels."""
    if not page.is_tiled or page.samplesperpixel != 1 or page.imagedepth != 1:
        raise ValueError('Expected tiled single-band raster')
    if min(col, row) < 0 or col+width > page.imagewidth or row+height > page.imagelength:
        raise ValueError('Complete crop is outside source coverage')
    tw, th = page.tilewidth, page.tilelength
    if tw > 2048 or th > 2048:
        raise ValueError('Unexpectedly large compression tiles')
    across = math.ceil(page.imagewidth/tw)
    output = np.empty((height, width), dtype=page.dtype)
    for ty in range(row//th, (row+height-1)//th+1):
        for tx in range(col//tw, (col+width-1)//tw+1):
            index = ty*across+tx
            length = page.databytecounts[index]
            if not 0 < length <= 16*1024*1024:
                raise ValueError('Unexpected compressed tile size')
            stream.seek(page.dataoffsets[index])
            payload = stream.read(length)
            decoded = page.decode(payload, index)[0]
            if decoded is None:
                raise ValueError('Missing decoded tile')
            tile = decoded[0, :, :, 0]
            x0, x1 = max(col, tx*tw), min(col+width, (tx+1)*tw)
            y0, y1 = max(row, ty*th), min(row+height, (ty+1)*th)
            output[y0-row:y1-row, x0-col:x1-col] = tile[y0-ty*th:y1-ty*th, x0-tx*tw:x1-tx*tw]
    return output


def crop_asset(asset, budget, bbox, expected_crs):
    with HTTPRangeReader(asset['href'], budget) as stream:
        with tifffile.TiffFile(stream) as tif:
            page = tif.pages[0]
            epsg, affine = georeference(page)
            if epsg != expected_crs:
                raise ValueError('Bands do not share the expected CRS')
            left, bottom, right, top = bbox
            c, r = (~affine)*(left, top)
            width, height = (right-left)/affine.a, (top-bottom)/(-affine.e)
            if any(abs(x-round(x)) > 1e-6 for x in (c, r, width, height)):
                raise ValueError('Crop does not align with native grid')
            data = read_window(page, stream, int(round(c)), int(round(r)), int(round(width)), int(round(height)))
            return data, affine * Affine.translation(round(c), round(r))


def write_json(path, report):
    with path.open('x', encoding='utf-8') as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.cache_dir, limit_bytes=100*1024*1024)
    catalog = json.loads(args.catalog.read_text(encoding='utf-8'))
    query = next(q for q in catalog['queries'] if q['facility_id']=='56216')
    item = min(query['items'], key=lambda i: (i['cloud_cover'], i['id']))
    report = {'facility_id': query['facility_id'], 'item_id': item['id'], 'datetime': item['datetime'],
              'scope': 'one-location engineering crop pilot; no training labels',
              'selection': 'lowest catalog scene cloud among previously bounded candidates',
              'catalog_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              'crop_width_m': 1280, 'methane_truth_available': False,
              'status': 'started', 'bands': {}}
    try:
        if item['collection'] != 'sentinel-2-l2a':
            raise ValueError('Unexpected source product')
        for name in ('scl',)+BANDS:
            if name not in item['assets']:
                raise ValueError(f'Missing catalog asset {name}')
        with HTTPRangeReader(item['assets']['scl']['href'], budget) as stream:
            with tifffile.TiffFile(stream) as tif:
                epsg, _ = georeference(tif.pages[0])
        x, y = transform('EPSG:4326', f'EPSG:{epsg}', [query['longitude']], [query['latitude']])
        left, top = math.floor((x[0]-640)/20)*20, math.ceil((y[0]+640)/20)*20
        bbox = (left, top-1280, left+1280, top)
        report.update(crs=f'EPSG:{epsg}', bbox_projected=list(bbox),
                      coordinate_assumption='CSV latitude/longitude treated as WGS84; inventory datum unconfirmed')
        arrays = {}
        for name in ('scl',)+BANDS:
            asset = item['assets'][name]
            data, affine = crop_asset(asset, budget, bbox, epsg)
            raster_meta = asset.get('raster_bands') or []
            if len(raster_meta) != 1:
                raise ValueError('Need explicit raster-band metadata')
            meta = raster_meta[0]
            if float(meta['spatial_resolution']) != affine.a:
                raise ValueError('Native resolution differs from catalog')
            nodata = meta.get('nodata')
            if nodata is None:
                raise ValueError('No explicit nodata')
            if name == 'scl':
                clear = np.isin(data, (4, 5, 6))
                report['scl_counts'] = {str(int(v)): int((data==v).sum()) for v in np.unique(data)}
                report['strict_clear_fraction'] = float(clear.mean())
            else:
                if 'scale' not in meta or 'offset' not in meta:
                    raise ValueError('Missing reflectance scale/offset')
            path = args.output_dir / f'{name}.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                               count=1, dtype=data.dtype, crs=f'EPSG:{epsg}', transform=affine,
                               nodata=nodata, compress='deflate') as dst:
                dst.write(data, 1)
                dst.scales = (float(meta.get('scale', 1)),)
                dst.offsets = (float(meta.get('offset', 0)),)
            report['bands'][name] = {'shape': list(data.shape), 'resolution_m': affine.a,
                                     'source': asset['href'], 'scale': meta.get('scale'),
                                     'offset': meta.get('offset'), 'nodata': nodata,
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if name=='scl' and report['strict_clear_fraction'] < .9:
                raise ValueError('Crop clear fraction below 90%; stop before spectral transfer')
            arrays[name] = data
        from PIL import Image, ImageDraw
        rgb = []
        for name in ('red', 'green', 'blue'):
            b = report['bands'][name]
            reflectance = arrays[name].astype(float)*b['scale']+b['offset']
            rgb.append(np.uint8(np.clip(reflectance/.3, 0, 1)**(1/2.2)*255))
        preview = Image.fromarray(np.stack(rgb, axis=-1)).resize((512, 512), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(preview)
        cx, cy = (x[0]-left)/1280*512, (top-y[0])/1280*512
        draw.ellipse((cx-8,cy-8,cx+8,cy+8), outline='yellow', width=2)
        preview.save(args.output_dir/'rgb_preview.png')
        report['status'] = 'crop_complete_pending_visual_review'
    except Exception as error:
        report.update(status='stopped', error=f'{type(error).__name__}: {error}')
    finally:
        report['budget'] = budget.snapshot()
        write_json(args.output_dir/'report.json', report)
    print(json.dumps({'status': report['status'], 'budget': report['budget'], 'error': report.get('error')}))


if __name__ == '__main__':
    main()
