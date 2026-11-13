"""Read-only archive/raster review and RGB previews of the announced CAFOSat pilot.

Reject unsafe archive members, excessive expansion, unsupported raster formats,
or an existing review directory. Filenames are not promoted into verified labels.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/GTM_Model0_hf_pilot_20260925'
ORIGINALS = STAGE / 'oishee3003__CAFOSat/originals'
REVIEW = STAGE / 'review/CAFOSat'


def font(size):
    path = Path('C:/Windows/Fonts/segoeui.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def review():
    if REVIEW.exists():
        raise FileExistsError('Refusing to overwrite an existing intake review')
    records, previews = [], []
    for archive_path in sorted(ORIGINALS.glob('*.tar.gz')):
        receipt = json.loads(archive_path.with_suffix('.gz.receipt.json').read_text())
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != receipt['sha256']:
            raise ValueError('Archive differs from its verified download')
        expanded, members = 0, 0
        with tarfile.open(archive_path, 'r|gz') as archive:
            for member in archive:
                members += 1
                expanded += member.size
                name = PurePosixPath(member.name)
                if (name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name
                        or member.issym() or member.islnk()):
                    raise ValueError('Unsafe archive member')
                if members > 2000 or expanded > 256 * 1024 * 1024 or member.size > 12 * 1024 * 1024:
                    raise ValueError('Archive exceeds inspection limits')
                if member.isdir():
                    continue
                if not member.isfile() or name.suffix.lower() != '.tif':
                    raise ValueError('Unexpected non-TIFF archive payload')
                with archive.extractfile(member) as handle:
                    payload = handle.read(member.size)
                if len(payload) != member.size:
                    raise ValueError('Short archive member')
                with MemoryFile(payload) as mem, mem.open() as ds:
                    if ds.width * ds.height > 4_000_000 or ds.count not in (3, 4) or set(ds.dtypes) != {'uint8'}:
                        raise ValueError('Unexpected image format')
                    rgb = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
                    bounds = list(transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)) if ds.crs else None
                    center = [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2] if bounds else None
                    state = archive_path.name.split('_')[0]
                    flags = []
                    # Coarse necessary-condition checks, not a state boundary lookup or repaired coordinate.
                    if center and state == 'CA' and not 32 <= center[1] <= 42.1:
                        flags.append('CA archive latitude outside broad California envelope')
                    if center and state == 'WA' and not 45.5 <= center[1] <= 49.1:
                        flags.append('WA archive latitude outside broad Washington envelope')
                    records.append({
                        'index': len(records) + 1, 'archive': archive_path.name, 'member': member.name,
                        'archive_declared_state': state, 'file_sha256': hashlib.sha256(payload).hexdigest(),
                        'shape': [ds.count, ds.height, ds.width], 'dtype': list(ds.dtypes),
                        'crs': str(ds.crs), 'affine': list(ds.transform)[:6], 'pixel_spacing': list(ds.res),
                        'bounds_wgs84_from_file': bounds, 'center_lon_lat_from_file': center,
                        'color_interpretation': [c.name for c in ds.colorinterp], 'tags': ds.tags(),
                        'acquisition_date': None, 'date_status': 'not present in inspected default raster tags',
                        'label_status': 'filename hints only; master CSV and human verification flags not acquired',
                        'geolocation_status': 'unverified file metadata', 'geolocation_flags': flags,
                        'preview': f'cafosat_{len(records)+1:02d}.png',
                    })
                    previews.append(Image.fromarray(rgb))
    if not records:
        raise ValueError('No image members found')
    REVIEW.mkdir(parents=True)
    tile_width, tile_height = 360, 455
    sheet = Image.new('RGB', (tile_width * 3 + 40, math.ceil(len(records) / 3) * tile_height + 125), '#eef3f4')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), 'CAFOSat intake: 17 downloaded image patches', fill='#123442', font=font(28))
    draw.text((20, 61), 'Observed RGB only. No predictions, methane labels, or new facility annotations.', fill='#42616d', font=font(17))
    draw.text((20, 87), 'Coordinates below come from file headers and remain unverified.', fill='#42616d', font=font(16))
    cards = []
    for i, (row, image) in enumerate(zip(records, previews)):
        image.save(REVIEW / row['preview'])
        thumbnail = image.copy()
        thumbnail.thumbnail((336, 336))
        x, y = 20 + (i % 3) * tile_width, 125 + (i // 3) * tile_height
        sheet.paste(thumbnail, (x, y))
        draw.text((x, y + 341), f'{row["index"]:02d} | {row["archive_declared_state"]} archive | {row["shape"][0]} bands', fill='#123442', font=font(17))
        center = row['center_lon_lat_from_file']
        location = f'{center[1]:.5f}, {center[0]:.5f}' if center else 'Coordinates missing'
        draw.text((x, y + 367), location, fill='#42616d', font=font(15))
        label = PurePosixPath(row['member']).name
        for j, line in enumerate(textwrap.wrap(label, 42)[:2]):
            draw.text((x, y + 391 + j * 18), line, fill='#42616d', font=font(12))
        if row['geolocation_flags']:
            draw.text((x, y + 429), 'CHECK: archive state / latitude mismatch', fill='#a23a20', font=font(13))
        cards.append(f'<article><h2>{row["index"]:02d} | {html.escape(row["archive_declared_state"])} archive</h2>'
                     f'<a href="{row["preview"]}"><img src="{row["preview"]}" alt="Observed CAFOSat RGB patch {row["index"]}"></a>'
                     f'<p>{html.escape(label)}</p><p>File-header center: {location}</p>'
                     f'<p class="flag">{html.escape("; ".join(row["geolocation_flags"]))}</p></article>')
    sheet.save(REVIEW / 'GTM_Model0_CAFOSat_contact_sheet.png')
    summary = {'schema_version': 1, 'scope': 'read-only pilot suitability review', 'records': records,
               'count': len(records), 'geolocation_flags_count': sum(bool(r['geolocation_flags']) for r in records),
               'distinct_file_hashes': len({r['file_sha256'] for r in records}),
               'label_changes': 0, 'training_manifest_created': False,
               'missing': ['master CSV metadata', 'independently verified coordinates', 'acquisition timestamps',
                           'reviewed negative controls', 'methane masks/concentration', 'matched Sentinel/EMIT']}
    (REVIEW / 'intake_manifest.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (REVIEW / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAFOSat pilot: input data review</title><style>body{font:16px system-ui;background:#eef3f4;color:#163743;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}article{background:white;padding:16px;border-radius:10px}img{width:100%;height:auto}h2{font-size:18px}p{overflow-wrap:anywhere}.flag{color:#a23a20}aside{padding:16px;background:#fff5df;margin-bottom:24px}</style>
<h1>CAFOSat: downloaded input imagery</h1><aside>17 image patches from four small state archives. These are observed RGB images, not model predictions. No labels were changed. Master annotations, acquisition dates, and matched methane evidence are not available in this pilot. File-header coordinates are unverified; flagged state/latitude conflicts must be resolved before mapping or joining other datasets.</aside><main>'''
        + ''.join(cards) + '</main></html>', encoding='utf-8')
    return {k: v for k, v in summary.items() if k != 'records'}


if __name__ == '__main__':
    print(json.dumps(review(), indent=2))
