"""Expose the original team's committed RGBA overlay without inventing raw scores.

Failure cases checked before export: changed Git bytes, missing/incorrect CRS,
non-RGBA raster, or overwriting an existing export. The neighboring Sentinel file
is recorded for audit only: sharing a directory does not prove input pairing.
No inference, training, imagery acquisition, or inverse-colormap reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT.parent / 'Earth-Remote-Sensing-Rapid-Response-'
RELATIVE = 'ERSRR_Website/Predictions/testprediction.tif'
BLOB = 'ff471819d0c6240ee5343bc78d63b3bf07cdd882'
COMMIT = 'ffbd96f41f9fae454bf21bc8536a9c3e05711321'
IDENTIFIER = 'GTM_Model0_capstone_archive_20260327'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing archival export')
    source = HISTORICAL / RELATIVE
    actual_blob = subprocess.check_output(
        ['git', '-C', str(HISTORICAL), 'hash-object', str(source)], text=True).strip()
    if actual_blob != BLOB:
        raise ValueError('Saved raster no longer matches the original committed artifact')
    with rasterio.open(source) as ds:
        if ds.crs != rasterio.crs.CRS.from_epsg(3857) or ds.count != 4 or any(t != 'uint8' for t in ds.dtypes):
            raise ValueError('Expected the original EPSG:3857 uint8 RGBA overlay')
        rgba = np.moveaxis(ds.read(), 0, -1)
        west, south, east, north = transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)
        raster_info = dict(shape=list(ds.shape), crs=str(ds.crs), affine=list(ds.transform)[:6],
                           bounds=list(ds.bounds), bounds_wgs84=[west, south, east, north])
    notes = ('Authentic original-team output committed March 27, 2026; acquisition date is unknown. '
             'Saved red RGBA visualization only, not raw probabilities or methane concentration. '
             'Contemporary rendering code used a 0.30 display cutoff; the saved run has no checkpoint receipt. '
             'The later April checkpoint is not attributed to this older result. '
             'Matching Sentinel input and methane ground truth are unverified. '
             'No accuracy or superiority comparison with Model6 can be calculated from this image.')
    provenance = dict(schema_version=1, artifact_kind='historical_rendered_prediction',
                      source=str(source), source_git_path=RELATIVE, source_git_blob=BLOB,
                      first_commit=COMMIT, archive_date='2026-03-27', acquisition_date=None,
                      source_sha256=digest(source), raster=raster_info,
                      checkpoint_run_identity='not recorded', raw_probability_available=False,
                      matched_reference_available=False, notes=notes)
    companion = HISTORICAL / 'ERSRR_Website/temp_s2.tif'
    if companion.exists():
        with rasterio.open(companion) as ds:
            provenance['unpaired_neighbor'] = dict(path=str(companion), sha256=digest(companion),
                crs=str(ds.crs), shape=list(ds.shape), bounds=list(ds.bounds),
                bounds_wgs84=list(transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds)),
                disposition='Not used as background: input pairing and date unverified')
    output.mkdir(parents=True)
    image_path = output / 'GTM_Model0_original_team_overlay.png'
    Image.fromarray(rgba).save(image_path)
    # Behavioral verification: export preserves every color/alpha value, not just shape.
    if not np.array_equal(np.asarray(Image.open(image_path)), rgba):
        raise ValueError('PNG round-trip changed the committed rendering')
    provenance['export_sha256'] = digest(image_path)
    provenance['png_pixel_roundtrip_equal'] = True
    scene = dict(id='capstone_web_saved_20260327', name='Original team | saved March 27, 2026',
        location='United States | archive acquisition date unknown', date=None,
        bounds=[[south, west], [north, east]], outputKind='rendered_archive',
        predictionImage=image_path.name, predictionAvailable=True,
        predictionHash=provenance['source_sha256'], source=RELATIVE + ' | Git ' + BLOB,
        checkpointHash='Saved run checkpoint unrecorded',
        resolution=f'{rgba.shape[1]} x {rgba.shape[0]} archived EPSG:3857 rendering',
        split='Original team archive; training/test membership unknown',
        rgbNote='Matching source imagery unverified', truthLabel='No matched methane reference',
        truthNote='No reference supplied for this saved run', notes=notes)
    experiment = dict(schemaVersion=1, id=IDENTIFIER, name='Original capstone team | March 2026 archive',
        lifecycle='archived', status='ARCHIVED ORIGINAL-TEAM OUTPUT | visual comparison only',
        description='Recovered original-team red prediction overlay; distinct from our later U-Nets and Model6.',
        warning='ARCHIVED RENDERING | fixed colors; no raw probabilities or matched ground truth',
        notes=notes, reportUrl='GTM_Model0_archive_provenance.md', scenes=[scene])
    for filename, value in [('experiment.json', experiment), ('GTM_Model0_archive_provenance.json', provenance)]:
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'GTM_Model0_archive_provenance.md').write_text(
        '# Original capstone team: recovered output\n\n' + notes + '\n\n'
        + f'Git commit: `{COMMIT}`\n\nGit blob: `{BLOB}`\n\n'
        + f'Source SHA-256: `{provenance["source_sha256"]}`\n\n'
        + 'The PNG preserves every RGBA byte of the committed raster. Coordinates come from the GeoTIFF, '
          'not the historical KML, whose LatLonBox values are not valid longitude/latitude. '
          'The nearby temp_s2.tif has different bounds and lacks a run association; it is not silently substituted.\n\n'
        + '[Original team and Model6 visual comparison](/GTM_Model0_team_comparison.html)\n', encoding='utf-8')
    return dict(output=str(output), experiment=IDENTIFIER, source_blob_verified=True,
                png_pixel_roundtrip_equal=True, source_bounds=raster_info['bounds_wgs84'],
                unpaired_neighbor=provenance.get('unpaired_neighbor'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/GTM_viewer_bundles' / IDENTIFIER)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))
