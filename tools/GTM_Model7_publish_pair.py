"""Verify a complete Model7 pair receipt and publish its saved, unlabeled prediction.

Example, from the repository root:
  .venv/Scripts/python.exe tools/GTM_Model7_publish_pair.py --prediction-dir outputs/my_pair_prediction --output outputs/my_pair_viewer --id GTM_Model7_my_pair --name "My unlabeled pair" --publish

The experiment ID is derived from the receipt hash unless --experiment-id is given.
Both the prepared bundle and published ID must be new. No inference or labels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from GTM_Model0_local_viewer import BUNDLES, ID_PATTERN, load_bundle
from GTM_Model0_publish_viewer import publish

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'outputs/GTM_Model7_hard_negative_v11_20261025/selection/freeze.json'
SELECTION = FREEZE.parent / 'receipt.json'
TRAINING_RECEIPT = ROOT / 'outputs/GTM_Model7_hard_negative_v11_20261025/fit/receipt.json'
TRAINING_REPLAY = ROOT / 'outputs/GTM_Model7_hard_negative_v11_20261025/fit/e2e_receipt.json'
TRAINER = ROOT / 'tools/GTM_Model7_train_repaired_hard_v11.py'
PREDICTOR = ROOT / 'tools/GTM_Model7_predict_pair.py'
# The audited first release accepted only legacy NoData=0 cloud files. Keep
# its existing, immutable receipts usable after accepting canonical class 4.
# Every source, model, score, support and output is still checked below.
AUDITED_LEGACY_PREDICTOR_SHA256 = '4e500d0979df7a313d0d4cfdc275f135bb78d8b0c6db98a0c16f1d529879ff28'
ROLES = ('current', 'reference', 'current_cloud', 'reference_cloud')
OUTPUTS = ('probability.tif', 'mask.tif', 'valid.tif', 'prediction.npz')
BANDS = ('B02', 'B03', 'B04', 'B08', 'B11', 'B12')
SUPPORT_RULE = 'all four native pixels in each 20 m block: both dates six DN bands >0 and both CloudSEN12 codes 0'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def utc(value: str) -> datetime:
    require(isinstance(value, str), 'Missing declared UTC time')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid declared UTC time') from error
    require(parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), 'Acquisition time must be UTC')
    return parsed


def grid(ds: rasterio.io.DatasetReader) -> dict:
    return {'crs': ds.crs.to_string() if ds.crs else None,
            'transform': list(ds.transform)[:6], 'width': ds.width, 'height': ds.height,
            'count': ds.count, 'dtype': ds.dtypes[0], 'nodata': ds.nodata,
            **({'descriptions_first_six': list(ds.descriptions[:6])} if ds.count >= 6 else {})}


def same_grid(actual: dict, recorded: dict) -> bool:
    return actual == recorded


def verify_receipt(receipt_path: Path) -> tuple[dict, dict]:
    """Return verified receipt and arrays; reject a changed or incomplete chain."""
    receipt_path = receipt_path.resolve(strict=True)
    require(receipt_path.name == 'receipt.json', 'Expected receipt.json from predict_pair')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt.get('schema_version') == 1 and receipt.get('status') == 'complete' and
            receipt.get('scope') == 'local pair inference; no label used' and
            receipt.get('cpu_repeat_exact') is True and
            receipt.get('script_sha256') in (sha(PREDICTOR), AUDITED_LEGACY_PREDICTOR_SHA256),
            'Incomplete or drifted prediction receipt')
    input_record = receipt['input']
    require(input_record.get('radiometry_declaration') == 'mars_cached_l1c_dn_uint16' and
            input_record.get('support_rule') == SUPPORT_RULE and
            input_record.get('model_pixel_size_m') == 20 and
            input_record.get('output_grid_pixel_size_m') == 10 and
            receipt.get('score_grid') == '20m block-constant model score repeated to 10m output grid; not extra spatial detail',
            'Unsupported input or score-grid contract')
    require(receipt.get('input_recipe') ==
            'native 10m first-six source DN; all-four two-date clear support; float32 2x2 mean; frozen per-date log1p median/normalizer; 24 early-fusion channels',
            'Input recipe changed')
    times = receipt['acquisition_times']
    require(times.get('reference_before_current') is True and
            times.get('provenance') == 'caller declaration; TIFF pixels alone do not prove acquisition time' and
            utc(times['reference_utc_declared']) < utc(times['current_utc_declared']),
            'Reference time must be declared earlier; time provenance changed')

    frozen = json.loads(FREEZE.read_text(encoding='utf-8'))
    selected = json.loads(SELECTION.read_text(encoding='utf-8'))
    choice = frozen['models']['siamese']
    model = receipt['model']
    checkpoint = Path(choice['checkpoint']).resolve(strict=True)
    require(selected.get('passed') is True and selected.get('freeze_sha256') == sha(FREEZE) and
            selected.get('candidate_rows') == 88 and
            frozen.get('selection_kind') == 'hard_negative_F1_exposed_DEV_v11' and
            frozen.get('improved_development_gate') is True and frozen.get('useful_target_met') is False and
            frozen['metadata']['config']['architecture'] == 'v7_early_fusion' and
            frozen['metadata']['normalization']['radiometry'] == 'log1p_dn_date_band_median_centered' and
            sha(Path(frozen['metadata']['manifest'])) == frozen['metadata']['manifest_sha256'] and
            sha(TRAINING_RECEIPT) == frozen['training_receipt_sha256'] and
            sha(TRAINING_REPLAY) == frozen['training_replay_sha256'] and
            json.loads(TRAINING_REPLAY.read_text(encoding='utf-8')).get('passed') is True and
            choice.get('qualified_development_gate') is True and choice.get('steps') == 2400 and
            choice.get('threshold') == frozen.get('threshold') == 0.65 and
            model.get('model') == 'retained_v11' and model.get('development_exposure') == 'exposed DEV only' and
            model.get('nationwide_accuracy_established') is False and
            model.get('freeze_sha256') == sha(FREEZE) and
            model.get('selection_receipt_sha256') == sha(SELECTION) and
            Path(model['checkpoint']).resolve(strict=True) == checkpoint and
            model.get('checkpoint_sha256') == choice['checkpoint_sha256'] == sha(checkpoint) and
            model.get('trainer_sha256') == frozen['metadata']['trainer_sha256'] == sha(TRAINER) and
            model.get('normalization') == frozen['metadata']['normalization'] and
            model.get('cutoff') == choice['threshold'], 'Frozen model identity or cutoff mismatch')

    source = {}
    profiles = []
    paths = []
    for role in ROLES:
        record = input_record['inputs'][role]
        path = Path(record['path']).resolve(strict=True)
        require(path.suffix.lower() in ('.tif', '.tiff') and path.stat().st_size == record['bytes'] and
                sha(path) == record['sha256'], f'{role} source hash or size mismatch')
        with rasterio.open(path) as ds:
            actual = grid(ds)
            require(ds.driver == 'GTiff' and ds.shape == (200, 200) and ds.crs is not None and
                    ds.crs.is_projected and ds.crs.linear_units_factor[1] == 1 and
                    list(ds.transform)[:6][:2] == [10.0, 0.0] and
                    ds.transform.d == 0 and ds.transform.e == -10 and
                    same_grid(actual, record['grid']), f'{role} grid differs from receipt')
            if role in ('current', 'reference'):
                tags = {k.upper(): v.upper() for k, v in ds.tags().items()}
                for band in range(1, 7):
                    tags.update({k.upper(): v.upper() for k, v in ds.tags(band).items()})
                require(ds.count in (6, 12) and set(ds.dtypes) == {'uint16'} and
                        tuple(ds.descriptions[:6]) == BANDS and ds.nodata == 0 and
                        all(x == 1 for x in ds.scales[:6]) and all(x == 0 for x in ds.offsets[:6]) and
                        not any('RADIO_ADD_OFFSET' in k or 'BOA_ADD_OFFSET' in k for k in tags) and
                        not any(('L2A' in v or 'LEVEL-2A' in v) for k, v in tags.items() if 'PROCESSING_LEVEL' in k),
                        f'{role} radiometry unsupported')
                source[role] = ds.read(indexes=(1, 2, 3, 4, 5, 6))
            else:
                allowed_nodata = (0,) if receipt['script_sha256'] == AUDITED_LEGACY_PREDICTOR_SHA256 else (0, 4)
                require(ds.count == 1 and ds.dtypes[0] == 'uint8' and ds.nodata in allowed_nodata,
                        f'{role} cloud contract unsupported')
                source[role] = ds.read(1)
                require(np.isin(source[role], (0, 1, 2, 3, 4)).all(), f'{role} cloud code unsupported')
        profiles.append(actual)
        paths.append(path)
    require(len(set(paths)) == 4 and all(p['crs'] == profiles[0]['crs'] and
            p['transform'] == profiles[0]['transform'] for p in profiles) and
            input_record['input_grid'] == profiles[0] and
            not np.array_equal(source['current'], source['reference']), 'Pair source grid or identity mismatch')

    native = ((source['current'] > 0).all(0) & (source['reference'] > 0).all(0) &
              (source['current_cloud'] == 0) & (source['reference_cloud'] == 0))
    valid20 = native.reshape(100, 2, 100, 2).all((1, 3))
    valid = np.repeat(np.repeat(valid20, 2, 0), 2, 1)
    require(valid20.any() and int(valid20.sum()) == input_record['valid20_pixels'] and
            int(valid.sum()) == input_record['valid_native_pixels'] == receipt['valid_native_pixels'],
            'Source-derived support differs from receipt')
    require(set(receipt['outputs']) == set(OUTPUTS), 'Incomplete prediction outputs')
    folder = receipt_path.parent
    for name in OUTPUTS:
        require(sha(folder / name) == receipt['outputs'][name], f'{name} hash mismatch')
    with np.load(folder / 'prediction.npz', allow_pickle=False) as z:
        require(set(z.files) == {'probability20', 'probability_native', 'mask_native', 'valid20', 'valid_native'},
                'Unexpected prediction NPZ members')
        p20, p, mask = z['probability20'], z['probability_native'], z['mask_native']
        require(p20.dtype == p.dtype == np.float32 and p20.shape == (100, 100) and p.shape == mask.shape == (200, 200) and
                np.isfinite(p20).all() and ((p20 >= 0) & (p20 <= 1)).all() and
                np.array_equal(p, np.repeat(np.repeat(p20, 2, 0), 2, 1)) and
                np.array_equal(z['valid20'], valid20.astype(np.uint8)) and
                np.array_equal(z['valid_native'], valid.astype(np.uint8)) and
                np.array_equal(mask, ((p >= model['cutoff']) & valid).astype(np.uint8)) and
                int(mask.sum()) == receipt['marked_native_pixels'], 'Saved arrays, repetition, support or frozen mask differ')
        p, mask = p.copy(), mask.copy()
    for name, expected in (('probability.tif', np.where(valid, p, np.nan).astype(np.float32)),
                           ('mask.tif', mask), ('valid.tif', valid.astype(np.uint8))):
        with rasterio.open(folder / name) as ds:
            require(ds.driver == 'GTiff' and ds.shape == (200, 200) and ds.count == 1 and
                    ds.crs.to_string() == profiles[0]['crs'] and list(ds.transform)[:6] == profiles[0]['transform'] and
                    ds.dtypes[0] == expected.dtype.name and
                    (math.isnan(ds.nodata) if name == 'probability.tif' else ds.nodata is None) and
                    np.array_equal(ds.read(1), expected, equal_nan=True), f'{name} TIFF parity or grid mismatch')
    return receipt, {'source': source, 'probability': p, 'mask': mask, 'valid': valid,
                     'crs': profiles[0]['crs'], 'transform': rasterio.Affine(*profiles[0]['transform'])}


def _warp(values: np.ndarray, verified: dict, transform, shape, *, nodata=0) -> np.ndarray:
    target = np.zeros(shape, dtype=values.dtype)
    reproject(values, target, src_transform=verified['transform'], src_crs=verified['crs'],
              src_nodata=nodata, dst_transform=transform, dst_crs='EPSG:3857',
              dst_nodata=nodata, resampling=Resampling.nearest)
    return target


def render(receipt_path: Path, receipt: dict, verified: dict, output: Path,
           identifier: str, display_name: str) -> dict:
    """Write a new prepared bundle. All image layers share one projected display grid."""
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds as bounds_in_crs
    src = verified['transform']
    west, south, east, north = array_bounds(200, 200, src)
    dst_transform, width, height = calculate_default_transform(
        verified['crs'], 'EPSG:3857', 200, 200, west, south, east, north)
    require(0 < width * height <= 1048576, 'Invalid display reprojection')
    dst_shape = (height, width)
    dst_bounds = array_bounds(height, width, dst_transform)
    minlon, minlat, maxlon, maxlat = bounds_in_crs('EPSG:3857', 'EPSG:4326', *dst_bounds)
    bounds = [[minlat, minlon], [maxlat, maxlon]]
    require(-85 <= minlat < maxlat <= 85 and -180 <= minlon < maxlon <= 180, 'Viewer bounds unsupported')
    output.mkdir(parents=True, exist_ok=False)
    names = {}
    stretches = {}
    for band in (2, 1, 0):
        values = np.concatenate([verified['source'][role][band][verified['valid']]
                                 for role in ('current', 'reference')])
        values = values[values > 0]
        require(values.size > 0, 'No jointly valid RGB samples')
        lo, hi = np.percentile(values, (2, 98))
        stretches[BANDS[band]] = {'low': float(lo), 'high': float(hi)}
    for role, stem in (('current', 'current'), ('reference', 'reference')):
        image = verified['source'][role]
        rgb = []
        for band in (2, 1, 0):
            channel = image[band].astype(np.float32)
            lo, hi = stretches[BANDS[band]]['low'], stretches[BANDS[band]]['high']
            stretched = np.rint(np.clip((channel - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
            rgb.append(_warp(stretched, verified, dst_transform, dst_shape, nodata=None))
        name = stem + '_rgb.png'
        Image.fromarray(np.stack(rgb, axis=-1), 'RGB').save(output / name)
        names[stem] = name
    valid = _warp(verified['valid'].astype(np.uint8), verified, dst_transform, dst_shape).astype(bool)
    probability = _warp(verified['probability'], verified, dst_transform, dst_shape, nodata=None)
    mask = _warp(verified['mask'], verified, dst_transform, dst_shape)
    require(np.array_equal(mask, ((probability >= receipt['model']['cutoff']) & valid).astype(np.uint8)),
            'Reprojected score and mask disagree')
    score_rgba = np.zeros((*dst_shape, 4), dtype=np.uint8)
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                      [94, 201, 98], [253, 231, 37]], dtype=np.float64)
    scaled = np.minimum(3.99999, np.maximum(0, probability.astype(np.float64)) * 4)
    index = np.floor(scaled).astype(np.intp)
    fraction = (scaled - index)[..., None]
    score_rgba[..., :3] = np.floor(stops[index] + (stops[index + 1] - stops[index]) * fraction + 0.5).astype(np.uint8)
    score_rgba[..., 3] = np.where(valid, 240, 0)
    Image.fromarray(score_rgba, 'RGBA').save(output / 'score.png')
    mask_rgba = np.zeros((*dst_shape, 4), dtype=np.uint8)
    mask_rgba[mask.astype(bool)] = (255, 135, 35, 230)
    Image.fromarray(mask_rgba, 'RGBA').save(output / 'mask.png')
    display_grid = {'width': width, 'height': height,
                    'probability': [float(v) if valid.flat[i] else None for i, v in enumerate(probability.flat)],
                    'valid': valid.astype(np.uint8).ravel().tolist(),
                    'displayCrs': 'EPSG:3857', 'displayTransform': list(dst_transform)[:6],
                    'sourceCrs': verified['crs'], 'sourceTransform': list(verified['transform'])[:6],
                    'sourceWidth': 200, 'sourceHeight': 200,
                    'resampling': 'nearest; 20 m model scores repeated to native 10 m before display reprojection'}
    (output / 'grid.json').write_text(json.dumps(display_grid, separators=(',', ':')), encoding='utf-8')
    receipt_hash = sha(receipt_path)
    current = receipt['acquisition_times']['current_utc_declared'][:10]
    earlier = receipt['acquisition_times']['reference_utc_declared'][:10]
    metadata = {'receiptPath': str(receipt_path.resolve()), 'receiptSha256': receipt_hash,
                'sourceSha256': {role: receipt['input']['inputs'][role]['sha256'] for role in ROLES},
                'savedPredictionSha256': receipt['outputs'], 'checkpointSha256': receipt['model']['checkpoint_sha256'],
                'cutoff': receipt['model']['cutoff'], 'nativeMarkedPixels': receipt['marked_native_pixels'],
                'nativeValidPixels': receipt['valid_native_pixels'], 'labelUsed': False,
                'displayGrid': {'crs': 'EPSG:3857', 'transform': list(dst_transform)[:6], 'width': width, 'height': height},
                'sourceGrid': receipt['input']['input_grid'],
                'rgbStretch': {'method': 'jointly supported positive pixels from both dates; per-band 2nd–98th percentiles',
                               'channels': stretches}}
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    note = ('Unvalidated local inference; no label used. The earlier image is a temporal reference, '
            'and methane absence at that time is not confirmed. Scores are uncalibrated. '
            'Model scores are 20 m values repeated on the 10 m source grid; display is nearest-neighbor reprojected. '
            'Current and reference RGB share one per-band percentile stretch. '
            'No truth mask or accuracy metrics are available.')
    scene = {'id': 'pair', 'name': f'Current {current} · earlier reference {earlier}',
             'task': 'plume_segmentation', 'dataset': 'Local Sentinel-2 pair',
             'location': f'{(minlat + maxlat) / 2:.5f}, {(minlon + maxlon) / 2:.5f}',
             'date': current, 'referenceDate': earlier, 'bounds': bounds,
             'rgb': names['current'], 'reference90Rgb': names['reference'],
             'reference90Label': f'Earlier reference · {earlier} (methane status unknown)',
             'rgbNote': 'Original current six-band DN source; B04/B03/B02 RGB with a shared per-band 2–98% stretch across both dates.',
             'predictionImage': 'mask.png', 'probabilityImage': 'score.png',
             'predictionAvailable': True, 'decisionThreshold': receipt['model']['cutoff'],
             'grid': 'grid.json', 'predictionHash': receipt['outputs']['prediction.npz'],
             'checkpointHash': receipt['model']['checkpoint_sha256'], 'source': str(receipt_path.resolve()),
             'resolution': '20 m model scores repeated to 10 m; georeferenced projected display',
             'notes': note, 'truthLabel': 'No ground-truth annotation',
             'truthNote': 'No label was used; temporal reference is imagery, not methane truth.'}
    manifest = {'id': identifier, 'name': display_name,
                'lifecycle': 'experimental', 'status': 'UNVALIDATED LOCAL INFERENCE · NO LABEL',
                'description': note, 'notes': note, 'decisionThreshold': receipt['model']['cutoff'],
                'scenes': [scene], 'reportUrl': 'provenance.json',
                'source': metadata}
    (output / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    load_bundle(output / 'experiment.json', 'bundle-assets/')
    return {'preparedBundle': str(output), 'experimentId': identifier,
            'receiptSha256': receipt_hash, 'displaySize': [width, height],
            'nativeMarkedPixels': receipt['marked_native_pixels']}


def run(receipt_path: Path, *, output: Path | None = None, experiment_id: str | None = None,
        display_name: str | None = None, bundles_dir: Path = BUNDLES,
        publish_bundle: bool = False, validate_only: bool = False) -> dict:
    receipt_path = receipt_path.resolve(strict=True)
    receipt, verified = verify_receipt(receipt_path)
    identifier = experiment_id or 'GTM_Model7_pair_' + sha(receipt_path)[:16]
    require(ID_PATTERN.fullmatch(identifier) is not None, 'Unsafe experiment ID')
    display_name = display_name or 'Model7 v11 · unlabeled pair · ' + receipt['acquisition_times']['current_utc_declared'][:10]
    require(isinstance(display_name, str) and 0 < len(display_name.strip()) <= 160, 'Invalid display name')
    result = {'experimentId': identifier, 'receiptSha256': sha(receipt_path),
              'nativeMarkedPixels': receipt['marked_native_pixels'],
              'nativeValidPixels': receipt['valid_native_pixels'], 'verified': True}
    if validate_only:
        return result
    destination = bundles_dir.resolve() / identifier
    if publish_bundle:
        require(not destination.exists(), f'Published experiment ID exists; refusing overwrite: {destination}')
    output = (output or receipt_path.parent / ('viewer_bundle_' + identifier)).resolve()
    require(not output.exists(), f'Prepared bundle exists; refusing overwrite: {output}')
    result.update(render(receipt_path, receipt, verified, output, identifier, display_name))
    if publish_bundle:
        result['publication'] = publish(output, bundles_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', type=Path, help='Complete output directory from GTM_Model7_predict_pair.py')
    parser.add_argument('--receipt', type=Path, help='Alternative path to its receipt.json')
    parser.add_argument('--output', type=Path, help='New prepared bundle directory; default beside receipt')
    parser.add_argument('--id', '--experiment-id', dest='experiment_id', help='New, unique viewer experiment ID')
    parser.add_argument('--name', help='Viewer display name')
    parser.add_argument('--bundles-dir', type=Path, default=BUNDLES)
    parser.add_argument('--publish', action='store_true', help='Atomically publish to the running viewer bundle folder')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    require((args.prediction_dir is None) != (args.receipt is None), 'Specify one of --prediction-dir or --receipt')
    receipt_path = args.receipt or args.prediction_dir / 'receipt.json'
    print(json.dumps(run(receipt_path, output=args.output, experiment_id=args.experiment_id,
                         display_name=args.name, bundles_dir=args.bundles_dir,
                         publish_bundle=args.publish, validate_only=args.validate_only), indent=2))


if __name__ == '__main__':
    main()
