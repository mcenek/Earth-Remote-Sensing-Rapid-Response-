"""Prepare at most four already-evaluated Model6 scenes for paired cache comparison.

No inference. Preserve native grids, references and existing group provenance.
Without --baseline-run, explicit missing legacy output paths are left for the
separate CPU baseline command; missing predictions never become empty masks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from GTM_Model0_compare_plumes import digest, write_json


def prepare(run, output, scene_ids, baseline_run=None):
    import rasterio
    run = Path(run).resolve(); output = Path(output).resolve()
    if not 1 <= len(scene_ids) <= 4 or len(set(scene_ids)) != len(scene_ids):
        raise ValueError('Explicit one to four unique scene IDs required')
    if output.exists():
        raise ValueError('Refusing to overwrite prepared inputs')
    manifest = json.loads((run / 'GTM_Model6_plume_manifest.json').read_text(encoding='utf-8'))
    results = json.loads((run / 'GTM_Model6_plume_results.json').read_text(encoding='utf-8'))
    if not results.get('complete'):
        raise ValueError('Current run is incomplete')
    for sid in scene_ids:
        record = next((r for r in results['records'] if r['id'] == sid), None)
        if record is None or record['thresholds'].get('mean') != .5:
            raise ValueError('Adapter supports completed fixed-0.50 Model6 means only; preserve other protocols separately')
    rows = {r['id']: (i, r) for i, r in enumerate(manifest['observations'])}
    if any(i not in rows for i in scene_ids):
        raise ValueError('Scene not present in completed run')
    baseline = None
    if baseline_run:
        baseline_run = Path(baseline_run).resolve()
        bm = json.loads((baseline_run / 'GTM_Model6_plume_manifest.json').read_text(encoding='utf-8'))
        br = json.loads((baseline_run / 'GTM_Model6_plume_results.json').read_text(encoding='utf-8'))
        if not br.get('complete'):
            raise ValueError('Comparison run is incomplete')
        for sid in scene_ids:
            record = next((r for r in br['records'] if r['id'] == sid), None)
            if record is None or record['thresholds'].get('mean') != .5:
                raise ValueError('Comparison run does not have the declared fixed-0.50 mean cutoff')
        baseline = {r['id']: (i, r) for i, r in enumerate(bm['observations'])}
    output.mkdir(parents=True)
    cases = []
    with np.load(run / 'GTM_Model6_plume_inputs.npz', allow_pickle=False) as inputs:
        for order, sid in enumerate(scene_ids):
            i, row = rows[sid]; prefix = f'{order:02d}'
            source = Path(row['source'])
            # External EMIT source is a target TIFF, MARS source holds target+reference bands.
            if source.is_dir():
                source = source / 'target.l1c.tif'
            source_hash = digest(source)
            expected = row['hashes'].get('image', row['hashes'].get('target'))
            if expected is None or source_hash != expected:
                raise ValueError('Source image no longer matches the frozen run manifest')
            valid = inputs[f'{i}_valid'].astype(bool)
            with rasterio.open(source) as raster:
                if list(raster.shape) != row['shape'] or str(raster.crs) != row['crs'] or list(raster.transform)[:6] != row['affine']:
                    raise ValueError('Source image grid changed')
                bands = raster.read([1, 2, 3, 5, 6]).astype('float32')
            grid = dict(shape=row['shape'], crs=row['crs'], affine=row['affine'])
            # These two known inputs use metre-based UTM CRSs; do not assume other CRS units.
            if row['crs'].startswith(('EPSG:326', 'EPSG:327')):
                a, b, _, d, e, _ = row['affine']; grid['pixel_area_m2'] = abs(a * e - b * d)
            truth = np.where(inputs[f'{i}_known'].astype(bool) & valid, inputs[f'{i}_target'], np.nan).astype('float32')
            np.savez_compressed(output / f'{prefix}_evidence.npz', truth=truth, valid=valid, rgb=inputs[f'{i}_rgb'])
            input_path = output / f'{prefix}_input.npz'
            np.savez_compressed(input_path, bands=bands, valid=valid)
            write_json(output / f'{prefix}_scene.json', dict(id=sid, source_sha256=source_hash, grid=grid,
                       input_bands=['B2', 'B3', 'B4', 'B11', 'B12'], input_archive=input_path.name,
                       input_archive_sha256=digest(input_path)))
            specs = {}
            for role, chosen_run in [('current', run), ('legacy', baseline_run)]:
                specs[role] = dict(path=f'{prefix}_{role}.npz', metadata=f'{prefix}_{role}.json')
                if chosen_run is None:
                    continue
                j = i
                if role == 'legacy':
                    if sid not in baseline:
                        raise ValueError('Comparison run is missing selected scene')
                    j, other = baseline[sid]
                    for key in ['hashes', 'crs', 'affine', 'shape', 'group', 'partition', 'fully_labeled']:
                        if row[key] != other[key]:
                            raise ValueError(f'Runs disagree on {key}; no paired comparison')
                    with np.load(chosen_run / 'GTM_Model6_plume_inputs.npz', allow_pickle=False) as other_inputs:
                        for key in ['known', 'valid', 'target']:
                            if not np.array_equal(inputs[f'{i}_{key}'], other_inputs[f'{j}_{key}'], equal_nan=True):
                                raise ValueError(f'Runs disagree on actual {key} support')
                path = chosen_run / f'GTM_Model6_plume_prediction_{j}.npz'
                with np.load(path, allow_pickle=False) as cache:
                    probability = cache['mean']
                target = output / specs[role]['path']
                np.savez_compressed(target, probability=probability, valid=valid)
                write_json(output / specs[role]['metadata'], dict(scene_id=sid, source_sha256=source_hash,
                           grid=grid, model_id=chosen_run.name + '_mean', score_kind='plume_probability',
                           prediction_sha256=digest(target), provenance=dict(source_archive=str(path),
                           source_archive_sha256=digest(path), run_manifest_sha256=digest(chosen_run / 'GTM_Model6_plume_manifest.json'),
                           training_membership='existing held-group development prediction',
                           partition=row['partition'], group=row['group'])))
            cases.append(dict(id=sid, source_sha256=source_hash, grid=grid, evidence=f'{prefix}_evidence.npz',
                         reference_semantics='reviewed_mask' if row['fully_labeled'] else 'positive_only',
                         split=dict(group=row['group'], partition=row['partition'], stage='historical development'), **specs))
    write_json(output / 'manifest.json', dict(schema_version=1,
               selection_reason='Explicit preselected scene IDs; small diagnostic only, no model ranking. ' +
               ('The legacy slot holds the named saved Model6 comparator, NOT the previous team model.' if baseline_run else 'Legacy predictions are pending, never fabricated.'),
               thresholds=dict(legacy=.5, current=.5), cases=cases))
    return dict(manifest=str(output / 'manifest.json'), cases=len(cases), legacy_pending=baseline_run is None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene-id', action='append', required=True); p.add_argument('--baseline-run', type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(prepare(args.run, args.output, args.scene_id, args.baseline_run)))
    except (ValueError, KeyError, OSError) as e:
        p.exit(2, str(e) + '\n')


if __name__ == '__main__':
    main()
