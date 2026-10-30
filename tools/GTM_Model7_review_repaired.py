"""Independently verify and display both repaired models' saved predictions.

Acceptance was written first in docs/GTM_Model7_dataset_repair_acceptance.md.
No fitting, checkpoint selection, smoothing, or threshold tuning occurs here.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import re
from pathlib import Path

import h5py
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from matplotlib.figure import Figure

import GTM_Model7_review_temporal as v

MODELS = ('stable_residual', 'siamese')
NAMES = {'stable_residual': 'Normalized residual CNN', 'siamese': 'Temporal U-Net'}


def learning_chart(frozen, output):
    source = Path(frozen['run']) / 'receipt.json'
    if v.sha(source) != frozen['training_receipt_sha256']:
        raise ValueError('Training history changed')
    receipt = json.loads(source.read_text())
    figure = Figure(figsize=(12, 3.6), facecolor='#13242d')
    axes = figure.subplots(1, 2)
    for axis, metric, title in zip(axes, ('iou', 'false_positive_area_rate'),
                                   ('Plume overlap on positive scenes', 'False activation on no-plume scenes')):
        axis.set_facecolor('#13242d')
        for model, color in zip(MODELS, ('#70d5d0', '#f4b66d')):
            rows = receipt['models'][model]['milestones']
            steps = sorted(map(int, rows))
            for split, style in (('train', '--'), ('dev', '-')):
                section = 'positive' if metric == 'iou' else 'negative'
                values = [100 * (rows[str(step)][split][section][metric] or 0) for step in steps]
                axis.plot(steps, values, style, color=color, linewidth=2,
                          label=NAMES[model] + ' ' + split.upper())
        axis.set_title(title, color='#e4eef1', fontsize=11)
        axis.set_xlabel('Optimizer updates', color='#b7cbd3', fontsize=9)
        axis.set_ylabel('Percent', color='#b7cbd3', fontsize=9)
        axis.tick_params(colors='#b7cbd3', labelsize=8)
        axis.grid(alpha=.15, color='#b7cbd3')
        axis.set_ylim(bottom=0)
        for spine in axis.spines.values():
            spine.set_color('#415b67')
    axes[0].legend(fontsize=7, labelcolor='#e4eef1', facecolor='#13242d', edgecolor='#415b67')
    figure.tight_layout()
    path = output / 'learning.png'
    figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
    return v.image_uri(path)


def rgb_pair(d, factor):
    valid = d['valid'].astype(bool)
    ranges = [np.percentile(d['target_image'][..., band][valid], (2, 98)) for band in (2, 1, 0)]
    images = []
    for key in ('target_image', 'reference_image'):
        rgb = np.stack([np.clip((d[key][..., b] - lo) / max(float(hi - lo), 1), 0, 1)
                        for b, (lo, hi) in zip((2, 1, 0), ranges)], -1)
        rgb[~valid] = .22
        images.append(np.repeat(np.repeat((rgb * 255).astype('uint8'), factor, 0), factor, 1))
    return images


def balanced_rgb_pair(d, factor):
    """Two-date per-band display stretch with a floor, never used by inference."""
    valid = d['valid'].astype(bool)
    pooled = np.concatenate([d[key][..., (2, 1, 0)][valid]
                             for key in ('target_image', 'reference_image')])
    lo, hi = np.percentile(pooled, (2, 98), axis=0)
    center, width = (lo+hi)/2, np.maximum(hi-lo, 1000.)
    images = []
    for key in ('target_image', 'reference_image'):
        rgb = np.clip((d[key][..., (2, 1, 0)]-(center-width/2))/width, 0, 1)
        rgb[~valid] = .22
        images.append(np.repeat(np.repeat((rgb*255).astype('uint8'), factor, 0), factor, 1))
    return images


def source_rebuild(scene, d, h5=None, indices=None):
    if scene['source_level'] == 'L2A':
        index = indices[int(scene['source_row']['id'])]
        bands = [1, 2, 3, 7, 10, 11]
        a = np.moveaxis(h5['target'][index, bands], 0, -1).astype(np.float32)
        reference_key = scene.get('reference_dataset', 'reference90')
        if reference_key not in ('reference90', 'reference365'):
            raise ValueError('Unknown released reference dataset')
        b = np.moveaxis(h5[reference_key][index, bands], 0, -1).astype(np.float32)
        valid = (a > 0).all(-1) & (b > 0).all(-1) & np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
        label = h5['mask'][index]
        target = label.astype(np.float32)
        native_valid = valid
    else:
        paths = {r['role']: r['path'] for r in scene['sources']}
        with rasterio.open(paths['target_image']) as ta, rasterio.open(paths['reference_image']) as re, \
                rasterio.open(paths['target_cloud']) as tc, rasterio.open(paths['reference_cloud']) as rc:
            if ta.transform != re.transform or ta.crs != re.crs or tc.transform != ta.transform or rc.transform != ta.transform:
                raise ValueError('Repaired source grids differ')
            current = np.moveaxis(ta.read(range(1, 7)), 0, -1).astype(np.float32)
            reference = np.moveaxis(re.read(range(1, 7)), 0, -1).astype(np.float32)
            support = (tc.read(1) == 0) & (rc.read(1) == 0)
            support &= (current > 0).all(-1) & (reference > 0).all(-1)
            support &= np.isfinite(current).all(-1) & np.isfinite(reference).all(-1)
            for ds, values in ((ta, current), (re, reference)):
                if ds.nodata is not None:
                    support &= (values != ds.nodata).all(-1)
            valid = support.reshape(100, 2, 100, 2).all((1, 3))
            native_valid = np.repeat(np.repeat(valid, 2, 0), 2, 1)
            a = current.reshape(100, 2, 100, 2, 6).mean((1, 3))
            b = reference.reshape(100, 2, 100, 2, 6).mean((1, 3))
        if scene['scene_kind'] == 'PLUME':
            with rasterio.open(paths['plume_mask']) as ds:
                label = ds.read(1)
        else:
            label = np.zeros((200, 200), np.uint8)
        target = label.reshape(100, 2, 100, 2).max((1, 3)).astype(np.float32)
    expected = dict(target_image=a, reference_image=b, valid=valid,
                    valid_native=native_valid, label_native=label, target=target)
    if any(not np.array_equal(d[k], value) for k, value in expected.items()):
        raise ValueError('Independent source reconstruction differs: ' + scene['id'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    p.add_argument('--evaluation-manifest', type=Path)
    p.add_argument('--id-prefix')
    p.add_argument('--title-prefix')
    p.add_argument('--scope-file', type=Path)
    p.add_argument('--balanced-display', action='store_true')
    args = p.parse_args()
    frozen = json.loads(args.freeze.read_text())
    receipt = json.loads((args.inference / 'receipt.json').read_text())
    replay = json.loads((args.inference / 'replay.json').read_text())
    if not replay['passed'] or replay['receipt_sha256'] != v.sha(args.inference / 'receipt.json'):
        raise ValueError('Fresh-process replay required')
    if receipt['freeze_sha256'] != v.sha(args.freeze):
        raise ValueError('Checkpoint selection changed')
    meta = frozen['metadata']
    if args.id_prefix and not re.fullmatch(r'[A-Za-z0-9_]+', args.id_prefix):
        raise ValueError('Experiment ID prefix must contain only letters, numbers and underscores')
    manifest_path = args.evaluation_manifest or Path(meta['manifest'])
    if v.sha(manifest_path) != receipt['manifest_sha256']:
        raise ValueError('Manifest changed')
    external = args.evaluation_manifest is not None
    if external and (receipt.get('training_manifest_sha256') != meta['manifest_sha256'] or
                     replay.get('evaluation_manifest_sha256') != v.sha(manifest_path)):
        raise ValueError('External evaluation must retain separate training and evaluation identities')
    manifest = json.loads(manifest_path.read_text())
    fresh_sites = external and manifest.get('role') == 'metadata_reserved_fresh_site_final_evaluation'
    branch, split = meta['config']['branch'], receipt['split']
    cm = branch == 'methanes2cm'
    factor = 1 if cm else 2
    scenes = sorted((s for s in manifest['scenes'] if s['split'] == split),
                    key=lambda s: (s['scene_kind'] != 'PLUME', s.get('state', ''), s['id']))
    args.output.mkdir(parents=True, exist_ok=False)
    for m in MODELS:
        (args.output / m).mkdir()
        choice = frozen['models'][m]
        if v.sha(choice['checkpoint']) != choice['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
    source_hashes = {}
    for s in scenes:
        for source in s['sources']:
            if source['path'] not in source_hashes:
                source_hashes[source['path']] = v.sha(source['path'])
            if source_hashes[source['path']] != source['sha256']:
                raise ValueError('Original data source changed')
    h5 = h5py.File(manifest['source_h5']['path'], 'r') if cm else None
    indices = {int(sid): i for i, sid in enumerate(h5['sample_id'][:])} if cm else None
    totals = {m: collections.Counter() for m in MODELS}
    negs = {m: collections.Counter() for m in MODELS}
    groups = {m: collections.defaultdict(collections.Counter) for m in MODELS}
    controls = {name: collections.Counter() for name in ('all_positive', 'all_negative')}
    cases, records = [], []
    entries = {m: [] for m in MODELS}
    run_tag = Path(frozen['run']).name
    calibrated = frozen.get('selection_kind') in ('DEV_native_cutoff_calibration_v1', 'DEV_scene_recall_calibration_v6')
    if calibrated:
        run_tag += '_calibrated'
    if cm:
        variant = 'balanced pixel loss' if meta['config'].get('objective') else 'weighted pixel loss'
    else:
        input_names = {'raw': 'raw DN', 'log1p_dn': 'log DN',
                       'log1p_dn_date_band_median_centered': 'scene-centered log DN'}
        radiometry = meta['normalization']['radiometry']
        if radiometry not in input_names:
            raise ValueError('Unknown input preprocessing label: ' + radiometry)
        variant = input_names[radiometry]
    if calibrated:
        variant += ', DEV cutoff'
    ids = {m: f'GTM_Model7_repaired_20261024_{branch}_{run_tag}_{split}_{m}' for m in MODELS}
    if args.id_prefix:
        ids = {m: f'{args.id_prefix}_{m}' for m in MODELS}
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 14)
    sheet = None
    page_count = 0
    for index, scene in enumerate(scenes):
        path = manifest_path.parent / scene['array_file']
        if v.sha(path) != scene['array_sha256']:
            raise ValueError('Prepared array changed')
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        source_rebuild(scene, d, h5, indices)
        current, background = (balanced_rgb_pair if args.balanced_display else rgb_pair)(d, factor)
        label, support = d['label_native'], d['valid_native'].astype(bool)
        annotation = current.copy()
        hit = (label == 1) & support
        annotation[hit] = (annotation[hit] * .35 + np.array((46, 224, 219)) * .65).astype('uint8')
        title = ('Plume' if scene['scene_kind'] == 'PLUME' else 'No plume') + ' | ' + (scene.get('state') or scene.get('satellite_time', '')[:10]) + ' | ' + scene['id'].split('_', 1)[1][:12]
        if external:
            location = ' / '.join(filter(None, (scene.get('country'), scene.get('official_csv_row', {}).get('location_name'))))
            title = ('Plume' if scene['scene_kind'] == 'PLUME' else 'No plume') + ' | ' + location + ' | ' + scene['satellite_time'][:10] + ' | ' + scene['id'].removeprefix('MARS_')[:8]
        figures, measured, painted_images = {}, {}, []
        for control in controls:
            controls[control].update(v.counts(np.ones_like(label) if control == 'all_positive' else np.zeros_like(label), label, support))
        for m in MODELS:
            cutoff = frozen['models'][m].get('threshold', frozen['threshold'])
            pred = args.inference / m / (scene['id'] + '.npz')
            accepted = receipt['models'][m]['scenes'][scene['id']]
            if v.sha(pred) != accepted['probability_sha256']:
                raise ValueError('Prediction changed')
            with np.load(pred, allow_pickle=False) as z:
                probability = z['probability_native']
                if not np.array_equal(z['valid_native'], support) or not np.array_equal(probability, np.repeat(np.repeat(z['probability20'], factor, 0), factor, 1)):
                    raise ValueError('Model support or grid mapping changed')
            c = v.counts((probability >= cutoff).astype(np.float32), label, support)
            if c != accepted['native_confusion']:
                raise ValueError('Independent native counts differ')
            measured[m] = v.ratios(c)
            totals[m].update(c)
            groups[m][scene['group_id']].update(c)
            if scene['scene_kind'] == 'NO_PLUME':
                negs[m].update(scenes=1, activated=int(c['fp'] > 0), fp=c['fp'], tn=c['tn'])
            folder = args.output / m / f'scene_{index:04d}'
            folder.mkdir()
            overlay, errors, _ = v.overlay(current, (probability >= cutoff).astype(np.float32), label, support)
            score = v.overlay(current, probability, label, support)[2]
            painted_images.append(overlay)
            for name, data in dict(current=current, background=background, annotation=annotation,
                                   overlay=overlay, errors=errors, score=score).items():
                Image.fromarray(data).save(folder / (name + '.png'))
            figures[m] = {k: v.image_uri(folder / (k + '.png')) for k in ('overlay', 'errors', 'score')}
            info = dict(id=scene['id'], name=title, task='plume_segmentation',
                        rgb=folder.name + '/current.png', reference90Rgb=folder.name + '/background.png',
                        reference90Label=('Nominal T-365 reference' if scene.get('reference_dataset') == 'reference365' else 'Nominal T-90 reference') if cm else 'Earlier reviewed clear reference',
                        probabilityImage=folder.name + '/score.png', disagreementImage=folder.name + '/errors.png',
                        predictionAvailable=True, decisionThreshold=cutoff, dataKind='measured',
                        dataset='MethaneS2CM' if cm else 'MARS', label=scene['scene_kind'],
                        split=('External | withheld from this fit; historical exposure audited' if external else
                               'Geographic ' + split + (' | previously exposed regression' if not cm else ' | source TRAIN internal split')),
                        latitude=scene['latitude'], longitude=scene['longitude'],
                        locationPoint=[scene['latitude'], scene['longitude']],
                        checkpointHash=frozen['models'][m]['checkpoint_sha256'], predictionHash=v.sha(pred),
                        source=scene['target_scene_id'], outcome='Missed plume' if hit.any() and not c['tp'] else 'Partial overlap' if hit.any() else 'False positive' if c['fp'] else 'Correct negative',
                        truthLabel='Released source mask' if cm else 'Reviewed MARS annotation',
                        truthNote='Binary plume labels, not a quantitative methane map.',
                        rgbNote='Real Sentinel-2 ' + scene['source_level'] + ('; shared two-date per-band display stretch, minimum 1000 DN.' if args.balanced_display else '; display-only current-date percentile stretch.'),
                        notes='Both dates checked with supplied cloud masks; earlier no-plume reference <=60 days. These masks can miss haze and cloud. No-plume means no released plume annotation, not proof of zero methane.' if not cm else scene.get('reference_dataset', 'reference90') + ' nominal only. Exact dates, cloud masks and geographic footprint unavailable.',
                        qualityFlags=['reference_cloud_unverified', 'point_location_only'] if cm else ['both_dates_cloud_checked', 'historical_exposure_audited' if external else 'historical_test_exposure'],
                        nativeMetrics={**measured[m], 'source_valid_pixels': int(support.sum()), 'support_fraction': float(support.mean()), 'claim': 'Geographic ' + split})
            if external:
                info['notes'] += ' Unseen by these weights; historical project exposure is audited and is not ruled out.'
                info['nativeMetrics']['claim'] = 'Frozen separate-site diagnostic; historical project exposure disclosed'
            if fresh_sites:
                info['split'] = 'Fresh project sites | model frozen before image QA and final inference'
                info['notes'] = info['notes'].replace(
                    ' Unseen by these weights; historical project exposure is audited and is not ruled out.',
                    ' No known prior local project fit, inference or viewer exposure at these sites. Historical source inventory metadata is disclosed in the reservation audit. This cohort is now consumed for evaluation.')
                info['nativeMetrics']['claim'] = 'First project evaluation on metadata-reserved separate sites; now consumed'
            if cm:
                grid = dict(width=32, height=32, probability=probability.reshape(-1).tolist(),
                            truth=label.reshape(-1).tolist(), valid=support.astype('uint8').reshape(-1).tolist())
                v.save(folder / 'grid.json', grid)
                info.update(coordinateSystem='pixel', imageSize=[32, 32], georeference=None,
                            grid=folder.name + '/grid.json', resolution='32 x 32 native pixels; nominal 20 m; point location only')
            else:
                bounds = v.map_assets(folder, current, probability, label, support, scene['reference_grid'], threshold=cutoff)
                # Verify the static fallback and the interactive grid show the same saved mask.
                exported = json.loads((folder / 'grid.json').read_text(encoding='utf-8'))
                scores = np.asarray([np.nan if p is None else p for p in exported['probability']]).reshape(exported['height'], exported['width'])
                visible = np.asarray(Image.open(folder / 'prediction.png'))[..., 3] > 0
                if not np.array_equal(visible, np.isfinite(scores) & (scores >= cutoff)):
                    raise ValueError('Static map mask differs from the frozen decision threshold')
                info.update(bounds=bounds, rgb=folder.name + '/satellite.png',
                            predictionImage=folder.name + '/prediction.png', truthImage=folder.name + '/truth.png',
                            grid=folder.name + '/grid.json', date=scene['satellite_time'][:10],
                            referenceDate=scene['reference_time'][:10], resolution='20 m prediction; native 10 m label comparison')
            entries[m].append(info)
        common = args.output / MODELS[0] / f'scene_{index:04d}'
        cases.append(dict(id=scene['id'], title=title, kind=scene['scene_kind'], group=scene['group_id'],
                          current=v.image_uri(common / 'current.png'), background=v.image_uri(common / 'background.png'),
                          annotation=v.image_uri(common / 'annotation.png'), models=figures, metrics=measured))
        records.append(dict(id=scene['id'], group=scene['group_id'], state=scene.get('state'), models=measured))
        if index % 12 == 0:
            if sheet is not None:
                sheet.save(args.output / f'contact_{page_count - 1:03d}.png')
            sheet = Image.new('RGB', (1200, 12 * 226), '#101e25')
            page_count += 1
        y = (index % 12) * 226
        draw = ImageDraw.Draw(sheet)
        draw.text((10, y + 2), title, font=font, fill='#dce9ed')
        for j, image in enumerate([current, background, annotation] + painted_images):
            sheet.paste(Image.fromarray(image).convert('RGB').resize((192, 192), Image.Resampling.NEAREST), (10 + j * 240, y + 26))
    if h5 is not None:
        h5.close()
    if sheet is not None:
        sheet.save(args.output / f'contact_{page_count - 1:03d}.png')
    pooled = {m: v.ratios(totals[m]) for m in MODELS}
    negative = {m: {**negs[m], 'pixel_fpr': negs[m]['fp'] / max(negs[m]['fp'] + negs[m]['tn'], 1)} for m in MODELS}
    failed_evaluation = {m: split == 'test' and not (
        (pooled[m]['precision'] or 0) >= .5 and (pooled[m]['recall'] or 0) >= .3 and
        (pooled[m]['iou'] or 0) >= .2 and negative[m]['pixel_fpr'] <= .001) for m in MODELS}
    summary = dict(pooled=pooled, negatives=negative, controls={k: v.ratios(c) for k, c in controls.items()},
                   groups={m: {g: v.ratios(c) for g, c in groups[m].items()} for m in MODELS},
                   branch=branch, split=split, scenes=records, source_reconstructions=len(scenes),
                   manifest_sha256=v.sha(manifest_path), zero_new_downloads=True)
    v.save(args.output / 'comparison.json', summary)
    learning = learning_chart(frozen, args.output)
    scope = ('Real US Sentinel-2 L2A crops from the source TRAIN split. Geographic groups are separate between fitting, development and evaluation. Point locations are supplied, but exact dates, per-pixel cloud masks and geographic footprints are unavailable. The figures use native pixel coordinates. Prior project exposure is possible.' if cm else
             'Real US Sentinel-2 L1C scenes. Both date cloud masks are checked. Every reference is an earlier reviewed no-plume observation at the same site, within 60 days. Split groups are geographically separate, but the small evaluation locations were already used in earlier experiments. These are regression results, not a fresh national test.')
    title = ('Broader US crop data' if cm else 'Repaired MARS references') + ' | ' + variant + ' | ' + split
    if args.scope_file:
        scope = args.scope_file.read_text(encoding='utf-8').strip()
    if args.title_prefix:
        title = args.title_prefix + ' | ' + variant + ' | ' + ('external' if external else split)
    options = ''.join(f'<option value="{i}">{html.escape(c["title"])}</option>' for i, c in enumerate(cases))
    stats = ''.join(f'<tr><td>{NAMES[m]}</td><td>{v.percent(pooled[m]["iou"])}</td><td>{v.percent(pooled[m]["precision"])}</td><td>{v.percent(pooled[m]["recall"])}</td><td>{v.percent(negative[m]["pixel_fpr"])}</td><td>{negative[m]["activated"]}/{negative[m]["scenes"]}</td></tr>' for m in MODELS)
    body = HTML.replace('__TITLE__', html.escape(title)).replace('__SCOPE__', html.escape(scope)).replace('__OPTIONS__', options).replace('__STATS__', stats).replace('__CASES__', json.dumps(cases).replace('</', '<\\/')).replace('__IDS__', json.dumps(ids)).replace('__N__', str(len(scenes))).replace('__ALLPOS__', v.percent(summary['controls']['all_positive']['iou'])).replace('__LEARNING__', learning)
    if meta.get('schema') == 'GTM_Model7_repaired_recall_v6':
        body = body.replace('These curves test learning and transfer; they do not establish national accuracy.',
                            'The residual CNN is the retained v5 control and was not retrained. The U-Net uses the new recall objective. These curves do not establish national accuracy.')
    acceptance = '; '.join(NAMES[m] + ': ' + ('passed numeric DEV check, still experimental' if frozen['models'][m]['qualified_development_gate'] else 'FAILED development recall/false-alarm check, diagnostic only') for m in MODELS)
    if any(failed_evaluation.values()):
        acceptance = 'FAILED geographic evaluation. These models are not ready for statewide use. ' + acceptance
    thresholds = '; '.join(NAMES[m] + ' ' + f"{frozen['models'][m].get('threshold', frozen['threshold']):.3f}" for m in MODELS)
    body = body.replace('__ACCEPTANCE__', html.escape(acceptance)).replace('__THRESHOLDS__', html.escape(thresholds))
    for m in MODELS:
        folder = args.output / m
        (folder / 'index.html').write_text(body, encoding='utf-8')
        name = ('US crops' if cm else 'MARS clear') + ' | ' + NAMES[m] + ' | ' + variant + ' | ' + split
        if args.title_prefix:
            name = args.title_prefix + ' | ' + NAMES[m] + ' | ' + ('external' if external else split)
        if not frozen['models'][m]['qualified_development_gate']:
            name += ' | FAILED DEV GATE'
        if failed_evaluation[m]:
            name += ' | FAILED GEOGRAPHIC EVALUATION'
        experiment = dict(id=ids[m], name=name, lifecycle='experimental', priority=False,
                          dataKind='measured', architectureFamily='GTM_Model7_repaired', runGroup='GTM_Model7_dataset_repair_20261024',
                          status='DEVELOPMENT | not final validation' if split == 'dev' else ('FAILED GEOGRAPHIC EVALUATION | not for statewide use' if failed_evaluation[m] else 'GEOGRAPHIC EVALUATION | experimental'),
                          description=f'Pooled IoU {v.percent(pooled[m]["iou"])}; recall {v.percent(pooled[m]["recall"])}. ' + scope,
                          decisionThreshold=frozen['models'][m].get('threshold', frozen['threshold']), nativeAggregate={**pooled[m], 'scenes': len(scenes)},
                          metrics={'Pixel IoU': v.percent(pooled[m]['iou']), 'Recall': v.percent(pooled[m]['recall']), 'No-plume pixel FPR': v.percent(negative[m]['pixel_fpr'])},
                          scenes=entries[m], reportUrl='index.html')
        v.save(folder / 'experiment.json', experiment)
        v.load_bundle(folder / 'experiment.json', 'bundle-assets/')
    (args.output / 'index.html').write_text(body, encoding='utf-8')
    verification = dict(passed=True, source_reconstructions=len(scenes), counts_recomputed=True,
                        inference_replay_sha256=v.sha(args.inference / 'replay.json'), renderer_sha256=v.sha(__file__),
                        manifest_sha256=v.sha(manifest_path), external_evaluation=external,
                        training_manifest_sha256=meta['manifest_sha256'],
                        display='pooled two-date RGB per-band, minimum 1000 DN' if args.balanced_display else 'current-date RGB percentile',
                        scope_file_sha256=v.sha(args.scope_file) if args.scope_file else None)
    if args.publish:
        verification['publication'] = {m: v.publish(args.output / m, v.ROOT / 'outputs/GTM_viewer_bundles') for m in MODELS}
    v.save(args.output / 'verification.json', verification)
    print(json.dumps({'summary': {m: pooled[m] for m in MODELS}, 'verification': verification}, indent=2))


HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d191f;color:#e4eef1;font:15px/1.6 system-ui,sans-serif}main{max-width:1560px;margin:auto;padding:42px 28px}h1{font-size:42px;line-height:1.1;letter-spacing:-1.5px}h2{font-size:21px}p{color:#b7cbd3}.eyebrow{color:#70d5d0;font-size:11px;letter-spacing:2px}.scope{padding:16px 20px;border-left:3px solid #dfb779;background:#1b2b33;margin:24px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:end}label{display:grid;gap:5px;color:#aabfc8;font-size:12px}button,select{background:#1a2d37;border:1px solid #415b67;color:#e4eef1;padding:10px 14px;border-radius:6px;max-width:90vw}button:disabled{opacity:.35}.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0;background:#152833;border-radius:7px;overflow:hidden}figure img{display:block;width:100%;aspect-ratio:1;image-rendering:pixelated;cursor:zoom-in}figcaption{padding:10px;font-size:12px;color:#9eb9c6}figcaption b{display:block;color:#e4eef1}.links{display:flex;gap:20px;margin:12px 0}a{color:#70d5d0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:13px 10px;border-bottom:1px solid #30454f}th{color:#8da8b5;font-weight:500}.legend{font-size:12px}.dialog{display:none;position:fixed;inset:0;background:#000d;place-items:center;z-index:5}.dialog img{width:min(80vw,80vh);image-rendering:pixelated}.mini{font-size:12px}.footer{margin-top:32px;border-top:1px solid #30454f;padding-top:16px;color:#8da8b5;font-size:12px}@media(max-width:950px){.images{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:600px){main{padding:20px 12px}.images{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:32px}}
</style><main><div class="eyebrow">ERSRR / REAL DATA REPAIR / OCTOBER 24, 2026</div><h1>__TITLE__</h1><p>Two models. Actual saved predictions. Every evaluated case remains available, including misses and false alarms.</p><div class="scope">__SCOPE__</div><p class="mini">__ACCEPTANCE__</p><div class="controls"><label>Scene<select id="scene">__OPTIONS__</select></label><button id="previous">Previous</button><button id="next">Next</button><label>Display<select id="mode"><option value="overlay">Mask and annotation</option><option value="errors">Errors against annotation</option><option value="score">Model score, zero to one</option></select></label><span id="counter"></span></div><h2 id="title"></h2><div class="links"><a id="map-stable_residual">Open residual in viewer</a><a id="map-siamese">Open U-Net in viewer</a></div><div class="images"><figure><img id="current" alt="Real Sentinel target image"><figcaption><b>Current observation</b>Observed RGB, display stretch only</figcaption></figure><figure><img id="background" alt="Real Sentinel temporal reference"><figcaption><b>Reference observation</b>Same stretch as current image</figcaption></figure><figure><img id="annotation" alt="Released binary plume annotation"><figcaption><b>Released annotation</b>Cyan marks source plume labels</figcaption></figure><figure><img id="stable_residual" alt="Saved residual CNN prediction"><figcaption><b>Normalized residual CNN</b><span id="metric-stable_residual"></span></figcaption></figure><figure><img id="siamese" alt="Saved temporal U-Net prediction"><figcaption><b>Temporal U-Net</b><span id="metric-siamese"></span></figcaption></figure></div><p class="legend">Mask: orange prediction, cyan annotation boundary. Errors: teal correct overlap, pink extra prediction, purple missed plume. Score: purple zero to yellow one, fixed scale. Gray pixels are unscored. Frozen decision cutoffs: __THRESHOLDS__. Scores are not calibrated probabilities.</p><h2>Learning check</h2><p class="mini">Curves use the model grid at the original 0.50 cutoff. The aggregate table uses the native annotation grid at each frozen cutoff. Dashed lines show fitting data. Solid lines show separate development locations used for selection. These curves test learning and transfer; they do not establish national accuracy.</p><img src="__LEARNING__" alt="Actual training and development overlap and false-alarm histories" style="width:100%;height:auto;border-radius:7px"><h2>All __N__ evaluated observations</h2><div class="table"><table><thead><tr><th>Model</th><th>Plume IoU</th><th>Precision</th><th>Recall</th><th>No-plume pixel false alarms</th><th>No-plume scenes activated</th></tr></thead><tbody>__STATS__</tbody></table></div><p class="mini">All-positive control IoU: __ALLPOS__. All-negative control recall: 0%. A model must recover plume pixels and reject clear scenes. These scores do not measure emission rates, source identity, EMIT upscaling or national detection accuracy.</p><p class="mini">Evaluation groups stay separate from fitting. Development scores are used for checkpoint selection and are not independent validation. Historical project exposure and missing source metadata are disclosed above. The models compare these specific runs; training-data volume alone cannot establish reliability.</p><div class="footer">No new imagery downloads. Source reconstructions, split receipts, checkpoints, saved probabilities, independent native counts and fresh-process replay are saved in the local repository.</div></main><div class="dialog" id="large"><img alt="Expanded actual output"></div><script>
const cases=__CASES__,ids=__IDS__,q=id=>document.getElementById(id),fmt=x=>x==null?'n/a':(100*x).toFixed(2)+'%';function update(){const i=+q('scene').value,c=cases[i];q('title').textContent=c.title;q('counter').textContent=(i+1)+' / '+cases.length;for(const k of ['current','background','annotation'])q(k).src=c[k];for(const k of ['stable_residual','siamese']){q(k).src=c.models[k][q('mode').value];q('metric-'+k).textContent='IoU '+fmt(c.metrics[k].iou)+' | extra pixels '+c.metrics[k].fp.toLocaleString();q('map-'+k).href='/?catalog=all&experiment='+ids[k]+'&scene='+encodeURIComponent(c.id)+'#compare';}q('previous').disabled=i===0;q('next').disabled=i===cases.length-1;}q('scene').onchange=update;q('mode').onchange=update;q('previous').onclick=()=>{q('scene').value=+q('scene').value-1;update()};q('next').onclick=()=>{q('scene').value=+q('scene').value+1;update()};for(const im of document.querySelectorAll('figure img'))im.onclick=()=>{q('large').querySelector('img').src=im.src;q('large').style.display='grid'};q('large').onclick=()=>q('large').style.display='none';document.onkeydown=e=>{if(e.key==='Escape')q('large').style.display='none'};update();</script></html>'''


if __name__ == '__main__':
    main()
