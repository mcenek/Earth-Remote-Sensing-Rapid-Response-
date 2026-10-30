"""Save and publish real baseline predictions at fixed, predeclared checkpoints.

No fitting, scene exclusions, threshold search, dilation or smoothing. The local
checkpoint is evaluated in FP32. The released model is an external reference,
with its training-location overlap disclosed rather than a fair contest claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom
from shapely import make_valid, wkt
from shapely.geometry import mapping
import torch

import GTM_Model8_mars_baseline as training
from GTM_Model0_local_viewer import load_bundle
from GTM_Model0_publish_viewer import publish
from evaluate_released_marss2l import RELEASE_SPECS, ReleasedUNet, load_released_model

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'tool-diagnostics/mars-baseline-20261030'
PLAN_SHA = '3b29c62bb250dd9155b1c2af90637b605f2709f7ec6c64b18662a0c9ed2a6345'
THRESHOLD = .5


def confusion(probability, target, valid):
    p, y = probability[valid] >= THRESHOLD, target[valid].astype(bool)
    return dict(tp=int((p & y).sum()), fp=int((p & ~y).sum()),
                fn=int((~p & y).sum()), tn=int((~p & ~y).sum()))


def metrics(c):
    tp, fp, fn, tn = (c[k] for k in ('tp', 'fp', 'fn', 'tn'))
    return dict(**c, f1=2*tp/max(2*tp+fp+fn, 1), iou=tp/max(tp+fp+fn, 1),
                precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                evaluated_pixels=tp+fp+fn+tn)


def review_html(title, notice, score, negative, entries, checkpoint_note):
    """Use the same saved images as the viewer; retain every review scene."""
    figures = []
    for scene in entries:
        filename = lambda key: html.escape(scene[key], quote=True)
        n = scene['nativeMetrics']
        assessment = (f"F1 {n['f1']:.1%}; precision {n['precision']:.1%}; recall {n['recall']:.1%}") if scene['label'] == 'PLUME' else f"{n['fp']} false-positive pixels"
        figures.append('<section><h2>'+html.escape(scene['name'])+'</h2><p>'+assessment+'</p><div class="panels">'
            '<figure><img loading="lazy" src="'+filename('rgb')+'"><figcaption>Current image</figcaption></figure>'
            '<figure><div class="stack"><img loading="lazy" src="'+filename('rgb')+'"><img loading="lazy" src="'+filename('truthImage')+'"><img loading="lazy" src="'+filename('predictionImage')+'"></div><figcaption>Cyan reference and orange prediction</figcaption></figure>'
            '<figure><div class="stack"><img loading="lazy" src="'+filename('rgb')+'"><img loading="lazy" src="'+filename('disagreementImage')+'"></div><figcaption>Green overlap, pink false mark, purple miss</figcaption></figure>'
            '</div></section>')
    return '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(title)+'</title><style>body{font:17px system-ui;background:#101c25;color:#e7f0f4;max-width:1200px;margin:30px auto;padding:20px}p{line-height:1.6}h2{font-size:18px}section{border-top:1px solid #375361;margin-top:30px;padding-top:12px}.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}figure{margin:0}img{display:block;width:100%;aspect-ratio:1;image-rendering:pixelated}.stack{display:grid}.stack img{grid-area:1/1}figcaption{font-size:14px;color:#aecbd8;padding-top:8px}a{color:#62ded6}@media(max-width:650px){.panels{grid-template-columns:1fr}}</style><h1>'+html.escape(title)+'</h1><p>'+html.escape(notice)+'</p><p>Pixel F1 '+format(score['f1'],'.1%')+'; precision '+format(score['precision'],'.1%')+'; recall '+format(score['recall'],'.1%')+'. Fixed cutoff 0.50. '+str(negative['activated'])+' of '+str(negative['scenes'])+' no-plume scenes have marked pixels. No dilation or area expansion was applied.</p><p>'+html.escape(checkpoint_note)+' All '+str(len(entries))+' reserved scenes are below. Predictions are saved model outputs. Native scores use observable pixels before map reprojection.</p>'+''.join(figures)


def independent_input(raw, cloud, row):
    """Reconstruct source channels independently, including lower-median MBMP."""
    a = np.clip(np.nan_to_num(raw.astype(np.float32), nan=0, posinf=10000, neginf=0)/5000, 0, 2)
    ratios = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for offset in (0, 6):
            ratio = np.ones(raw.shape[1:], np.float32)
            good = a[offset+4] != 0
            values = a[offset+5, good]/a[offset+4, good]
            if values.size:
                middle = (values.size-1)//2
                ratio[good] = values/np.partition(values, middle)[middle]
            ratios.append(np.clip(ratio, 0, 10))
        mbmp = ratios[0]/ratios[1]
        mbmp[~np.isfinite(mbmp)] = 1
        mbmp[np.abs(mbmp) > 10] = 10
    winds = []
    for key in ('wind_u', 'wind_v'):
        value = float(row[key]) if row.get(key) else np.nan
        winds.append((np.clip(value, -20, 20) if np.isfinite(value) else 4)/8)
    cm = cloud.astype(np.float32).copy()
    cm[cm > 0] = 1
    cm[np.isnan(cm)] = 0
    cm[np.isinf(cm)] = 1
    return np.concatenate([mbmp[None], a, np.broadcast_to(np.asarray(winds, np.float32)[:, None, None], (2, *cm.shape)), cm[None]])


def rgb_pair(raw, valid):
    ranges = [np.percentile(raw[k][valid], (2, 98)) for k in (2, 1, 0)]
    result = []
    for offset in (0, 6):
        image = np.stack([np.clip((raw[offset+k]-lo)/max(hi-lo, 1), 0, 1) for k, (lo, hi) in zip((2, 1, 0), ranges)], -1)
        result.append((np.nan_to_num(image)*255).astype(np.uint8))
    return result


def map_files(folder, raw, probability, truth, valid, affine, crs):
    """Project both dates and all overlays to the same Web Mercator grid."""
    h, width = truth.shape
    box = transform_bounds(crs, 'EPSG:3857', *array_bounds(h, width, affine), densify_pts=21)
    destination = from_bounds(*box, width, h)

    def project(a, smooth=False):
        b = np.full((h, width), np.nan, np.float32)
        reproject(a.astype(np.float32), b, src_transform=affine, src_crs=crs,
                  dst_transform=destination, dst_crs='EPSG:3857', src_nodata=np.nan,
                  dst_nodata=np.nan, resampling=Resampling.bilinear if smooth else Resampling.nearest)
        return b

    p = project(np.where(valid, probability, np.nan))
    y = project(np.where(valid, truth, np.nan))
    rgb, earlier = rgb_pair(raw, valid)
    for name, image in (('satellite', rgb), ('background', earlier)):
        pixels = np.stack([project(image[..., k], True) for k in range(3)], -1)
        Image.fromarray(np.clip(np.nan_to_num(pixels), 0, 255).astype(np.uint8)).save(folder / f'{name}.png')
    scored = np.isfinite(p)
    predicted, positive = scored & (p >= THRESHOLD), scored & (y == 1)
    colors = {'prediction': (predicted, (255, 151, 56, 210)),
              'truth': (positive, (46, 224, 219, 200))}
    for name, (mask, color) in colors.items():
        rgba = np.zeros((h, width, 4), np.uint8)
        rgba[mask] = color
        Image.fromarray(rgba).save(folder / f'{name}.png')
    error = np.zeros((h, width, 4), np.uint8)
    error[predicted & positive] = (32, 179, 157, 220)
    error[predicted & ~positive & scored] = (244, 92, 129, 220)
    error[~predicted & positive] = (168, 140, 240, 220)
    Image.fromarray(error).save(folder / 'errors.png')
    values = lambda x: [float(v) if np.isfinite(v) else None for v in x.ravel()]
    training.atomic_json(folder / 'grid.json', dict(width=width, height=h, probability=values(p),
        truth=values(y), valid=scored.astype(np.uint8).ravel().tolist(), crs='EPSG:3857'))
    west, south, east, north = transform_bounds('EPSG:3857', 'EPSG:4326', *box)
    return [[south, west], [north, east]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--role', choices=('validation', 'holdout'), required=True)
    parser.add_argument('--completed', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if training.sha256(PLAN/'plan.json') != PLAN_SHA:
        raise ValueError('Frozen plan changed')
    plan = json.loads((PLAN/'plan.json').read_text())
    filename = 'select_ids.json' if args.role == 'validation' else 'final_candidate_ids.json'
    if training.sha256(PLAN/filename) != plan['id_lists'][filename]:
        raise ValueError('Frozen evaluation membership changed')
    wanted = set(json.loads((PLAN/filename).read_text()))
    checkpoint_sha = training.sha256(args.checkpoint)
    # The training fingerprint stores Torch's string subclass, not executable code.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['fingerprint']['manifest_sha256'] != training.sha256(PLAN/'train_manifest.json'):
        raise ValueError('Checkpoint belongs to another training split')
    if checkpoint['fingerprint']['source_file_sha256'] != training.pinned_source_hashes():
        raise ValueError('Checkpoint producer source changed')
    rows, fields = {}, None
    with (args.data_root/'validated_images_all.csv').open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        for row in reader:
            if row['id_loc_image'] in wanted:
                rows[row['id_loc_image']] = row
    if set(rows) != wanted:
        raise ValueError('An evaluation row is missing')
    if args.role == 'validation':
        selected = []
        for positive in (True, False):
            group = [key for key, row in rows.items() if (row['isplume'].lower() == 'true') == positive]
            selected += sorted(group, key=lambda key: __import__('hashlib').sha256(('visual20261030:'+key).encode()).hexdigest())[:12]
        rows = {key: rows[key] for key in selected}
    ordered = sorted(rows, key=lambda key: (rows[key]['country'] != 'United States of America', rows[key]['isplume'].lower() != 'true', key))
    args.output.mkdir(parents=True, exist_ok=False)
    process, read_images, read_plumes, Dataset, Loss, Model = training.source_modules()
    with tempfile.TemporaryDirectory(dir=args.output, prefix='metadata_') as temporary:
        metadata_path = Path(temporary)/'scenes.csv'
        with metadata_path.open('w', newline='', encoding='utf-8') as writer:
            csv_writer = csv.DictWriter(writer, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(rows[key] for key in ordered)
        frame = read_images(str(metadata_path), path_prepend_data=str(args.data_root.resolve()))
    if set(frame.id_loc_image) != set(rows):
        raise ValueError('Source reader filtered frozen evaluation rows')
    dataset = Dataset(mode='test', image_dataframe=frame, do_simulation=False,
        multipass=True, wind=True, cloud_mask=True, cat_mbmp=True, norm_wind=True,
        bands_l8=True, cache=False, rotate_data_augmentation=False)
    local = ReleasedUNet(16).eval()
    local.load_state_dict(checkpoint['model_state_dict'], strict=True)
    spec = RELEASE_SPECS['mars-s2l']
    release_root = ROOT/spec['directory']
    release_path = release_root/'best_epoch'
    if training.sha256(release_path) != spec['checkpoint_sha256']:
        raise ValueError('Released reference checkpoint changed')
    reference = load_released_model(release_path, torch.device('cpu'), 16)
    if training.sha256(release_root/'config_experiment.json') != spec['config_sha256']:
        raise ValueError('Released reference configuration changed')
    reference_config = json.loads((release_root/'config_experiment.json').read_text())
    references_seen = set(reference_config['all_locs_train'])
    methods = {'baseline': local, 'released_reference': reference}
    totals = {key: Counter() for key in methods}
    negative = {key: Counter() for key in methods}
    entries = {key: [] for key in methods}
    records = []
    for method in methods:
        (args.output/method).mkdir()
    frame_index = {str(row.id_loc_image): i for i, (_, row) in enumerate(dataset.image_dataframe.iterrows())}
    for number, identifier in enumerate(ordered):
        row = rows[identifier]
        item = dataset[frame_index[identifier]]
        x = item['y_context_ls0_0'].numpy()
        target = item['y_target'].numpy().astype(np.uint8)
        with rasterio.open(args.data_root/row['s2path']) as image, rasterio.open(args.data_root/row['cloudmaskpath']) as cloud_file:
            raw, cloud = image.read(), cloud_file.read(1)
            affine, crs = image.transform, image.crs
            if raw.shape != (12, 200, 200) or cloud.shape != target.shape or image.transform != cloud_file.transform:
                raise ValueError('Scene grid mismatch')
            valid = np.isfinite(raw).all(0) & (cloud == 0) & (raw != 0).all(0)
        if not valid.any():
            raise ValueError('Evaluation scene has no observable pixels')
        rebuilt = independent_input(raw, cloud, row)
        if not np.allclose(x, rebuilt, rtol=0, atol=2e-6):
            raise ValueError(f'Independent input reconstruction differs: {identifier}, max {np.max(np.abs(x-rebuilt))}')
        if row['isplume'].lower() == 'true':
            geometry = transform_geom('EPSG:4326', crs, mapping(make_valid(wkt.loads(row['plume']))))
            independent_target = rasterize([(geometry, 1)], out_shape=target.shape, transform=affine, all_touched=True, dtype='uint8')
        else:
            independent_target = np.zeros_like(target)
        if not np.array_equal(target, independent_target):
            raise ValueError('Source and independent label rasterization differ')
        record = {'id': identifier, 'source_row': row, 'input_max_abs_error': float(np.max(np.abs(x-rebuilt))),
                  'valid_pixels': int(valid.sum()), 'reference_training_location_overlap': row['location_name'] in references_seen, 'models': {}}
        with torch.no_grad():
            for method, model in methods.items():
                probability = torch.sigmoid(model(torch.from_numpy(x[None])))[0].numpy()
                if probability.shape != target.shape or not np.isfinite(probability).all():
                    raise ValueError('Invalid prediction')
                folder = args.output/method/identifier
                folder.mkdir()
                path = folder/'prediction.npz'
                np.savez_compressed(path, probability=probability, target=target, valid=valid, input=x)
                with np.load(path, allow_pickle=False) as saved:
                    if not np.array_equal(saved['probability'], probability):
                        raise ValueError('Saved prediction differs from inference')
                c = confusion(probability, target, valid)
                measured = metrics(c)
                totals[method].update(c)
                if row['isplume'].lower() != 'true':
                    negative[method].update(scenes=1, activated=int(c['fp'] > 0), fp=c['fp'], tn=c['tn'])
                record['models'][method] = dict(**measured, prediction_sha256=training.sha256(path))
                bounds = map_files(folder, raw, probability, target, valid, affine, crs)
                name = ('Plume' if row['isplume'].lower() == 'true' else 'No plume')+' | '+row['country']+' | '+row['tile_date'][:10]
                entries[method].append(dict(id=identifier, name=name, task='plume_segmentation',
                    rgb=identifier+'/satellite.png', reference90Rgb=identifier+'/background.png',
                    reference90Label='Provider reference image', truthImage=identifier+'/truth.png',
                    predictionImage=identifier+'/prediction.png', disagreementImage=identifier+'/errors.png',
                    grid=identifier+'/grid.json', bounds=bounds, predictionAvailable=True,
                    dataKind='measured', dataset='MARS-S2L', label='PLUME' if row['isplume'].lower() == 'true' else 'NO_PLUME',
                    split=args.role, latitude=float(row['lat']), longitude=float(row['lon']),
                    locationPoint=[float(row['lat']), float(row['lon'])], date=row['tile_date'][:10],
                    source=row['tile'], resolution='10 m output grid; Sentinel SWIR information is 20 m',
                    truthLabel='MARS reviewed plume boundary', truthNote='Binary plume label; no emission-rate estimate.',
                    notes=('Plume-labeled observation.' if row['isplume'].lower() == 'true' else 'Reviewed observation without a labeled plume.')+' Saved model output; native scores exclude cloud and missing input pixels.',
                    decisionThreshold=THRESHOLD, predictionHash=training.sha256(path),
                    checkpointHash=checkpoint_sha if method == 'baseline' else spec['checkpoint_sha256'],
                    nativeMetrics=dict(**measured, source_valid_pixels=int(valid.sum()), support_fraction=float(valid.mean())),
                    outcome=('FN' if c['tp'] == 0 else 'TP') if target.any() else ('FP' if c['fp'] else 'TN')))
        records.append(record)
        print(json.dumps({'scene': number+1, 'total': len(ordered), 'id': identifier}), flush=True)
    summary = dict(role=args.role, checkpoint_sha256=checkpoint_sha, epoch=checkpoint['epoch'],
        reviewer_sha256=training.sha256(Path(__file__)),
        completed=args.completed, threshold=THRESHOLD, scenes=len(records),
        pooled={key: metrics(value) for key, value in totals.items()}, negative={key: dict(value, pixel_fpr=value['fp']/max(value['fp']+value['tn'], 1)) for key, value in negative.items()},
        reference_training_location_overlap=sum(row['reference_training_location_overlap'] for row in records),
        plan_sha256=PLAN_SHA, trainer_fingerprint=checkpoint['fingerprint'],
        scope='Fresh-weights model location holdout. Public regions have previous project exposure; not pristine external validation.' if args.role == 'holdout' else 'Validation used for checkpoint selection. These are development results, not an independent test.')
    training.atomic_json(args.output/'summary.json', summary)
    training.atomic_json(args.output/'scene_records.json', {'scenes': records})
    publication = {}
    for method in methods:
        identifier = 'GTM_Model8_mars_20261030_'+args.role+'_'+method
        name = 'Model8 MARS baseline' if method == 'baseline' else 'Released MARS reference'
        notice = summary['scope']
        if method != 'baseline':
            notice += ' Released weights have seen some of these locations; this is not a controlled architecture comparison.'
        score = summary['pooled'][method]
        title = name+' | '+args.role+(' | interim' if method == 'baseline' and not args.completed else '')
        checkpoint_note = ('Local checkpoint after pass '+str(checkpoint['epoch']+1)+'.') if method == 'baseline' else 'Released pretrained checkpoint; no local fitting.'
        report = review_html(title, notice, score, summary['negative'][method], entries[method], checkpoint_note)
        (args.output/method/'index.html').write_text(report, encoding='utf-8')
        experiment = dict(id=identifier, name=title, lifecycle='experimental', priority=False,
            dataKind='measured', architectureFamily='early_fusion_unet', decisionThreshold=.5,
            task='plume_segmentation', status=('Completed exploratory baseline' if args.completed else 'Interim training checkpoint') if method == 'baseline' else 'Released pretrained reference',
            scope=notice, notes=notice, outcomeRule='TP requires at least one predicted pixel to overlap the labeled plume; use native pixel scores to assess its extent.',
            nativeAggregate=dict(**score, scenes=len(ordered)),
            metrics={'Pixel F1':format(score['f1'],'.1%'),'Precision':format(score['precision'],'.1%'),'Recall':format(score['recall'],'.1%')},
            run={'state':'completed' if args.completed or method != 'baseline' else 'running','message':'Snapshot from a frozen checkpoint. '+notice},
            scenes=entries[method], reportUrl='index.html')
        training.atomic_json(args.output/method/'experiment.json', experiment)
        load_bundle(args.output/method/'experiment.json', 'bundle-assets/')
        if args.publish:
            publication[method] = publish(args.output/method, ROOT/'outputs/GTM_viewer_bundles')
    training.atomic_json(args.output/'publication.json', publication)
    if training.sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError('Checkpoint changed during review; publication must not be used')
    print(json.dumps({'summary':summary, 'publication':publication}, indent=2))


if __name__ == '__main__':
    main()
