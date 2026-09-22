"""Compare saved plume predictions on identical native scenes. No model execution.

Versioned JSON manifest, compact NPZ per scene, strict common reference/grid.
Default limit is four scenes; at most sixteen scenes per diagnostic invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > 32 * 1024**2:
            raise ValueError('Per-scene archive exceeds 32 MiB uncompressed')
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def binary(array, shape, name):
    a = np.asarray(array)
    if a.shape != shape or not np.isin(a, [0, 1]).all():
        raise ValueError(f'{name} must be an explicit binary mask on the native grid')
    return a.astype(bool)


def score(probability, truth, support, threshold, positive_only=False, pixel_area=None):
    p = probability >= threshold
    y = truth == 1
    tp = int(np.sum(p & y & support)); fn = int(np.sum(~p & y & support))
    fp = int(np.sum(p & ~y & support)); tn = int(np.sum(~p & ~y & support))
    div = lambda a, b: a / b if b else None
    result = dict(tp=tp, fn=fn, recall=div(tp, tp + fn), support_pixels=int(support.sum()))
    if positive_only:
        result.update(fp=None, tn=None, precision=None, iou=None, dice=None,
                      pixel_false_positive_rate=None, area_error_pixels=None,
                      absolute_area_error_pixels=None, relative_area_error=None,
                      area_error_m2=None)
    else:
        area_error = fp - fn
        result.update(fp=fp, tn=tn, precision=div(tp, tp + fp), iou=div(tp, tp + fp + fn),
                      dice=div(2 * tp, 2 * tp + fp + fn), pixel_false_positive_rate=div(fp, fp + tn),
                      area_error_pixels=area_error, absolute_area_error_pixels=abs(area_error),
                      relative_area_error=div(area_error, tp + fn),
                      area_error_m2=area_error * pixel_area if pixel_area is not None else None)
    return result


def load_case(case, base, thresholds, minimum_coverage):
    """Load only this case. No resizing, reprojection, label creation or threshold fitting."""
    evidence_path = (base / case['evidence']).resolve()
    evidence = read_archive(evidence_path)
    truth = np.asarray(evidence['truth'], dtype=np.float32)
    shape = truth.shape
    if len(shape) != 2 or max(shape) > 512 or min(shape) < 1:
        raise ValueError('Only 2D native grids up to 512x512 are allowed')
    if not np.isin(truth[np.isfinite(truth)], [0, 1]).all() or np.isinf(truth).any():
        raise ValueError('Reference must be 0/1/NaN, never continuous EMIT or a scene score')
    valid = binary(evidence['valid'], shape, 'input validity')
    known = valid & np.isfinite(truth)
    positive_only = case['reference_semantics'] == 'positive_only'
    if case['reference_semantics'] not in ['reviewed_mask', 'positive_only']:
        raise ValueError('Explicit reference semantics required')
    if positive_only and np.any(truth[known] != 1):
        raise ValueError('Positive-only exterior must be NaN, not negative labels')
    if not known.any():
        raise ValueError('No observed reference support')
    grid = case['grid']
    if grid['shape'] != list(shape) or not grid.get('crs'):
        raise ValueError('Reference grid metadata disagrees with array')
    affine = grid['affine']
    if len(affine) != 6 or not all(math.isfinite(x) for x in affine):
        raise ValueError('Six finite affine coefficients required')
    if affine[0] * affine[4] - affine[1] * affine[3] == 0:
        raise ValueError('Singular affine transform')
    area = grid.get('pixel_area_m2')
    if area is not None and (not math.isfinite(area) or area <= 0):
        raise ValueError('Pixel area must be positive square metres or omitted')
    rgb = np.asarray(evidence['rgb'])
    if rgb.shape != (*shape, 3) or not np.isfinite(rgb).all():
        raise ValueError('Finite native HxWx3 RGB required for visual review')
    predictions = {}; masks = {}; provenance = {}
    for name in ['legacy', 'current']:
        spec = case[name]
        meta_path = (base / spec['metadata']).resolve()
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        for key, expected in [('scene_id', case['id']), ('source_sha256', case['source_sha256']), ('grid', grid)]:
            if meta.get(key) != expected:
                raise ValueError(f'{name} {key} mismatch; cannot pair different scenes or grids')
        if meta.get('score_kind') != 'plume_probability' or not meta.get('model_id') or not meta.get('provenance'):
            raise ValueError('Prediction must identify its mask score and model provenance')
        path = (base / spec['path']).resolve()
        if meta.get('prediction_sha256') != digest(path):
            raise ValueError(f'{name} prediction hash mismatch')
        data = read_archive(path)
        p = np.asarray(data[spec.get('key', 'probability')], dtype=np.float32)
        mask = binary(data['valid'], shape, name + ' validity') & valid
        if p.shape != shape or not np.isfinite(p[mask]).all() or np.any((p[mask] < 0) | (p[mask] > 1)):
            raise ValueError(f'{name} needs finite [0,1] probabilities on its declared validity')
        predictions[name] = p; masks[name] = mask
        provenance[name] = dict(metadata=meta, metadata_sha256=digest(meta_path))
    common = known & masks['legacy'] & masks['current']
    coverage = float(common.sum() / known.sum())
    if not common.any() or coverage < minimum_coverage:
        raise ValueError(f'Common reference coverage {coverage:.3f} below required {minimum_coverage:.3f}')
    results = {name: score(p, truth, common, thresholds[name], positive_only, area)
               for name, p in predictions.items()}
    for name, p in predictions.items():
        unknown = masks[name] & ~np.isfinite(truth)
        results[name]['unknown_pixels_flagged'] = int(((p >= thresholds[name]) & unknown).sum())
        results[name]['individual_reference_coverage'] = float((known & masks[name]).sum() / known.sum())
    return dict(id=case['id'], reference_semantics=case['reference_semantics'], grid=grid,
                split=case.get('split', 'not provided'), evidence_sha256=digest(evidence_path),
                source_sha256=case['source_sha256'], common_reference_coverage=coverage,
                metrics=results, provenance=provenance), (rgb, truth, valid, predictions, masks)


def render_case(folder, index, arrays, thresholds):
    from PIL import Image
    rgb, truth, valid, predictions, masks = arrays
    display = np.zeros_like(rgb, dtype=np.uint8)
    for c in range(3):
        values = rgb[..., c][valid]
        lo, hi = np.percentile(values, [2, 98])
        display[..., c] = np.clip((rgb[..., c] - lo) * 255 / max(hi - lo, 1e-6), 0, 255).astype('uint8')
    yy, xx = np.indices(truth.shape)
    unknown = valid & ~np.isfinite(truth)
    base = display.copy(); base[~valid] = [35, 45, 53]
    panels = {'rgb': base.copy(), 'reference': base.copy()}
    panels['reference'][truth == 1] = (.45 * base[truth == 1] + .55 * np.array([0, 220, 205])).astype('uint8')
    panels['reference'][unknown & ((xx + yy) % 8 < 2)] = [170, 120, 205]
    for name in ['legacy', 'current']:
        im = base.copy(); mask = (predictions[name] >= thresholds[name]) & masks[name]
        im[mask] = (.45 * im[mask] + .55 * np.array([255, 140, 30])).astype('uint8')
        y = truth == 1
        inner = y.copy()
        inner[1:] &= y[:-1]; inner[:-1] &= y[1:]; inner[:, 1:] &= y[:, :-1]; inner[:, :-1] &= y[:, 1:]
        inner[[0, -1], :] = False; inner[:, [0, -1]] = False
        im[y & ~inner & valid] = [0, 240, 220]
        im[valid & ~masks[name]] = [130, 65, 135]
        panels[name] = im
    for name, im in panels.items():
        Image.fromarray(im).save(folder / f'{index:02d}_{name}.png')


def compare(manifest_path, output, limit=4, minimum_coverage=.95):
    if not 1 <= limit <= 16 or not 0 < minimum_coverage <= 1:
        raise ValueError('Limit must be 1..16 and minimum coverage in (0,1]')
    manifest_path = Path(manifest_path).resolve(); base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('selection_reason'):
        raise ValueError('Versioned manifest and explicit selection reason required')
    cases = manifest['cases']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Nonempty unique scene IDs required')
    if len(cases) > limit:
        raise ValueError('Manifest exceeds explicit scene limit; no silent cherry-picked truncation')
    thresholds = manifest['thresholds']
    if set(thresholds) != {'legacy', 'current'} or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds.values()):
        raise ValueError('Predeclared legacy/current thresholds in [0,1] required')
    output = Path(output)
    if output.exists():
        raise ValueError('Refusing to overwrite an existing comparison')
    # Preflight every compact case before creating output. Process serially.
    for case in cases:
        load_case(case, base, thresholds, minimum_coverage)
    output.mkdir(parents=True)
    rows = []; cards = []
    for index, case in enumerate(cases):
        row, arrays = load_case(case, base, thresholds, minimum_coverage)
        rows.append(row); render_case(output, index, arrays, thresholds)
        images = ''.join(f'<figure><img src="{index:02d}_{name}.png"><figcaption>{html.escape(label)}</figcaption></figure>'
                         for name, label in [('rgb', 'Sentinel RGB'), ('reference', 'Shared reference; purple = unknown'),
                                             ('legacy', row['provenance']['legacy']['metadata']['model_id'] + ' + reference outline'),
                                             ('current', row['provenance']['current']['metadata']['model_id'] + ' + reference outline')])
        metrics = [('iou', 'Plume IoU'), ('precision', 'Pixel precision'), ('recall', 'Plume recall'),
                   ('pixel_false_positive_rate', 'Background pixels flagged'), ('area_error_pixels', 'Signed area error (pixels)'),
                   ('absolute_area_error_pixels', 'Absolute area error (pixels)'), ('area_error_m2', 'Signed area error (m²)')]
        table = '<table><tr><th>On shared reference support</th><th>Left model</th><th>Right model</th></tr>'
        for key, label in metrics:
            values = [row['metrics'][name][key] for name in ['legacy', 'current']]
            cells = [('Not measured' if v is None else f'{v:.2%}' if key in ['iou', 'precision', 'recall', 'pixel_false_positive_rate'] else f'{v:,.1f}') for v in values]
            table += f'<tr><td>{label}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>'
        table += '</table>'
        cards.append(f'<section><h2>{html.escape(case["id"])}</h2><p>{html.escape(case["reference_semantics"])} · common support {row["common_reference_coverage"]:.1%}</p><div class="panels">{images}</div>{table}<details><summary>Exact metrics and provenance</summary><pre>{html.escape(json.dumps(row, indent=2))}</pre></details></section>')
    result = dict(schema_version=1, status='diagnostic_comparison_not_model_promotion',
                  selection_reason=manifest['selection_reason'], manifest_sha256=digest(manifest_path),
                  thresholds=thresholds, cases=rows,
                  notes=['No threshold tuning or prediction execution.', 'Metrics use identical reference and joint valid pixels.',
                         'Unknown exterior is visible but is never a negative label.',
                         'Empty-positive IoU/recall and undefined ratios are null, never perfect scores.',
                         'Legacy training membership may be unknown: this is not a paired held-out benchmark.',
                         'Four-panel previews are visual review aids; inspect actual outputs before promotion.'])
    write_json(output / 'comparison.json', result)
    page = '<!doctype html><meta charset="utf-8"><title>GTM paired plume comparison</title><style>body{font:15px system-ui;background:#eef3f6;color:#19303b;margin:24px}section{background:white;padding:20px;margin:24px 0}.panels{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}img{width:100%;image-rendering:pixelated}figcaption,h2{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}table{border-collapse:collapse;margin:20px 0;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}@media(max-width:800px){.panels{grid-template-columns:repeat(2,1fr)}}</style><h1>Paired plume comparison</h1><p>Diagnostic only. Orange: prediction. Cyan: reference. Common native-grid scoring; no automatic alignment or threshold tuning. This report does not establish which model generalizes better.</p><p>' + html.escape(manifest['selection_reason']) + '</p>' + ''.join(cards)
    (output / 'index.html').write_text(page, encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=4); parser.add_argument('--minimum-coverage', type=float, default=.95)
    args = parser.parse_args()
    try:
        result = compare(args.manifest, args.output, args.limit, args.minimum_coverage)
        print(json.dumps({'cases': len(result['cases']), 'report': str(args.output / 'index.html')}))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, str(error) + '\n')


if __name__ == '__main__':
    main()
