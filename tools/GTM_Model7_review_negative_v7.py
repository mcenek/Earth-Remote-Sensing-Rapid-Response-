"""Publish real reserved-clear predictions with no plume-recall claims.

Failure modes precede code in the negative-only test protocol.
"""
import argparse
import collections
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
import GTM_Model7_review_repaired as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads(a.gate.read_text(encoding='utf-8'))
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    receipt = json.loads((a.inference/'receipt.json').read_text(encoding='utf-8'))
    replay = json.loads((a.inference/'replay.json').read_text(encoding='utf-8'))
    if (not replay['passed'] or replay['receipt_sha256'] != r.v.sha(a.inference/'receipt.json') or
            receipt['gate_sha256'] != r.v.sha(a.gate) or receipt['manifest_sha256'] != r.v.sha(a.manifest)):
        raise ValueError('Verified frozen final predictions required')
    scenes = manifest['scenes']
    if any(s['scene_kind'] != 'NO_PLUME' or s['split'] != 'test' for s in scenes):
        raise ValueError('Only reserved no-plume test scenes')
    a.output.mkdir(parents=True, exist_ok=False)
    ids = {k: 'GTM_Model7_broader_v7_20261024_fresh_clear_'+k for k in ('baseline', 'selected')}
    names = dict(baseline='Original v5 U-Net', selected='Selected fusion U-Net')
    entries, totals, sites, cases, records = {k: [] for k in ids}, {}, {}, [], []
    for key in ids:
        (a.output/key).mkdir()
        totals[key] = collections.Counter()
        sites[key] = set()
    for index, s in enumerate(scenes):
        for source in s['sources']:
            if r.v.sha(source['path']) != source['sha256']:
                raise ValueError('Original source changed')
        with np.load(a.manifest.parent/s['array_file'], allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        r.source_rebuild(s, d)
        current, reference = r.balanced_rgb_pair(d, 2)
        truth, valid = d['label_native'], d['valid_native'].astype(bool)
        if truth.any():
            raise ValueError('Unexpected plume')
        title = f"No plume | {s.get('country', '')} | {s['satellite_time'][:10]} | {s['id'][5:13]}"
        figures, measured = {}, {}
        for key in ids:
            saved = receipt['models'][key]
            cutoff = saved['cutoff']
            pred = a.inference/key/(s['id']+'.npz')
            if r.v.sha(pred) != saved['scenes'][s['id']]['probability_sha256']:
                raise ValueError('Prediction changed')
            with np.load(pred, allow_pickle=False) as z:
                probability = z['probability_native']
                if not np.array_equal(z['valid_native'], valid):
                    raise ValueError('Support changed')
            c = r.v.counts((probability >= cutoff).astype(np.float32), truth, valid)
            if c != saved['scenes'][s['id']]['native_confusion']:
                raise ValueError('Independent native counts differ')
            totals[key].update(c)
            if c['fp']:
                sites[key].add(s['physical_location_id'])
            measured[key] = r.v.ratios(c)
            folder = a.output/key/f'scene_{index:04d}'
            folder.mkdir()
            overlay, errors, _ = r.v.overlay(current, (probability >= cutoff).astype(np.float32), truth, valid)
            score = r.v.overlay(current, probability, truth, valid)[2]
            for name, pixels in dict(current=current, background=reference, annotation=current,
                                     overlay=overlay, errors=errors, score=score).items():
                Image.fromarray(pixels).save(folder/(name+'.png'))
            figures[key] = {k: r.v.image_uri(folder/(k+'.png')) for k in ('overlay', 'errors', 'score')}
            bounds = r.v.map_assets(folder, current, probability, truth, valid, s['reference_grid'], threshold=cutoff)
            entries[key].append(dict(id=s['id'], name=title, task='plume_segmentation',
                rgb=folder.name+'/satellite.png', reference90Rgb=folder.name+'/background.png',
                reference90Label='Earlier explicit no-plume reference',
                predictionImage=folder.name+'/prediction.png', truthImage=folder.name+'/truth.png',
                probabilityImage=folder.name+'/score.png', disagreementImage=folder.name+'/errors.png',
                grid=folder.name+'/grid.json', bounds=bounds, predictionAvailable=True,
                decisionThreshold=cutoff, dataKind='measured', dataset='MARS', label='NO_PLUME',
                split='Reserved new no-plume sites | no plume recall claim', latitude=s['latitude'],
                longitude=s['longitude'], locationPoint=[s['latitude'], s['longitude']],
                source=s['target_scene_id'], date=s['satellite_time'][:10], referenceDate=s['reference_time'][:10],
                resolution='20 m prediction; native 10 m label comparison',
                truthLabel='Explicit MARS no-plume annotation', truthNote='No annotation is not proof of zero methane.',
                outcome='False positive' if c['fp'] else 'Correct negative',
                checkpointHash=saved['checkpoint_sha256'], predictionHash=r.v.sha(pred),
                nativeMetrics={**measured[key], 'source_valid_pixels': int(valid.sum()),
                               'support_fraction': float(valid.mean()), 'claim':'New no-plume sites only'},
                notes='Metadata reserved before final source decoding. Both-date source clouds checked. No plume recall is measured.'))
        common = a.output/'baseline'/f'scene_{index:04d}'
        cases.append(dict(id=s['id'], title=title, current=r.v.image_uri(common/'current.png'),
                          background=r.v.image_uri(common/'background.png'), annotation=r.v.image_uri(common/'annotation.png'),
                          models=figures, metrics=measured))
        records.append(dict(id=s['id'], models=measured))
    summary = {key: dict(**r.v.ratios(totals[key]), activated_scenes=sum(s['nativeMetrics']['fp'] > 0 for s in entries[key]),
                scenes=len(scenes), sites=len({s['physical_location_id'] for s in scenes}),
                activated_sites=len(sites[key]), clear_pixel_fpr=totals[key]['fp']/max(totals[key]['fp']+totals[key]['tn'], 1)) for key in ids}
    figure = Figure(figsize=(10, 2.5), facecolor='#13242d')
    axis = figure.subplots(); axis.set_facecolor('#13242d')
    axis.bar([names[k] for k in ids], [100*summary[k]['clear_pixel_fpr'] for k in ids], color=['#70d5d0', '#f4b66d'])
    axis.set_ylabel('False-positive pixels (%)', color='white'); axis.tick_params(colors='white')
    figure.tight_layout(); figure.savefig(a.output/'false_alarms.png', facecolor=figure.get_facecolor())
    body = r.HTML.replace('stable_residual', 'baseline').replace('siamese', 'selected')
    body = body.replace('Normalized residual CNN', names['baseline']).replace('Temporal U-Net', names['selected'])
    body = body.replace('Open residual in viewer', 'Open original U-Net in viewer')
    body = body.replace('Open U-Net in viewer', 'Open selected U-Net in viewer')
    body = body.replace('Saved residual CNN prediction', 'Saved original U-Net prediction')
    body = body.replace('Actual training and development overlap and false-alarm histories', 'False-positive pixel rates on reserved no-plume scenes')
    body = body.replace('Evaluation groups stay separate from fitting. Development scores are used for checkpoint selection and are not independent validation. Historical project exposure and missing source metadata are disclosed above.',
                        'These reserved locations were excluded from known prior local fitting and evaluation. Models and cutoffs were frozen before final source decoding. All accepted cases are retained; three metadata-reserved observations failed fixed image QA and were not replaced.')
    body = body.replace('<h2>Learning check</h2>', '<h2>False alarms on reserved sites</h2>')
    begin = body.index('<p class="mini">Curves use')
    end = body.index('</p>', begin)+4
    body = body[:begin]+'<p class="mini">Only negative annotations are present. This check measures false alarms, not plume recall or national accuracy.</p>'+body[end:]
    body = body.replace('All-positive control IoU: __ALLPOS__. All-negative control recall: 0%.', 'Plume recall and plume IoU are not measurable in this all-negative set.')
    options = ''.join(f'<option value="{i}">{html.escape(c["title"])}</option>' for i,c in enumerate(cases))
    rows = ''.join(f'<tr><td>{names[k]}</td><td>n/a</td><td>n/a</td><td>n/a</td><td>{r.v.percent(summary[k]["clear_pixel_fpr"])}</td><td>{summary[k]["activated_scenes"]}/{len(scenes)}; {summary[k]["activated_sites"]}/{summary[k]["sites"]} sites</td></tr>' for k in ids)
    scope = 'Real cached Sentinel-2 L1C, MARS-S2L annotations. New sites excluded from known prior local fitting, evaluation, viewers and broader DEV before pixel QA. No-plume sites only. No further selection or fitting used these outcomes. Binary masks are not quantitative EMIT upscaling.'
    values = dict(TITLE='Reserved new no-plume sites | frozen U-Net comparison', SCOPE=scope,
        OPTIONS=options, STATS=rows, CASES=json.dumps(cases).replace('</','<\\/'), IDS=json.dumps(ids),
        N=str(len(scenes)), LEARNING=r.v.image_uri(a.output/'false_alarms.png'),
        ACCEPTANCE='A false-alarm check only. Performance on unseen plumes remains untested for this revision.',
        THRESHOLDS='; '.join(names[k]+' '+str(receipt['models'][k]['cutoff']) for k in ids))
    for key,value in values.items():
        body = body.replace('__'+key+'__', value)
    publication = {}
    for key in ids:
        folder = a.output/key
        (folder/'index.html').write_text(body, encoding='utf-8')
        stats = summary[key]
        r.v.save(folder/'experiment.json', dict(id=ids[key], name='Fresh clear sites | '+names[key],
            lifecycle='experimental', priority=False, dataKind='measured', architectureFamily='GTM_Model7_broader_development',
            runGroup='GTM_Model7_broader_v7_fresh_clear', status='NEW NO-PLUME SITES ONLY | plume recall unknown',
            description=scope, decisionThreshold=receipt['models'][key]['cutoff'], nativeAggregate=stats,
            metrics={'No-plume pixel FPR':r.v.percent(stats['clear_pixel_fpr']), 'No-plume scenes activated':f"{stats['activated_scenes']}/{stats['scenes']}"},
            scenes=entries[key], reportUrl='index.html'))
        r.v.load_bundle(folder/'experiment.json', 'bundle-assets/')
        publication[key] = r.v.publish(folder, r.v.ROOT/'outputs/GTM_viewer_bundles')
    (a.output/'index.html').write_text(body, encoding='utf-8')
    r.v.save(a.output/'comparison.json', dict(pooled={k:r.v.ratios(totals[k]) for k in ids},
        summary=summary, scenes=records))
    r.v.save(a.output/'verification.json', dict(passed=True, counts_recomputed=True,
        source_reconstructions=len(scenes), evaluation_role='reserved_no_plume_only',
        renderer_sha256=r.v.sha(Path(__file__)), template_sha256=r.v.sha(r.__file__),
        manifest_sha256=r.v.sha(a.manifest), publication=publication))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
