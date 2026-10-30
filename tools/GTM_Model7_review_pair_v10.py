"""Publish two verified saved prediction methods on the exposed broader DEV.

No inference or fitting occurs here. Failure and visual controls precede code
in the v10 protocol. Every source is reconstructed. The v8 renderer is retained.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    recipe = json.loads(a.recipe.read_text(encoding='utf-8'))
    for identity in recipe['sources']:
        path = Path(identity['path'])
        if r.v.sha(path) != identity['sha256']:
            raise ValueError('Verified source changed: '+str(path))
        if identity.get('require_passed') and not json.loads(path.read_text()).get('passed'):
            raise ValueError('Required verification did not pass')
    manifest_path = Path(recipe['manifest'])
    if r.v.sha(manifest_path) != recipe['manifest_sha256']:
        raise ValueError('Manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if any(s['split'] not in ('train', 'dev') for s in manifest['scenes']):
        raise ValueError('Only the broader TRAIN/DEV manifest is allowed')
    scenes = sorted((s for s in manifest['scenes'] if s['split'] == 'dev'),
                    key=lambda s: (s['scene_kind'] != 'PLUME', s.get('country', ''), s['id']))
    if len(scenes) != 198 or sum(s['scene_kind'] == 'PLUME' for s in scenes) != 46:
        raise ValueError('Broader DEV membership changed')
    methods = recipe['methods']
    if set(methods) != {'baseline', 'candidate'}:
        raise ValueError('Exactly two named methods required')
    ids = {k: recipe['id_prefix']+'_'+k for k in methods}
    names = {k: m['name'] for k,m in methods.items()}
    a.output.mkdir(parents=True, exist_ok=False)
    entries = {k: [] for k in methods}
    totals = {k: collections.Counter() for k in methods}
    negatives = {k: collections.Counter() for k in methods}
    countries = {}
    cases, records, source_hashes = [], [], {}
    for k in methods:
        (a.output/k).mkdir()
    for index, s in enumerate(scenes):
        country=s['source_row']['country']
        if country not in countries:
            countries[country]={'plume':0,'clear':0,'models':{k:collections.Counter() for k in methods}}
        countries[country]['plume' if s['scene_kind']=='PLUME' else 'clear']+=1
        for source in s['sources']:
            source_path = source['path']
            if source_path not in source_hashes:
                source_hashes[source_path] = r.v.sha(source_path)
            if source_hashes[source_path] != source['sha256']:
                raise ValueError('Original source changed')
        prepared = manifest_path.parent/s['array_file']
        if r.v.sha(prepared) != s['array_sha256']:
            raise ValueError('Prepared array changed')
        with np.load(prepared, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        r.source_rebuild(s, data)
        current, reference = r.balanced_rgb_pair(data, 2)
        truth, valid = data['label_native'], data['valid_native'].astype(bool)
        annotation = current.copy()
        hit = (truth == 1) & valid
        annotation[hit] = (.35*annotation[hit]+.65*np.array([46,224,219])).astype('uint8')
        title = f"{'Plume' if s['scene_kind'] == 'PLUME' else 'No plume'} | {s.get('country', '')} | {s['satellite_time'][:10]} | {s['id'][5:13]}"
        figures, measured = {}, {}
        for k, method in methods.items():
            cutoff = method['cutoff']
            pred = Path(method['folder'])/(s['id']+'.npz')
            accepted = method['records'][s['id']]
            if r.v.sha(pred) != accepted['probability_sha256']:
                raise ValueError('Verified prediction changed')
            with np.load(pred, allow_pickle=False) as z:
                probability = z['probability_native']
                if (not np.array_equal(z['valid_native'], valid) or
                    not np.array_equal(probability, np.repeat(np.repeat(z['probability20'],2,0),2,1)) or
                    not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1):
                    raise ValueError('Prediction grid/support/range mismatch')
            mask = (probability >= cutoff).astype(np.float32)
            c = r.v.counts(mask, truth, valid)
            if c != accepted['native_confusion']:
                raise ValueError('Independent native counts differ')
            measured[k] = r.v.ratios(c)
            totals[k].update(c)
            countries[country]['models'][k].update(c)
            if s['scene_kind']=='NO_PLUME':
                countries[country]['models'][k].update(clear_fp=c['fp'],clear_activated=int(c['fp']>0))
            if s['scene_kind'] == 'NO_PLUME':
                negatives[k].update(scenes=1,activated=int(c['fp']>0),fp=c['fp'],tn=c['tn'])
            folder = a.output/k/f'scene_{index:04d}'
            folder.mkdir()
            overlay, errors, _ = r.v.overlay(current, mask, truth, valid)
            score = r.v.overlay(current, probability, truth, valid)[2]
            for name, pixels in dict(current=current,background=reference,annotation=annotation,
                                     overlay=overlay,errors=errors,score=score).items():
                Image.fromarray(pixels).save(folder/(name+'.png'))
            figures[k] = {name:r.v.image_uri(folder/(name+'.png')) for name in ('overlay','errors','score')}
            bounds = r.v.map_assets(folder,current,probability,truth,valid,s['reference_grid'],threshold=cutoff)
            entries[k].append(dict(id=s['id'],name=title,task='plume_segmentation',
                rgb=folder.name+'/satellite.png',reference90Rgb=folder.name+'/background.png',
                reference90Label='Earlier reviewed no-plume reference',predictionImage=folder.name+'/prediction.png',
                probabilityImage=folder.name+'/score.png',disagreementImage=folder.name+'/errors.png',
                truthImage=folder.name+'/truth.png',grid=folder.name+'/grid.json',bounds=bounds,
                predictionAvailable=True,decisionThreshold=cutoff,dataKind='measured',dataset='MARS',label=s['scene_kind'],
                split='Exposed broader DEV | separate from fitting; previously evaluated project sites',
                latitude=s['latitude'],longitude=s['longitude'],locationPoint=[s['latitude'],s['longitude']],
                source=s['target_scene_id'],date=s['satellite_time'][:10],referenceDate=s['reference_time'][:10],
                resolution='20 m prediction; native 10 m label comparison',truthLabel='Reviewed MARS annotation',
                truthNote='Binary plume label, not quantitative methane concentration.',
                outcome=('Missed plume' if not c['tp'] else 'Partial overlap') if hit.any() else ('False positive' if c['fp'] else 'Correct negative'),
                checkpointHash=method['checkpoint_sha256'],predictionHash=r.v.sha(pred),
                nativeMetrics={**measured[k],'source_valid_pixels':int(valid.sum()),'support_fraction':float(valid.mean()),
                               'claim':'Exposed DEV used for selection, not independent validation'},
                notes=recipe['scope']+' Both dates use supplied cloud masks, which do not rule out haze.'))
        common = a.output/'baseline'/f'scene_{index:04d}'
        cases.append(dict(id=s['id'],title=title,current=r.v.image_uri(common/'current.png'),
            background=r.v.image_uri(common/'background.png'),annotation=r.v.image_uri(common/'annotation.png'),
            models=figures,metrics=measured))
        records.append(dict(id=s['id'],models=measured))
    pooled = {k:{**r.v.ratios(c),'f1':2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1)} for k,c in totals.items()}
    negative = {k:dict(**c,pixel_fpr=c['fp']/max(c['fp']+c['tn'],1)) for k,c in negatives.items()}
    # Explicitly show inference-method cost or sampling context, never invented learning curves.
    figure = Figure(figsize=(12,2.2),facecolor='#13242d')
    axis=figure.subplots();axis.set_axis_off()
    axis.text(.02,.7,recipe['chart_title'],color='#e4eef1',fontsize=13,transform=axis.transAxes)
    axis.text(.02,.3,recipe['chart_detail'],color='#b7cbd3',fontsize=10,transform=axis.transAxes,wrap=True)
    figure.savefig(a.output/'context.png',facecolor=figure.get_facecolor())
    if recipe.get('learning_image'):
        learning_path=Path(recipe['learning_image']['path'])
        if r.v.sha(learning_path)!=recipe['learning_image']['sha256']:
            raise ValueError('Verified learning plot changed')
        (a.output/'learning.png').write_bytes(learning_path.read_bytes())
    body = r.HTML.replace('stable_residual','baseline').replace('siamese','candidate')
    body=body.replace('OCTOBER 24, 2026','OCTOBER 25, 2026').replace('<th>Plume IoU</th>','<th>Plume F1</th><th>Plume IoU</th>')
    for old,new in [('Normalized residual CNN',names['baseline']),('Temporal U-Net',names['candidate']),
                    ('Open residual in viewer','Open baseline in viewer'),('Open U-Net in viewer','Open candidate in viewer'),
                    ('Saved residual CNN prediction','Saved baseline prediction'),('Saved temporal U-Net prediction','Saved candidate prediction'),
                    ('Two models. Actual saved predictions.','Two prediction methods. Actual saved predictions.'),
                    ('<h2>Learning check</h2>','<h2>What changed</h2>'),
                    ('Actual training and development overlap and false-alarm histories','Recorded experiment context')]:
        body=body.replace(old,new)
    start=body.index('<p class="mini">Curves use'); end=body.index('</p>',start)+4
    body=body[:start]+'<p class="mini">'+html.escape(recipe['context'])+'</p>'+body[end:]
    options=''.join(f'<option value="{i}">{html.escape(c["title"])}</option>' for i,c in enumerate(cases))
    stats=''.join(f'<tr><td>{html.escape(names[k])}</td><td>{r.v.percent(pooled[k]["f1"])}</td><td>{r.v.percent(pooled[k]["iou"])}</td><td>{r.v.percent(pooled[k]["precision"])}</td><td>{r.v.percent(pooled[k]["recall"])}</td><td>{r.v.percent(negative[k]["pixel_fpr"])}</td><td>{negative[k]["activated"]}/{negative[k]["scenes"]}</td></tr>' for k in methods)
    values=dict(TITLE=html.escape(recipe['title']),SCOPE=html.escape(recipe['scope']),OPTIONS=options,STATS=stats,
        CASES=json.dumps(cases).replace('</','<\\/'),IDS=json.dumps(ids),N=str(len(scenes)),
        LEARNING=r.v.image_uri(a.output/'context.png'),ACCEPTANCE=html.escape(recipe['acceptance']),
        THRESHOLDS='; '.join(html.escape(names[k])+' '+str(methods[k]['cutoff']) for k in methods))
    if recipe.get('learning_image'):
        values['LEARNING']=r.v.image_uri(a.output/'learning.png')
    # Compute the all-positive control from the actual same native support.
    denom=sum(totals['baseline'][key] for key in ('tp','fp','fn','tn'))
    values['ALLPOS']=r.v.percent((totals['baseline']['tp']+totals['baseline']['fn'])/max(denom,1))
    for key,value in values.items():
        body=body.replace('__'+key+'__',value)
    country_rows=[]
    for country,data in sorted(countries.items()):
        cells=[html.escape(country),str(data['plume']),str(data['clear'])]
        for k,c in data['models'].items():
            c['f1']=2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1)
            cells.extend([r.v.percent(c['f1']) if data['plume'] else 'No positive labels',str(c['clear_fp']),str(c['clear_activated'])+'/'+str(data['clear'])])
        country_rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
    country_headers=['Country','Plume scenes','Clear scenes']
    for k in methods:
        country_headers.extend([names[k]+' F1','Clear FP pixels','Clear scenes activated'])
    country_table='<h2>All countries, descriptive breakdown</h2><p class="mini">The US group has only three positive observations. These previously evaluated sites do not establish national performance. No country subset replaces the all-198 target.</p><div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in country_headers)+'</tr></thead><tbody>'+''.join(country_rows)+'</tbody></table></div>'
    body=body.replace('<div class="footer">',country_table+'<div class="footer">')
    publication={}
    for k in methods:
        folder=a.output/k;(folder/'index.html').write_text(body,encoding='utf-8')
        r.v.save(folder/'experiment.json',dict(id=ids[k],name=recipe['title']+' | '+names[k],
            lifecycle='experimental',priority=False,dataKind='measured',architectureFamily=recipe['architecture_family'],
            runGroup=recipe['id_prefix'],status=recipe['status'],description=recipe['scope'],
            decisionThreshold=methods[k]['cutoff'],nativeAggregate={**pooled[k],'scenes':len(scenes)},
            metrics={'Plume F1':r.v.percent(pooled[k]['f1']),'Pixel IoU':r.v.percent(pooled[k]['iou']),'Recall':r.v.percent(pooled[k]['recall']),
                     'No-plume pixel FPR':r.v.percent(negative[k]['pixel_fpr'])},scenes=entries[k],reportUrl='index.html'))
        r.v.load_bundle(folder/'experiment.json','bundle-assets/')
        publication[k]=r.v.publish(folder,r.v.ROOT/'outputs/GTM_viewer_bundles')
    (a.output/'index.html').write_text(body,encoding='utf-8')
    r.v.save(a.output/'comparison.json',dict(pooled=pooled,negatives=negative,scenes=records))
    r.v.save(a.output/'country_comparison.json',countries)
    r.v.save(a.output/'verification.json',dict(passed=True,counts_recomputed=True,source_reconstructions=len(scenes),
        evaluation_role='exposed_broader_dev',manifest_sha256=r.v.sha(manifest_path),recipe_sha256=r.v.sha(a.recipe),
        renderer_sha256=r.v.sha(Path(__file__)),template_sha256=r.v.sha(r.__file__),publication=publication))
    print(json.dumps(dict(pooled=pooled,negatives=negative,publication=publication)),flush=True)


if __name__=='__main__':
    main()
