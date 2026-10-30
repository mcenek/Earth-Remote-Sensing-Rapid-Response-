"""Independently reconstruct native inputs and publish v13 development comparisons.

The v13 protocol fixes failure/visual controls before implementation. Historical
renderers remain unchanged. No fitting, thresholds or scene selection occurs here.
"""
import argparse
import collections
import html
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from matplotlib.figure import Figure

import GTM_Model7_review_repaired as r


def independent_features(scene, target, reference, valid, paths, released=False):
    """Rebuild all 16 channels without calling the training feature helper."""
    target=np.clip(target.transpose(2,0,1)/5000,0,2)
    reference=np.clip(reference.transpose(2,0,1)/5000,0,2)
    normalized=[]
    for image in (target,reference):
        ratio=np.ones(valid.shape,np.float32)
        good=((image[4]!=0) & np.isfinite(image[4]) & np.isfinite(image[5])) if released else valid & (image[4]!=0)
        ratio[good]=image[5,good]/image[4,good]
        median=float(np.median(ratio[good])) if good.any() else 1.
        if not np.isfinite(median) or abs(median)<1e-8:median=1.
        ratio[good]/=median
        normalized.append(np.clip(ratio,0,10))
    mbmp=np.ones(valid.shape,np.float32)
    np.divide(normalized[0],normalized[1],out=mbmp,where=normalized[1]!=0)
    mbmp[~np.isfinite(mbmp)]=1
    np.clip(mbmp,0,10,out=mbmp)
    if not released: mbmp[~valid]=1
    with rasterio.open(paths['target_cloud']) as a,rasterio.open(paths['reference_cloud']) as b:
        cloud=((a.read(1)>0) if released else ((a.read(1)>0)|(b.read(1)>0))).astype(np.float32)
    wind=np.clip(np.array([float(scene['official_csv_row'][k]) for k in ('wind_u','wind_v')],np.float32),-20,20)/8
    return np.concatenate((mbmp[None],target,reference,np.broadcast_to(wind[:,None,None],(2,200,200)),cloud[None]))


def independent_residual_features(target, reference, valid, teacher, norm):
    """Reconstruct the frozen residual inputs independently of its trainer."""
    dates=[]
    for image in (target,reference):
        raw=image.transpose(2,0,1)
        logged=np.log1p(raw[:,valid].astype(np.float64))
        centered=logged-np.median(logged,axis=1)[:,None]
        center=np.asarray(norm['center'],np.float64)[:,None]
        scale=np.asarray(norm['scale'],np.float64)[:,None]
        output=np.zeros((6,200,200),np.float32)
        output[:,valid]=np.clip((centered-center)/scale,-8,8).astype(np.float32)
        dates.append(output)
    ratios=[]
    for image in (target,reference):
        ratio=np.ones((200,200),np.float32)
        good=valid & (image[...,4]!=0)
        ratio[good]=image[...,5][good]/image[...,4][good]
        median=float(np.median(ratio[good])) if good.any() else 1.
        if not np.isfinite(median) or abs(median)<1e-8: median=1.
        ratio[good]/=median
        ratios.append(np.clip(ratio,0,10))
    contrast=np.ones((200,200),np.float32)
    np.divide(ratios[0],ratios[1],out=contrast,where=ratios[1]!=0)
    contrast[~np.isfinite(contrast)]=1
    np.clip(contrast,0,10,out=contrast)
    signal=np.zeros((200,200),np.float32)
    signal[valid]=np.clip(-10*np.log(np.maximum(contrast[valid],.0001)),-8,8)
    clipped=np.clip(teacher,1e-6,1-1e-6)
    logit=np.log(clipped/(1-clipped)).astype(np.float32)
    current,earlier=dates
    x=np.concatenate((current,earlier,current-earlier,np.abs(current-earlier),signal[None],(logit/6)[None]))
    x[:,~valid]=0
    return x,logit


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
    if recipe['evaluation_split'] != 'dev':
        raise ValueError('Native v13 only publishes the exposed 198 DEV observations')
    scenes = sorted((s for s in manifest['scenes'] if s['split'] == recipe['evaluation_split']),
                    key=lambda s: (s['scene_kind'] != 'PLUME', s['source_row']['country'], s['id']))
    expected=(198,46)
    if (len(scenes),sum(s['scene_kind']=='PLUME' for s in scenes))!=expected:
        raise ValueError('Frozen evaluation membership changed')
    methods = recipe['methods']
    if set(methods) != {'baseline', 'candidate', 'ratio'}:
        raise ValueError('Historical v11 and the two fresh native models are required')
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
        prepared = (manifest_path.parent / s['array_file']).resolve()
        if r.v.sha(prepared) != s['array_sha256']:
            raise ValueError('Prepared array changed')
        with np.load(prepared, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        r.source_rebuild(s, data)
        # Read original native bands independently of the trained feature cache.
        paths = {q['role']:Path(q['path']) for q in s['sources']}
        with rasterio.open(paths['target_image']) as target_ds, rasterio.open(paths['reference_image']) as reference_ds:
            target_native=np.moveaxis(target_ds.read(range(1,7)),0,-1).astype(np.float32)
            reference_native=np.moveaxis(reference_ds.read(range(1,7)),0,-1).astype(np.float32)
        native_cache = recipe['native_data'][s['id']]
        native_path=Path(native_cache['path'])
        if r.v.sha(native_path)!=native_cache['sha256']:
            raise ValueError('Native training cache changed')
        contract=recipe.get('feature_contract','native16')
        if contract=='residual26':
            teacher_method=methods['baseline']
            teacher_path=Path(teacher_method['folder'])/(s['id']+'.npz')
            if r.v.sha(teacher_path)!=teacher_method['records'][s['id']]['probability_sha256']:
                raise ValueError('Frozen teacher changed')
            with np.load(teacher_path,allow_pickle=False) as teacher_data:
                teacher=teacher_data['probability_native']
                if not np.array_equal(teacher_data['valid_native'].astype(bool),data['valid_native'].astype(bool)):
                    raise ValueError('Teacher support mismatch')
            rebuilt_features,rebuilt_logit=independent_residual_features(target_native,reference_native,
                data['valid_native'].astype(bool),teacher,recipe['frozen_normalization'])
        elif contract=='native16':
            rebuilt_features=independent_features(s,target_native,reference_native,data['valid_native'].astype(bool),paths)
        elif contract=='released16':
            rebuilt_features=independent_features(s,target_native,reference_native,data['valid_native'].astype(bool),paths,released=True)
        else:
            raise ValueError('Unknown native feature contract')
        with np.load(native_path,allow_pickle=False) as z:
            if (not np.array_equal(z['y'],data['label_native']) or
                not np.array_equal(z['valid'].astype(bool),data['valid_native'].astype(bool)) or
                not np.array_equal(z['x16' if contract=='released16' else 'x'],rebuilt_features)):
                raise ValueError('Native source reconstruction differs from fitted input')
            if contract=='residual26' and (not np.array_equal(z['teacher_logit'],rebuilt_logit) or
                                          not np.array_equal(z['teacher_probability'],teacher)):
                raise ValueError('Residual teacher input mismatch')
        current, reference = r.balanced_rgb_pair(dict(target_image=target_native,
            reference_image=reference_native,valid=data['valid_native']),1)
        truth, valid = data['label_native'], data['valid_native'].astype(bool)
        annotation = current.copy()
        hit = (truth == 1) & valid
        annotation[hit] = (.35*annotation[hit]+.65*np.array([46,224,219])).astype('uint8')
        title = f"{'Plume' if s['scene_kind'] == 'PLUME' else 'No plume'} | {country} | {s['satellite_time'][:10]} | {s['id'][5:13]}"
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
                    probability.shape != (200,200) or
                    not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1):
                    raise ValueError('Prediction grid/support/range mismatch')
            if method['prediction_grid'] == '20m_repeat':
                with np.load(pred,allow_pickle=False) as z:
                    if not np.array_equal(probability,np.repeat(np.repeat(z['probability20'],2,0),2,1)):
                        raise ValueError('Historical 20m model display changed')
            elif method['prediction_grid'] != 'native10m':
                raise ValueError('Undeclared prediction grid')
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
                reference90Label='Earlier selected reference; methane absence not confirmed',predictionImage=folder.name+'/prediction.png',
                probabilityImage=folder.name+'/score.png',disagreementImage=folder.name+'/errors.png',
                truthImage=folder.name+'/truth.png',grid=folder.name+'/grid.json',bounds=bounds,
                predictionAvailable=True,decisionThreshold=cutoff,dataKind='measured',dataset='MARS',label=s['scene_kind'],
                split=recipe['evaluation_role'],
                latitude=s['latitude'],longitude=s['longitude'],locationPoint=[s['latitude'],s['longitude']],
                source=s['target_scene_id'],date=s['satellite_time'][:10],referenceDate=s['reference_time'][:10],
                resolution=method['resolution'],truthLabel='Reviewed MARS annotation',
                truthNote='Binary plume label, not quantitative methane concentration.',
                outcome=('Missed plume' if not c['tp'] else 'Partial overlap') if hit.any() else ('False positive' if c['fp'] else 'Correct negative'),
                checkpointHash=method['checkpoint_sha256'],predictionHash=r.v.sha(pred),
                nativeMetrics={**measured[k],'source_valid_pixels':int(valid.sum()),'support_fraction':float(valid.mean()),
                               'claim':recipe['evaluation_role']},
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
    body=body.replace('OCTOBER 24, 2026','OCTOBER 26, 2026').replace('<th>Plume IoU</th>','<th>Plume F1</th><th>Plume IoU</th>')
    for old,new in [('Normalized residual CNN',names['baseline']),('Temporal U-Net',names['candidate']),
                    ('Open residual in viewer','Open baseline in viewer'),('Open U-Net in viewer','Open candidate in viewer'),
                    ('Saved residual CNN prediction','Saved baseline prediction'),('Saved temporal U-Net prediction','Saved candidate prediction'),
                    ('Two models. Actual saved predictions.',recipe.get('comparison_heading','Historical v11 and two fresh native-grid models. Actual saved predictions.')),
                    ('<h2>Learning check</h2>','<h2>What changed</h2>'),
                    ('Actual training and development overlap and false-alarm histories','Recorded experiment context')]:
        body=body.replace(old,new)
    start=body.index('<p class="mini">Curves use'); end=body.index('</p>',start)+4
    body=body[:start]+'<p class="mini">'+html.escape(recipe['context'])+'</p>'+body[end:]
    body=body.replace('repeat(5,minmax(0,1fr))','repeat(6,minmax(0,1fr))')
    body=body.replace("for(const k of ['baseline','candidate'])","for(const k of Object.keys(ids))")
    body=body.replace("'IoU '+fmt(c.metrics[k].iou)+' | extra pixels '+c.metrics[k].fp.toLocaleString()",
        "'F1 '+fmt(2*c.metrics[k].tp/Math.max(2*c.metrics[k].tp+c.metrics[k].fp+c.metrics[k].fn,1))+' | P '+fmt(c.metrics[k].precision)+' | R '+fmt(c.metrics[k].recall)+' | extra pixels '+c.metrics[k].fp.toLocaleString()")
    body=body.replace('<a id="map-candidate">Open candidate in viewer</a>', '<a id="map-candidate">Open '+html.escape(names['candidate'])+' in viewer</a><a id="map-ratio">Open '+html.escape(names['ratio'])+' in viewer</a>')
    marker='<p class="legend">'
    end=body.index(marker)
    insertion='<figure><img id="ratio" alt="'+html.escape(names['ratio'],quote=True)+' saved prediction"><figcaption><b>'+html.escape(names['ratio'])+'</b><span id="metric-ratio"></span></figcaption></figure>'
    body=body[:end].removesuffix('</div>')+insertion+'</div>'+body[end:]
    body=body.replace('Development scores are used for checkpoint selection and are not independent validation.', 'These 198 observations select checkpoints and cutoffs. They have historical project exposure and are not an independent US test. Prospective US data is reserved separately.')
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
    country_table='<h2>All countries, descriptive breakdown</h2><p class="mini">These are descriptive country groups. Every evaluated observation contributes to the main score. Historical project exposure prevents an independent national performance claim.</p><div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in country_headers)+'</tr></thead><tbody>'+''.join(country_rows)+'</tbody></table></div>'
    body=body.replace('<div class="footer">',country_table+'<div class="footer">')
    publication={}
    for k in methods:
        folder=a.output/k;(folder/'index.html').write_text(body,encoding='utf-8')
        r.v.save(folder/'experiment.json',dict(id=ids[k],name=recipe['title']+' | '+names[k],
            lifecycle='experimental',priority=False,dataKind='measured',architectureFamily=recipe['architecture_family'],
            runGroup=recipe['id_prefix'],status=methods[k].get('status',recipe['status']),description=recipe['scope'],
            decisionThreshold=methods[k]['cutoff'],nativeAggregate={**pooled[k],'scenes':len(scenes)},
            metrics={'Plume F1':r.v.percent(pooled[k]['f1']),'Pixel IoU':r.v.percent(pooled[k]['iou']),'Recall':r.v.percent(pooled[k]['recall']),
                     'No-plume pixel FPR':r.v.percent(negative[k]['pixel_fpr'])},scenes=entries[k],reportUrl='index.html'))
        r.v.load_bundle(folder/'experiment.json','bundle-assets/')
        publication[k]=r.v.publish(folder,r.v.ROOT/'outputs/GTM_viewer_bundles')
    (a.output/'index.html').write_text(body,encoding='utf-8')
    r.v.save(a.output/'comparison.json',dict(pooled=pooled,negatives=negative,scenes=records))
    r.v.save(a.output/'country_comparison.json',countries)
    r.v.save(a.output/'verification.json',dict(passed=True,counts_recomputed=True,source_reconstructions=len(scenes),
        native_feature_contract=recipe.get('feature_contract','native16'),
        all_native_features_independently_reconstructed=True,
        evaluation_role=recipe['evaluation_role'],manifest_sha256=r.v.sha(manifest_path),recipe_sha256=r.v.sha(a.recipe),
        renderer_sha256=r.v.sha(Path(__file__)),template_sha256=r.v.sha(r.__file__),publication=publication))
    print(json.dumps(dict(pooled=pooled,negatives=negative,publication=publication)),flush=True)


if __name__=='__main__':
    main()
