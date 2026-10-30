"""Bounded offline comparison of cached R5 outputs and two CPU pixel baselines.

No model runtime, acquisition, held-label fitting, threshold search or spatial
reprojection. Arrays stay on their native grids; the viewer gets pixel coordinates.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '2'
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / 'outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r5'
CACHE = ROOT / 'outputs/GTM_Model6/GTM_Model6_plume_evidence_02'
BRANCHES = ('S16', 'S32', 'S64', 'S128')
LABELS = {'r5': 'R5 saved mean', **{key: key for key in BRANCHES},
          'probability_mean': 'Probability mean', 'agree3': 'At least 3 of 4',
          'agree4': 'All 4 agree', 'swir1': 'Linear: SWIR contrast',
          'linear16': 'Linear: all 16 features', 'all_negative': 'Always negative',
          'all_positive': 'Always positive'}
PROTOCOL = dict(seed=20261003, threshold=.5, per_parent_class_cap=512, l2=.01,
                max_newton_steps=30, fit_features={'swir1': [15], 'linear16': list(range(16))},
                scope='Reused development comparison, not independent validation',
                class_weight='One half of objective per class',
                selection='All positive scenes plus top two false alarms for R5, linear16 and swir1; deduplicated',
                network_requests=0, model_runtime_required=False)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda: source.read(1048576), b''): value.update(part)
    return value.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def require(condition, message):
    if not condition: raise ValueError(message)


def counts(score, target, valid):
    predicted, truth = score[valid] >= .5, target[valid] > 0
    return {key: int(value.sum()) for key, value in {
        'tp': predicted & truth, 'fp': predicted & ~truth,
        'fn': ~predicted & truth, 'tn': ~predicted & ~truth}.items()}


def metrics(c):
    tp, fp, fn, tn = (c[key] for key in ('tp', 'fp', 'fn', 'tn'))
    divide = lambda a, b: a / b if b else None
    return dict(c, iou=divide(tp, tp+fp+fn), precision=divide(tp, tp+fp), recall=divide(tp, tp+fn),
                dice=divide(2*tp, 2*tp+fp+fn), evaluated_pixels=tp+fp+fn+tn,
                pixel_false_positive_rate=divide(fp, fp+tn))


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


class Study:
    def __init__(self, cache_dir, r5_dir, minutes):
        self.cache_dir, self.r5_dir = cache_dir, r5_dir
        self.deadline = time.monotonic() + minutes * 60
        self.input_files = {}
        self.manifest = self.read(r5_dir / 'GTM_Model6_mapper_mars_pilot_manifest.json')
        cache_manifest = self.read(cache_dir / 'GTM_Model6_plume_manifest.json')
        self.fit_rows, self.eval_rows = self.manifest['training_observations'], self.manifest['observations']
        all_rows = self.fit_rows + self.eval_rows
        self.cache_rows = {row['id']: (i, row) for i, row in enumerate(cache_manifest['observations'])}
        require(len(self.cache_rows) == len(cache_manifest['observations']), 'Duplicate cache IDs')
        require(len({row['id'] for row in all_rows}) == len(all_rows), 'Duplicate or shared parent IDs')
        require(len(self.fit_rows) == self.manifest['training_records'] and len(self.eval_rows) == self.manifest['held_records'], 'Manifest counts differ')
        for rows, held in ((self.fit_rows, False), (self.eval_rows, True)):
            require(bool(rows), 'Empty partition')
            for row in rows:
                require(row['dataset'] == 'MARS' and row['fully_labeled'], 'Only fully labeled MARS parents allowed')
                require((row['partition'] == self.manifest['holdout_fold']) == held, 'Incorrect partition')
                require(row['id'] in self.cache_rows, 'Missing cache ID: ' + row['id'])
                cached = self.cache_rows[row['id']][1]
                for key in ('dataset','label','site','group','partition','target_scene','reference_scene','crs','affine','shape','hashes'):
                    require(row[key] == cached[key], 'Cache metadata mismatch: ' + key + ' ' + row['id'])
        for key in ('id', 'site', 'group'):
            require(not {r[key] for r in self.fit_rows} & {r[key] for r in self.eval_rows}, 'Training/evaluation leakage: ' + key)
        acquisition = lambda rows: {r[k] for r in rows for k in ('target_scene','reference_scene')}
        require(not acquisition(self.fit_rows) & acquisition(self.eval_rows), 'Shared acquisition across partitions')
        hashes = lambda rows: {r['hashes']['image'] for r in rows}
        require(not hashes(self.fit_rows) & hashes(self.eval_rows), 'Shared image hash across partitions')
        self.cache_path = cache_dir / 'GTM_Model6_plume_inputs.npz'
        self.track(self.cache_path)

    def check_time(self):
        if time.monotonic() > self.deadline: raise TimeoutError('Compute deadline reached')

    def track(self, path):
        self.check_time(); self.input_files[str(path)] = digest(path)

    def read(self, path):
        self.track(path); return json.loads(path.read_text(encoding='utf-8'))

    def parent(self, cache, row, target=True):
        self.check_time()
        index = self.cache_rows[row['id']][0]
        x, valid = cache[f'{index}_x'][:16], cache[f'{index}_valid']
        shape = tuple(row['shape'])
        require(x.shape == (16, *shape) and valid.shape == shape, 'Wrong input shape')
        require(np.isin(valid, [0, 1]).all(), 'Nonbinary validity')
        valid = valid.astype(bool)
        require(valid.any() and np.isfinite(x[:, valid]).all(), 'Nonfinite supported input or empty support')
        x = np.where(valid[None], x, 0)
        y = cache[f'{index}_target'] if target else None
        if target:
            require(y.shape == shape and np.isin(y, [0, 1]).all(), 'Nonbinary target or wrong target shape')
        return x, y, valid

    def fit(self):
        rng = np.random.default_rng(PROTOCOL['seed'])
        features, targets = [], []
        with np.load(self.cache_path, allow_pickle=False) as cache:
            for row in self.fit_rows:
                x, y, valid = self.parent(cache, row)
                for label in (0, 1):
                    candidates = np.flatnonzero(valid.ravel() & (y.ravel() == label))
                    selected = rng.choice(candidates, min(len(candidates), PROTOCOL['per_parent_class_cap']), replace=False)
                    features.append(x.reshape(16, -1)[:, selected].T)
                    targets.append(np.full(len(selected), label, dtype=np.float64))
        x, y = np.concatenate(features).astype(np.float64), np.concatenate(targets)
        require(0 < y.sum() < len(y), 'Both training classes are required')
        weights = np.where(y > 0, .5 / y.sum(), .5 / (len(y)-y.sum()))
        models = {}
        for name, columns in PROTOCOL['fit_features'].items():
            self.check_time()
            raw = x[:, columns]; center = raw.mean(0); scale = np.maximum(raw.std(0), 1e-4)
            design = np.column_stack([(raw-center)/scale, np.ones(len(raw))])
            beta = np.zeros(design.shape[1]); penalty = np.full(len(beta), PROTOCOL['l2']); penalty[-1] = 0
            objective = lambda b: float(np.dot(weights, np.logaddexp(0, design @ b)-y*(design @ b)) + .5*np.dot(penalty*b, b))
            losses = [objective(beta)]; converged = False
            for iteration in range(PROTOCOL['max_newton_steps']):
                self.check_time(); probability = sigmoid(design @ beta)
                gradient = design.T @ (weights*(probability-y)) + penalty*beta
                if np.max(np.abs(gradient)) < 1e-7:
                    converged = True; break
                hessian = design.T @ (design*(weights*probability*(1-probability))[:, None]) + np.diag(penalty+1e-9)
                step = np.linalg.solve(hessian, gradient); factor = 1.0
                while factor >= 2**-15:
                    candidate = beta-factor*step; loss = objective(candidate)
                    if loss <= losses[-1]: break
                    factor *= .5
                require(factor >= 2**-15, 'Training objective backtracking failed')
                beta = candidate; losses.append(loss)
            require(converged, 'Fixed training iteration limit reached without convergence')
            models[name] = dict(columns=columns, center=center.tolist(), scale=scale.tolist(),
                                weights=beta[:-1].tolist(), intercept=float(beta[-1]),
                                converged=converged, objective_history=losses)
        return dict(models=models, sampled_positive_pixels=int(y.sum()), sampled_negative_pixels=int(len(y)-y.sum()),
                    training_ids=[r['id'] for r in self.fit_rows], evaluation_ids=[r['id'] for r in self.eval_rows])

    @staticmethod
    def predict(x, model):
        values = (x[model['columns']].astype(np.float64) - np.array(model['center'])[:, None, None]) / np.array(model['scale'])[:, None, None]
        logits = np.tensordot(np.array(model['weights']), values, axes=(0, 0)) + model['intercept']
        return sigmoid(logits).astype(np.float32)

    def evaluate(self, models, output):
        saved = self.read(self.r5_dir / 'GTM_Model6_mapper_mars_pilot_results.json')
        records = {r['id']: r for r in saved['observations']}
        require(set(records) == {r['id'] for r in self.eval_rows}, 'Saved evaluation cohort differs')
        result_rows = []
        with np.load(self.cache_path, allow_pickle=False) as cache:
            for row in self.eval_rows:
                x, y, source_valid = self.parent(cache, row)
                record = records[row['id']]; path = self.r5_dir / f"prediction_{record['index']:04d}.npz"
                self.track(path)
                with np.load(path, allow_pickle=False) as native:
                    require(np.array_equal(native['target'], y), 'Cached/saved target mismatch')
                    valid = native['valid']; context = native['context_support']
                    require(valid.shape == y.shape and context.shape == y.shape and np.isin(valid,[0,1]).all() and np.isin(context,[0,1]).all(), 'Invalid prediction support')
                    valid = valid.astype(bool)
                    require(np.array_equal(valid, source_valid & context.astype(bool)), 'Source/context support mismatch')
                    prediction = {'r5': native['probability'].copy(), **{key: native[key].copy() for key in BRANCHES}}
                for key, p in prediction.items():
                    require(p.shape == y.shape and np.isfinite(p[valid]).all() and ((p[valid]>=0)&(p[valid]<=1)).all(), 'Invalid saved score')
                    expected = record['per_method']['mean' if key == 'r5' else key]
                    require(all(counts(p,y,valid)[k] == expected[k] for k in ('tp','fp','fn','tn')), 'Saved native counts differ: ' + key)
                stacked = np.stack([prediction[key] for key in BRANCHES]); votes = (stacked >= .5).sum(0)
                prediction.update(probability_mean=stacked.mean(0), agree3=(votes>=3).astype(np.float32),
                                  agree4=(votes==4).astype(np.float32), all_negative=np.zeros_like(y), all_positive=np.ones_like(y))
                prediction.update({name: self.predict(x, model) for name, model in models.items()})
                for p in prediction.values(): require(np.isfinite(p[valid]).all(), 'Nonfinite new prediction')
                per_method = {key: metrics(counts(prediction[key], y, valid)) for key in LABELS}
                out_path = output / f"prediction_{record['index']:04d}.npz"
                np.savez_compressed(out_path, target=y, valid=valid.astype(np.uint8), **prediction)
                result_rows.append(dict(id=row['id'], index=record['index'], label=row['label'], group=row['group'],
                                        date=row['date'], per_method=per_method, native_prediction_sha256=digest(out_path)))
        pooled = {}
        negatives = [row for row in result_rows if row['label'] == 'NO_PLUME']
        for name in LABELS:
            total = {k: sum(r['per_method'][name][k] for r in result_rows) for k in ('tp','fp','fn','tn')}
            pooled[name] = metrics(total)
            pixels = sum(r['per_method'][name]['evaluated_pixels'] for r in negatives)
            alarms = sum(r['per_method'][name]['fp'] for r in negatives)
            pooled[name].update(negative_scene_activation=alarms/pixels if pixels else None,
                                negative_scenes_with_any_activation=sum(r['per_method'][name]['fp']>0 for r in negatives),
                                negative_scenes=len(negatives), scenes=len(result_rows))
        return dict(status='completed_development_comparison_not_promoted', methods=pooled, observations=result_rows,
                    training_scenes=len(self.fit_rows), evaluation_scenes=len(self.eval_rows),
                    positive_evaluation_groups=sorted({r['group'] for r in result_rows if r['label']=='PLUME'}),
                    scope=PROTOCOL['scope'], threshold=.5, network_requests=0)


def pct(value):
    return 'Not defined' if value is None else f'{100*value:.2f}%'


def image_rgb(raw):
    lo, hi = np.percentile(raw, [2,98], axis=(0,1))
    return (255*np.clip((raw-lo)/np.maximum(hi-lo,1),0,1)).astype(np.uint8)


def overlay(rgb, target, valid, prediction=None):
    canvas = rgb.astype(np.float32).copy()
    if prediction is not None:
        active = valid & (prediction>=.5); canvas[active] = .45*canvas[active]+.55*np.array([255,135,35])
    truth = (target>0)&valid; padded = np.pad(truth, 1)
    inside = padded[:-2,1:-1]&padded[2:,1:-1]&padded[1:-1,:-2]&padded[1:-1,2:]
    canvas[truth & ~inside] = [0,235,240]
    yy, xx = np.indices(valid.shape); striped = ~valid & ((xx+yy)%8<3)
    canvas[~valid] *= .5; canvas[striped] = .5*canvas[striped]+.5*np.array([170,185,195])
    return Image.fromarray(canvas.clip(0,255).astype(np.uint8))


def export_review(study, result, output):
    chosen = [r for r in result['observations'] if r['label']=='PLUME']
    negatives = [r for r in result['observations'] if r['label']=='NO_PLUME']
    for key in ('r5','linear16','swir1'):
        for row in sorted(negatives,key=lambda r:(-r['per_method'][key]['fp'],r['id']))[:2]:
            if row not in chosen: chosen.append(row)
    assets = output / 'review'; assets.mkdir()
    rows = {r['id']:r for r in study.eval_rows}; scenes, options = [], []
    panels = ['r5','S128','linear16','swir1','agree3','agree4']
    with np.load(study.cache_path, allow_pickle=False) as cache:
        for rec in chosen:
            study.check_time(); row=rows[rec['id']]; stem=f"case_{rec['index']:04d}"
            with np.load(output/f"prediction_{rec['index']:04d}.npz") as pred:
                y,valid=pred['target'],pred['valid'].astype(bool)
                rgb=image_rgb(cache[f"{study.cache_rows[row['id']][0]}_rgb"])
                Image.fromarray(rgb).save(assets/f'{stem}_rgb.png')
                overlay(rgb,np.zeros_like(y),valid).save(assets/f'{stem}_input.png')
                overlay(rgb,y,valid).save(assets/f'{stem}_reference.png')
                for key in panels: overlay(rgb,y,valid,pred[key]).save(assets/f'{stem}_{key}.png')
                rgba=np.zeros((*y.shape,4),np.uint8);rgba[valid&(y>0)]=[0,225,230,80]
                Image.fromarray(rgba).save(assets/f'{stem}_truth.png')
                rgba[:]=0;rgba[valid&(pred['linear16']>=.5)]=[255,135,35,230]
                Image.fromarray(rgba).save(assets/f'{stem}_prediction.png')
                grid={'width':y.shape[1],'height':y.shape[0],
                      'probability':[float(v) if ok else None for v,ok in zip(pred['linear16'].ravel(),valid.ravel())],
                      'truth':[int(v) if ok else None for v,ok in zip(y.ravel(),valid.ravel())], 'valid':valid.astype(int).ravel().tolist()}
                write(assets/f'{stem}_grid.json',grid)
                native=rec['per_method']['linear16'] | dict(support_fraction=float(valid.mean()))
            scenes.append(dict(id=row['id'],name=row['label']+' | '+row['date'][:10]+' | '+row['id'][-8:],
                               coordinateSystem='pixel',imageSize=row['shape'],locationPoint=[row['lat'],row['lon']],
                               rgb=f'review/{stem}_rgb.png',truthImage=f'review/{stem}_truth.png',predictionImage=f'review/{stem}_prediction.png',
                               grid=f'review/{stem}_grid.json',predictionAvailable=True,decisionThreshold=.5,
                               date=row['date'][:10],split='Original R5 evaluation fold; reused development data',
                               source='Cached Sentinel inputs; SHA-256 in provenance.json',truthLabel='MARS reviewed plume mask',
                               nativeMetrics=native, notes='New CPU linear16 output. Native image coordinates, not a reprojected map. Scores are uncalibrated. No emission-origin or quantitative methane claim.'))
            options.append(dict(stem=stem,label=scenes[-1]['name'],metrics=rec['per_method']))
    table=''.join('<tr><td>'+html.escape(LABELS[key])+'</td>'+''.join('<td>'+html.escape(value)+'</td>' for value in
                 [pct(m['iou']),pct(m['precision']),pct(m['recall']),f"{m['fp']:,}",pct(m['negative_scene_activation'])])+'</tr>' for key,m in result['methods'].items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local model comparison</title>
<style>body{margin:0;background:#0d1920;color:#e1ecef;font:16px system-ui}main{max-width:1300px;margin:auto;padding:32px}h1{font-size:32px;margin-bottom:12px}p{color:#abc0c9;line-height:1.6}.note{border-left:3px solid #edb45f;padding:8px 20px;background:#172730}a{color:#58d4d7}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #29414c}thead{color:#86dbd5}select{padding:12px;background:#20343d;color:white;border:1px solid #45636f;border-radius:5px;width:min(100%,660px)}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}article{background:#17272f;padding:12px;border:1px solid #29414c;border-radius:6px}article img{width:100%;image-rendering:pixelated}h3{font-size:15px;margin:0 0 10px}.micro{font-size:12px;margin:8px 0 0}.table{overflow:auto}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:18px}}</style>
<main><p>ERSRR / LOCAL CPU COMPARISON / 3 OCTOBER 2026</p><h1>Do simpler predictors work better?</h1>
<p>Same training parents. Same evaluation scenes. Same native scoring pixels. No downloads.</p>
<p class="note">Development comparison only. TRAIN training scenes and EVAL evaluation scenes. Only GROUP positive evaluation group(s). This run cannot establish national performance, quantitative upscaling, or emission sources.</p>
<div class="table"><table><thead><tr><th>Method</th><th>Pixel IoU</th><th>Precision</th><th>Recall</th><th>False-positive pixels</th><th>Negative-scene pixel activation</th></tr></thead><tbody>TABLE</tbody></table></div>
<h2>Inspect the actual outputs</h2><p>Orange: prediction. Cyan: reference boundary. Gray stripes: no evaluated model output. All panels show the same native pixels, with no geographic reprojection. These are scored methane-plume candidates, not measured methane concentrations.</p>
<label for="scene">Scene </label><select id="scene"></select><div id="panels" class="grid"></div>
<p>Selection: all positive scenes, plus the two largest false alarms for R5 and each linear model, with duplicates removed. Full-cohort scores above include every evaluation scene.</p>
<p>Fixed cutoff 0.50. The two CPU models use class-balanced logistic fitting, L2 strength 0.01, training-only standardization, and no evaluation-based tuning. Agreement methods combine saved branch decisions without fitting. All 65 source-resolution outputs are retained in the run directory.</p>
<p><a href="results.json">Full results</a> · <a href="models.json">Fitted model parameters</a> · <a href="provenance.json">Input hashes and runtime</a> · <a href="/?experiment=RUNID&catalog=all#compare">Open linear16 in the experiment viewer</a></p></main>
<script>const cases=CASES;const methods=[['input','Sentinel input'],['reference','Reviewed reference'],['r5','R5 saved mean'],['S128','S128 branch'],['linear16','Linear: all 16 features'],['swir1','Linear: SWIR contrast'],['agree3','At least 3 of 4'],['agree4','All 4 agree']];const choice=document.getElementById('scene');cases.forEach((c,i)=>choice.add(new Option(c.label,i)));function render(){const c=cases[Number(choice.value)];document.getElementById('panels').replaceChildren(...methods.map(([key,name])=>{const panel=document.createElement('article'),heading=document.createElement('h3'),image=document.createElement('img'),note=document.createElement('p');heading.textContent=name;image.src='review/'+c.stem+'_'+key+'.png';image.alt=name+' '+c.label;note.className='micro';const m=c.metrics[key];note.textContent=m?'Native IoU '+(m.iou===null?'undefined':(m.iou*100).toFixed(2)+'%')+' | FP '+m.fp.toLocaleString()+' | FN '+m.fn.toLocaleString():'Same image and evaluated support';panel.append(heading,image,note);return panel}));}choice.onchange=render;render();</script></html>'''
    page=page.replace('TRAIN',str(result['training_scenes'])).replace('EVAL',str(result['evaluation_scenes'])).replace('GROUP',str(len(result['positive_evaluation_groups'])))
    page=page.replace('TABLE',table).replace('CASES',json.dumps(options).replace('<','\\u003c')).replace('RUNID',output.name)
    (output/'comparison.html').write_text(page,encoding='utf-8')
    result['review_ids']=[r['id'] for r in chosen]
    experiment=dict(id=output.name,name='Local CPU baseline | all 16 features | October 3',status='DEVELOPMENT COMPARISON ONLY',
                    description='Two CPU linear baselines and fixed branch agreement compared with saved R5 predictions on identical native pixels.',
                    notes='Original evaluation fold reused. No national, source-origin or quantitative upscaling claim. Open Evidence report for paired visual comparisons.',
                    decisionThreshold=.5,metricScope='reused_development_fold',nativeAggregate=result['methods']['linear16'],
                    reportUrl='comparison.html',scenes=scenes)
    return experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=CACHE);parser.add_argument('--r5-dir',type=Path,default=R5)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['fit','all'],default='all')
    parser.add_argument('--minutes',type=float,default=5)
    args=parser.parse_args();output=args.output.resolve()
    require(not output.exists(),'Refusing existing output directory')
    require(0<args.minutes<=5,'Compute cap must be in (0,5] minutes')
    started=time.monotonic();study=Study(args.cache_dir.resolve(),args.r5_dir.resolve(),args.minutes)
    output.mkdir(parents=True);write(output/'protocol.json',PROTOCOL)
    fitted=study.fit();write(output/'models.json',fitted)
    print(json.dumps({'stage':'fit_complete','seconds':round(time.monotonic()-started,2),
                      'sampled_positive_pixels':fitted['sampled_positive_pixels'],'sampled_negative_pixels':fitted['sampled_negative_pixels']}),flush=True)
    experiment=None
    if args.phase=='all':
        result=study.evaluate(fitted['models'],output);experiment=export_review(study,result,output);write(output/'results.json',result)
    for file, before in study.input_files.items(): require(digest(Path(file))==before,'Input changed during execution: '+file)
    provenance=dict(input_sha256=study.input_files,inputs_unchanged=True,cli_sha256=digest(Path(__file__)),
                    elapsed_seconds=round(time.monotonic()-started,3),numpy_version=np.__version__,python=sys.version,
                    network_requests=0,phase=args.phase,training_only_fit=True)
    write(output/'provenance.json',provenance)
    if experiment:
        write(output/'experiment.json.tmp',experiment)
        (output/'experiment.json.tmp').replace(output/'experiment.json')
    print(json.dumps({'result':'completed','output':str(output),'seconds':provenance['elapsed_seconds']}))


if __name__=='__main__':
    try: main()
    except (ValueError,KeyError,FileNotFoundError,TimeoutError) as error:
        print(str(error),file=sys.stderr);sys.exit(2)
