"""Export actual native-grid cross-acquisition reconstruction comparisons."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
from rasterio.transform import Affine,array_bounds,from_bounds
from rasterio.warp import reproject,Resampling,transform_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GTM_Model6_reconstruction import ROOT,write_json,digest
from GTM_Model6_export import rgba

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);args=ap.parse_args()
    run=ROOT/'outputs/GTM_Model6'/args.run_id
    result=json.loads((run/'GTM_Model6_native_results.json').read_text());manifest=json.loads((run/'GTM_Model6_native_manifest.json').read_text())
    methods=['pred16','pred32','pred64','pred128','uniform_mean','nearest'];names=['S16','S32','S64','S128','Four-branch mean','Nearest baseline']
    records=result['observations'];summary={}
    for method in methods+['bilinear','nearest','constant']:
        summary[method]={}
        for subset in ['all_observed','vetted_plume_support','other_observed_not_negative_labels']:
            groups=[]
            for fold in range(3):
                eligible=[r['metrics'][method][subset]['mae'] for r in records if r['fold']==fold and r['metrics'][method][subset]['mae'] is not None]
                groups.append(float(np.mean(eligible)) if eligible else None)
            summary[method][subset]={'per_acquisition_mae':groups,'macro_acquisition_mae':float(np.mean([g for g in groups if g is not None]))}
    write_json(run/'GTM_Model6_native_summary.json',summary)
    limits='Native EMIT-only controlled 2x reconstruction, nominal 120→60 m. Three US acquisitions; held out by entire acquisition. No Sentinel input, no learned fusion, no 20 m validation, no source localization. Background observations are not verified negative emissions labels. Colors show enhancement, not plume probability.'
    report=['# Native Model6 multiscale reconstruction','',limits,'',
      '| Method | All-observed MAE | Vetted plume-support MAE |','|---|---:|---:|']
    for method,entry in summary.items():report.append(f'| {method} | {entry["all_observed"]["macro_acquisition_mae"]:.2f} | {entry["vetted_plume_support"]["macro_acquisition_mae"]:.2f} |')
    report+=['','Units: ppm·m. Means first average windows within each acquisition, then average the three acquisitions. Windows and pixels are not independent replicates. Each acquisition contributes one catalog-peak window plus four seeded random >95%-observed windows. Plume-support metrics use the aligned CH4PLM vetted footprint; other observations do not become NO_PLUME labels.',
      '', 'Four independently trained U-Nets receive the same degraded observations. Each uses its own 16/32/64/128 context and overlapping Hann-weighted tiling. The combined result is a fixed uniform mean, not a trained merger. All checkpoints use 300 fixed updates, eight training transforms, and training-only scaling.',
      '', 'An aligned uncertainty/sensitivity/ENH validity intersection excludes nodata; original signed retrieval values, including negative retrieval noise, are retained. Reported metrics use native arrays, not resampled web display grids. Fine detail below the native observation resolution is not evaluated.']
    report_text='\n'.join(report)+'\n';(run/'GTM_Model6_native_report.md').write_text(report_text,encoding='utf-8')
    for method,name in zip(methods,names):
        bundle=ROOT/'outputs/GTM_viewer_bundles'/f'{args.run_id}_{method}';bundle.mkdir(parents=True,exist_ok=True);scenes=[]
        for row in records:
            z=np.load(run/row['prediction_file'],allow_pickle=False);target=z['target'];pred=z[method];valid=z['valid']>0
            source_affine=Affine(*row['affine']);bbox=transform_bounds(row['crs'],'EPSG:3857',*array_bounds(128,128,source_affine));dst=from_bounds(*bbox,128,128)
            def warp(a):
                out=np.full((128,128),np.nan,dtype='float32')
                reproject(np.where(valid,a,np.nan).astype('float32'),out,src_transform=source_affine,src_crs=row['crs'],dst_transform=dst,dst_crs='EPSG:3857',src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
                return out
            layers={k:warp(a) for k,a in [('observed',target),('coarse',z['nearest']),('prediction',pred),('residual',pred-target)]}
            mask=np.isfinite(layers['observed'])&np.isfinite(layers['prediction']);stem=row['id']
            for key,a in layers.items():rgba(a,mask,bundle/f'{stem}_{key}.png',lo=-500,hi=5000,residual=key=='residual')
            grid={'task':'methane_reconstruction','width':128,'height':128,'units':'ppm·m','display':{'min':-500,'max':5000,'residualMax':1000},'valid':mask.astype('uint8').ravel().tolist()}
            grid.update({k:[float(v) if good else None for v,good in zip(a.ravel(),mask.ravel())] for k,a in layers.items()});write_json(bundle/f'{stem}_grid.json',grid)
            west,south,east,north=transform_bounds('EPSG:3857','EPSG:4326',*bbox)
            source=next(s for s in manifest['observations'] if s['stamp']==row['stamp']);fold=result['folds'][row['fold']]
            checks=fold['checkpoints'] if method=='uniform_mean' else [c for c in fold['checkpoints'] if str(c['window'])==method.removeprefix('pred')]
            place='Georgia' if row['stamp'].startswith('202410') else 'Texas'
            scenes.append({'id':stem,'name':f'{place} {row["stamp"][:8]} · {"catalog plume" if row["index"]==0 else "random window "+str(row["index"])}',
              'task':'methane_reconstruction','units':'ppm·m','coarseLabel':'Nearest baseline · same coarse measurements','predictionLabel':'Nearest interpolation baseline' if method=='nearest' else 'Actual reconstruction','bounds':[[south,west],[north,east]],'location':place+' · held acquisition',
              'date':row['stamp'][:4]+'-'+row['stamp'][4:6]+'-'+row['stamp'][6:8],
              'observedImage':f'{stem}_observed.png','coarseImage':f'{stem}_coarse.png','predictionImage':f'{stem}_prediction.png','residualImage':f'{stem}_residual.png',
              'grid':f'{stem}_grid.json','predictionAvailable':True,'truthLabel':'Withheld native EMIT enhancement','truthNote':'Measured native CH4ENH with uncertainty/sensitivity validity; not finer-resolution truth',
              'resolution':'Original EMIT grid, nominal 60 m; input degraded 2x. Display reprojected to EPSG:3857.',
              'source':source['paths']['enh']+' SHA256 '+source['hashes']['enh'],
              'checkpointHash':'; '.join(c['sha256'] for c in checks),'predictionHash':digest(run/row['prediction_file']),
              'split':f'Entire acquisition held out of fold {row["fold"]+1}; two other US acquisitions train',
              'notes':limits+f' Native-grid MAE {row["metrics"][method]["all_observed"]["mae"]:.2f}; bilinear {row["metrics"]["bilinear"]["all_observed"]["mae"]:.2f}. View range −500 to 5000 ppm·m, clipped for display only.',
              'nativeMetrics':row['metrics'][method]})
        exp={'id':f'GTM_Model6_native_{method}','name':f'Model6 native · {name} · EMIT only','lifecycle':'diagnostic','status':('INTERPOLATION BASELINE' if method=='nearest' else 'ACTUAL NATIVE-GRID RECONSTRUCTION')+' · three held acquisitions',
          'description':'Nearest-neighbor interpolation of identical coarse measurements; no learned weights.' if method=='nearest' else 'Independently trained context model; observed enhancement reconstructed from 2x degraded EMIT.' if method!='uniform_mean' else 'Uniform average of four independently trained context models; no learned merger.',
          'notes':limits,'scenes':scenes,'reportUrl':'GTM_Model6_native_report.md',
          'metrics':{'All-observed MAE':round(summary[method]['all_observed']['macro_acquisition_mae'],2),'Plume-support MAE':round(summary[method]['vetted_plume_support']['macro_acquisition_mae'],2)}}
        write_json(bundle/'experiment.json',exp);(bundle/'GTM_Model6_native_report.md').write_text(report_text,encoding='utf-8')
    # Every peak window and every random window reviewed, no best-case selection.
    for subset,label in [(records,'all_windows'),([r for r in records if r['index']==0],'catalog_peaks')]:
        columns=['target','bilinear','nearest','pred16','pred32','pred64','pred128','uniform_mean'];fig,axes=plt.subplots(len(subset),len(columns),figsize=(21,3*len(subset)),squeeze=False,layout='constrained')
        for i,row in enumerate(subset):
            z=np.load(run/row['prediction_file'],allow_pickle=False)
            for j,key in enumerate(columns):
                axes[i,j].imshow(np.where(z['valid']>0,z[key],np.nan),cmap='viridis',vmin=-500,vmax=5000);axes[i,j].axis('off');axes[i,j].set_title(f'{key}\n{row["stamp"][:8]} / {row["kind"]}',fontsize=8)
        fig.suptitle('Actual native-grid held-acquisition outputs · same −500…5000 ppm·m scale; no finer-truth claim')
        fig.savefig(run/f'GTM_Model6_native_{label}.png',dpi=100 if label=='catalog_peaks' else 70);plt.close(fig)
    for source,name in [(Path(__file__),'GTM_Model6_export_source.py'),(Path(__file__).with_name('GTM_Model6_native.py'),'GTM_Model6_native_source.py'),(Path(__file__).with_name('GTM_Model6_reconstruction.py'),'GTM_Model6_parent_source.py')]:shutil.copy2(source,run/name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
