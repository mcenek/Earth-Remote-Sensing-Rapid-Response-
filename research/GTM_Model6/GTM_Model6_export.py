"""Export measured held-location predictions with georeferenced display layers."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GTM_Model6_reconstruction import ROOT, inventory, degrade, write_json, digest
import json

def rgba(a,valid,path,lo=-500,hi=2500,residual=False):
    cmap=plt.get_cmap('RdBu_r' if residual else 'viridis')
    if residual: lo,hi=-1000,1000
    out=cmap(np.clip((a-lo)/(hi-lo),0,1),bytes=True)
    out[:,:,3]=np.where(valid,255,0).astype('uint8')
    Image.fromarray(out).save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',required=True); args=ap.parse_args()
    run=ROOT/'outputs/GTM_Model6'/args.run_id
    results=json.loads((run/'GTM_Model6_results.json').read_text())
    _,items=inventory(); lookup={r['id']:(r,s,t,v) for r,s,t,v in items}
    z=np.load(run/'GTM_Model6_predictions.npz',allow_pickle=False)
    manifest=json.loads((run/'GTM_Model6_manifest.json').read_text())
    if any(digest(ROOT/r['source'])!=r['sha256'] for r in manifest['observations']): raise ValueError('Source hash changed since training')
    all_rows=results['observations']; summary={}
    for variant in ['emit_only','sentinel_emit']:
        records=[x for x in all_rows if x['variant']==variant]
        grouped={g:np.mean([r['metrics']['bilinear']['mae']-r['metrics'][variant]['mae'] for r in records if r['group']==g]) for g in sorted({r['group'] for r in records})}
        rng=np.random.default_rng(61019); values=np.array(list(grouped.values()))
        samples=rng.choice(values,(10000,len(values)),replace=True).mean(1)
        summary[variant]={'macro_observation_mae':results['summary_macro_observation_mae'][variant],
          'group_mean_mae_improvement_vs_bilinear':grouped,'group_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),
          'groups_improved':int((values>0).sum()),'groups_total':len(values)}
    write_json(run/'GTM_Model6_group_summary.json',summary)
    limitations='Engineering diagnostic on legacy resampled observations; not native EMIT or validated 20 m methane. Negative-scene performance untested. Held-site development results, not new confirmation. Display is reprojected; native-grid metrics are in the report. Shared display range −500 to 2500 legacy CH4 units; units expected ppm*m but unverified.'
    bundle_paths=[]
    for variant in ['emit_only','sentinel_emit']:
      for fold in range(5):
        bundle=ROOT/'outputs/GTM_viewer_bundles'/f'{args.run_id}_{variant}_f{fold}'
        bundle.mkdir(parents=True,exist_ok=True); scenes=[]
        records=[x for x in all_rows if x['variant']==variant and x['fold']==fold]
        checkpoint=next(x for x in results['trained'] if x['variant']==variant and x['fold']==fold)
        for rec in records:
            sid=rec['id']; row,s,t,v=lookup[sid]
            pred=z[f'f{fold}_{variant}_{sid}']; target=t.numpy()[0,0]; valid=v.numpy()[0,0]>0
            base=degrade(t,v)[0].numpy()[0,0]
            with rasterio.open(ROOT/row['source']) as src:
                rgb=src.read([3,2,1]).transpose(1,2,0)
                lo=np.percentile(rgb,2,axis=(0,1)); hi=np.percentile(rgb,98,axis=(0,1)); rgb=np.clip((rgb-lo)/np.maximum(hi-lo,1),0,1)
                dst_bounds=transform_bounds(src.crs,'EPSG:3857',*src.bounds)
                src_transform=src.transform*src.transform.scale(8,8)
                dst_transform=from_bounds(*dst_bounds,32,32)
                def warp(array,mask=True):
                    out=np.full((32,32),np.nan,dtype='float32')
                    reproject(np.where(valid,array,np.nan).astype('float32') if mask else array.astype('float32'),out,src_transform=src_transform,src_crs=src.crs,dst_transform=dst_transform,dst_crs='EPSG:3857',src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear)
                    return out
                display={k:warp(a) for k,a in [('observed',target),('coarse',base),('prediction',pred),('residual',pred-target)]}
                mask=np.isfinite(display['observed'])&np.isfinite(display['prediction'])
                warped_rgb=np.zeros((256,256,3),dtype='float32')
                for band in range(3): reproject(rgb[:,:,band].astype('float32'),warped_rgb[:,:,band],src_transform=src.transform,src_crs=src.crs,dst_transform=from_bounds(*dst_bounds,256,256),dst_crs='EPSG:3857',resampling=Resampling.bilinear)
                w,south,e,north=transform_bounds('EPSG:3857','EPSG:4326',*dst_bounds)
            stem=sid; Image.fromarray(np.rint(warped_rgb*255).astype('uint8')).save(bundle/f'{stem}_rgb.png')
            for key,a in display.items(): rgba(a,mask,bundle/f'{stem}_{key}.png',residual=key=='residual')
            grid={'task':'methane_reconstruction','width':32,'height':32,'units':'legacy CH4 units (ppm*m unverified)','display':{'min':-500,'max':2500},'valid':mask.astype('uint8').ravel().tolist()}
            grid.update({k:[float(a) if good else None for a,good in zip(arr.ravel(),mask.ravel())] for k,arr in display.items()})
            write_json(bundle/f'{stem}_grid.json',grid)
            scenes.append({'id':sid,'name':row['us_interior']+' · '+row['sentinel_time'][:10]+' · held out',
              'task':'methane_reconstruction','units':grid['units'],'bounds':[[south,w],[north,e]],
              'location':row['us_interior']+' · US interior diagnostic','date':row['sentinel_time'][:10],
              'referenceDate':row['emit_time'][:10],'rgb':f'{stem}_rgb.png','observedImage':f'{stem}_observed.png',
              'coarseImage':f'{stem}_coarse.png','predictionImage':f'{stem}_prediction.png','residualImage':f'{stem}_residual.png',
              'grid':f'{stem}_grid.json','predictionAvailable':True,'truthLabel':'Observed legacy aggregate (withheld target)',
              'truthNote':'8x aggregation of resampled export; not a native EMIT or fine-resolution reference',
              'source':row['source']+' SHA256 '+row['sha256'],'checkpointHash':checkpoint['sha256'],
              'predictionHash':digest(run/'GTM_Model6_predictions.npz'),
              'split':f'Location group {row["group"]}; held out of fold {fold}; training partition {fold}',
              'resolution':'32x32 aggregated target / 16x16 degraded input; ~80 m / ~160 m nominal, not recovered native EMIT grids',
              'notes':limitations+f' Sentinel/EMIT lag {row["pairing_lag_days"]:.2f} days. Native MAE {rec["metrics"][variant]["mae"]:.2f}; bilinear {rec["metrics"]["bilinear"]["mae"]:.2f}.',
              'nativeMetrics':rec['metrics']})
        exp={'id':f'GTM_Model6_{variant}_f{fold}','name':f'Model6 · {"EMIT only" if variant=="emit_only" else "Sentinel + EMIT"} · fold {fold+1}',
          'lifecycle':'diagnostic','status':'ACTUAL HELD-LOCATION OUTPUT · engineering diagnostic, not promoted',
          'description':'S32 residual U-Net, 16→32 controlled reconstruction of legacy aggregates; eight training transforms.',
          'notes':limitations,'scenes':scenes,'reportUrl':'GTM_Model6_run_report.md'}
        write_json(bundle/'experiment.json',exp); bundle_paths.append(bundle)
    # Fixed, complete visual cohort: one predetermined held-out fold per observation.
    fig,axes=plt.subplots(len(items),5,figsize=(15,2.4*len(items)),layout='constrained')
    for i,(row,s,t,v) in enumerate(items):
        fold=(row['partition']+1)%5; valid=v.numpy()[0,0]>0; y=t.numpy()[0,0]; base=degrade(t,v)[0].numpy()[0,0]
        p=z[f'f{fold}_emit_only_{row["id"]}']; q=z[f'f{fold}_sentinel_emit_{row["id"]}']
        for j,(a,title) in enumerate([(y,'Observed aggregate'),(base,'Bilinear'),(p,'EMIT-only U-Net'),(q,'Sentinel + EMIT'),(p-y,'EMIT residual')]):
            axes[i,j].imshow(np.where(valid,a,np.nan),cmap='RdBu_r' if j==4 else 'viridis',vmin=-1000 if j==4 else -500,vmax=1000 if j==4 else 2500)
            axes[i,j].set_title(title+'\n'+row['us_interior']+f' held out of fold {fold+1}',fontsize=8); axes[i,j].axis('off')
    fig.suptitle('Actual predictions — every eligible observation; shared scales, missing support blank. Engineering only.')
    fig.savefig(run/'GTM_Model6_all_heldout_outputs.png',dpi=90); plt.close(fig)
    lines=['# Model6 reconstruction diagnostic','',limitations,'',
      '13 observations in 11 spatial groups; five group rotations, one partition trains and four evaluate. Fixed 600 updates per model. All held-out cases exported, no best-case filtering.',
      '', '| Method | MAE | Bilinear MAE | Groups improved | Group bootstrap MAE-gain interval |','|---|---:|---:|---:|---|']
    for variant,x in summary.items():
        m=x['macro_observation_mae']; lines.append(f'| {variant} | {m[variant]:.3f} | {m["bilinear"]:.3f} | {x["groups_improved"]}/{x["groups_total"]} | {x["group_bootstrap_95_percentile"]} |')
    lines+=['','Bootstrap resamples the 11 location-group mean gains, not pixels or repeated predictions. Small-cohort diagnostic only; it does not repair missing independent truth or missing negatives.',
      '', 'The 16/64/128 branches, learned fusion, fine-grid upscaling and Gaussian source localization were not run: original EMIT grids and representative source/negative supervision are missing. Tiny-set memorization is a plumbing test and is excluded from these held-out figures.',
      '', 'Visual review is recorded separately in the research workspace. No model is promoted by this report.']
    report='\n'.join(lines)+'\n'; (run/'GTM_Model6_run_report.md').write_text(report,encoding='utf-8')
    for bundle in bundle_paths: (bundle/'GTM_Model6_run_report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'bundles':len(bundle_paths),'summary':summary}))

if __name__=='__main__':main()
