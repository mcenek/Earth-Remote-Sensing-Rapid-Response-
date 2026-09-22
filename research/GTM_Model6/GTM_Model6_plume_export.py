"""Export genuine plume predictions to the local viewer, with shared georeferencing.
EMIT references are positive-only: unknown exterior never clips prediction display.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, transform_bounds, transform, Resampling
from rasterio.transform import Affine, from_bounds, array_bounds
from PIL import Image, ImageDraw
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[2]


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(p,d):p.write_text(json.dumps(d,separators=(',',':'),allow_nan=False),encoding='utf-8')


def rgb_image(a):
    lo,hi=np.percentile(a,[2,98],axis=(0,1));return (np.clip((a-lo)/np.maximum(hi-lo,1),0,1)*255).astype('uint8')


def geometry(row,shape):
    native=Affine(*row['affine']);b=array_bounds(*shape,native)
    mb=transform_bounds(row['crs'],'EPSG:3857',*b,densify_pts=21)
    w,s,e,n=transform_bounds('EPSG:3857','EPSG:4326',*mb)
    return native,from_bounds(*mb,160,160),[[s,w],[n,e]]


def warp(a,row,native,dst,method=Resampling.nearest,nodata=None):
    out=np.full((160,160),np.nan if nodata is not None else 0,dtype='float32')
    reproject(a.astype('float32'),out,src_transform=native,src_crs=row['crs'],dst_transform=dst,dst_crs='EPSG:3857',
              src_nodata=nodata,dst_nodata=nodata,resampling=method)
    return out


def rgba(value,valid,kind):
    out=np.zeros((*value.shape,4),dtype='uint8')
    if kind=='probability':
        out[:,:,:3]=np.stack([255*value,135*value,45+100*(1-value)],-1).clip(0,255).astype('uint8');out[:,:,3]=np.where(valid,220,0)
    else:
        on=valid&(value>0);out[on]=[0,225,230,120] if kind=='truth' else [255,135,35,210]
    return out


def candidates(p,valid,row,threshold):
    components,n=label((p>=threshold)&valid);out=[];aff=Affine(*row['affine'])
    for k in range(1,n+1):
        mask=components==k;count=int(mask.sum())
        if count<100:continue
        index=np.argmax(np.where(mask,p,-1));r,c=np.unravel_index(index,p.shape);x,y=aff*(c+.5,r+.5)
        lon,lat=transform(row['crs'],'EPSG:4326',[x],[y])
        out.append({'lat':lat[0],'lon':lon[0],'score':float(p[r,c]),'pixels':count,'label':'Predicted plume hotspot; emission origin unverified'})
    return sorted(out,key=lambda c:-c['score'])


def montage(bundle,scenes,indices,title,name):
    folder=bundle/'montages';folder.mkdir(exist_ok=True)
    for page,start in enumerate(range(0,len(indices),8)):
        group=indices[start:start+8];sheet=Image.new('RGB',(960,80+len(group)*270),'#f5f7fa');d=ImageDraw.Draw(sheet)
        d.text((12,8),title,fill='#16212b');d.text((12,28),'Sentinel RGB                 Reference overlay            Prediction mask             Probability (fixed 0..1)',fill='#16212b')
        for j,idx in enumerate(group):
            s=scenes[idx];rgb=Image.open(bundle/s['rgb']).convert('RGBA').resize((230,230))
            for col,key in enumerate([None,'truthImage','predictionImage','probabilityImage']):
                im=rgb.copy()
                if key:im=Image.alpha_composite(im,Image.open(bundle/s[key]).convert('RGBA').resize((230,230)))
                sheet.paste(im.convert('RGB'),(12+col*238,60+j*270))
            d.text((12,294+j*270),s['name'][:120],fill='#16212b')
        sheet.save(folder/f'{name}_{page:02d}.png')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-id',required=True);args=ap.parse_args()
    run=ROOT/'outputs/GTM_Model6'/args.run_id;m=json.loads((run/'GTM_Model6_plume_manifest.json').read_text(encoding='utf-8'));result=json.loads((run/'GTM_Model6_plume_results.json').read_text(encoding='utf-8'))
    if not result['complete']:raise ValueError('Incomplete experiment')
    z=np.load(run/'GTM_Model6_plume_inputs.npz');records={r['index']:r for r in result['records']};rows=m['observations']
    methods=['S16','S32','S64','S128','mean','spatial_prior']
    if m.get('variant')=='spectral_evidence_bounded_refinement':methods+=['spectral_baseline']
    byfold={}
    for c in result['models']:byfold.setdefault(c['fold'],[]).append(c)
    for method in methods:
        folder=ROOT/'outputs/GTM_viewer_bundles'/f'{args.run_id}_{method}';folder.mkdir(parents=True,exist_ok=False);scenes=[];features=[]
        for i,row in enumerate(rows):
            rec=records[i];pfile=run/f'GTM_Model6_plume_prediction_{i}.npz';p=np.load(pfile)[method]
            valid=z[f'{i}_valid'].astype(bool);known=z[f'{i}_known'].astype(bool);gt=z[f'{i}_target'];rgb=z[f'{i}_rgb'];threshold=rec['thresholds'][method]
            native,dst,bounds=geometry(row,p.shape)
            v=warp(valid,row,native,dst)>.5
            pd=warp(np.where(valid,p,np.nan),row,native,dst,Resampling.bilinear,np.nan)
            td=warp(np.where(known,gt,np.nan),row,native,dst,Resampling.nearest,np.nan)
            v &= np.isfinite(pd);pd=np.nan_to_num(pd)
            rgbd=np.stack([warp(rgb[:,:,b],row,native,dst,Resampling.bilinear) for b in range(3)],-1)
            stem=f'{i:04d}';Image.fromarray(rgb_image(rgbd)).save(folder/f'{stem}_rgb.png')
            truth=rgba(np.nan_to_num(td),np.isfinite(td)&v,'truth');pred=rgba(pd>=threshold,v,'prediction');heat=rgba(pd,v,'probability')
            Image.fromarray(truth).save(folder/f'{stem}_truth.png');Image.fromarray(pred).save(folder/f'{stem}_prediction.png');Image.fromarray(heat).save(folder/f'{stem}_probability.png')
            grid={'width':160,'height':160,'crs':'EPSG:3857','probability':np.round(pd,5).ravel().tolist(),
                  'truth':[int(t) if np.isfinite(t) and good else None for t,good in zip(td.ravel(),v.ravel())],
                  'valid':v.astype('uint8').ravel().tolist()}
            write(folder/f'{stem}_grid.json',grid)
            cp=[c for c in byfold[rec['fold']] if method=='mean' or method=='S'+str(c['window'])]
            points=candidates(p,valid,row,threshold)
            for point in points:features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[point['lon'],point['lat']]},'properties':{'scene':row['id'],**point}})
            labeltext='Reviewed plume' if row['label']=='PLUME' else 'Reviewed no plume' if row['label']=='NO_PLUME' else 'EMIT positive support'
            sid=row['id'];title=f"{row['dataset']} | {labeltext} | {row['date'][:10]} | {sid[-8:]}"
            notes=('Held-location development output; threshold frozen at 0.50 before fitting. '+
                   ('EMIT exterior is unknown: support recall only; no verified FP/precision or IoU. Timing offset '+str(round(row['lag_hours'],2))+' h. ' if row['dataset']=='EMIT' else 'MARS reviewed plume/background reference. ')+
                   '10m source file grid; Sentinel SWIR resolves about 20m and EMIT about 60m. Map reprojection is for display; report scores use source grid. Hotspots are candidate plume peaks, not verified emission origins.')
            scenes.append({'id':sid,'name':title,'task':'plume_segmentation','location':f"{row['lat']:.3f}, {row['lon']:.3f} | United States",'date':row['date'][:10],
              'dataset':row['dataset'],'label':row['label'],'positiveOnlyReference':not row['fully_labeled'],'bounds':bounds,
              'rgb':f'{stem}_rgb.png','truthImage':f'{stem}_truth.png','predictionImage':f'{stem}_prediction.png','probabilityImage':f'{stem}_probability.png',
              'grid':f'{stem}_grid.json','predictionAvailable':True,'decisionThreshold':threshold,
              'truthLabel':'EMIT catalog plume outline (positive only)' if row['dataset']=='EMIT' else 'MARS reviewed plume mask',
              'truthNote':'Unknown exterior is excluded from comparison' if row['dataset']=='EMIT' else 'Reviewed 0/1 mask on clear Sentinel support',
              'referenceDate':row.get('emit_date',row['date'])[:10],'notes':notes,'source':row['source'],
              'checkpointHash':', '.join(c['sha256'] for c in cp) or 'No trained weights: baseline','predictionHash':digest(pfile),
              'split':f"Outer held partition {rec['fold']}; group {rec['group']}; historical development data",
              'resolution':'200x200 source grid (10m sampling) / 160x160 EPSG:3857 display','nativeMetrics':rec['metrics'][method],
              'candidatePoints':points[:5],'candidateCount':len(points)})
        # EMIT first makes the project target easy to inspect, without hiding failures.
        scenes.sort(key=lambda s:(s['dataset']!='EMIT',s['label']=='NO_PLUME',s['date'],s['id']))
        stage='Evidence constrained' if 'evidence' in args.run_id else 'Unconstrained (failed pilot)'
        exp={'id':f'{args.run_id}_{method}','name':f'Model6 | {stage} | {method}','lifecycle':'experimental',
             'status':'EXPLORATORY | actual held-location outputs, not validated nationally','description':'Independent Sentinel-only plume segmentation; EMIT positive supervision + reviewed US controls.',
             'notes':'Fixed 0.50 threshold. All held cases included. EMIT coverage is not verified specificity. See report and background failures before using candidate locations.',
             'decisionThreshold':.5,'scenes':scenes,'reportUrl':'GTM_Model6_plume_report.md'}
        write(folder/'experiment.json',exp);write(folder/'GTM_Model6_candidate_hotspots.geojson',{'type':'FeatureCollection','features':features})
        report=run/'GTM_Model6_plume_report.md'
        (folder/report.name).write_text(report.read_text(encoding='utf-8') if report.exists() else 'Visual review and native aggregate results pending. This model is not promoted.',encoding='utf-8')
        if method=='mean':
            for dataset in ['EMIT','MARS']:
                ids=[i for i,s in enumerate(scenes) if s['dataset']==dataset and (dataset=='EMIT' or s['label']=='PLUME')]
                montage(folder,scenes,ids,f'{stage} | every {dataset} positive case; held-location output',dataset)
            neg=[i for i,s in enumerate(scenes) if s['label']=='NO_PLUME'];montage(folder,scenes,neg[:8],f'{stage} | first eight chronological negative controls','negative_controls')
        print(json.dumps({'method':method,'scenes':len(scenes),'bundle':str(folder)}),flush=True)
    z.close()


if __name__=='__main__':main()
