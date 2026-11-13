"""Package verified local model exports for the GTM static viewer; no inference/downloads."""
import json, shutil
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from PIL import Image
site=Path(__file__).resolve().parents[1]
root=site.parent/'ersrr-ordinal-win/outputs/GTM_Model1_viewer_export'
assets=site/'dist/assets'; assets.mkdir(exist_ok=True)
registry_path=site/'dist/GTM_Model0_data.js'
registry=json.loads(registry_path.read_text(encoding='utf-8').split('=',1)[1].strip().rstrip(';'))
manifest=json.loads((root/'manifest.json').read_text())
variants=json.loads((root/'model_variants.json').read_text())['variants']
experiments=[]; stats=[]
for label,variant in variants.items():
    number=1 if label=='compact' else 5
    exp={'id':f'GTM_Model{number}','name':('Compact ResUNet' if label=='compact' else 'Physics-feature ResUNet'), 'status':'Real checkpoint replay · diagnostic examples','decisionThreshold':variant['decision_threshold'],'description':variant['feature_contract'],'notes':'Two legacy validation-folder scenes in Texas. Fresh Keras/Torch inference, not a rerun of a frozen benchmark. Reference acquisition dates and product-level metadata are unverified; geographic holdout independence has not been established. These examples cannot validate US-wide performance.','reportUrl':'GTM_Model0_visual_review.md','scenes':[]}
    for row in manifest['cases']:
        sid=row['sample_id']; folder=root/sid; prefix=f'GTM_Model{number}_{sid}'
        with rasterio.open(folder/f'display_rgb_{label}_3857.tif') as r:
            rgb=r.read(); bounds=transform_bounds(r.crs,'EPSG:4326',*r.bounds); transform=r.transform; crs=r.crs
        Image.fromarray(np.moveaxis(rgb,0,-1)).save(assets/f'{prefix}_RGB.png')
        g=json.loads((folder/f'overlay_{label}_3857.json').read_text())
        assert all(len(g[k])==g['width']*g['height'] for k in ['probability','truth','valid'])
        p=np.asarray(g['probability']); truth=np.array([np.nan if v is None else v for v in g['truth']]); valid=np.asarray(g['valid'],bool)
        rgba=np.zeros((g['height'],g['width'],4),dtype='uint8'); t=truth.reshape(g['height'],g['width']); rgba[t==1]=[0,225,230,230]
        Image.fromarray(rgba).save(assets/f'{prefix}_reference.png')
        with rasterio.open(folder/'emit_reference.tif') as r:
            em=np.full((g['height'],g['width']),np.nan,dtype='float32')
            reproject(r.read(1),em,src_transform=r.transform,src_crs=r.crs,src_nodata=np.nan,dst_transform=transform,dst_crs=crs,dst_nodata=np.nan,resampling=Resampling.nearest)
        rgba=np.zeros((*em.shape,4),dtype='uint8'); support=np.isfinite(em); rgba[support,:3]=[0,225,230]; rgba[support,3]=np.clip(40+np.clip(em[support],0,3000)/3000*215,40,255).astype('uint8')
        Image.fromarray(rgba).save(assets/f'{prefix}_EMIT.png')
        shutil.copyfile(folder/f'overlay_{label}_3857.json',assets/f'{prefix}_grid.json')
        s={'id':sid,'name':sid.split('_')[-1]+' · '+row['sentinel_acquisition_date'][:4]+'-'+row['sentinel_acquisition_date'][4:6]+'-'+row['sentinel_acquisition_date'][6:],'location':'Texas, United States','date':row['sentinel_acquisition_date'][:4]+'-'+row['sentinel_acquisition_date'][4:6]+'-'+row['sentinel_acquisition_date'][6:],'bounds':[[bounds[1],bounds[0]],[bounds[3],bounds[2]]], 'rgb':f'assets/{prefix}_RGB.png','emit':f'assets/{prefix}_EMIT.png','truthImage':f'assets/{prefix}_reference.png','grid':f'assets/{prefix}_grid.json','predictionAvailable':True,'checkpointHash':variant['artifact_sha256'],'source':sid+' · SHA256 '+row['source_sha256'],'resolution':'Approx. 20 m native; 128×128 Mercator display (~40 m)','split':'Legacy validation folder; not independently certified held-out sites','referenceDate':'Unknown; not encoded in source filename','rgbNote':'Observed B4/B3/B2; 2–98% display stretch','emitNote':'Paired EMIT band; cyan intensity 0–3000 ppm·m, clipped for display','truthLabel':'EMIT-derived reference mask','truthNote':'EMIT >300 ppm·m where observed; missing pixels are unknown','notes':'Diagnostic replay using four native 128×128 windows; no overlap blending. Artifact threshold '+str(variant['decision_threshold'])+'. Approximately 20 m source grid; physical resolution was not resampled to an exact metric grid. EMIT date and Sentinel product level unverified. Most pixels lack EMIT support; no-data is not a negative. Output is plume probability, not methane concentration.'}
        exp['scenes'].append(s)
        known=valid & np.isfinite(truth); pred=p>=variant['decision_threshold']; tp=int((pred&known&(truth==1)).sum()); fp=int((pred&known&(truth==0)).sum()); fn=int((~pred&known&(truth==1)).sum()); tn=int((~pred&known&(truth==0)).sum())
        stats.append({'model':label,'scene':sid,'threshold':variant['decision_threshold'],'predicted_fraction':float(pred[valid].mean()),'reference_support':float(known.mean()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'iou':tp/(tp+fp+fn) if tp+fp+fn else None})
    experiments.append(exp)
registry['experiments']=experiments+[e for e in registry['experiments'] if e['id'] not in ['GTM_Model1','GTM_Model5']]
registry_path.write_text('window.GTM_DATA = '+json.dumps(registry,indent=2)+';\n',encoding='utf-8')
(assets/'GTM_Model1_display_review_stats.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(json.dumps(stats,indent=2))
