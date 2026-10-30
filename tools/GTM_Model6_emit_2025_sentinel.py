"""Bounded Sentinel context for the frozen EMIT positive; never trains a model.

Preflight reads SCL first. Refuse a crop below 90% clear (SCL classes 4/5/6),
a crop not containing every provider-positive native cell, a changed grid,
or a conservative transfer bound exceeding the persistent/local allowance.
Raw DN is preserved; ambiguous Earth Search offset metadata is not applied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, transform, Resampling
from GTM_Model0_hf_bounded_stage import Budget, LEDGER, save_json
from GTM_Model6_emit_2025_overlap import OUT
from GTM_Model6_emit_2025_stage import ROLES
from GTM_Model6_repair_native_pair import AttemptBudget, _probe_asset
from acquire_cafo_cog_pilot import crop_asset

DEST=OUT/'sentinel'
LIMIT=12*1024*1024


def main(mode):
    selection=json.loads((OUT/'native_selection.json').read_text())['samples'][0]
    row=selection['row'];sample_id=row['id']
    queries=json.loads((OUT/'sentinel_query_summary.json').read_text())['queries']
    query=next(q for q in queries if q['row']['id']==sample_id)
    match=query['items'][0];item=match['item'];assets=item['assets']
    if abs(match['offset_minutes'])>15:raise ValueError('Outside dense-pair timing gate')
    with rasterio.open(OUT/'native'/sample_id/'plumemask.tif') as ds:
        mask=ds.read(1);native_affine=ds.transform;crs=ds.crs
        if str(crs)!=row['crs'] or crs.to_epsg()!=item['properties']['proj:epsg']:
            raise ValueError('Native/manifest/Sentinel CRS mismatch')
    yy,xx=np.where(mask==1)
    if len(xx)==0:raise ValueError('Selected positive has no positive mask pixels')
    x,y=transform('EPSG:4326',crs,[float(row['location_lon'])],[float(row['location_lat'])])
    left,top=native_affine*(int(xx.min()),int(yy.min()))
    right,bottom=native_affine*(int(xx.max()+1),int(yy.max()+1))
    bbox=(math.floor((min(left,x[0])-300)/20)*20,math.floor((min(bottom,y[0])-300)/20)*20,
          math.ceil((max(right,x[0])+300)/20)*20,math.ceil((max(top,y[0])+300)/20)*20)
    DEST.mkdir(exist_ok=True)
    shared=Budget(LEDGER,100*1024*1024)
    starting=json.loads((DEST/'preflight.json').read_text())['budget_before'] if (DEST/'preflight.json').exists() else shared.snapshot()
    spent=shared.charged_bytes-starting['charged_bytes']
    budget=AttemptBudget(shared,max(0,LIMIT-spent))
    report={'sample':sample_id,'sentinel_id':item['id'],'offset_minutes':match['offset_minutes'],
        'scope':'provider-mask-centered development context; not unbiased evaluation or fine-resolution ground truth',
        'crop_rule':'full positive-mask bounding box plus listed location, 300 m margin, outward snapped to 20 m grid',
        'bbox_projected':bbox,'crs':str(crs),'budget_before':starting,'local_ceiling_bytes':LIMIT,
        'mode':mode,'assets':{},'ready_for_training':False,'status':'started',
        'radiometry':'raw DN only; scale/offset metadata retained in receipt and not applied'}
    try:
        if mode=='preflight':
            if (DEST/'preflight.json').exists():raise FileExistsError('Preserve existing preflight')
            scl_probe=_probe_asset('scl',assets['scl'],budget,bbox,crs.to_epsg())
            report['assets']['scl']={'probe':scl_probe,'source':assets['scl']}
            if scl_probe['crop_range_block_upper_bound']*262144>budget.limit_bytes-budget.charged_bytes:
                raise ValueError('SCL transfer alone exceeds local budget')
            scl,affine=crop_asset(assets['scl'],budget,bbox,crs.to_epsg())
            positive=np.zeros(scl.shape,dtype=np.uint8)
            reproject(mask,positive,src_transform=native_affine,src_crs=crs,
                      dst_transform=affine,dst_crs=crs,resampling=Resampling.nearest)
            clear=np.isin(scl,(4,5,6))
            report.update(scl_counts={str(v):int((scl==v).sum()) for v in np.unique(scl)},
                          clear_fraction=float(clear.mean()),positive_area_clear_fraction=float(clear[positive==1].mean()))
            write_raster('scl',scl,affine,crs,assets['scl'],report)
            if min(report['clear_fraction'],report['positive_area_clear_fraction'])<.9:
                raise ValueError('Local crop or annotated plume area is below 90% strict clear; stop spectral acquisition')
            for name in ROLES[1:]:
                report['assets'][name]={'source':assets[name],
                    'probe':_probe_asset(name,assets[name],budget,bbox,crs.to_epsg())}
            report['spectral_upper_bound_bytes']=sum(report['assets'][name]['probe']['crop_range_block_upper_bound'] for name in ROLES[1:])*262144
            report['rgb_upper_bound_bytes']=sum(report['assets'][name]['probe']['crop_range_block_upper_bound'] for name in ('red','green','blue'))*262144
            report['status']='preflight_complete'
        else:
            prior=json.loads((DEST/'preflight.json').read_text())
            if prior['status']!='preflight_complete' or list(prior['bbox_projected'])!=list(bbox):
                raise ValueError('Successful unchanged preflight required')
            if (DEST/'acquisition.json').exists():raise FileExistsError('Preserve acquisition receipt')
            names=ROLES[1:] if mode=='acquire-all' else ('red','green','blue')
            upper=prior['spectral_upper_bound_bytes'] if mode=='acquire-all' else prior['rgb_upper_bound_bytes']
            if upper>min(budget.limit_bytes,shared.snapshot()['remaining_bytes']):
                raise ValueError('Conservative range-block bound exceeds remaining allowance')
            report['preflight_sha256']=hashlib.sha256((DEST/'preflight.json').read_bytes()).hexdigest()
            for name in names:
                report['assets'][name]={'source':assets[name],'probe':prior['assets'][name]['probe']}
                arr,affine=crop_asset(assets[name],budget,bbox,crs.to_epsg())
                write_raster(name,arr,affine,crs,assets[name],report)
            report['status']='acquired_pending_visual_review'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
    report['budget_after']=shared.snapshot()
    path=DEST/('preflight.json' if mode=='preflight' else 'acquisition.json')
    if path.exists():raise FileExistsError('Do not overwrite original receipt')
    save_json(path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='assets'},indent=2))


def write_raster(name,arr,affine,crs,asset,report):
    path=DEST/f'{name}.tif'
    if path.exists():raise FileExistsError(path)
    meta=asset['raster:bands'][0]
    if abs(float(meta['spatial_resolution'])-affine.a)>1e-6:raise ValueError('Catalog/native pixel spacing conflict')
    with rasterio.open(path,'w',driver='GTiff',width=arr.shape[1],height=arr.shape[0],count=1,
                       dtype=arr.dtype,crs=crs,transform=affine,nodata=meta['nodata'],compress='deflate') as ds:
        ds.write(arr,1)
        ds.update_tags(acquisition_scope='development-only raw DN; see acquisition receipt')
    report['assets'][name].update(path=str(path),shape=list(arr.shape),transform=list(affine),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_dn=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','acquire-all','acquire-rgb'])
    main(parser.parse_args().mode)
