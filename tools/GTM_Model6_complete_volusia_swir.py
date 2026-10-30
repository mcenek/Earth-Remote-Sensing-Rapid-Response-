"""Complete two native SWIR crops in the user-authorized September 30 pass.

Failure cases checked before acquisition: changed earlier receipt; wrong scene;
different bounds/grid/CRS; changed range source; excessive transfer bound;
existing output; no-data; and accidentally interpreting raw DN as reflectance.
No original receipt, budget limit, raster, label or training split is changed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from GTM_Model0_hf_bounded_stage import Budget, LEDGER, save_json
from GTM_Model6_emit_2025_overlap import OUT
from GTM_Model6_repair_native_pair import AttemptBudget, _probe_asset
from acquire_cafo_cog_pilot import crop_asset

PREFLIGHT_SHA='d5bde1c851783d8f76a8101cafd376526902972a5d0496d713899c3f34cdd039'
DEST=OUT/'swir_20260930'


def main():
    source=OUT/'sentinel/preflight.json'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=PREFLIGHT_SHA:
        raise ValueError('Frozen preflight changed')
    prior=json.loads(source.read_text());names=('swir16','swir22')
    if prior['status']!='preflight_complete' or prior['sentinel_id']!='S2B_17RMN_20250601_0_L2A':
        raise ValueError('Unexpected preflight')
    shared=Budget(LEDGER,100*1024*1024); budget=AttemptBudget(shared,3*1024*1024)
    upper=sum(prior['assets'][n]['probe']['crop_range_block_upper_bound'] for n in names)*262144
    if upper>min(3*1024*1024,shared.snapshot()['remaining_bytes']):
        raise ValueError('SWIR transfer estimate exceeds unchanged shared allowance or 3 MiB pass cap')
    if DEST.exists():raise FileExistsError('Existing pass preserved; inspect receipt rather than re-run')
    DEST.mkdir()
    report={'scope':'September 30 bounded five-band input completion; no inference or training',
        'preflight_sha256':PREFLIGHT_SHA,'scene':prior['sentinel_id'],'bbox_projected':prior['bbox_projected'],
        'crs':prior['crs'],'budget_before':shared.snapshot(),'upper_bound_bytes':upper,
        'local_pass_limit_bytes':3*1024*1024,'radiometry':'raw DN only; scale/offset unapproved',
        'status':'started','bands':{}}
    try:
        with rasterio.open(OUT/'sentinel/scl.tif') as ds:
            expected_affine=ds.transform;expected_shape=ds.shape;expected_crs=ds.crs
        for name in names:
            asset=prior['assets'][name]['source']
            probe=_probe_asset(name,asset,budget,tuple(prior['bbox_projected']),expected_crs.to_epsg())
            if probe!=prior['assets'][name]['probe']:raise ValueError('COG header/probe changed')
            arr,affine=crop_asset(asset,budget,tuple(prior['bbox_projected']),expected_crs.to_epsg())
            if affine!=expected_affine or arr.shape!=expected_shape or arr.dtype!=np.uint16:
                raise ValueError('SWIR differs from native SCL grid or uint16 contract')
            path=DEST/f'{name}.tif';meta=asset['raster:bands'][0]
            with rasterio.open(path,'w',driver='GTiff',width=arr.shape[1],height=arr.shape[0],
                count=1,dtype=arr.dtype,crs=expected_crs,transform=affine,nodata=meta['nodata'],compress='deflate') as ds:
                ds.write(arr,1);ds.update_tags(value_semantics='raw DN; no scale or offset applied')
            report['bands'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'source':asset,'shape':list(arr.shape),'transform':list(affine),
                'non_nodata_fraction':float((arr!=meta['nodata']).mean()),
                'min_raw_dn':int(arr.min()),'max_raw_dn':int(arr.max())}
        report['status']='acquired_pending_radiometric_and_target_semantics_QA'
    except Exception as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
    report['budget_after']=shared.snapshot()
    report['this_pass_received_bytes']=shared.received_bytes-report['budget_before']['received_bytes']
    save_json(DEST/'receipt.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='bands'},indent=2))


if __name__=='__main__':main()
