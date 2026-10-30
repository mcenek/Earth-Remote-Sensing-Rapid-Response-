"""Repeatable offline inspection of the two frozen native EMIT intake samples.

Never uses mask nodata=0 as validity: provider zero is shown as unannotated,
not as a verified methane-free pixel. Constant RGB is explicitly unavailable.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import rasterio
from rasterio.warp import transform, reproject, Resampling
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from GTM_Model0_hf_bounded_stage import save_json
from GTM_Model6_emit_2025_overlap import OUT, ROOT


def main():
    selection=json.loads((OUT/'native_selection.json').read_text())
    report={'scope':'provider training observations and annotations; no model prediction',
            'mask_policy':'Raw 0/1 codes preserved. Header nodata=0 conflicts with background encoding. Zero is unannotated, not a verified negative.',
            'retrieval_units':'No physical units declared in downloaded rasters; show stored values only.',
            'samples':[]}
    review=OUT/'review';review.mkdir(exist_ok=True)
    for selected in selection['samples']:
        row=selected['row'];base=OUT/'native'/row['id']; rasters={}; details={}
        for path in sorted(base.glob('*.tif')):
            with rasterio.open(path) as ds:
                arr=ds.read()
                rasters[path.stem]=arr
                details[path.stem]={'shape':list(arr.shape),'dtype':str(arr.dtype),'crs':str(ds.crs),
                    'transform':list(ds.transform),'bounds':list(ds.bounds),'resolution_m':list(ds.res),
                    'nodata':ds.nodata,'units':list(ds.units),'descriptions':list(ds.descriptions),
                    'min':float(np.nanmin(arr)),'max':float(np.nanmax(arr)),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                if path.stem=='plumemask':affine=ds.transform; crs=ds.crs
        ref=details['plumemask']
        for name,meta in details.items():
            if any(meta[k]!=ref[k] for k in ('crs','transform','bounds')) or meta['shape'][1:]!=ref['shape'][1:]:
                raise ValueError(f'{row["id"]}: {name} grid mismatch')
        mask=rasters['plumemask'][0]
        if not set(np.unique(mask)) <= {0,1}: raise ValueError('Unsupported raw mask codes')
        x,y=transform('EPSG:4326',crs,[float(row['location_lon'])],[float(row['location_lat'])])
        c,r=(~affine)*(x[0],y[0]); inside=0<=c<mask.shape[1] and 0<=r<mask.shape[0]
        if not inside: raise ValueError('Manifest point lies outside native raster')
        ys,xs=np.where(mask==1)
        mask_bounds=None
        if len(xs):
            left,top=affine*(int(xs.min()),int(ys.min()))
            right,bottom=affine*(int(xs.max()+1),int(ys.max()+1))
            mask_bounds=[left,bottom,right,top]
        rgb=rasters['rgb'];constant=all(float(np.ptp(b))==0 for b in rgb[:3])
        result={'id':row['id'],'location':row['location_name'],'role':selected['role'],
                'declared_isplume':row['isplume'],'sentinel_matches':selected['sentinel'],
                'rasters':details,'all_native_grids_equal':True,
                'source_coordinate_inside':inside,'location_pixel_col_row':[c,r],
                'positive_pixels':int(len(xs)),'total_pixels':int(mask.size),
                'positive_fraction':float((mask==1).mean()),'positive_bbox_projected':mask_bounds,
                'positive_touches_edge':bool(np.any(mask[[0,-1],:]) or np.any(mask[:,[0,-1]])),
                'rgb_constant':constant,'rgb_status':'unavailable: constant arrays' if constant else 'requires visual review',
                'mask_zero_is_confirmed_negative':False,'ready_for_training':False}
        fig,axes=plt.subplots(1,3,figsize=(15.3,5.7),dpi=130)
        # Same complete 15.36 km raster extent, native cells, no smoothing.
        extent=[0,mask.shape[1]*60/1000,mask.shape[0]*60/1000,0]
        for axis,name,title in zip(axes[:2],('mf','mag1c'),('Matched-filter retrieval','MAG1C retrieval')):
            data=rasters[name][0]
            image=axis.imshow(data,origin='upper',extent=extent,interpolation='nearest',cmap='magma')
            if len(xs): axis.contour((mask==1).astype(float),levels=[.5],colors=['#58eeff'],linewidths=.8,
                                    origin='upper',extent=extent)
            axis.set_title(title+' + provider mask boundary',fontsize=10)
            fig.colorbar(image,ax=axis,shrink=.67,label='Stored value; physical units unconfirmed')
        axes[2].imshow(mask,origin='upper',extent=extent,interpolation='nearest',vmin=0,vmax=1,
                       cmap=ListedColormap(['#152039','#58eeff']))
        axes[2].set_title(f'Provider mask: {len(xs):,} positive pixels',fontsize=10)
        for axis in axes:
            axis.scatter([c*.06],[r*.06],marker='+',color='white',s=70,linewidths=1.1)
            axis.set_xlabel('km east from crop edge');axis.set_ylabel('km south from crop edge')
        title='Volusia landfill' if 'Volusia' in row['id'] else 'Black Oak landfill'
        fig.suptitle(f'{title} | EMIT {row["tile_date"][:10]} | 60 m native grid | DATA REVIEW, NOT A MODEL OUTPUT',fontsize=13)
        fig.text(.02,.035,'Cyan = provider-annotated plume. Dark mask background = unannotated, not verified methane-free. White + = listed facility location, not source truth.\nBoth supplied RGB files are constant arrays and cannot supply context. Plume-mask header declares zero as nodata; validity remains unresolved.',fontsize=9)
        fig.subplots_adjust(left=.055,right=.97,top=.85,bottom=.22,wspace=.32)
        path=review/f'{selected["role"]}.png';fig.savefig(path);plt.close(fig)
        result['preview']=str(path);report['samples'].append(result)
        print(json.dumps({k:result[k] for k in ('location','positive_pixels','positive_bbox_projected','rgb_constant','positive_touches_edge')}))
    save_json(review/'native_review.json',report)
    if (OUT/'sentinel/acquisition.json').exists(): export_pair(selection,report)


def export_pair(selection,native_report):
    receipt=json.loads((OUT/'sentinel/acquisition.json').read_text())
    preflight=json.loads((OUT/'sentinel/preflight.json').read_text())
    if receipt['status']!='acquired_pending_visual_review':raise ValueError('No complete Sentinel crop')
    positive=selection['samples'][0];row=positive['row'];native=OUT/'native'/row['id']
    channels=[];stretch=[]
    for name in ('red','green','blue'):
        path=OUT/'sentinel'/f'{name}.tif'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['assets'][name]['sha256']:
            raise ValueError('Acquired Sentinel bytes changed')
        with rasterio.open(path) as ds:
            arr=ds.read(1);affine=ds.transform;crs=ds.crs;bounds=list(ds.bounds)
            good=(arr!=ds.nodata)&np.isfinite(arr)
            if not good.all():raise ValueError('RGB crop contains missing pixels')
            low,high=np.percentile(arr[good],[2,98])
            if high<=low:raise ValueError('Constant Sentinel band')
            channels.append(np.uint8(np.clip((arr.astype(float)-low)/(high-low),0,1)*255))
            stretch.append({'band':name,'p2_dn':float(low),'p98_dn':float(high)})
    rgb=np.stack(channels,axis=-1); shape=rgb.shape[:2];arrays={}
    for name in ('mf','plumemask'):
        with rasterio.open(native/f'{name}.tif') as ds:
            src=ds.read(1);dest=np.zeros(shape,dtype=src.dtype)
            # Zero nodata on provider masks is not passed as a validity mask.
            reproject(src,dest,src_transform=ds.transform,src_crs=ds.crs,
                      dst_transform=affine,dst_crs=crs,resampling=Resampling.nearest)
            arrays[name]=dest
            if name=='mf':vmin,vmax=float(src.min()),float(src.max())
            if name=='plumemask':native_count=int((src==1).sum())
    mask=arrays['plumemask']==1
    if int(mask.sum())!=native_count*36:raise ValueError('Display grid does not preserve all native positive-cell area')
    interior=mask.copy()
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        neighbor=np.roll(mask,(dy,dx),(0,1));neighbor[[0,-1],:]=False;neighbor[:,[0,-1]]=False
        interior &= neighbor
    border=mask&~interior
    mask_png=np.zeros((*shape,4),dtype=np.uint8);mask_png[border]=[88,238,255,255]
    retrieval=np.uint8(plt.get_cmap('magma')(np.clip((arrays['mf']-vmin)/(vmax-vmin),0,1))*255)
    folder=ROOT/'outputs/GTM_viewer_bundles/GTM_Model6_emit_2025_intake';folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(folder/'sentinel_rgb.png')
    Image.fromarray(retrieval).save(folder/'mf.png')
    Image.fromarray(mask_png).save(folder/'mask.png')
    for role in ('declared_positive','declared_control'):
        shutil.copyfile(OUT/'review'/f'{role}.png',folder/f'{role}.png')
    x,y=transform('EPSG:4326',crs,[float(row['location_lon'])],[float(row['location_lat'])])
    fig,axes=plt.subplots(1,3,figsize=(15.2,5.6),dpi=135)
    extent=[0,shape[1]/100,shape[0]/100,0]
    c,r=(~affine)*(x[0],y[0])
    for i,axis in enumerate(axes):
        axis.imshow(rgb if i<2 else retrieval,extent=extent,interpolation='nearest')
        if i:axis.imshow(mask_png,extent=extent,interpolation='nearest')
        axis.scatter(c/100,r/100,marker='+',color='white',s=80,linewidths=1)
        axis.set_xlabel('km east from crop edge');axis.set_ylabel('km south from crop edge')
    axes[0].set_title('Actual Sentinel-2 RGB | 10 m pixels')
    axes[1].set_title('Sentinel + native EMIT annotation')
    axes[2].set_title('EMIT matched-filter retrieval + annotation')
    fig.suptitle('Volusia landfill | 1 June 2025 | DATA REVIEW, NOT A MODEL PREDICTION',fontsize=14)
    fig.text(.025,.025,'Sentinel is 12.7 minutes later. Cyan = provider annotation at native 60 m resolution; regridding adds no measured detail.\nWhite + = listed facility, not a verified emission origin. RGB is percentile-stretched raw DN. Retrieval units and cross-sensor alignment need further QA.',fontsize=10)
    fig.subplots_adjust(left=.045,right=.985,top=.86,bottom=.21,wspace=.29)
    fig.savefig(folder/'paired_review.png');plt.close(fig)
    save_json(folder/'review.json',{'bounds':bounds,'crs':str(crs),'rgb':'sentinel_rgb.png','retrieval':'mf.png','mask':'mask.png',
        'location':{'x':x[0],'y':y[0]},'title':'Volusia landfill','offsetMinutes':receipt['offset_minutes'],
        'clearPercent':preflight['clear_fraction']*100,'fullPositive':'declared_positive.png','fullControl':'declared_control.png',
        'retrievalRange':[vmin,vmax],'stretch':stretch,'sourceSample':row['id'],'sentinelId':receipt['sentinel_id'],
        'display_shape':list(shape),'display_spacing_m':10,'native_label_spacing_m':60,'native_positive_pixels':native_count,
        'display_positive_pixels':int(mask.sum()),'ready_for_training':False,'prediction':False,
        'caveats':['No new model result.','RGB is raw DN stretched for display; scale/offset conflict remains.',
                   'Provider RGB is constant ones. Mask header nodata=0 leaves background validity unconfirmed.',
                   'Only RGB and SCL were acquired. NIR/SWIR/reference scene absent.',
                   'Timing passes 15 minutes but does not prove plume stationarity or spatial alignment.',
                   '39 historical US evaluation locations remain unresolved. Development inspection only.'],
        'acquisition':receipt,'preflight':preflight,'native_review':native_report,
        'assets_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.png'))}})
    print(json.dumps({'pair_preview':str(folder/'paired_review.png'),'full_native_positive_area_retained':True,'shape':shape}))


if __name__=='__main__':main()
