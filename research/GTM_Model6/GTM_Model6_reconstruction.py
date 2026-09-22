"""Bounded reconstruction diagnostic on legacy resampled measurements, never fine truth.

No acquisition. A 256-pixel export is aggregated 8x before the 2x reconstruction
task. This does not recover the missing native EMIT grid or validate 20 m output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
# Conservative interior rectangles, deliberately excluding uncertain border areas.
# These select this diagnostic cohort; they are not a general US boundary dataset.
US_INTERIORS = {'AZ':(-114,31.6,-109.1,36.9), 'CO':(-109,37.1,-102.1,40.9),
 'UT':(-113.9,37.1,-109.1,41.9), 'AL':(-88.3,31,-85.1,34.9),
 'NC':(-83,35,-77,36.4), 'IN':(-87.5,38,-84.9,41.7),
 'TX':(-103.8,30,-94,33.5), 'TX_SOUTH':(-99.6,27.8,-98,29.5),
 'LA':(-93.6,30.1,-91,32.9), 'CA_SAN_DIEGO':(-117.3,32.7,-116.9,33.1)}
SEED = 61019

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def pool_valid(value, valid, factor):
    """Area means over observed cells; missing pixels are never zero targets."""
    weight = F.avg_pool2d(valid, factor)
    mean = F.avg_pool2d(value * valid, factor) / weight.clamp_min(1e-8)
    return mean, weight

def degrade(target, valid):
    coarse, coverage = pool_valid(target, valid, 2)
    known = (coverage > 0).float()
    numerator = F.interpolate(coarse * known, size=target.shape[-2:], mode='bilinear', align_corners=False)
    denominator = F.interpolate(known, size=target.shape[-2:], mode='bilinear', align_corners=False)
    baseline = numerator / denominator.clamp_min(1e-8)
    nearest = F.interpolate(coarse, size=target.shape[-2:], mode='nearest')
    return baseline, F.interpolate(coverage, size=target.shape[-2:], mode='nearest'), nearest

def augment(sentinel, target, valid, angle, mirror):
    """Transform numerator and support together; padded borders remain unknown."""
    if mirror:
        sentinel, target, valid = [torch.flip(x, [-1]) for x in (sentinel,target,valid)]
    radians=math.radians(angle)
    theta=target.new_tensor([[[math.cos(radians),-math.sin(radians),0],
                              [math.sin(radians), math.cos(radians),0]]]).expand(len(target),-1,-1)
    grid=F.affine_grid(theta, target.shape, align_corners=False)
    support=F.grid_sample(valid,grid,align_corners=False)
    value=F.grid_sample(target*valid,grid,align_corners=False)/support.clamp_min(1e-8)
    mask=(support>=0.999).float()
    imagery=F.grid_sample(sentinel,grid,align_corners=False)
    return imagery, value*mask, mask

def inventory():
    rows=[]; arrays=[]
    for path in sorted((ROOT/'EarthRemoteSensingRapidResponse/Dataset').rglob('*.tif')):
        with rasterio.open(path) as src:
            data=src.read().astype('float32'); bounds=transform_bounds(src.crs,'EPSG:4326',*src.bounds)
            target=data[-1]; valid=np.isfinite(target)&(target!=-9999)
            region=next((k for k,(w,s,e,n) in US_INTERIORS.items() if w<bounds[0]<bounds[2]<e and s<bounds[1]<bounds[3]<n),None)
            match=re.search(r'EMIT_L2B_CH4PLM_\d+_(\d{8}T\d{6})_(\d+)',path.stem)
            stamp=datetime.strptime(path.name[:15],'%Y%m%dT%H%M%S')
            emit_stamp=datetime.strptime(match[1],'%Y%m%dT%H%M%S') if match else None
            reasons=[]
            if not region: reasons.append('outside conservative verified US interior selection')
            if not match: reasons.append('EMIT acquisition identity absent')
            if data.shape!=(6,256,256): reasons.append('unexpected band or grid shape')
            if valid.mean()<0.05: reasons.append('under 5 percent observed target support')
            if not np.isfinite(data[:5]).all() or np.any(data[:5]==-9999): reasons.append('invalid Sentinel input')
            row={'id':path.stem,'source':path.relative_to(ROOT).as_posix(),'sha256':digest(path),
                 'bounds_wgs84':list(bounds),'crs':str(src.crs),'affine':list(src.transform)[:6],
                 'us_interior':region,'sentinel_time':stamp.isoformat(),
                 'emit_time':emit_stamp.isoformat() if emit_stamp else None,
                 'pairing_lag_days':abs((stamp-emit_stamp).total_seconds())/86400 if emit_stamp else None,
                 'valid_fraction':float(valid.mean()),'target_unique_values':int(len(np.unique(target[valid]))),
                 'source_grid':'legacy resampled export, not native EMIT',
                 'units':'legacy CH4 product values; ppm*m expected, metadata not independently verified',
                 'eligible_engineering_only':not reasons,'exclusions':reasons}
            rows.append(row)
            if not reasons:
                a=torch.tensor(data[:5][None]); t=torch.tensor(np.where(valid,target,0)[None,None]); v=torch.tensor(valid[None,None].astype('float32'))
                t,coverage=pool_valid(t,v,8); v=(coverage>=0.5).float(); t=t*v
                a=F.avg_pool2d(a,8)
                arrays.append((row,a,t,v))
    # Spatial connected components: repeated sites and all locations within 20 km
    # are grouped before augmentation. Share parent acquisition IDs as well.
    parents=list(range(len(arrays)))
    def find(i):
        while parents[i]!=i: i=parents[i]
        return i
    for i,(a,*_) in enumerate(arrays):
        for j,(b,*_) in enumerate(arrays[:i]):
            ax=(a['bounds_wgs84'][0]+a['bounds_wgs84'][2])/2; ay=(a['bounds_wgs84'][1]+a['bounds_wgs84'][3])/2
            bx=(b['bounds_wgs84'][0]+b['bounds_wgs84'][2])/2; by=(b['bounds_wgs84'][1]+b['bounds_wgs84'][3])/2
            dist=111.2*math.hypot(ay-by,(ax-bx)*math.cos(math.radians((ay+by)/2)))
            if dist<20 or a['emit_time']==b['emit_time']:
                parents[find(i)]=find(j)
    for i,(row,*_) in enumerate(arrays): row['group']=f'US_group_{find(i):02d}'
    groups=sorted({r['group'] for r,*_ in arrays},key=lambda x:hashlib.sha256((str(SEED)+x).encode()).hexdigest())
    for row,*_ in arrays: row['partition']=groups.index(row['group'])%5
    return rows,arrays

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())

class ReconstructionUNet(nn.Module):
    def __init__(self,channels=7):
        super().__init__(); self.e1=Block(channels,16); self.e2=Block(16,32); self.mid=Block(32,64)
        self.d2=Block(96,32); self.d1=Block(48,16); self.head=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x):
        a=self.e1(x); b=self.e2(F.avg_pool2d(a,2)); c=self.mid(F.avg_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return x[:,-2:-1]+self.head(e)

def score(pred,target,valid):
    error=(pred-target)[valid>0]
    return {'mae':float(error.abs().mean()),'rmse':float(error.square().mean().sqrt()),'observed_pixels':int(error.numel())}

def fit(train,steps,device,use_sentinel,seed,aug=True):
    torch.manual_seed(seed); np.random.seed(seed)
    s=torch.cat([x[1] for x in train]).to(device); t=torch.cat([x[2] for x in train]).to(device); v=torch.cat([x[3] for x in train]).to(device)
    mean=s.mean((0,2,3),keepdim=True); std=s.std((0,2,3),keepdim=True).clamp_min(1)
    scale=t[v>0].abs().quantile(0.95).clamp_min(1)
    s=(s-mean)/std; t=t/scale
    views=[]
    for mirror in ([False,True] if aug else [False]):
        for angle in ([0,15,30,45] if aug else [0]):
            a,y,m=augment(s,t,v,angle,mirror); base,cov,_=degrade(y,m)
            if not use_sentinel: a=torch.zeros_like(a)
            views.append((torch.cat([a,base,cov],1),y,m))
    x=torch.cat([z[0] for z in views]); y=torch.cat([z[1] for z in views]); mask=torch.cat([z[2] for z in views])
    model=ReconstructionUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    log=[]
    for step in range(steps):
        index=torch.randint(len(x),(min(16,len(x)),),device=device)
        p=model(x[index]); m=mask[index]; loss=(F.smooth_l1_loss(p,y[index],reduction='none',beta=0.1)*m).sum()/m.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if step%100==0 or step==steps-1: log.append({'step':step+1,'loss':float(loss.detach())})
    return model,{'mean':mean.cpu(),'std':std.cpu(),'scale':scale.cpu(),'use_sentinel':use_sentinel},log

@torch.no_grad()
def predict(model,norm,item,device,shuffled=None,zero_emit=False):
    _,s,t,v=item; a=(s.to(device)-norm['mean'].to(device))/norm['std'].to(device)
    if shuffled is not None: a=(shuffled[1].to(device)-norm['mean'].to(device))/norm['std'].to(device)
    if not norm['use_sentinel']: a=torch.zeros_like(a)
    base,cov,nearest=degrade(t,v)
    x=torch.cat([a,base.to(device)/norm['scale'].to(device),cov.to(device)],1)
    if zero_emit: x[:,-2:]=0
    p=model(x).cpu()*norm['scale']
    return p,base,nearest

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run-id',required=True); ap.add_argument('--steps',type=int,default=600); ap.add_argument('--minutes',type=float,default=25); ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args(); out=ROOT/'outputs/GTM_Model6'/args.run_id
    if out.exists(): raise SystemExit('Refusing to overwrite a run')
    out.mkdir(parents=True); start=time.monotonic(); torch.set_num_threads(4)
    rows,items=inventory(); write_json(out/'GTM_Model6_inventory.json',rows)
    manifest={'task':'legacy_aggregate_controlled_degradation_engineering_only','seed':SEED,
     'fine_resolution_validated':False,'native_emit_grid_recovered':False,
     'source_export_pixels':256,'aggregation':8,'target_pixels':32,'coarse_input_pixels':16,
     'target_validity':'at least 50 percent valid legacy pixels per 8x8 block; -9999 excluded',
     'degradation':'2x2 masked area mean; coarse support only enters model; no target support input',
     'split':'five predetermined group partitions; one trains, remaining four evaluate per rotation',
     'steps_fixed_before_evaluation':args.steps,'negative_controls':'no representative true-negative coverage available',
     'units':'legacy CH4 values, expected ppm*m but not independently verified',
     'limitations':['resampled legacy product; not original EMIT observations','historically inspected development data','Sentinel pairing may differ by weeks; context only','sparse plume-selected coverage cannot evaluate background false positives'],
     'code_sha256':digest(__file__),'observations':[r for r,*_ in items]}
    write_json(out/'GTM_Model6_manifest.json',manifest)
    print(json.dumps({'eligible':len(items),'groups':len({r['group'] for r,*_ in items}),'partitions':{i:sum(r['partition']==i for r,*_ in items) for i in range(5)}}),flush=True)
    if args.audit_only: return
    if len({r['group'] for r,*_ in items})<5: raise SystemExit('Insufficient independent groups even for five-way diagnostic')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # Memorization is a plumbing check only, always labeled training output.
    model,norm,log=fit(items[:1],min(300,args.steps),device,True,SEED,aug=False)
    smoke,base,_=predict(model,norm,items[0],device); smoke_metrics={'trained':score(smoke,items[0][2],items[0][3]),'bilinear':score(base,items[0][2],items[0][3]),'log':log}
    write_json(out/'GTM_Model6_tiny_fit.json',smoke_metrics)
    print('TINY_FIT '+json.dumps(smoke_metrics),flush=True)
    if smoke_metrics['trained']['mae']>=smoke_metrics['bilinear']['mae']*0.8: raise SystemExit('Tiny-set fit failed; stop before folds')
    records=[]; predictions={}; trained=[]
    for fold in range(5):
        train=[x for x in items if x[0]['partition']==fold]; held=[x for x in items if x[0]['partition']!=fold]
        for use_sentinel in (False,True):
            if time.monotonic()-start>args.minutes*60: raise SystemExit('Compute cap reached; partial run only')
            label='sentinel_emit' if use_sentinel else 'emit_only'
            model,norm,log=fit(train,args.steps,device,use_sentinel,SEED+fold)
            checkpoint=out/f'GTM_Model6_S32_{label}_fold{fold}.pt'
            torch.save({'state_dict':model.cpu().state_dict(),'normalization':norm,'fold':fold,'training_ids':[x[0]['id'] for x in train],'log':log},checkpoint); model.to(device).eval()
            trained.append({'fold':fold,'variant':label,'checkpoint':checkpoint.name,'sha256':digest(checkpoint),'train_count':len(train),'held_count':len(held),'log':log})
            for idx,item in enumerate(held):
                p,b,n=predict(model,norm,item,device); row,s,t,v=item
                controls={'bilinear':b,'nearest':n,'constant_crop':torch.ones_like(t)*(b[v>0].mean())}
                if use_sentinel:
                    controls['shuffled_sentinel']=predict(model,norm,item,device,shuffled=held[(idx+1)%len(held)])[0]
                    controls['zero_emit']=predict(model,norm,item,device,zero_emit=True)[0]
                metrics={label:score(p,t,v),**{k:score(z,t,v) for k,z in controls.items()}}
                records.append({'fold':fold,'id':row['id'],'group':row['group'],'variant':label,'metrics':metrics})
                key=f'f{fold}_{label}_{row["id"]}'
                predictions[key]=p.numpy()[0,0]
            print(f'FOLD {fold} {label} complete; elapsed {time.monotonic()-start:.1f}s',flush=True)
            write_json(out/'GTM_Model6_metrics_partial.json',records)
    np.savez_compressed(out/'GTM_Model6_predictions.npz',**predictions)
    totals={}
    for variant in ['emit_only','sentinel_emit']:
        selected=[r for r in records if r['variant']==variant]
        totals[variant]={k:float(np.mean([r['metrics'][k]['mae'] for r in selected])) for k in selected[0]['metrics']}
    result={'status':'completed_engineering_diagnostic_not_promoted','summary_macro_observation_mae':totals,
     'note':'Repeated held-out predictions across rotations are correlated; not independent samples or confirmation.',
     'runtime_seconds':time.monotonic()-start,'device':device,'trained':trained,'observations':records}
    write_json(out/'GTM_Model6_results.json',result); print('RESULT '+json.dumps(totals),flush=True)

if __name__=='__main__': main()
