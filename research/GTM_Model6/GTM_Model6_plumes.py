"""US Sentinel-only plume learning; EMIT positive supervision, reviewed MARS controls.

No acquisition and no EMIT raster, polygon, coordinate or support enters inference.
Catalog-exterior EMIT pixels are unlabeled, never training negatives. The historical
development cohort is reused openly; these are exploratory held-group results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from GTM_Model6_reconstruction import ROOT, Block, digest, write_json

DATA = ROOT.parent/'Earth-Remote-Sensing-Rapid-Response-'
PAIRS = DATA/'EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs'
MARS = PAIRS/'publication-v1/external/MARS-S2L'
SEED = 6192026
WINDOWS = (16,32,64,128)
BANDS = [0,1,2,4,5]


def read(path):
    with rasterio.open(path) as s:
        return s.read(), str(s.crs), list(s.transform)[:6]


def features(target, reference, valid):
    """Same-date and temporal log contrasts; scale factors cancel per scene.

    Fit medians over Sentinel-valid pixels only. Never use the plume label to
    select background. Both radiometry products must already share a grid.
    """
    t=np.log(np.maximum(target.astype('float32'),1))
    r=np.log(np.maximum(reference.astype('float32'),1))
    a=t-np.median(t[:,valid],axis=1)[:,None,None]
    b=r-np.median(r[:,valid],axis=1)[:,None,None]
    d=t-r; d-=np.median(d[:,valid],axis=1)[:,None,None]
    swir=d[-1:]-d[-2:-1]
    x=np.concatenate([a,b,d,swir])*4
    x=np.where(valid[None],np.clip(x,-8,8),0).astype('float32')
    contrast=-x[-1];center=np.median(contrast[valid])
    sigma=max(float(1.4826*np.median(np.abs(contrast[valid]-center))),.01)
    evidence=np.clip((contrast-center)/sigma-3,-12,12)
    return np.concatenate([x,np.where(valid,evidence,-12)[None]]).astype('float32')


def pack(row,t,r,mask,valid,fully_labeled,crs,affine):
    # Preserve real 200x200 scene context. The 10m file grid does not imply 10m
    # methane measurements (EMIT is nominally 60m; Sentinel SWIR is 20m).
    x=features(t[BANDS],r[BANDS],valid)
    v=valid;truth=mask.astype('float32')
    known=v if fully_labeled else v&(truth>0)
    row.update(crs=crs,affine=affine,shape=[200,200],fully_labeled=fully_labeled,
               known_pixels=int(known.sum()),positive_pixels=int((truth*known).sum()),valid_fraction=float(v.mean()))
    return {'row':row,'x':x,'target':truth,'known':known,'valid':v,
            'reference_mask':truth,'reference_valid':v,'rgb':t[[2,1,0]].astype('float32').transpose(1,2,0)}


def inventory(negative_cap=128):
    items=[];excluded=[]
    manifest=MARS/'publication_v3_training_samples.jsonl'
    records=[json.loads(s) for s in manifest.read_text().splitlines()]
    records=[r for r in records if r.get('country')=='United States of America' and r.get('product_level')=='L1C']
    positives=sorted([r for r in records if r['label_state']=='PLUME'],key=lambda r:r['sample_id'])
    # Round-robin negative locations so frequent revisits do not consume the cap.
    locations={}
    for r in sorted(records,key=lambda r:hashlib.sha256((str(SEED)+r['sample_id']).encode()).hexdigest()):
        if r['label_state']=='NO_PLUME':locations.setdefault(r['physical_location_id'],[]).append(r)
    negatives=[]
    for depth in range(max(map(len,locations.values()),default=0)):
        for group in locations.values():
            if depth<len(group) and len(negatives)<negative_cap:negatives.append(group[depth])
    for r in positives+negatives:
        paths={a['role']:MARS/a['path'] for a in r['assets']}
        if not all(p.exists() for p in paths.values()):raise FileNotFoundError(r['sample_id'])
        a,crs,aff=read(paths['image']);cloud,c2,a2=read(paths['cloud_mask'])
        if crs!=c2 or aff!=a2 or a.shape!=(12,200,200):raise ValueError('MARS grid mismatch')
        valid=(a!=0).all(0)&(cloud[0]==0)
        mask=np.zeros((200,200),bool)
        if r['label_state']=='PLUME':
            m,c2,a2=read(paths['plume_mask'])
            if c2!=crs or a2!=aff:raise ValueError('MARS mask alignment')
            mask=m[0]>0
        if valid.mean()<.7 or (r['label_state']=='PLUME' and not (mask&valid).any()):
            excluded.append({'id':r['sample_id'],'reason':'under 70% clear support or no visible positive'});continue
        row={'id':'MARS_'+r['sample_id'],'dataset':'MARS','label':r['label_state'],'lon':r['longitude'],'lat':r['latitude'],
             'site':r['physical_location_id'],'target_scene':r['target_scene_id'],'reference_scene':r['reference_scene_id'],
             'date':r['target_datetime'],'source':str(paths['image']),'hashes':{k:digest(p) for k,p in paths.items()},
             'license':r['license'],'label_semantics':'reviewed MARS plume/no-plume; segmentation reference, not fine-resolution EMIT measurement'}
        items.append(pack(row,a[:6],a[6:],mask,valid,True,crs,aff))
    candidates=json.loads((ROOT/'reports/acquisition/emit_v002_time_aligned_candidates.json').read_text())['candidates']
    seal=json.loads((ROOT/'reports/acquisition/emit_v002_external_cohort_seal.json').read_text())['records']
    passing={r['group_id'] for r in seal if r['final_gate_pass']}
    for c in sorted(candidates,key=lambda r:r['group_id']):
        lon,lat=c['center']
        # These twelve candidates are interior US locations, manually audited.
        if not(-125<lon<-66 and 25<lat<49):continue
        folder=PAIRS/'emit-v002-external-l1c-2026-07'/c['granule_id']
        if c['group_id'] not in passing:
            excluded.append({'id':c['group_id'],'reason':'failed original Sentinel quality gate'});continue
        paths={k:folder/v for k,v in {'image':'target.l1c.tif','reference':'reference.l1c.tif','mask':'plume_mask.tif',
               'cloud':'target.cloudsen12.tif','scl':'target.l2a_scl.tif','reference_scl':'reference.l2a_scl.tif'}.items()}
        values={};signature=None
        for k,p in paths.items():
            a,crs,aff=read(p);values[k]=a
            if signature is None:signature=(crs,aff)
            elif signature!=(crs,aff):raise ValueError('EMIT/Sentinel cropped asset grid mismatch')
        t=values['image'];r=values['reference'];m=values['mask'][0]>0
        valid=(t>0).all(0)&(r>0).all(0)&(values['cloud'][0]==0)
        for k in ['scl','reference_scl']:valid &= ~np.isin(values[k][0],[0,1,3,8,9,10,11])
        if valid.mean()<.7 or not(m&valid).any():
            excluded.append({'id':c['group_id'],'reason':'current common support gate'});continue
        meta=json.loads((folder/'manifest.json').read_text())
        row={'id':c['group_id'],'dataset':'EMIT','label':'POSITIVE_ONLY','lon':lon,'lat':lat,'site':c['group_id'],
             'target_scene':meta['target_scene_id'],'reference_scene':meta['reference_scene_id'],
             'date':c['s2_datetime'],'emit_date':c['emit_datetime'],'lag_hours':c['offset_hours'],
             'source':str(paths['image']),'hashes':{k:digest(p) for k,p in paths.items()},'granule':c['granule_id'],
             'label_semantics':'NASA CMR irregular EMIT plume-complex geometry; exterior unknown, only positive support trains'}
        items.append(pack(row,t,r,m,valid,False,crs,aff))
    # Connected geographic/site/acquisition groups are assigned BEFORE any views.
    parent=list(range(len(items)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def scene_key(s):
        # Different naming conventions across MARS and Earth Search are not used
        # to assert identity; spatial linking still groups colocated scenes.
        return s
    for i,a in enumerate(items):
        a=a['row']
        for j,b in enumerate(items[:i]):
            b=b['row'];dist=111.2*math.hypot(a['lat']-b['lat'],(a['lon']-b['lon'])*math.cos(math.radians((a['lat']+b['lat'])/2)))
            shared=set([scene_key(a['target_scene']),scene_key(a['reference_scene'])])&set([scene_key(b['target_scene']),scene_key(b['reference_scene'])])
            if dist<25 or a['site']==b['site'] or shared:parent[find(i)]=find(j)
    groups={}
    for i,item in enumerate(items):groups.setdefault(find(i),[]).append(item)
    # Greedy group-size balance uses no labels or outcomes.
    loads=[0]*5
    for g,vals in sorted(groups.items(),key=lambda kv:(-len(kv[1]),hashlib.sha256(kv[1][0]['row']['id'].encode()).hexdigest())):
        fold=int(np.argmin(loads));loads[fold]+=len(vals)
        group='US_'+hashlib.sha256('|'.join(sorted(x['row']['id'] for x in vals)).encode()).hexdigest()[:10]
        for x in vals:x['row'].update(group=group,partition=fold)
    if len(groups)<5:raise ValueError('Fewer than five independent groups')
    return items,excluded


class PlumeUNet(nn.Module):
    def __init__(self,evidence=False):
        super().__init__();self.e1=Block(16,16);self.e2=Block(16,32);self.mid=Block(32,64)
        self.d2=Block(96,32);self.d1=Block(48,16);self.head=nn.Conv2d(16,1,1)
        self.evidence=evidence
        if evidence:nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        else:nn.init.constant_(self.head.bias,-2)
    def forward(self,x):
        a=self.e1(x[:,:16]);b=self.e2(F.avg_pool2d(a,2));c=self.mid(F.avg_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        logits=self.head(self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))
        # Shared scene-wide Sentinel-only evidence across all window sizes.
        # Bounded correction cannot invent a detection absent a SWIR decrease.
        return x[:,-1:]+1.5*torch.tanh(logits) if self.evidence else logits


def dense_loss(logits,y,known,full):
    pixel=F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=logits.new_tensor(3.))
    bce=(pixel*known).sum((1,2,3))/known.sum((1,2,3)).clamp_min(1)
    p=logits.sigmoid()*known
    dice=1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+(y*known).sum((1,2,3))+1)
    # No Dice over positive-only EMIT supervision: exterior is unknown.
    return (bce+.5*dice*full).mean()


def training_views(items,device):
    x=torch.tensor(np.stack([a['x'] for a in items]),device=device)
    y=torch.tensor(np.stack([a['target'] for a in items])[:,None],device=device)
    v=torch.tensor(np.stack([a['known'] for a in items])[:,None],device=device,dtype=torch.float32)
    full=torch.tensor([a['row']['fully_labeled'] for a in items],device=device,dtype=torch.float32)
    result=[]
    for mirror in [False,True]:
        for angle in [0,15,30,45]:
            xx,yy,vv=[torch.flip(a,[-1]) if mirror else a for a in [x,y,v]]
            theta=x.new_tensor([[[math.cos(math.radians(angle)),-math.sin(math.radians(angle)),0],
                                  [math.sin(math.radians(angle)),math.cos(math.radians(angle)),0]]]).expand(len(x),-1,-1)
            grid=F.affine_grid(theta,y.shape,align_corners=False)
            xx=F.grid_sample(xx,grid,align_corners=False,padding_mode='reflection')
            yy=F.grid_sample(yy,grid,mode='nearest',align_corners=False)
            vv=(F.grid_sample(vv,grid,align_corners=False)>=.999).float()
            result.append((xx,yy,vv,full))
    return tuple(torch.cat([a[i] for a in result]) for i in range(4))


def fit(items,window,steps,device,seed,evidence=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x=torch.tensor(np.stack([a['x'] for a in items]),device=device)
    y=torch.tensor(np.stack([a['target'] for a in items])[:,None],device=device)
    v=torch.tensor(np.stack([a['known'] for a in items])[:,None],device=device,dtype=torch.float32)
    full=torch.tensor([a['row']['fully_labeled'] for a in items],device=device,dtype=torch.float32)
    if min(x.shape[-2:])<window:raise ValueError('Requested context exceeds real imagery')
    positives=torch.where(((y*v).sum((1,2,3))>0))[0]
    negatives=torch.where((full>0)&((y*v).sum((1,2,3))==0))[0]
    if not len(positives) or not len(negatives):raise ValueError('Training split lacks positive or reviewed negative controls')
    model=PlumeUNet(evidence=evidence).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);log=[]
    for step in range(steps):
        ids=torch.cat([positives[torch.randint(len(positives),(4,),device=device)],negatives[torch.randint(len(negatives),(4,),device=device)]])
        xx=[];yy=[];vv=[]
        for idx in ids:
            i=int(idx);positions=torch.nonzero((y[i,0]*v[i,0])>0)
            if len(positions) and rng.random()<.7:
                cy,cx=positions[int(rng.integers(len(positions)))].tolist()
                top=int(np.clip(cy-rng.integers(window),0,x.shape[-2]-window));left=int(np.clip(cx-rng.integers(window),0,x.shape[-1]-window))
            else:
                top=int(rng.integers(x.shape[-2]-window+1));left=int(rng.integers(x.shape[-1]-window+1))
            xx.append(x[i,:,top:top+window,left:left+window]);yy.append(y[i,:,top:top+window,left:left+window]);vv.append(v[i,:,top:top+window,left:left+window])
        xx,yy,vv=torch.stack(xx),torch.stack(yy),torch.stack(vv)
        # Cycle all eight Martin transforms on training patches only. Bilinear
        # support excludes rotation wedges; labels use nearest-neighbor.
        view=step%8;angle=[0,15,30,45][view%4]
        if view>=4:xx,yy,vv=[torch.flip(a,[-1]) for a in [xx,yy,vv]]
        theta=x.new_tensor([[[math.cos(math.radians(angle)),-math.sin(math.radians(angle)),0],
                              [math.sin(math.radians(angle)),math.cos(math.radians(angle)),0]]]).expand(len(ids),-1,-1)
        grid=F.affine_grid(theta,yy.shape,align_corners=False)
        xx=F.grid_sample(xx,grid,align_corners=False,padding_mode='reflection')
        yy=F.grid_sample(yy,grid,mode='nearest',align_corners=False)
        vv=(F.grid_sample(vv,grid,align_corners=False)>=.999).float()
        pred=model(xx);loss=dense_loss(pred,yy,vv,full[ids])
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%200==0 or step==steps-1:log.append({'step':step+1,'loss':float(loss.detach())})
    return model.eval(),log


@torch.no_grad()
def predict(model,x,window,device):
    """Dense tiled inference consumes only an array of Sentinel features."""
    t=torch.tensor(x[None],device=device);h,w=x.shape[-2:]
    if min(h,w)<window:raise ValueError('Requested context exceeds real imagery')
    hh,ww=t.shape[-2:]
    ys=sorted(set(list(range(0,hh-window+1,max(1,window//2)))+[hh-window]));xs=sorted(set(list(range(0,ww-window+1,max(1,window//2)))+[ww-window]))
    out=torch.zeros((hh,ww),device=device);weight=torch.zeros_like(out)
    taper=(torch.hann_window(window,periodic=False,device=device)+.05);taper=taper[:,None]*taper[None,:]
    positions=[(y,x) for y in ys for x in xs]
    for start in range(0,len(positions),32):
        pp=positions[start:start+32];batch=torch.cat([t[:,:,y:y+window,x:x+window] for y,x in pp]);p=model(batch).sigmoid()[:,0]
        for i,(y,x) in enumerate(pp):out[y:y+window,x:x+window]+=p[i]*taper;weight[y:y+window,x:x+window]+=taper
    assert (weight>0).all()
    return (out/weight)[:h,:w].cpu().numpy()


def metrics(p,item,threshold):
    y=item['reference_mask']>0;v=item['reference_valid'];b=p>=threshold
    tp=int((b&y&v).sum());fp=int((b&~y&v).sum());fn=int((~b&y&v).sum());tn=int((~b&~y&v).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'iou':tp/max(tp+fp+fn,1),'dice':2*tp/max(2*tp+fp+fn,1),
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'area_ratio':(tp+fp)/max(tp+fn,1),
            'predicted_fraction':float(b[item['valid']].mean()),'reference_fraction':float(y[v].mean()),
            'pixel_ap':float(average_precision_score(y[v],p[v])) if y[v].any() else None,
            'interpretation':'reviewed mask agreement' if item['row']['fully_labeled'] else 'EMIT footprint agreement proxy; fp is outside catalog, NOT verified false methane'}


def threshold_from_validation(predictions,items):
    # Only reviewed MARS masks tune the cutoff. No outer held labels involved.
    selected=[(p,x) for p,x in zip(predictions,items) if x['row']['fully_labeled']]
    if not selected or not any(x['target'].any() for _,x in selected):return .5,[]
    table=[]
    for threshold in np.arange(.1,.91,.05):
        ms=[metrics(p,x,float(threshold)) for p,x in selected];tp=sum(m['tp'] for m in ms);fp=sum(m['fp'] for m in ms);fn=sum(m['fn'] for m in ms)
        table.append({'threshold':float(threshold),'dice':2*tp/max(2*tp+fp+fn,1)})
    best=max(table,key=lambda r:(r['dice'],r['threshold']))
    return best['threshold'],table


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-id',required=True);ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--negative-cap',type=int,default=128);ap.add_argument('--minutes',type=float,default=20);ap.add_argument('--audit-only',action='store_true');ap.add_argument('--evidence',action='store_true');args=ap.parse_args()
    out=ROOT/'outputs/GTM_Model6'/args.run_id
    if out.exists():raise SystemExit('Refusing to overwrite run')
    out.mkdir(parents=True);torch.set_num_threads(4);start=time.monotonic();items,excluded=inventory(args.negative_cap)
    manifest={'task':'Sentinel target/reference -> plume probability','seed':SEED,'windows':WINDOWS,'steps':args.steps,
              'variant':'spectral_evidence_bounded_refinement' if args.evidence else 'free_dense_baseline',
              'evidence_contract':'scene-wide robust z score of decreasing temporal B12/B11 log ratio, minus 3; learned logit correction bounded +/-1.5' if args.evidence else None,
              'training_augmentation':'0/15/30/45 degrees, with/without horizontal mirror; primary evaluation untransformed',
              'split':'5 outer held geographic/acquisition partitions; remaining 4 fit; fixed 0.5 cutoff, no checkpoint/threshold selection',
              'inner_validation':'not used: reviewed positive examples occupy only two connected groups',
              'EMIT_input':False,'EMIT_exterior_training':'unknown, no loss','EMIT_metrics':'catalog footprint agreement proxy, not verified precision/FPR',
              'quantitative_enhancement_trained':False,'source_origin_trained':False,'nationally_validated':False,
              'observations':[x['row'] for x in items],'excluded':excluded,'code_sha256':digest(__file__)}
    write_json(out/'GTM_Model6_plume_manifest.json',manifest)
    np.savez_compressed(out/'GTM_Model6_plume_inputs.npz',**{f'{i}_{k}':x[k] for i,x in enumerate(items) for k in ['x','target','known','valid','reference_mask','reference_valid','rgb']})
    (out/'GTM_Model6_plumes_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'items':len(items),'groups':len({x['row']['group'] for x in items}),
       'partitions':{f:{d:sum(x['row']['partition']==f and x['row']['dataset']==d for x in items) for d in ['MARS','EMIT']} for f in range(5)},'excluded':len(excluded)}),flush=True)
    if args.audit_only:return
    device='cuda' if torch.cuda.is_available() else 'cpu';results=[];models=[]
    for fold in range(5):
        train=[x for x in items if x['row']['partition']!=fold]
        val=[];held=[x for x in items if x['row']['partition']==fold]
        predictions={};val_predictions={}
        for size in WINDOWS:
            if time.monotonic()-start>args.minutes*60:raise SystemExit('Compute cap reached; partial run preserved')
            model,log=fit(train,size,args.steps,device,SEED+100*fold+size,evidence=args.evidence)
            path=out/f'GTM_Model6_plume_S{size}_fold{fold}.pt'
            torch.save({'state_dict':model.cpu().state_dict(),'window':size,'fold':fold,'evidence':args.evidence,'training_ids':[x['row']['id'] for x in train],
                        'feature_contract':'five-band target and reference scene-centered log reflectance, temporal log ratio, differential SWIR; 16 channels','log':log},path)
            model.to(device);predictions[f'S{size}']=[predict(model,x['x'],size,device) for x in held]
            val_predictions[f'S{size}']=[predict(model,x['x'],size,device) for x in val]
            models.append({'fold':fold,'window':size,'path':path.name,'sha256':digest(path),'log':log})
            print(json.dumps({'fold':fold,'window':size,'loss':log[-1]['loss'],'elapsed':round(time.monotonic()-start,1)}),flush=True)
        predictions['mean']=list(np.mean([predictions[f'S{s}'] for s in WINDOWS],axis=0))
        val_predictions['mean']=list(np.mean([val_predictions[f'S{s}'] for s in WINDOWS],axis=0))
        thresholds={k:threshold_from_validation(val_predictions[k],val) for k in predictions}
        # Image-independent spatial prior exposes crop-centering shortcuts.
        reviewed=[x for x in train if x['row']['fully_labeled'] and x['target'].any()]
        prior=np.mean([x['reference_mask'] for x in reviewed],axis=0) if reviewed else np.zeros(items[0]['target'].shape,dtype='float32')
        prior_t,prior_table=threshold_from_validation([prior for x in val],val)
        for i,x in enumerate(held):
            idx=items.index(x) if False else next(j for j,z in enumerate(items) if z['row']['id']==x['row']['id'])
            arrays={k:p[i] for k,p in predictions.items()};arrays['spatial_prior']=prior
            if args.evidence:arrays['spectral_baseline']=1/(1+np.exp(-x['x'][-1]))
            np.savez_compressed(out/f'GTM_Model6_plume_prediction_{idx}.npz',**arrays)
            record={'index':idx,'id':x['row']['id'],'fold':fold,'dataset':x['row']['dataset'],'label':x['row']['label'],
                    'group':x['row']['group'],'thresholds':{k:v[0] for k,v in thresholds.items()},
                    'metrics':{k:metrics(p,x,thresholds[k][0]) for k,p in arrays.items() if k in thresholds}}
            record['thresholds']['spatial_prior']=prior_t
            record['metrics']['spatial_prior']=metrics(prior,x,prior_t)
            record['metrics']['all_positive']=metrics(np.ones_like(x['target']),x,.5)
            record['metrics']['all_negative']=metrics(np.zeros_like(x['target']),x,.5)
            if args.evidence:
                record['metrics']['spectral_baseline']=metrics(arrays['spectral_baseline'],x,.5)
                record['thresholds']['spectral_baseline']=.5
            results.append(record)
        write_json(out/f'GTM_Model6_plume_thresholds_fold{fold}.json',{'validation_ids':[x['row']['id'] for x in val],
             'thresholds':thresholds,'spatial_prior':(prior_t,prior_table)})
        write_json(out/'GTM_Model6_plume_results.json',{'records':results,'models':models,'elapsed_seconds':time.monotonic()-start,'complete':False})
    write_json(out/'GTM_Model6_plume_results.json',{'records':results,'models':models,'elapsed_seconds':time.monotonic()-start,'complete':True})
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__':main()
