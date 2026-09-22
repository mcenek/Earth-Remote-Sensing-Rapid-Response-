"""Bounded native EMIT-only reconstruction experiment.

This module is deliberately a controlled degradation experiment.  It uses the
three independent US acquisitions for which original EMIT CH4ENH grids are
available locally, with no Sentinel context and no learned merger or source
head.  The target is withheld except for the observed 2x coarse measurement
fed to the reconstruction model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import torch
import torch.nn.functional as F

try:
    from GTM_Model6_reconstruction import ROOT, ReconstructionUNet, digest
except ImportError:  # direct execution from the repository root
    from research.GTM_Model6.GTM_Model6_reconstruction import ROOT, ReconstructionUNet, digest

SEED = 61020
TARGET_STAMPS = ("20241020T170504", "20241130T180310", "20250922T204933")
WINDOWS = (16, 32, 64, 128)
DATA_GLOB = "Earth-Remote-Sensing-Rapid-Response-"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def discover_root() -> Path:
    candidates = [
        ROOT.parent / DATA_GLOB / "EarthRemoteSensingRapidResponse" / "Data Collection" / "EMIT_Plumes" / "emit-v002-authenticated-2026-07",
        ROOT / DATA_GLOB / "EarthRemoteSensingRapidResponse" / "Data Collection" / "EMIT_Plumes" / "emit-v002-authenticated-2026-07",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    # Keep ignored local data discoverable if its parent directory was renamed.
    for p in ROOT.parent.glob("*/EarthRemoteSensingRapidResponse/Data Collection/EMIT_Plumes/emit-v002-authenticated-2026-07"):
        if p.is_dir():
            return p
    raise FileNotFoundError("native EMIT v002 data directory was not found")


def source_record(stamp: str, data_root: Path) -> dict:
    enh_dir = data_root / "CH4ENH"
    enh = next(enh_dir.rglob(f"EMIT_L2B_CH4ENH_*_{stamp}_*.tif"), None)
    if enh is None:
        raise FileNotFoundError(f"CH4ENH missing for {stamp}")
    # The matching CH4PLM is a sibling directory, and is provenance only.
    plm_dir = next(data_root.glob(f"EMIT_L2B_CH4PLM_*_{stamp}_*"), None)
    plm = next(plm_dir.glob("*.tif"), None) if plm_dir else None
    stem = enh.stem.replace("CH4ENH", "CH4SENS")
    sens = enh.with_name(stem + ".tif")
    uncert = enh.with_name(enh.stem.replace("CH4ENH", "CH4UNCERT") + ".tif")
    if not sens.exists() or not uncert.exists() or plm is None or not plm.exists():
        raise FileNotFoundError(f"incomplete ENH/SENS/UNCERT/PLM set for {stamp}")
    paths = {"enh": enh, "uncert": uncert, "sens": sens, "plm": plm}
    arrays = {}
    metadata = {}
    for key, path in paths.items():
        with rasterio.open(path) as src:
            arrays[key] = src.read(1).astype("float32")
            metadata[key] = {"shape": [src.height, src.width], "crs": str(src.crs),
                             "affine": list(src.transform)[:6], "resolution": list(src.res),
                             "nodata": src.nodata, "tags": dict(src.tags()),
                             "dtype": src.dtypes[0]}
    ref = metadata["enh"]
    for key in ("sens", "uncert"):
        if metadata[key]["shape"] != ref["shape"] or metadata[key]["crs"] != ref["crs"] or metadata[key]["affine"] != ref["affine"]:
            raise ValueError(f"grid mismatch between ENH and {key} for {stamp}")
    valid = np.ones(arrays['enh'].shape, dtype=bool)
    for key in ('enh','uncert','sens'):
        value=arrays[key]; valid &= np.isfinite(value) & (value != -9999)
        if metadata[key]['nodata'] is not None: valid &= value != metadata[key]['nodata']
        if key!='enh': valid &= value>0
    if valid.mean()<.01: raise ValueError('Insufficient joint QA support')
    tags=metadata['plm']['tags']
    lat=float(tags['Latitude of max concentration']); lon=float(tags['Longitude of max concentration'])
    if not (-105<lon<-80 and 29<lat<36): raise ValueError('Unexpected US cohort coordinates')
    with rasterio.open(enh) as src: pr,pc=src.index(lon,lat)
    plume=np.zeros(arrays['enh'].shape,dtype='uint8')
    plume_valid=np.isfinite(arrays['plm'])&(arrays['plm']!=-9999)
    if metadata['plm']['nodata'] is not None: plume_valid &= arrays['plm'] != metadata['plm']['nodata']
    reproject(plume_valid.astype('uint8'),plume,src_transform=Affine(*metadata['plm']['affine']),src_crs=metadata['plm']['crs'],dst_transform=Affine(*ref['affine']),dst_crs=ref['crs'],resampling=Resampling.nearest)
    plume = plume.astype(bool)&valid
    arrays['enh']=np.where(valid,arrays['enh'],0).astype('float32')
    return {'id':f'native_{stamp}','stamp':stamp,'paths':{k:str(v) for k,v in paths.items()},
            'hashes':{k:digest(v) for k,v in paths.items()},'metadata':metadata,'shape':list(arrays['enh'].shape),
            'valid_fraction':float(valid.mean()),'peak_rc':[int(pr),int(pc)],'peak_latlon':[lat,lon],
            'arrays':arrays,'valid':valid,'plume_support':plume}



def inventory():
    data_root = discover_root()
    records = [source_record(stamp, data_root) for stamp in TARGET_STAMPS]
    if len(records) != 3:
        raise ValueError("native experiment requires exactly three independent US acquisitions")
    return data_root, records


def crop(a: np.ndarray, top: int, left: int, size: int, fill=np.nan) -> np.ndarray:
    out = np.full((size, size), fill, dtype="float32")
    r0, r1 = max(0, top), min(a.shape[0], top + size)
    c0, c1 = max(0, left), min(a.shape[1], left + size)
    if r1 > r0 and c1 > c0:
        out[r0 - top:r1 - top, c0 - left:c1 - left] = a[r0:r1, c0:c1]
    return out


def patch_positions(record, size: int, held_out: bool, seed: int):
    h, w = record["shape"]
    pr, pc = record["peak_rc"]
    positions = [(max(0, min(h - size, pr - size // 2)), max(0, min(w - size, pc - size // 2)))]
    if held_out:
        rng = np.random.default_rng(seed)
        attempts = 0
        while len(positions) < 5 and attempts < 1000:
            attempts += 1
            r = int(rng.integers(0, max(1, h - size + 1))); c = int(rng.integers(0, max(1, w - size + 1)))
            if float(crop(record["valid"].astype("float32"), r, c, size, 0).mean()) > 0.95 and (r, c) not in positions:
                positions.append((r, c))
        if len(positions) < 5:
            raise ValueError(f"could not find five >95% valid held-out patches for {record['stamp']}")
    return positions


def make_view(y: torch.Tensor, valid: torch.Tensor, angle: float, mirror: bool):
    if mirror:
        y, valid = torch.flip(y, [-1]), torch.flip(valid, [-1])
    rad = math.radians(angle)
    theta = y.new_tensor([[[math.cos(rad), -math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0]]]).expand(len(y), -1, -1)
    grid = F.affine_grid(theta, y.shape, align_corners=False)
    support = F.grid_sample(valid, grid, align_corners=False)
    value = F.grid_sample(y * valid, grid, align_corners=False) / support.clamp_min(1e-8)
    mask = (support >= 0.999).float()
    return value * mask, mask


def degrade(y: torch.Tensor, valid: torch.Tensor):
    coarse = F.avg_pool2d(y * valid, 2) / F.avg_pool2d(valid, 2).clamp_min(1e-8)
    support = F.avg_pool2d(valid, 2)
    known = (support > 0).float()
    up = F.interpolate(coarse * known, size=y.shape[-2:], mode="bilinear", align_corners=False) / F.interpolate(known, size=y.shape[-2:], mode="bilinear", align_corners=False).clamp_min(1e-8)
    cov = F.interpolate(support, size=y.shape[-2:], mode="nearest")
    return up, cov


def tensor_patch(record, pos, size):
    y = crop(record["arrays"]["enh"], *pos, size, 0)
    v = crop(record["valid"].astype("float32"), *pos, size, 0)
    return torch.from_numpy(y[None, None]), torch.from_numpy(v[None, None])


def fit(train, window, steps, seed, device, started, deadline):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    patches = []
    for rec in train:
        rng_local=np.random.default_rng(seed)
        h,w=rec['shape']; plume_r,plume_c=np.where(rec['plume_support'])
        positions=[]; attempts=0
        while len(positions)<32 and attempts<3000:
            attempts+=1
            if len(positions)%2==0 and len(plume_r):
                j=int(rng_local.integers(len(plume_r))); r=int(plume_r[j])-window//2; c=int(plume_c[j])-window//2
                r=max(0,min(h-window,r));c=max(0,min(w-window,c))
            else: r=int(rng_local.integers(h-window+1));c=int(rng_local.integers(w-window+1))
            if rec['valid'][r:r+window,c:c+window].mean()<.8: continue
            positions.append((r,c))
        if len(positions)<32: raise ValueError('Insufficient valid training windows')
        for pos in positions:
            y,v=tensor_patch(rec,pos,window)
            for mirror in (False,True):
                for angle in (0,15,30,45):
                    yy,mm=make_view(y,v,angle,mirror);base,cov=degrade(yy,mm)
                    if mm.sum()==0: raise ValueError('Empty transformed training target')
                    patches.append((torch.cat([torch.zeros((1,5,window,window)),base,cov],1),yy,mm))
    x = torch.cat([p[0] for p in patches]); y = torch.cat([p[1] for p in patches]); m = torch.cat([p[2] for p in patches])
    scale = y[m > 0].abs().quantile(0.95).clamp_min(1).to(device)
    x[:, -2:-1] /= scale.cpu(); y /= scale.cpu()
    model = ReconstructionUNet(channels=7).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); log=[]
    for step in range(steps):
        if time.monotonic() > deadline: raise TimeoutError("compute cap reached during training")
        idx = torch.randint(len(x), (min(16, len(x)),)); xb, yb, mb = x[idx].to(device), y[idx].to(device), m[idx].to(device)
        pred = model(xb); loss = (F.smooth_l1_loss(pred, yb, reduction="none", beta=.1) * mb).sum() / mb.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if step == 0 or step == steps-1 or (step+1) % 100 == 0: log.append({"step": step+1, "loss": float(loss.detach())})
    return model.eval(), {"target_scale": float(scale.cpu())}, log


@torch.no_grad()
def predict_patch(model, norm, record, pos, window, device):
    y, v = tensor_patch(record, pos, window); base, cov = degrade(y, v)
    x = torch.cat([torch.zeros((1, 5, window, window)), base / norm["target_scale"], cov], 1).to(device)
    return model(x).cpu()[0, 0].numpy() * norm["target_scale"], y[0, 0].numpy(), v[0, 0].numpy(), base[0, 0].numpy()


@torch.no_grad()
def tiled(model,norm,base,cov,window,device):
    size=base.shape[-1];pred=np.zeros((size,size),dtype='float32');weights=np.zeros_like(pred)
    hann=np.outer(np.hanning(window),np.hanning(window)).astype('float32')+1e-3
    positions=[(r,c) for r in range(0,size-window+1,window//2) for c in range(0,size-window+1,window//2)]
    for start in range(0,len(positions),32):
        batch=positions[start:start+32]
        xb=torch.cat([torch.cat([torch.zeros((1,5,window,window)),base[:,:,r:r+window,c:c+window]/norm['target_scale'],cov[:,:,r:r+window,c:c+window]],1) for r,c in batch]).to(device)
        ys=model(xb).cpu().numpy()[:,0]*norm['target_scale']
        for (r,c),y in zip(batch,ys):pred[r:r+window,c:c+window]+=y*hann;weights[r:r+window,c:c+window]+=hann
    if not np.all(weights>0):raise ValueError('Tiling left uncovered pixels')
    return pred/weights


def metrics(pred, target, valid):
    e = (pred-target)[valid > 0];
    if e.size==0:return {'mae':None,'rmse':None,'observed_pixels':0}
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.mean(e*e))), "observed_pixels": int(e.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--run-id", required=True); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args(); out = ROOT / "outputs/GTM_Model6" / args.run_id
    if out.exists(): raise SystemExit("Refusing to overwrite an existing run")
    out.mkdir(parents=True); started=time.monotonic(); deadline=started + args.minutes*60; torch.set_num_threads(4)
    data_root, records = inventory(); device = "cuda" if torch.cuda.is_available() else "cpu"
    parent = ROOT / "research/GTM_Model6/GTM_Model6_reconstruction.py"
    manifest = {"task":"native_emit_only_reconstruction", "status":"engineering_diagnostic", "seed":SEED, "steps":args.steps, "minutes":args.minutes, "device":device, "source_root":str(data_root), "acquisition_split":"three independent acquisitions; leave-one-acquisition-out (3 folds)", "windows":list(WINDOWS), "native_resolution_note":"original EMIT grid; expected approximately 0.000542 degrees / 60 m", "inputs":["five zero Sentinel channels","2x masked coarse ENH upsampled bilinear","coarse support coverage"], "target":"withheld native CH4ENH ppm*m; joint finite positive ancillary support, nodata excluded", "augmentation":"mirror x rotations [0,15,30,45] = 8 views", "learned_merger":{"status":"blocked","reason":"only two training acquisitions per fold; requires more independent groups"}, "source_head":{"status":"not_run","reason":"no Gaussian/source claim"}, "code_sha256":digest(__file__), "parent_reconstruction_sha256":digest(parent), "observations":[]}
    for rec in records:
        manifest["observations"].append({k:rec[k] for k in ("id","stamp","paths","hashes","shape","valid_fraction","peak_rc","peak_latlon","metadata")})
    write_json(out/"GTM_Model6_native_manifest.json", manifest)
    if args.audit_only: print(json.dumps({"acquisitions":len(records),"data_root":str(data_root)})); return
    folds=[];all_metrics=[];pred_files=[]
    for fold,held in enumerate(records):
        train=[r for i,r in enumerate(records) if i!=fold];models={};fold_checkpoints=[]
        for window in WINDOWS:
            model,norm,log=fit(train,window,args.steps,SEED+fold*10+window,device,started,deadline)
            checkpoint=out/f'GTM_Model6_native_emit_only_w{window}_fold{fold}.pt'
            torch.save({'state_dict':model.cpu().state_dict(),'normalization':norm,'fold':fold,'window':window,'training_stamps':[r['stamp'] for r in train],'log':log},checkpoint)
            model.to(device);models[window]=(model,norm)
            fold_checkpoints.append({'window':window,'path':checkpoint.name,'sha256':digest(checkpoint),'training_log':log})
            print(f'FOLD {fold+1} WINDOW {window} trained, {time.monotonic()-started:.1f}s',flush=True)
        patches=[]
        for patch_index,pos in enumerate(patch_positions(held,128,True,SEED+fold)):
            if time.monotonic()>deadline:raise TimeoutError('Compute cap reached')
            y,v=tensor_patch(held,pos,128);base,cov=degrade(y,v);target=y.numpy()[0,0];valid=v.numpy()[0,0]
            predictions={f'pred{w}':tiled(*models[w],base,cov,w,device) for w in WINDOWS}
            predictions['uniform_mean']=np.mean(list(predictions.values()),axis=0)
            coarse=base.numpy()[0,0];small=F.avg_pool2d(y*v,2)/F.avg_pool2d(v,2).clamp_min(1e-8)
            predictions['bilinear']=coarse;predictions['nearest']=F.interpolate(small,size=(128,128),mode='nearest').numpy()[0,0]
            predictions['constant']=np.full((128,128),float(small[F.avg_pool2d(v,2)>0].mean()),dtype='float32')
            support=crop(held['plume_support'].astype('float32'),*pos,128,0)
            metrics_row={k:{'all_observed':metrics(a,target,valid),'vetted_plume_support':metrics(a,target,valid*support),'other_observed_not_negative_labels':metrics(a,target,valid*(1-support))} for k,a in predictions.items()}
            key=f'{held["stamp"]}_f{fold}_p{patch_index}';name=f'GTM_Model6_native_prediction_{key}.npz'
            np.savez_compressed(out/name,target=target,valid=valid,coarse=coarse,plume_support=support,**predictions)
            affine=Affine(*held['metadata']['enh']['affine'])*Affine.translation(pos[1],pos[0])
            row={'id':key,'fold':fold,'stamp':held['stamp'],'index':patch_index,'kind':'catalog_peak' if patch_index==0 else 'seeded_random_observed','top_left':list(pos),'affine':list(affine)[:6],'crs':held['metadata']['enh']['crs'],'metrics':metrics_row,'prediction_file':name}
            patches.append(row);all_metrics.append(row);pred_files.append(name)
        folds.append({'fold':fold,'held_out':held['stamp'],'training_stamps':[r['stamp'] for r in train],'checkpoints':fold_checkpoints,'patches':patches})
        write_json(out/'GTM_Model6_native_metrics_partial.json',all_metrics)
    result={'status':'completed_engineering_diagnostic_not_promoted','folds':folds,'observations':all_metrics,'prediction_files':pred_files,'runtime_seconds':time.monotonic()-started,'device':device,'uniform_mean_note':'uniform mean across four separately trained branch outputs, not learned fusion'}
    write_json(out/'GTM_Model6_native_results.json',result)
    print(json.dumps({'status':result['status'],'folds':len(folds),'predictions':len(pred_files),'runtime':result['runtime_seconds']}),flush=True)



if __name__ == "__main__": main()
