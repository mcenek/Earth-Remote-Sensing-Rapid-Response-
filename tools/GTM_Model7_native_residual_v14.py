"""Frozen v11 teacher plus bounded native residual corrector; exposed DEV only."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import rasterio
import torch
from torch import nn
from torch.nn import functional as F

import GTM_Model7_native_transfer_v13 as v13

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/GTM_Model7_native_residual_v14_20261026"
PROTOCOL = ROOT / "docs/GTM_Model7_native_residual_v14_protocol_20261026.md"
TEACHER = ROOT / "outputs/GTM_Model7_hard_negative_v11_20261025"
CHECKPOINT = TEACHER / "fit/siamese/step_2400.pt"
CHECKPOINT_SHA = "8a36587380ea14b86501b5bc245b3b26f82f52f380aa5e5e156900491ae39483"
SCHEDULE = v13.OUT / "preflight/draw_schedule.json"
SCHEDULE_SHA = "3efd3da71f6b1e80297903a145afdc329cde553b632d8e305a88d49fb36e5611"
ARMS = ("residual_original_loss", "residual_balanced_loss")
STEPS = (1200, 2400, 3600, 4800)
SEED = 20261026
NORM = None


def sha(path: Path) -> str:
    return v13.sha(path)


def save(path: Path, value: object) -> None:
    v13.save_json(path, value)


def frozen_sources() -> tuple[list[dict], dict, dict, dict]:
    if sha(CHECKPOINT) != CHECKPOINT_SHA or sha(SCHEDULE) != SCHEDULE_SHA:
        raise ValueError("Frozen source changed")
    items = v13.index()
    meta_path = TEACHER / "fit/metadata.json"
    fit_path = TEACHER / "fit/receipt.json"
    inference_path = TEACHER / "inference/receipt.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if (meta["manifest_sha256"] != v13.SOURCE_SHA or
        sha(Path(meta["manifest"])) != v13.SOURCE_SHA or
        fit["metadata_sha256"] != sha(meta_path) or
        fit["models"]["siamese"]["milestones"]["2400"]["checkpoint_sha256"] != CHECKPOINT_SHA or
        inference["manifest_sha256"] != v13.SOURCE_SHA or
        inference["models"]["siamese"]["checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("Teacher checkpoint, split, or receipt mismatch")
    train_ids = set(fit["models"]["siamese"]["milestones"]["2400"]["train"]["scenes"])
    dev_ids = set(inference["models"]["siamese"]["scenes"])
    if train_ids != {r["id"] for r in items if r["split"] == "train"} or dev_ids != {r["id"] for r in items if r["split"] == "dev"}:
        raise ValueError("Teacher/FIT split mismatch")
    if meta["normalization"]["radiometry"] != "log1p_dn_date_band_median_centered":
        raise ValueError("Wrong frozen teacher normalizer")
    return items, meta, fit, inference


def teacher_path(item: dict) -> Path:
    if item["split"] == "train":
        return TEACHER / "fit/siamese/train_step_2400" / f"{item['id']}.npz"
    return TEACHER / "inference/siamese" / f"{item['id']}.npz"


def teacher_expected(item: dict, fit: dict, inference: dict) -> str:
    if item["split"] == "train":
        return fit["models"]["siamese"]["milestones"]["2400"]["train"]["scenes"][item["id"]]["probability_sha256"]
    return inference["models"]["siamese"]["scenes"][item["id"]]["probability_sha256"]


def build_features(a: np.ndarray, b: np.ndarray, valid: np.ndarray, p20: np.ndarray, norm: dict) -> tuple[np.ndarray, np.ndarray]:
    good = valid.astype(bool)
    center = np.asarray(norm["center"], np.float64)[:, None]
    scale = np.asarray(norm["scale"], np.float64)[:, None]
    dates = []
    for raw in (a, b):
        z = np.log1p(raw[:, good].astype(np.float64))
        median = np.median(z, axis=1)[:, None]
        channel = np.zeros((6, 200, 200), np.float32)
        channel[:, good] = np.clip((z - median - center) / scale, -8, 8).astype(np.float32)
        dates.append(channel)
    cur, ref = dates
    mbmp = v13.compute_mbmp(a.astype(np.float32), b.astype(np.float32), valid_mask=good)
    contrast = np.zeros((200, 200), np.float32)
    contrast[good] = np.clip(-10 * np.log(np.maximum(mbmp[good], .0001)), -8, 8)
    p = np.repeat(np.repeat(p20, 2, 0), 2, 1).astype(np.float32)
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    features = np.concatenate((cur, ref, cur-ref, np.abs(cur-ref), contrast[None], (logit/6)[None])).astype(np.float32)
    features[:, ~good] = 0
    if features.shape != (26, 200, 200) or not np.isfinite(features).all():
        raise ValueError("Bad 26-channel native features")
    return features, logit.astype(np.float32)


def prepare() -> None:
    items, meta, fit, inference = frozen_sources()
    dest = OUT / "data"
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    result = []
    for i, item in enumerate(items):
        if sha(teacher_path(item)) != teacher_expected(item, fit, inference):
            raise ValueError("Changed teacher prediction: " + item["id"])
        sources = {s["role"]: s for s in item["sources"]}
        for role in ("target_image", "reference_image"):
            if sha(Path(sources[role]["path"])) != sources[role]["sha256"]:
                raise ValueError("Changed native source: " + item["id"])
        with rasterio.open(sources["target_image"]["path"]) as ds:
            if ds.shape != (200, 200) or tuple(ds.descriptions)[:6] != v13.BANDS:
                raise ValueError("Target band/grid drift")
            a = ds.read(indexes=(1,2,3,4,5,6))
        with rasterio.open(sources["reference_image"]["path"]) as ds:
            if ds.shape != (200, 200) or tuple(ds.descriptions)[:6] != v13.BANDS:
                raise ValueError("Reference band/grid drift")
            b = ds.read(indexes=(1,2,3,4,5,6))
        native = v13.scene(item)
        with np.load(teacher_path(item), allow_pickle=False) as z:
            p20, pn = z["probability20"], z["probability_native"]
            if (not np.array_equal(np.repeat(np.repeat(p20,2,0),2,1), pn) or
                not np.array_equal(z["valid_native"].astype(bool), native["valid"].astype(bool))):
                raise ValueError("Teacher grid/support drift: " + item["id"])
        x, logit = build_features(a, b, native["valid"], p20, meta["normalization"])
        path = dest / f"{item['id']}.npz"
        np.savez_compressed(path, x=x, teacher_logit=logit, teacher_probability=pn,
                            y=native["y"], valid=native["valid"])
        result.append({"id": item["id"], "split": item["split"], "scene_kind": item["scene_kind"], "site": item["site"],
                       "cache": path.name, "sha256": sha(path), "teacher_sha256": teacher_expected(item,fit,inference),
                       "valid_pixels": int(native["valid"].sum()), "label_pixels": int((native["valid"]*native["y"]).sum())})
        if (i+1) % 100 == 0:
            print(json.dumps({"prepared": i+1, "total": len(items)}), flush=True)
    save(dest / "index.json", result)
    save(dest / "receipt.json", {"index_sha256": sha(dest/"index.json"), "script_sha256": sha(Path(__file__)),
          "protocol_sha256": sha(PROTOCOL), "teacher_checkpoint_sha256": CHECKPOINT_SHA,
          "teacher_fit_receipt_sha256": sha(TEACHER/"fit/receipt.json"),
          "teacher_inference_receipt_sha256": sha(TEACHER/"inference/receipt.json"),
          "v13_data_receipt_sha256": sha(v13.OUT/"data/receipt.json"), "source_manifest_sha256": v13.SOURCE_SHA,
          "scenes": len(result), "bytes": sum((dest/r["cache"]).stat().st_size for r in result)})


def index() -> list[dict]:
    path = OUT / "data/index.json"
    receipt = json.loads((OUT/"data/receipt.json").read_text(encoding="utf-8"))
    if (sha(path) != receipt["index_sha256"] or receipt["script_sha256"] != sha(Path(__file__)) or
        receipt["protocol_sha256"] != sha(PROTOCOL) or receipt["teacher_checkpoint_sha256"] != sha(CHECKPOINT) or
        receipt["source_manifest_sha256"] != v13.SOURCE_SHA):
        raise ValueError("v14 data receipt drift")
    return json.loads(path.read_text(encoding="utf-8"))


def scene(item: dict, verify: bool = False) -> dict[str, np.ndarray]:
    path = OUT / "data" / item["cache"]
    if verify and sha(path) != item["sha256"]:
        raise ValueError("v14 cache drift: " + item["id"])
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("x", "teacher_logit", "teacher_probability", "y", "valid")}


def settings() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


class Block(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.GroupNorm(4,co),nn.SiLU(),
                                    nn.Conv2d(co,co,3,padding=1),nn.GroupNorm(4,co),nn.SiLU())
    def forward(self,x):
        return self.layers(x)


class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1,self.enc2,self.bridge = Block(26,16),Block(16,32),Block(32,64)
        self.dec2,self.dec1 = Block(96,32),Block(48,16)
        self.head = nn.Conv2d(16,1,1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(F.avg_pool2d(a,2)); c=self.bridge(F.avg_pool2d(b,2))
        d=self.dec2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode="bilinear",align_corners=False),b),1))
        e=self.dec1(torch.cat((F.interpolate(d,size=a.shape[-2:],mode="bilinear",align_corners=False),a),1))
        return self.head(e)


def new_model() -> Corrector:
    torch.manual_seed(SEED)
    return Corrector().float()


def forward(model: Corrector, x: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    correction = 6*torch.tanh(model(x)/6)
    return teacher + correction, correction


def loss(logits, correction, y, valid, arm):
    if arm not in ARMS or logits.shape != y.shape or y.shape != valid.shape or correction.shape != y.shape:
        raise ValueError("v14 loss contract")
    if not all(torch.isfinite(t).all() for t in (logits, correction, y, valid)):
        raise FloatingPointError("Nonfinite v14 loss inputs")
    positive = y*valid
    negative = (1-y)*valid
    if positive.sum() == 0 or negative.sum() == 0:
        raise ValueError("Missing pixel class")
    if arm == ARMS[0]:
        bce = (F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(2.,device=logits.device),reduction="none")*valid).sum()/valid.sum()
    else:
        raw = F.binary_cross_entropy_with_logits(logits,y,reduction="none")
        bce = .25*(raw*positive).sum()/positive.sum()+.75*(raw*negative).sum()/negative.sum()
    selected = valid*((positive.sum((1,2,3))>0).float())[:,None,None,None]
    p = torch.sigmoid(logits)
    dice = 1-(2*(p*y*selected).sum()+1)/((p*selected).sum()+(y*selected).sum()+1)
    regularizer = .001*(correction.square()*valid).sum()/valid.sum()
    return bce + .1*dice + regularizer


def batch(items, draw, device):
    xs,ts,ys,vs=[],[],[],[]
    for d in draw:
        item=items[d["index"]]
        if item["split"] != "train": raise ValueError("DEV draw")
        data=scene(item)
        x,t,y,v=(data[k] for k in ("x","teacher_logit","y","valid"))
        turns,mirror=d["turns"],d["mirror"]
        x=np.rot90(x,turns,axes=(-2,-1)).copy(); t=np.rot90(t,turns).copy()
        y=np.rot90(y,turns).copy(); v=np.rot90(v,turns).copy()
        if mirror:
            x=x[...,::-1].copy(); t=t[:,::-1].copy(); y=y[:,::-1].copy(); v=v[:,::-1].copy()
        xs.append(x);ts.append(t[None]);ys.append(y[None]);vs.append(v[None])
    return tuple(torch.from_numpy(np.stack(z).astype(np.float32)).to(device) for z in (xs,ts,ys,vs))


def numpy_loss(logits, correction, y, valid, arm):
    pos=y*valid; neg=(1-y)*valid
    raw=np.logaddexp(0,logits)-y*logits
    if arm==ARMS[0]: bce=((raw*(1+y))*valid).sum()/valid.sum()
    else: bce=.25*(raw*pos).sum()/pos.sum()+.75*(raw*neg).sum()/neg.sum()
    selected=valid*((pos.sum(axis=(1,2,3))>0).astype(np.float32))[:,None,None,None]
    prob=1/(1+np.exp(-np.clip(logits,-80,80)))
    dice=1-(2*(prob*y*selected).sum()+1)/((prob*selected).sum()+(y*selected).sum()+1)
    return float(bce+.1*dice+.001*(correction**2*valid).sum()/valid.sum())


def preflight() -> None:
    settings()
    items = index()
    source_items, meta, fit, inference = frozen_sources()
    if [r["id"] for r in items] != [r["id"] for r in source_items]:
        raise ValueError("Scene order mismatch")
    draws = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    if len(draws) != 4800 or any(len(d)!=8 or sum(items[x["index"]]["scene_kind"]=="PLUME" for x in d)!=4 for d in draws):
        raise ValueError("Frozen draw balance mismatch")
    if any(items[x["index"]]["split"]!="train" for d in draws for x in d):
        raise ValueError("Non-TRAIN draw")
    a,b=new_model(),new_model()
    initial_a=hashlib.sha256(b"".join(z.detach().numpy().tobytes() for z in a.state_dict().values())).hexdigest()
    initial_b=hashlib.sha256(b"".join(z.detach().numpy().tobytes() for z in b.state_dict().values())).hexdigest()
    if initial_a!=initial_b or torch.count_nonzero(a.head.weight) or torch.count_nonzero(a.head.bias):
        raise ValueError("Initial identity mismatch")
    max_error=0.; different_masks=0; total=0
    for item in items:
        data=scene(item,verify=True)
        if data["x"].shape!=(26,200,200) or not np.isfinite(data["x"]).all():
            raise ValueError("Feature cache corruption")
        if int(data["valid"].sum())!=item["valid_pixels"] or int((data["y"]*data["valid"]).sum())!=item["label_pixels"]:
            raise ValueError("Native label/support corruption")
        p=1/(1+np.exp(-np.clip(data["teacher_logit"],-80,80)))
        max_error=max(max_error,float(np.max(np.abs(p-data["teacher_probability"]))))
        different_masks+=int(np.count_nonzero((p>=.65)!=(data["teacher_probability"]>=.65)))
        total+=1
    if total!=1156 or max_error>1.1e-6 or different_masks:
        raise ValueError(f"Initial teacher identity failed: {total}, {max_error}, {different_masks}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x,t,y,v=batch(items,draws[0],device)
    model=new_model().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    with torch.no_grad():
        logits,correction=forward(model,x,t)
        if torch.count_nonzero(correction): raise ValueError("Nonzero initial correction")
    checks={}
    for arm in ARMS:
        model.zero_grad(set_to_none=True)
        logits,correction=forward(model,x,t)
        objective=loss(logits,correction,y,v,arm)
        independent=numpy_loss(logits.detach().cpu().numpy().astype(np.float64),correction.detach().cpu().numpy().astype(np.float64),
                               y.cpu().numpy().astype(np.float64),v.cpu().numpy().astype(np.float64),arm)
        if abs(float(objective.detach())-independent)>2e-5:
            raise ValueError("Independent NumPy loss mismatch")
        objective.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Preflight gradient")
        checks[arm]={"torch":float(objective.detach()),"numpy":independent,
                     "head_gradient_norm":float(model.head.weight.grad.norm())}
        if checks[arm]["head_gradient_norm"]<=0: raise ValueError("No correction gradient")
    # One local optimizer update confirms an actual weight change; it is not part of a fit.
    before=model.head.weight.detach().clone()
    optimizer.step()
    changed=float((model.head.weight.detach()-before).abs().max())
    if changed<=0: raise ValueError("No correction update")
    trial=OUT/"preflight/roundtrip.pt"
    trial.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":model.cpu().state_dict(),"initial_state_sha256":initial_a},trial)
    copy=new_model(); copy.load_state_dict(torch.load(trial,map_location="cpu",weights_only=True)["model_state_dict"])
    if any(not torch.equal(z,copy.state_dict()[k]) for k,z in model.state_dict().items()):
        raise ValueError("Checkpoint roundtrip mismatch")
    # Warmed real-batch forward/backward timing, without optimizer updates.
    model=new_model().to(device).train()
    if device.type=="cuda": torch.cuda.synchronize()
    timings=[]
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        start=time.monotonic()
        l,c=forward(model,x,t); loss(l,c,y,v,ARMS[0]).backward()
        if device.type=="cuda": torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
    benchmark=float(np.median(timings[1:]))
    estimated=benchmark*4800
    preflight_result={"passed":True,"scene_count":total,"initial_max_abs_probability_error":max_error,
        "initial_mask_differences_at_065":different_masks,"initial_state_sha256":initial_a,
        "loss_checks":checks,"changed_head_weight_max":changed,
        "device":str(device),"warm_batch_seconds":timings,"estimated_forward_backward_seconds_per_fit":estimated,
        "runtime_suitable_20min":estimated<1200,
        "data_bytes":sum((OUT/"data"/r["cache"]).stat().st_size for r in items),
        "data_receipt_sha256":sha(OUT/"data/receipt.json"),"schedule_sha256":sha(SCHEDULE),
        "teacher_checkpoint_sha256":sha(CHECKPOINT),"teacher_fit_receipt_sha256":sha(TEACHER/"fit/receipt.json"),
        "teacher_inference_receipt_sha256":sha(TEACHER/"inference/receipt.json"),
        "roundtrip_sha256":sha(trial),"script_sha256":sha(Path(__file__)),"protocol_sha256":sha(PROTOCOL)}
    save(OUT/"preflight/receipt.json",preflight_result)
    print(json.dumps({k:preflight_result[k] for k in ("passed","scene_count","initial_max_abs_probability_error","initial_mask_differences_at_065","loss_checks","device","warm_batch_seconds","estimated_forward_backward_seconds_per_fit","runtime_suitable_20min","data_bytes","script_sha256")}),flush=True)


def preflight_receipt() -> dict:
    r=json.loads((OUT/"preflight/receipt.json").read_text(encoding="utf-8"))
    if not r["passed"] or not r["runtime_suitable_20min"] or r["script_sha256"]!=sha(Path(__file__)) or r["protocol_sha256"]!=sha(PROTOCOL) or r["schedule_sha256"]!=sha(SCHEDULE) or r["data_receipt_sha256"]!=sha(OUT/"data/receipt.json"):
        raise ValueError("Unsuitable or changed preflight")
    return r


def checkpoint_path(arm: str, step: int) -> Path:
    return OUT/f"fit_{arm}"/f"step_{step}.pt"


def predict_scene(model, item, device):
    data=scene(item)
    with torch.inference_mode():
        x=torch.from_numpy(data["x"][None]).to(device)
        t=torch.from_numpy(data["teacher_logit"][None,None]).to(device)
        logits,_=forward(model,x,t)
        p=torch.sigmoid(logits)[0,0].cpu().numpy().astype(np.float32)
    if not np.isfinite(p).all(): raise FloatingPointError("Nonfinite scene score")
    return p,data


def save_predictions(model, items, folder, device):
    folder.mkdir(parents=True,exist_ok=False)
    model.eval(); result={}
    for item in items:
        if item["split"]!="dev": continue
        p,data=predict_scene(model,item,device)
        path=folder/f"{item['id']}.npz"
        np.savez_compressed(path,probability_native=p,probability20=p.reshape(100,2,100,2).mean((1,3)),
                            target_native=data["y"],valid_native=data["valid"])
        result[item["id"]]={"probability_sha256":sha(path),"valid_native_pixels":int(data["valid"].sum()),
                            "positive_native_pixels":int((data["y"]*data["valid"]).sum())}
    if len(result)!=198: raise ValueError("Incomplete DEV scores")
    return result


def train(arm: str) -> None:
    if arm not in ARMS: raise ValueError("Unknown arm")
    settings(); pre=preflight_receipt(); items=index()
    draws=json.loads(SCHEDULE.read_text(encoding="utf-8"))
    folder=OUT/f"fit_{arm}"
    if folder.exists(): raise FileExistsError(folder)
    folder.mkdir(parents=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=new_model().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    start=time.monotonic(); history=[]; milestones={}
    for step,draw in enumerate(draws,1):
        if time.monotonic()-start>1200: raise TimeoutError("v14 fit exceeded 20 minutes")
        model.train(); x,t,y,v=batch(items,draw,device)
        optimizer.zero_grad(set_to_none=True)
        logits,correction=forward(model,x,t)
        objective=loss(logits,correction,y,v,arm)
        if not torch.isfinite(objective): raise FloatingPointError("Fit loss")
        objective.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("Fit gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%100==0:
            history.append({"step":step,"loss":float(objective.detach()),"seconds":time.monotonic()-start})
            print(json.dumps({"arm":arm,**history[-1]}),flush=True)
        if step in STEPS:
            path=checkpoint_path(arm,step)
            torch.save({"model_state_dict":model.cpu().state_dict(),"step":step,"arm":arm,
                        "initial_state_sha256":pre["initial_state_sha256"],"protocol_sha256":sha(PROTOCOL)},path)
            model.to(device)
            # CPU is canonical, even if fit uses CUDA.
            records=save_predictions(model.cpu(),items,folder/f"dev_step_{step}",torch.device("cpu"))
            model.to(device)
            milestones[str(step)]={"checkpoint_sha256":sha(path),"dev":{"scenes":records}}
    save(folder/"history.json",history)
    save(folder/"metadata.json",{"arm":arm,"initial_state_sha256":pre["initial_state_sha256"],
          "milestones":milestones,"schedule_sha256":sha(SCHEDULE),"data_index_sha256":sha(OUT/"data/index.json"),
          "preflight_sha256":sha(OUT/"preflight/receipt.json"),"script_sha256":sha(Path(__file__)),
          "protocol_sha256":sha(PROTOCOL),"teacher_checkpoint_sha256":CHECKPOINT_SHA,
          "duration_seconds":time.monotonic()-start,"device":str(device)})
    save(folder/"receipt.json",{"complete":True,"metadata_sha256":sha(folder/"metadata.json"),"history_sha256":sha(folder/"history.json")})


def fit_meta(arm):
    folder=OUT/f"fit_{arm}"
    receipt=json.loads((folder/"receipt.json").read_text(encoding="utf-8"))
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    if (not receipt["complete"] or receipt["metadata_sha256"]!=sha(folder/"metadata.json") or
        meta["script_sha256"]!=sha(Path(__file__)) or meta["protocol_sha256"]!=sha(PROTOCOL) or
        meta["schedule_sha256"]!=sha(SCHEDULE) or meta["data_index_sha256"]!=sha(OUT/"data/index.json")):
        raise ValueError("Changed fit metadata")
    return meta


def load_model(arm,step,meta):
    path=checkpoint_path(arm,step)
    if sha(path)!=meta["milestones"][str(step)]["checkpoint_sha256"]: raise ValueError("Checkpoint hash")
    state=torch.load(path,map_location="cpu",weights_only=True)
    if state["arm"]!=arm or state["step"]!=step or state["initial_state_sha256"]!=meta["initial_state_sha256"]: raise ValueError("Checkpoint identity")
    model=Corrector().float(); model.load_state_dict(state["model_state_dict"])
    return model.eval()


def replay(arm):
    settings(); meta=fit_meta(arm); items=index(); result={}
    for step in STEPS:
        model=load_model(arm,step,meta)
        count=0; difference=0.
        for item in items:
            if item["split"]!="dev": continue
            path=OUT/f"fit_{arm}"/f"dev_step_{step}"/f"{item['id']}.npz"
            if sha(path)!=meta["milestones"][str(step)]["dev"]["scenes"][item["id"]]["probability_sha256"]:
                raise ValueError("Saved DEV score hash")
            fresh,data=predict_scene(model,item,torch.device("cpu"))
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(fresh,z["probability_native"]) or not np.array_equal(data["y"],z["target_native"]) or not np.array_equal(data["valid"],z["valid_native"]):
                    raise ValueError("Non-exact fresh CPU replay: "+item["id"])
            count+=1
        if count!=198: raise ValueError("DEV count")
        result[str(step)]={"scene_count":count,"exact_probability_support_label":True,"max_abs_probability_error":difference}
    save(OUT/f"replay_{arm}/receipt.json",{"passed":True,"arm":arm,"milestones":result,
         "fit_metadata_sha256":sha(OUT/f"fit_{arm}/metadata.json"),"script_sha256":sha(Path(__file__)),"protocol_sha256":sha(PROTOCOL)})


def saved_predictions(arm, step, items, meta):
    result={}
    for item in items:
        if item["split"]!="dev": continue
        path=OUT/f"fit_{arm}"/f"dev_step_{step}"/f"{item['id']}.npz"
        if sha(path)!=meta["milestones"][str(step)]["dev"]["scenes"][item["id"]]["probability_sha256"]:
            raise ValueError("Changed saved DEV prediction")
        with np.load(path,allow_pickle=False) as z:
            result[item["id"]]={k:z[k] for k in ("probability_native","target_native","valid_native")}
    if len(result)!=198: raise ValueError("Incomplete DEV predictions")
    return result


def teacher_predictions(items):
    result={}
    for item in items:
        if item["split"]!="dev": continue
        d=scene(item)
        result[item["id"]]={"probability_native":d["teacher_probability"],"target_native":d["y"],"valid_native":d["valid"]}
    return result


def selection_measure(predictions,items,cutoff,step,baseline):
    row=v13.measure(predictions,items,cutoff,step)
    pooled=row["pooled"]; original=baseline["pooled"]
    clear_fp=sum(row["scene_native_confusion"][r["id"]]["fp"] for r in items if r["split"]=="dev" and r["scene_kind"]=="NO_PLUME")
    base_clear_fp=sum(baseline["scene_native_confusion"][r["id"]]["fp"] for r in items if r["split"]=="dev" and r["scene_kind"]=="NO_PLUME")
    extra={"f1_at_least_teacher_plus_002":pooled["f1"]>=original["f1"]+.02,
           "recall_at_least_teacher":pooled["recall"]>=original["recall"],
           "at_most_18_activated_clear":row["activated_clear_scenes"]<=18,
           "clear_false_pixels_at_most_110pct_teacher":clear_fp<=1.10*base_clear_fp,
           "added_false_pixels_at_most_added_true_pixels":max(0,pooled["fp"]-original["fp"])<=max(0,pooled["tp"]-original["tp"])}
    row["replacement_gates"]=extra
    row["selection_eligible"]=row["selection_eligible"] and all(extra.values())
    row["failed_selection_gates"] += [k for k,v in extra.items() if not v]
    row["clear_false_positive_pixels"]=clear_fp
    row["teacher_delta"]={"tp":pooled["tp"]-original["tp"],"fp":pooled["fp"]-original["fp"],
                          "clear_fp":clear_fp-base_clear_fp}
    return row


def select() -> None:
    items=index(); destination=OUT/"selection"
    if destination.exists(): raise FileExistsError(destination)
    baseline=v13.measure(teacher_predictions(items),items,.65,0)
    measurements={}; choices={}; initial=[]
    for arm in ARMS:
        meta=fit_meta(arm); initial.append(meta["initial_state_sha256"])
        replay_path=OUT/f"replay_{arm}/receipt.json"
        receipt=json.loads(replay_path.read_text(encoding="utf-8"))
        if (not receipt["passed"] or receipt["fit_metadata_sha256"]!=sha(OUT/f"fit_{arm}/metadata.json") or
            set(receipt["milestones"])!={str(s) for s in STEPS} or receipt["script_sha256"]!=sha(Path(__file__))):
            raise ValueError("Fresh CPU replay required")
        candidates=[]
        for step in STEPS:
            preds=saved_predictions(arm,step,items,meta)
            candidates.extend(selection_measure(preds,items,cutoff,step,baseline) for cutoff in v13.CUTOFFS)
        if len(candidates)!=88: raise ValueError("Candidate count")
        eligible=[r for r in candidates if r["selection_eligible"]]
        best=max(candidates,key=v13.rank)
        chosen=max(eligible,key=v13.rank) if eligible else baseline
        choices[arm]={"step":chosen["step"],"cutoff":chosen["cutoff"],"eligible":bool(eligible),
                      "checkpoint":str(checkpoint_path(arm,chosen["step"])) if eligible else str(CHECKPOINT),
                      "checkpoint_sha256":meta["milestones"][str(chosen["step"])]["checkpoint_sha256"] if eligible else CHECKPOINT_SHA,
                      "fallback_teacher_identity":not bool(eligible)}
        measurements[arm]={"candidates":candidates,"eligible_rows":len(eligible),"selected":chosen,
                           "unconstrained_best_rejected_diagnostic":best if not best["selection_eligible"] else None}
    if len(set(initial))!=1: raise ValueError("Different arm initializations")
    eligible_arms=[arm for arm in ARMS if choices[arm]["eligible"]]
    winner=max(eligible_arms,key=lambda arm:v13.rank(measurements[arm]["selected"])) if eligible_arms else None
    destination.mkdir(parents=True)
    save(destination/"measurements.json",{"teacher":baseline,"arms":measurements})
    save(destination/"all_arms_frozen.json",{"passed":True,"choices":choices,"selected_arm":winner,
         "retain_historical_v11":winner is None,"teacher_step":0,"teacher_cutoff":.65,
         "teacher_checkpoint_sha256":CHECKPOINT_SHA,"source_manifest_sha256":v13.SOURCE_SHA,
         "protocol_sha256":sha(PROTOCOL),"script_sha256":sha(Path(__file__)),
         "data_receipt_sha256":sha(OUT/"data/receipt.json"),"schedule_sha256":sha(SCHEDULE),
         "measurements_sha256":sha(destination/"measurements.json"),
         "fit_metadata_sha256":{arm:sha(OUT/f"fit_{arm}/metadata.json") for arm in ARMS},
         "replay_receipt_sha256":{arm:sha(OUT/f"replay_{arm}/receipt.json") for arm in ARMS},
         "selection_kind":"historically exposed DEV 198; no v12 ASSESS or protected final imagery"})


def infer() -> None:
    frozen_path=OUT/"selection/all_arms_frozen.json"
    frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
    if (not frozen["passed"] or frozen["script_sha256"]!=sha(Path(__file__)) or
        frozen["protocol_sha256"]!=sha(PROTOCOL) or frozen["measurements_sha256"]!=sha(OUT/"selection/measurements.json")):
        raise ValueError("Changed selection freeze")
    destination=OUT/"evaluation"
    if destination.exists(): raise FileExistsError(destination)
    items=index(); measurement=json.loads((OUT/"selection/measurements.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True); models={}
    for arm in ARMS:
        choice=frozen["choices"][arm]; folder=destination/arm; folder.mkdir()
        preds=teacher_predictions(items) if choice["fallback_teacher_identity"] else saved_predictions(arm,choice["step"],items,fit_meta(arm))
        scenes={}
        for item in items:
            if item["split"]!="dev": continue
            sid=item["id"]; data=preds[sid]
            path=folder/f"{sid}.npz"; p=data["probability_native"]
            np.savez_compressed(path,probability_native=p,
                probability20=p.reshape(100,2,100,2).mean((1,3)),target_native=data["target_native"],valid_native=data["valid_native"])
            scenes[sid]={"probability_sha256":sha(path),
                "native_confusion":v13.native_confusion(p,data["target_native"],data["valid_native"],choice["cutoff"]),
                "valid_native_pixels":int(data["valid_native"].sum()),
                "positive_native_pixels":int((data["target_native"]*data["valid_native"]).sum())}
        models[arm]={"step":choice["step"],"cutoff":choice["cutoff"],"eligible":choice["eligible"],"scenes":scenes}
    save(destination/"summary.json",{"models":models,"measurements":{arm:measurement["arms"][arm]["selected"] for arm in ARMS},
         "teacher_measurement":measurement["teacher"],"selected_arm":frozen["selected_arm"],
         "selection_freeze_sha256":sha(frozen_path),"source_manifest_sha256":v13.SOURCE_SHA,
         "script_sha256":sha(Path(__file__)),"scene_count":198,
         "grid":"native 200x200 annotation; Sentinel-2 SWIR physical resolution remains 20m"})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","preflight","train","replay","select","infer"))
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.command=="prepare": prepare()
    elif args.command=="preflight": preflight()
    elif args.command=="train":
        if not args.arm: parser.error("--arm required")
        train(args.arm)
    elif args.command=="replay":
        if not args.arm: parser.error("--arm required")
        replay(args.arm)
    elif args.command=="select": select()
    elif args.command=="infer": infer()


if __name__=="__main__": main()
