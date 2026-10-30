"""Real FIT preflight and fresh-process CPU replay for frozen v12 arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_validation_v12 as t


def expect_failure(name, action):
    try:
        action()
    except (ValueError, FloatingPointError, KeyError):
        return name
    raise AssertionError("Failure control unexpectedly passed: " + name)


def independent_loss(logits, target, valid, positive):
    z, y, m = (a.astype(np.float64) for a in (logits, target, valid))
    raw = (1-y)*np.logaddexp(0, z) + 2*y*np.logaddexp(0, -z)
    bce = float((raw*m).sum()/m.sum())
    p = 1/(1+np.exp(-z))
    selected = positive[:, None, None, None]*m
    dice = 1-(2*(p*y*selected).sum()+1)/((p*selected).sum()+(y*selected).sum()+1)
    return np.array([bce+.1*dice, bce, dice])


def batch_loss(sample):
    x, y, valid, flag = sample
    torch.manual_seed(912)
    logits = torch.randn_like(y) * .8
    actual = np.array([float(v.item()) for v in t.loss_parts(logits, y, valid, flag)])
    independent = independent_loss(logits.numpy(), y.numpy(), valid.numpy(), flag.numpy())
    if not np.allclose(actual, independent, atol=2e-5, rtol=1e-5):
        raise AssertionError(f"Independent fractional loss mismatch: {actual} vs {independent}")
    return actual.tolist()


def expanded_native_bce(logits, target, valid):
    z = np.repeat(np.repeat(logits.astype(np.float64), 2, 2), 2, 3)
    y = target.astype(np.float64)
    m = valid.astype(np.float64)
    raw = (1-y)*np.logaddexp(0, z) + 2*y*np.logaddexp(0, -z)
    return float((raw*m).sum()/m.sum())


def preflight(manifest: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rows, receipt, split_path, receipt_path = t.frozen_inputs(manifest)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    members = split["members"]
    for role, expected in (("fit", "train"), ("select", "dev"), ("assess", "assessment")):
        if set(members[role]) != {s["id"] for s in rows if s["split"] == expected}:
            raise ValueError("Split membership drift: " + role)
    if any(set(members[a]) & set(members[b]) for a,b in (("fit","select"),("fit","assess"),("select","assess"))):
        raise ValueError("Cross-split observation overlap")
    # The split receipt itself pins physical, proximity, acquisition, observation,
    # and spectral-source boundary checks before any pixels are read here.
    if receipt["boundary_checks"] != "physical, 25km, acquisition, observation, spectral image hash clear":
        raise ValueError("Split boundary audit missing")
    fit = [s for s in rows if s["split"] == "train"]
    by_id = {s["id"]: s for s in rows}
    controls = []
    controls.append(expect_failure("wrong_arm", lambda: t.configure({"arm": "invalid"})))
    controls.append(expect_failure("changed_manifest", lambda: t.frozen_inputs(t.PROTOCOL)))
    t.configure({"arm": "control"})
    norm = t.normalization(manifest, rows)
    ratio_norm = t.ratio_normalization(manifest, rows)
    norm["mbmp"] = ratio_norm
    member_ids = sorted(s["id"] for s in fit)
    member_hash = hashlib.sha256(json.dumps([(s["id"],s["array_sha256"]) for s in sorted(fit,key=lambda x:x["id"])], separators=(",", ":")).encode()).hexdigest()
    if set(member_ids) != set(members["fit"]) or len(member_ids) != 670:
        raise ValueError("Normalizer includes non-FIT identities")
    draws = t.draw_schedule(rows)
    if len(draws) != 38400 or {d[0] for d in draws} - set(member_ids):
        raise ValueError("Draw schedule includes held-out identity")
    if any(sum(d[2] for d in draws[i:i+8]) != 4 or
           [d[2] for d in draws[i:i+8]] != [1]*4+[0]*4 for i in range(0,len(draws),8)):
        raise ValueError("Four/four batch schedule changed")
    seen = {True:set(),False:set()}
    for sid,_,flag in draws:
        seen[bool(flag)].add(by_id[sid]["physical_location_id"])
    inventory = {flag:{s["physical_location_id"] for s in fit if (s["scene_kind"]=="PLUME") == flag} for flag in (True,False)}
    if any(seen[flag] != inventory[flag] for flag in (True,False)):
        raise ValueError("Draw schedule did not reach every FIT site")
    first = t.load_scene(manifest, by_id[draws[0][0]])
    valid = first["valid"].astype(bool)
    t.configure({"arm":"control"})
    control = t.prepare(first,norm)
    dates = t.v7.v5.prepare(first,norm)
    a,b = dates[...,:6],dates[...,6:]
    expected = np.concatenate((a,b,a-b,np.abs(a-b)),axis=-1)
    if not np.array_equal(control,expected) or np.any(control[~valid]):
        raise ValueError("v7 feature parity failed")
    t.configure({"arm":"fraction"})
    if not np.array_equal(t.prepare(first,norm),control):
        raise ValueError("Control/fraction feature parity failed")
    t.configure({"arm":"ratio"})
    feature = t.prepare(first,norm)
    raw_ratio = t.ratio_log(first)
    expected_ratio = np.zeros((100,100),np.float32)
    expected_ratio[valid] = np.clip((raw_ratio[valid]-ratio_norm["mean"])/ratio_norm["scale"],-8,8)
    if not np.array_equal(feature[...,:24],control) or not np.array_equal(feature[...,24],expected_ratio) or not np.isfinite(feature).all():
        raise ValueError("MBMP feature parity failed")
    controls.append(expect_failure("missing_support",lambda: t.load_scene(manifest,{**by_id[draws[0][0]],"array_sha256":"0"*64})))
    invalid = {k:v.copy() for k,v in first.items()}
    invalid["valid_native"][0,0] = ~invalid["valid_native"][0,0]
    # The shared validator is exercised on a temporary in-memory NPZ below.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as td:
        path = Path(td)/"malformed.npz"
        np.savez_compressed(path,**invalid)
        fake = {**by_id[draws[0][0]],"array_file":str(path),"source_array_file":str(path),"array_sha256":t.sha(path)}
        controls.append(expect_failure("invalid_native_support",lambda: t.load_scene(manifest,fake)))
        bad = {k:v.copy() for k,v in first.items()}
        bad["label_native"][0,0] = 2
        np.savez_compressed(path,**bad)
        fake["array_sha256"] = t.sha(path)
        controls.append(expect_failure("nonbinary_native_label",lambda: t.load_scene(manifest,fake)))
        bad["label_native"] = first["label_native"].copy()
        bad["target_image"] = first["target_image"].copy()
        yx = np.argwhere(valid)[0]
        bad["target_image"][yx[0],yx[1],0] = np.nan
        np.savez_compressed(path,**bad)
        fake["array_sha256"] = t.sha(path)
        controls.append(expect_failure("nonfinite_supported_radiometry",lambda: t.load_scene(manifest,fake)))
    arms = {}
    initial = {}
    for arm in t.ARMS:
        t.configure({"arm":arm})
        torch.manual_seed(20261026)
        model = t.new_model("siamese")
        initial[arm] = {k:v.detach().clone() for k,v in model.state_dict().items()}
        batch = t.make_batch(manifest,by_id,draws[:8],norm,{})
        x,y,m,flag = batch
        if x.shape != (8,25 if arm=="ratio" else 24,100,100) or y.shape != (8,1,100,100):
            raise ValueError("Real balanced batch shape changed")
        for i,(sid,transform,_) in enumerate(draws[:8]):
            data = t.load_scene(manifest,by_id[sid])
            expected_x = t.v7.v5.v4.v3.old.d4(t.prepare(data,norm),transform)
            expected_y = t.v7.v5.v4.v3.old.d4(t.target20(data),transform)
            expected_m = t.v7.v5.v4.v3.old.d4(data["valid"].astype(bool),transform)
            if (not np.array_equal(x[i].permute(1,2,0).numpy(),expected_x) or
                not np.array_equal(y[i,0].numpy(),expected_y) or
                not np.array_equal(m[i,0].numpy().astype(bool),expected_m)):
                raise ValueError("D4 real input/target/support misalignment")
        if arm=="fraction":
            logits = torch.randn_like(y).numpy()
            native = np.stack([t.v7.v5.v4.v3.old.d4(t.load_scene(manifest,by_id[sid])["label_native"], transform)
                               for sid,transform,_ in draws[:8]])[:,None]
            native_valid = np.repeat(np.repeat(m.numpy(),2,2),2,3)
            frac_bce = independent_loss(logits,y.numpy(),m.numpy(),flag.numpy())[1]
            native_bce = expanded_native_bce(logits,native,native_valid)
            if abs(frac_bce-native_bce)>1e-6:
                raise ValueError("Fraction BCE does not equal expanded native BCE")
        controls.append(expect_failure(arm+"_wrong_class_flag",lambda: t.loss_parts(torch.zeros_like(y),y,m,1-flag)))
        controls.append(expect_failure(arm+"_empty_support",lambda: t.loss_parts(torch.zeros_like(y),y,torch.zeros_like(m),flag)))
        loss_values = batch_loss(batch)
        optim = torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
        before = {k:v.detach().clone() for k,v in model.state_dict().items()}
        values = t.fit_step(model,optim,batch)
        if not np.isfinite(list(values.values())).all() or not any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):
            raise ValueError("Real optimizer update did not change finite weights")
        model.eval()
        with torch.no_grad():
            direct = model(x[:1]).numpy()
        checkpoint = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        fresh = t.new_model("siamese")
        fresh.load_state_dict(checkpoint)
        fresh.eval()
        with torch.no_grad():
            replay = fresh(x[:1]).numpy()
        if not np.array_equal(direct,replay):
            raise ValueError("CPU checkpoint roundtrip changed predictions")
        arms[arm] = {"loss":loss_values,"step":values,"input_shape":list(x.shape),
                     "parameter_count":sum(p.numel() for p in model.parameters()),
                     "cpu_roundtrip_exact":True}
    for k,v in initial["control"].items():
        if not torch.equal(v,initial["fraction"][k]):
            raise ValueError("Control and fraction initial tensors differ")
        if k=="enc1.layers.0.weight":
            if not torch.equal(v,initial["ratio"][k][:,:24]) or not torch.all(initial["ratio"][k][:,24]==0):
                raise ValueError("Ratio input extension changed common initial tensor")
        elif not torch.equal(v,initial["ratio"][k]):
            raise ValueError("Ratio downstream initial tensor changed")
    if arms["control"]["parameter_count"] != 121937:
        raise ValueError("Control architecture parameter count changed")
    output.mkdir(parents=True)
    t.write_json(output/"normalization.json",norm)
    t.write_json(output/"draw_schedule.json",draws)
    receipt_out = {"schema":"GTM_Model7_validation_v12_preflight", "passed":True,
                   "manifest_sha256":t.sha(manifest),"split_sha256":t.sha(split_path),
                   "split_receipt_sha256":t.sha(receipt_path),**t.source_hashes(),
                   "normalization_sha256":t.sha(output/"normalization.json"),
                   "draw_schedule_sha256":t.sha(output/"draw_schedule.json"),
                   "normalization_member_ids":member_ids,
                   "normalization_member_sha256":member_hash,
                   "fit_only_arrays":670,"select_array_access":"none",
                   "assessment_array_access":"none","regression_array_access":"none",
                   "split_boundary_checks":receipt["boundary_checks"],
                   "schedule_site_inventory":{"plume":len(inventory[True]),"clear":len(inventory[False])},
                   "controls":controls,"arms":arms}
    t.write_json(output/"receipt.json",receipt_out)
    print(json.dumps({"passed":True,"receipt":str(output/"receipt.json"),
                      "trainer_sha256":t.sha(t.HERE),"schedule_sha256":receipt_out["draw_schedule_sha256"],
                      "controls":len(controls),"arms":{k:v["parameter_count"] for k,v in arms.items()}}))


def replay(run: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    metadata_path = run/"metadata.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("schema")!="GTM_Model7_validation_v12" or t.sha(metadata_path)!=json.loads((run/"receipt.json").read_text(encoding="utf-8"))["metadata_sha256"]:
        raise ValueError("Run metadata mismatch")
    t.configure(meta)
    manifest = Path(meta["manifest"])
    rows,_,_,_ = t.frozen_inputs(manifest)
    norm = meta["normalization"]
    if t.sha(run/"draw_schedule.json")!=meta["draw_schedule_file_sha256"]:
        raise ValueError("Run draw schedule changed")
    progress = json.loads((run/"siamese/receipt.json").read_text(encoding="utf-8"))["milestones"]
    output.mkdir(parents=True)
    replay_rows={}
    for step in t.MILESTONES:
        checkpoint=run/"siamese"/f"step_{step}.pt"
        if t.sha(checkpoint)!=progress[str(step)]["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model=t.new_model("siamese")
        model.load_state_dict(saved["model"])
        model.eval()
        comparison={}
        for split in ("train","dev"):
            folder=run/"siamese"/f"{split}_step_{step}"
            expected=progress[str(step)][split]
            if len(list(folder.glob("*.npz")))!=expected["scene_count"]:
                raise ValueError("Saved probability file count changed")
            maximum=0.
            for s in rows:
                if s["split"]!=split:
                    continue
                data=t.load_scene(manifest,s)
                p=t.predict(model,t.prepare(data,norm))
                path=folder/f"{s['id']}.npz"
                if t.sha(path)!=expected["scenes"][s["id"]]["probability_sha256"]:
                    raise ValueError("Saved probability hash changed")
                with np.load(path,allow_pickle=False) as z:
                    maximum=max(maximum,float(np.max(np.abs(p-z["probability20"]))))
                    native=np.repeat(np.repeat(p,2,0),2,1)
                    if (not np.array_equal(z["valid20"],data["valid"].astype(bool)) or
                        not np.array_equal(z["valid_native"],data["valid_native"].astype(bool)) or
                        not np.allclose(z["probability_native"],native,atol=t.REPLAY_ATOL,rtol=0)):
                        raise ValueError("Saved support or native repeat changed")
                c=t.confusion(p,data["target"],data["valid"].astype(bool))
                nc=t.confusion(native,data["label_native"],data["valid_native"].astype(bool))
                if c!=expected["scenes"][s["id"]]["confusion"] or nc!=expected["scenes"][s["id"]]["native_confusion"]:
                    raise ValueError("Fresh CPU native counts changed")
            if maximum>t.REPLAY_ATOL:
                raise ValueError("Fresh CPU score changed")
            comparison[split]={"scene_count":expected["scene_count"],"max_abs_probability20_error":maximum,
                               "support_and_native_counts_exact":True}
        replay_rows[str(step)]={"checkpoint_sha256":t.sha(checkpoint),**comparison}
    result={"schema":"GTM_Model7_validation_v12_replay","passed":True,
            "arm":meta["arm"],"run_metadata_sha256":t.sha(metadata_path),
            "manifest_sha256":t.sha(manifest),"trainer_sha256":t.sha(t.HERE),
            "milestones":replay_rows,"assessment_array_access":"none",
            "regression_array_access":"none"}
    t.write_json(output/"receipt.json",result)
    t.write_json(run/"e2e_receipt.json",result)
    print(json.dumps({"passed":True,"arm":meta["arm"],"receipt":str(output/"receipt.json")}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("preflight","replay"))
    p.add_argument("--manifest",type=Path,default=t.DATA/"combined_manifest.json")
    p.add_argument("--run",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.mode=="preflight":preflight(a.manifest.resolve(),a.output.resolve())
    else:
        if a.run is None:raise ValueError("--run required for replay")
        replay(a.run.resolve(),a.output.resolve())


if __name__=="__main__":main()
