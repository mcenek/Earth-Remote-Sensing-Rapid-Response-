"""Bounded native-CPU PyTorch rehearsal of the three-CNN synthetic experiment.

The NumPy bridge of the installed torch build is unavailable. All transfers use
Python lists deliberately; no ndarray is passed to torch or requested from it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Use one recorded numerical policy for fitting and all later replay. Earlier
# float32 attempts failed replay and remain rejected in their original folders.
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import GTM_Model6_three_cnn_data as data
import GTM_Model6_three_cnn_model as reference
import GTM_Model6_three_cnn_runtime as rt

PREFIX = "GTM_Model6_three_cnn"
STAGES = ("airborne", "emit", "fusion")
CONFIG = dict(seed=41, epochs=6, base_channels=4, batch_size=4,
              learning_rate=.003, max_minutes=4.0, tile_size=48, halo=32,
              threshold=.5)


def digest(path):
    return data.sha256(path)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    return {name: digest(HERE / name) for name in
            ("GTM_Model6_three_cnn_torch.py", "GTM_Model6_three_cnn_data.py",
             "GTM_Model6_three_cnn_model.py", "GTM_Model6_three_cnn_runtime.py")}


def runtime_info():
    return dict(backend="pytorch", python=platform.python_version(),
                pytorch=torch.__version__, numpy=np.__version__, device="cpu",
                cpu_threads=torch.get_num_threads(), mkldnn_enabled=torch.backends.mkldnn.enabled,
                computation_dtype="float64", saved_probability_dtype="float32",
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def tensor(array, dtype=torch.float64):
    return torch.tensor(np.asarray(array).tolist(), dtype=dtype)


def array(value, dtype=np.float32):
    return np.asarray(value.detach().cpu().tolist(), dtype=dtype)


def chw(image):
    return tensor(np.moveaxis(image, -1, 0).copy()).unsqueeze(0)


def hwc(value):
    return np.moveaxis(array(value.squeeze(0)), 0, -1)


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            ChannelNorm(outgoing), nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            ChannelNorm(outgoing), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class Branch(nn.Module):
    def __init__(self, base=4):
        super().__init__()
        self.enc0 = Block(5, base)
        self.enc1 = Block(base, base * 2)
        self.bottleneck = Block(base * 2, base * 4)
        self.dec1 = Block(base * 6, base * 2)
        self.feature = Block(base * 3, base)
        self.logits = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(F.max_pool2d(s0, 2))
        x = self.bottleneck(F.max_pool2d(s1, 2))
        x = self.dec1(torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"), s1), 1))
        feature = self.feature(torch.cat((F.interpolate(x, scale_factor=2, mode="nearest"), s0), 1))
        return feature, self.logits(feature)


class Fusion(nn.Module):
    def __init__(self, airborne, emit, base=4):
        super().__init__()
        self.airborne = airborne
        self.emit = emit
        for branch in (airborne, emit):
            branch.eval()
            for parameter in branch.parameters():
                parameter.requires_grad_(False)
        self.airborne_projection = nn.Conv2d(base, base, 1)
        self.emit_projection = nn.Conv2d(base, base, 1)
        self.context = Block(base * 2, base * 2)
        self.decoder = Block(base * 2, base)
        self.logits = nn.Conv2d(base, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        self.airborne.eval()
        self.emit.eval()
        return self

    def intermediates(self, a, e):
        fa, la = self.airborne(a)
        fe, le = self.emit(e)
        joined = torch.cat((F.silu(self.airborne_projection(fa)),
                            F.silu(self.emit_projection(fe))), 1)
        decoded = self.decoder(self.context(joined))
        return fa, fe, joined, decoded, la, le, self.logits(decoded)

    def forward(self, a, e):
        return self.intermediates(a, e)[-1]


def weight_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(np.asarray(value.detach().cpu().tolist()).tobytes())
    return h.hexdigest()


def prediction(model, stage, image, valid, norms):
    h, w = image.shape[:2]
    if stage == "fusion":
        aa, _, _ = rt._pad(rt._normalized(image, valid, norms["airborne"]))
        ee, _, _ = rt._pad(rt._normalized(image, valid, norms["emit"]))
        logits = model(chw(aa), chw(ee))
    else:
        xx, _, _ = rt._pad(rt._normalized(image, valid, norms[stage]))
        _, logits = model(chw(xx))
    return torch.sigmoid(logits[0, 0, :h, :w])


def native_loss(model, stage, a, norms):
    valid = a["input_valid"].astype(bool)
    target = a["target"]
    nh, nw = target.shape
    h, w = valid.shape
    fy, fx = h // nh, w // nw
    support = a["target_valid"].astype(bool) & valid.reshape(nh, fy, nw, fx).all(axis=(1, 3))
    if not support.any():
        raise ValueError("Empty reviewed native support")
    p = prediction(model, stage, a["image"], valid, norms)
    coarse = p.reshape(nh, fy, nw, fx).mean(dim=(1, 3))
    y = tensor(np.nan_to_num(target))
    m = tensor(support.astype(np.float32))
    clamped = coarse.clamp(1e-6, 1 - 1e-6)
    bce = -(m * (y * clamped.log() + (1-y) * (1-clamped).log())).sum() / m.sum()
    dice = 1 - (2 * (m * coarse * y).sum() + 1) / ((m * coarse).sum() + (m * y).sum() + 1)
    return bce + dice


def tiled(model, stage, image, valid, norms, config=CONFIG):
    h, w = valid.shape
    size, halo = config["tile_size"], config["halo"]
    overlap = min(halo, max(0, (size - 1) // 2))
    stride = size - overlap
    summed = np.zeros((h, w), np.float64)
    weights = np.zeros((h, w), np.float64)
    model.eval()
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, x1 = min(y + size, h), min(x + size, w)
                ay, ax = max(0, ((y - halo) // 4) * 4), max(0, ((x - halo) // 4) * 4)
                by = min(h, ((y1 + halo + 3) // 4) * 4)
                bx = min(w, ((x1 + halo + 3) // 4) * 4)
                p = array(prediction(model, stage, image[ay:by, ax:bx],
                                     valid[ay:by, ax:bx], norms))
                core = p[y-ay:y1-ay, x-ax:x1-ax]
                wy, wx = np.ones(y1-y), np.ones(x1-x)
                if overlap:
                    if y > 0:
                        wy[:min(overlap, len(wy))] *= np.linspace(0, 1, min(overlap, len(wy)), endpoint=False) + 1 / overlap
                    if y1 < h:
                        wy[-min(overlap, len(wy)):] *= np.linspace(1, 0, min(overlap, len(wy)), endpoint=False)
                    if x > 0:
                        wx[:min(overlap, len(wx))] *= np.linspace(0, 1, min(overlap, len(wx)), endpoint=False) + 1 / overlap
                    if x1 < w:
                        wx[-min(overlap, len(wx)):] *= np.linspace(1, 0, min(overlap, len(wx)), endpoint=False)
                weight = wy[:, None] * wx[None, :]
                summed[y:y1, x:x1] += core * weight
                weights[y:y1, x:x1] += weight
    if np.any(weights <= 0):
        raise RuntimeError("Uncovered tile pixel")
    result = (summed / weights).astype(np.float32)
    result[~valid] = np.nan
    return result


def stage_paths(run, stage):
    return (run / f"{PREFIX}_{stage}_initial.pt",
            run / f"{PREFIX}_{stage}_checkpoint.pt",
            run / f"{PREFIX}_{stage}_training.json",
            run / f"{PREFIX}_{stage}_results.json")


def load_weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build(stage, run=None):
    if stage != "fusion":
        return Branch(4).double()
    branches = []
    for parent in ("airborne", "emit"):
        branch = Branch(4).double()
        branch.load_state_dict(load_weights(stage_paths(run, parent)[1]))
        branches.append(branch)
    return Fusion(*branches, 4).double()


def stage_rows(manifest, stage, role):
    kind = "emit" if stage == "fusion" else stage
    fit_role = "fusion_fit" if stage == "fusion" else "branch_fit"
    return [r for r in manifest["observations"] if r["role"] == (fit_role if role == "fit" else role)
            and r["target_kind"] == kind]


def stage_norms(manifest, stage, run):
    if stage != "fusion":
        return {stage: rt._normalization(stage_rows(manifest, stage, "fit"))}
    return {parent: json.loads(stage_paths(run, parent)[2].read_text())["normalization"][parent]
            for parent in ("airborne", "emit")}


def metrics(native, target, support):
    return rt._metrics(native, target, support, .5)


def native(p, a):
    valid = a["input_valid"].astype(bool)
    n, complete = reference.native_probability(p, valid, a["target"].shape)
    return n, complete & a["target_valid"].astype(bool)


def validation_loss(model, stage, rows, norms):
    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            losses.append(float(native_loss(model, stage, data.load_arrays(row), norms).item()))
    return float(np.mean(losses))


def train(manifest, run, stage):
    initial, checkpoint, receipt_path, _ = stage_paths(run, stage)
    if any(p.exists() for p in (initial, checkpoint, receipt_path)):
        raise FileExistsError(f"Existing {stage} artifact")
    fit, val = stage_rows(manifest, stage, "fit"), stage_rows(manifest, stage, "validation")
    if not fit or not val:
        raise ValueError("Missing fitting or validation cohort")
    for cohort in (fit, val):
        positive = [bool(np.any(data.load_arrays(r)["target"] == 1)) for r in cohort]
        if not any(positive) or all(positive):
            raise ValueError("Each cohort needs positive and negative scenes")
    norms = stage_norms(manifest, stage, run)
    torch.manual_seed(CONFIG["seed"])
    model = build(stage, run)
    parent_hash = {s: weight_hash(getattr(model, s)) for s in ("airborne", "emit")} if stage == "fusion" else {}
    torch.save(model.state_dict(), initial)
    initial_hash = weight_hash(model)
    initial_val = validation_loss(model, stage, val, norms)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=CONFIG["learning_rate"])
    rng = np.random.default_rng(CONFIG["seed"])
    start = time.monotonic()
    deadline = start + 60 * CONFIG["max_minutes"]
    best, best_loss, stale, history = None, float("inf"), 0, []
    for epoch in range(CONFIG["epochs"]):
        if time.monotonic() >= deadline:
            break
        examples = [(r, aug) for r in fit for aug in range(8)]
        rng.shuffle(examples)
        fit_losses = []
        for offset in range(0, len(examples), CONFIG["batch_size"]):
            if time.monotonic() >= deadline:
                break
            model.train()
            optimizer.zero_grad()
            per = []
            for row, aug in examples[offset:offset+CONFIG["batch_size"]]:
                a = rt._sample_tile(data.load_arrays(row), CONFIG["tile_size"], CONFIG["halo"], rng, aug % 2)
                a = {k: rt._d4(v, aug) for k, v in a.items()}
                per.append(native_loss(model, stage, a, norms))
            loss = torch.stack(per).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fitting loss")
            loss.backward()
            optimizer.step()
            fit_losses.append(float(loss.item()))
        if not fit_losses:
            break
        current = validation_loss(model, stage, val, norms)
        if not np.isfinite(current):
            raise ValueError("Nonfinite validation loss")
        item = dict(epoch=epoch+1, fit_loss=float(np.mean(fit_losses)), validation_loss=current,
                    fit_batches=len(fit_losses), planned_batches=(len(examples)+3)//4)
        history.append(item)
        print(json.dumps({"stage": stage, **item}), flush=True)
        if current < best_loss - 1e-5:
            best_loss = current
            best = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    if best is None:
        raise TimeoutError("No validated checkpoint within stage time cap")
    model.load_state_dict(best)
    if any(weight_hash(getattr(model, s)) != parent_hash[s] for s in parent_hash):
        raise RuntimeError("Frozen in-memory branch changed")
    torch.save(model.state_dict(), checkpoint)
    parents = {s: dict(receipt_sha256=digest(stage_paths(run, s)[2]),
                       checkpoint_sha256=digest(stage_paths(run, s)[1])) for s in parent_hash}
    full = sum(h["fit_batches"] == h["planned_batches"] for h in history)
    result = dict(schema="GTM_Model6_three_cnn_torch_v1", backend="pytorch", data_kind="synthetic",
                  stage=stage, manifest_sha256=manifest["manifest_sha256"],
                  manifest_path=manifest["path"], config=CONFIG,
                  initial=dict(path=initial.name, sha256=digest(initial), weight_hash=initial_hash),
                  checkpoint=dict(path=checkpoint.name, sha256=digest(checkpoint), weight_hash=weight_hash(model)),
                  source_hashes=source_hashes(), normalization=norms,
                  fit_ids=[r["id"] for r in fit], validation_ids=[r["id"] for r in val],
                  parents=parents, frozen_weights_unchanged={s: True for s in parent_hash},
                  requested_epochs=CONFIG["epochs"], completed_epochs=full,
                  partial_epochs=len(history)-full, early_stopped=stale >= 3,
                  history=history, initial_validation_loss=initial_val,
                  trained_validation_loss=validation_loss(model, stage, val, norms),
                  elapsed_seconds=round(time.monotonic()-start, 3), runtime=runtime_info(),
                  interpretation="synthetic engineering check only")
    save_json(receipt_path, result)
    return result


def predict_stage(manifest, run, stage):
    initial, checkpoint, receipt_path, result_path = stage_paths(run, stage)
    if result_path.exists():
        raise FileExistsError(result_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt["manifest_sha256"] != manifest["manifest_sha256"] or digest(checkpoint) != receipt["checkpoint"]["sha256"]:
        raise ValueError("Changed manifest or checkpoint")
    if receipt["source_hashes"] != source_hashes() or receipt["runtime"] != runtime_info():
        raise ValueError("Training source or numerical runtime changed")
    model = build(stage, run)
    model.load_state_dict(load_weights(checkpoint))
    records = []
    for row in stage_rows(manifest, stage, "test"):
        a = data.load_arrays(row)
        valid = a["input_valid"].astype(bool)
        p = tiled(model, stage, a["image"], valid, receipt["normalization"])
        n, support = native(p, a)
        m = metrics(n, a["target"], support)
        finite = p[valid]
        name = f"{PREFIX}_{stage}_{row['id']}_prediction.npz"
        path = run / name
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, probability=p, valid=valid.astype(np.uint8),
                            native_probability=n, target=a["target"], target_valid=support.astype(np.uint8))
        flags = ["synthetic_engineering_only"]
        if not np.any(finite >= .5): flags.append("empty_threshold_mask")
        if np.all(finite >= .5): flags.append("all_positive")
        if np.ptp(finite) < 1e-5: flags.append("constant_probability")
        records.append(dict(id=row["id"], role=row["role"], target_kind=row["target_kind"],
                            group=row["group"], prediction=dict(path=name, sha256=digest(path)),
                            native_metrics=m, quality_flags=flags,
                            probability_min=float(finite.min()), probability_max=float(finite.max()),
                            predicted_area_pixels=int(np.sum(finite >= .5)), valid_pixels=int(valid.sum()),
                            total_pixels=int(valid.size), native_supported_cells=int(support.sum())))
    result = dict(schema="GTM_Model6_three_cnn_torch_v1", backend="pytorch", data_kind="synthetic",
                  stage=stage, manifest_sha256=manifest["manifest_sha256"],
                  checkpoint=receipt["checkpoint"], threshold=.5, config=CONFIG,
                  source_hashes=source_hashes(), training_receipt_sha256=digest(receipt_path),
                  records=records, replay_atol=1e-5,
                  interpretation="synthetic engineering check only")
    save_json(result_path, result)
    return result


def aggregate(records, key):
    counts = {k: sum(r[key][k] for r in records) for k in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return dict(**counts, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                pixel_false_positive_rate=fp/(fp+tn) if fp+tn else None)


def audit(manifest, run, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cohort = Counter()
    for row in manifest["observations"]:
        a = data.load_arrays(row)
        positive = bool(np.any(a["target"][a["target_valid"].astype(bool)] == 1))
        cohort[(row["target_kind"], row["role"], "positive" if positive else "negative")] += 1
    report = dict(schema="GTM_Model6_learning_audit_v1", backend="pytorch", run_id=run.name,
                  data_kind="synthetic", claim="Artificial spectral-cue learning and software evidence only; no methane detection accuracy",
                  manifest_sha256=manifest["manifest_sha256"], source_hashes=source_hashes(),
                  cohort={"/".join(k): v for k,v in sorted(cohort.items())}, runtime=runtime_info(),
                  stages={}, invocation=sys.argv,
                  random_control="Saved initial state and fitted normalization; CNN 3 initial state has trained frozen branches")
    for stage in STAGES:
        initial, checkpoint, receipt_path, result_path = stage_paths(run, stage)
        receipt = json.loads(receipt_path.read_text())
        results = json.loads(result_path.read_text())
        if receipt["runtime"] != runtime_info():
            raise ValueError("Numerical runtime changed")
        if (results["manifest_sha256"] != manifest["manifest_sha256"]
                or receipt["manifest_sha256"] != manifest["manifest_sha256"]
                or results["training_receipt_sha256"] != digest(receipt_path)
                or results["source_hashes"] != source_hashes()):
            raise ValueError("Changed prediction lineage")
        if any(receipt["source_hashes"].get(k) != v for k,v in source_hashes().items()):
            raise ValueError("Changed source")
        if digest(initial) != receipt["initial"]["sha256"] or digest(checkpoint) != receipt["checkpoint"]["sha256"]:
            raise ValueError("Changed weights")
        for parent, lineage in receipt["parents"].items():
            if digest(stage_paths(run, parent)[2]) != lineage["receipt_sha256"] or digest(stage_paths(run, parent)[1]) != lineage["checkpoint_sha256"]:
                raise ValueError("Changed frozen parent")
        model = build(stage, run)
        model.load_state_dict(load_weights(initial))
        ih = weight_hash(model)
        if ih != receipt["initial"]["weight_hash"]:
            raise ValueError("Initial weight mismatch")
        frozen_before = {s: weight_hash(getattr(model,s)) for s in receipt["parents"]}
        rows = stage_rows(manifest, stage, "test")
        untrained = {r["id"]: tiled(model, stage, data.load_arrays(r)["image"],
                                     data.load_arrays(r)["input_valid"].astype(bool), receipt["normalization"]) for r in rows}
        model.load_state_dict(load_weights(checkpoint))
        th = weight_hash(model)
        if ih == th or th != receipt["checkpoint"]["weight_hash"]:
            raise ValueError("Trained weight mismatch or no optimization")
        if any(weight_hash(getattr(model,s)) != frozen_before[s] for s in frozen_before):
            raise ValueError("Frozen branch changed")
        saved = {r["id"]: r for r in results["records"]}
        if len(saved) != 80 or set(saved) != {r["id"] for r in rows}:
            raise ValueError("Incomplete held-out cohort")
        records, arrays_to_keep, max_replay = [], {}, 0.0
        for row in rows:
            a = data.load_arrays(row)
            valid = a["input_valid"].astype(bool)
            p = tiled(model, stage, a["image"], valid, receipt["normalization"])
            n, support = native(p, a)
            before, _ = native(untrained[row["id"]], a)
            rec = saved[row["id"]]
            path = run / rec["prediction"]["path"]
            if digest(path) != rec["prediction"]["sha256"]:
                raise ValueError("Changed saved prediction")
            with np.load(path, allow_pickle=False) as z:
                diff = float(np.max(np.abs(p[valid] - z["probability"][valid])))
            if diff > 1e-5:
                raise ValueError(f"Replay difference {row['id']}: {diff}")
            max_replay = max(max_replay, diff)
            measured = metrics(n, a["target"], support)
            if any(measured[k] != rec["native_metrics"][k] for k in ("tp","fp","fn","tn")):
                raise ValueError("Saved count mismatch")
            positive = bool(np.any(a["target"][support] == 1))
            records.append(dict(id=row["id"], group=row["group"], positive=positive,
                                trained=measured, untrained=metrics(before,a["target"],support),
                                always_negative=metrics(np.zeros_like(n),a["target"],support),
                                always_positive=metrics(np.ones_like(n),a["target"],support),
                                probability_max=float(np.nanmax(n[support])), replay_max_abs=diff,
                                fine_threshold_cells=int(np.sum(valid & (p >= .5))),
                                fine_supported_cells=int(valid.sum())))
            arrays_to_keep[row["id"]] = dict(image=a["image"], input_valid=a["input_valid"],
                target=a["target"], target_valid=a["target_valid"], native_probability=n,
                native_untrained=before, probability=p, untrained=untrained[row["id"]])
        positives = sorted((r for r in records if r["positive"]), key=lambda r:r["trained"]["iou"])
        negatives = sorted((r for r in records if not r["positive"]), key=lambda r:r["trained"]["fp"], reverse=True)
        selections = dict(worst_positive=positives[0], median_positive=positives[len(positives)//2],
                          best_positive=positives[-1], worst_negative=negatives[0],
                          median_negative=negatives[len(negatives)//2])
        for label, rec in selections.items():
            name = f"{stage}_{label}.npz"
            np.savez_compressed(output/name, **arrays_to_keep[rec["id"]])
            rec.setdefault("evidence_files", []).append(name)
        fitting = []
        for desired in (True, False):
            for row in stage_rows(manifest,stage,"fit"):
                a = data.load_arrays(row)
                positive = bool(np.any(a["target"][a["target_valid"].astype(bool)] == 1))
                if positive != desired: continue
                valid = a["input_valid"].astype(bool)
                payload = dict(a, probability=tiled(model,stage,a["image"],valid,receipt["normalization"]))
                if stage == "fusion":
                    na = rt._normalized(a["image"],valid,receipt["normalization"]["airborne"])
                    ne = rt._normalized(a["image"],valid,receipt["normalization"]["emit"])
                    with torch.no_grad():
                        fa, fe, *_ = model.intermediates(chw(na),chw(ne))
                    payload.update(airborne_features=hwc(fa),emit_features=hwc(fe))
                name = f"{stage}_fit_{'positive' if positive else 'negative'}.npz"
                np.savez_compressed(output/name,**payload)
                fitting.append(dict(id=row["id"],role=row["role"],positive=positive,file=name,input_sha256=row["input"]["sha256"]))
                break
        report["stages"][stage] = dict(history=receipt["history"],config=receipt["config"],
            completed_epochs=receipt["completed_epochs"],requested_epochs=receipt["requested_epochs"],
            partial_epochs=receipt["partial_epochs"],elapsed_seconds=receipt["elapsed_seconds"],
            initial_validation_loss=receipt["initial_validation_loss"],trained_validation_loss=receipt["trained_validation_loss"],
            initial_weight_hash=ih,trained_weight_hash=th,initial_checkpoint_sha256=receipt["initial"]["sha256"],
            checkpoint_sha256=receipt["checkpoint"]["sha256"],frozen_weights_unchanged=receipt["frozen_weights_unchanged"],
            replay_max_abs=max_replay,test_scenes=len(records),positive_scenes=len(positives),negative_scenes=len(negatives),
            trained=aggregate(records,"trained"),untrained=aggregate(records,"untrained"),
            always_negative=aggregate(records,"always_negative"),always_positive=aggregate(records,"always_positive"),
            negative_scenes_with_any_false_activation=sum(r["trained"]["fp"]>0 for r in negatives),
            negative_false_positive_cells=sum(r["trained"]["fp"] for r in negatives),
            negative_evaluated_cells=sum(r["trained"]["evaluated_cells"] for r in negatives),
            negative_scenes_with_any_fine_activation=sum(r["fine_threshold_cells"]>0 for r in negatives),
            negative_fine_threshold_cells=sum(r["fine_threshold_cells"] for r in negatives),
            negative_fine_supported_cells=sum(r["fine_supported_cells"] for r in negatives),
            fitting_examples=fitting,selections={k:v["id"] for k,v in selections.items()},records=records)
        print(json.dumps({"audit_stage":stage,"trained":report["stages"][stage]["trained"],
                          "untrained":report["stages"][stage]["untrained"]}),flush=True)
        if stage == "fusion":
            chosen = selections["median_positive"]
            row = next(r for r in rows if r["id"] == chosen["id"])
            a = data.load_arrays(row)
            valid = a["input_valid"].astype(bool)
            na = rt._normalized(a["image"],valid,receipt["normalization"]["airborne"])
            ne = rt._normalized(a["image"],valid,receipt["normalization"]["emit"])
            with torch.no_grad():
                fa,fe,joined,decoded,la,le,logits = model.intermediates(chw(na),chw(ne))
            payload = dict(image=a["image"],input_valid=a["input_valid"],target=a["target"],
                target_valid=a["target_valid"],normalized_airborne=na,normalized_emit=ne,
                airborne_features=hwc(fa),emit_features=hwc(fe),joined_features=hwc(joined),
                fusion_features=hwc(decoded),airborne_score=array(torch.sigmoid(la)[0,0]),
                emit_score=array(torch.sigmoid(le)[0,0]),fusion_probability=array(torch.sigmoid(logits)[0,0]),
                saved_probability=arrays_to_keep[row["id"]]["probability"])
            np.savez_compressed(output/"feed_forward.npz",**payload)
            diff = float(np.max(np.abs(payload["fusion_probability"][valid]-payload["saved_probability"][valid])))
            report["feed_forward"] = dict(id=row["id"],selection="median CNN3 test-plume IoU",
                shapes={k:list(v.shape) for k,v in payload.items()},full_scene_vs_tiled_max_abs=diff,
                note="CNN1/2 spatial features feed CNN3; references never enter forward pass")
    save_json(output/"audit.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("pipeline","replay"))
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--audit-dir",type=Path)
    args=parser.parse_args()
    manifest=data.load_manifest(args.manifest)
    if manifest["data_kind"]!="synthetic":
        raise ValueError("Native fallback accepts synthetic manifest only")
    if args.command=="pipeline":
        args.run_dir.mkdir(parents=True,exist_ok=True)
        if any(args.run_dir.iterdir()):
            raise FileExistsError("Run directory must be empty")
        for stage in STAGES:
            train(manifest,args.run_dir,stage)
            predict_stage(manifest,args.run_dir,stage)
        if args.audit_dir:
            audit(manifest,args.run_dir,args.audit_dir)
    else:
        if not args.audit_dir:
            raise ValueError("replay requires --audit-dir")
        audit(manifest,args.run_dir,args.audit_dir)


if __name__=="__main__":
    main()
