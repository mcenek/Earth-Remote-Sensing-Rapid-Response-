"""Bounded staged training and replayable Sentinel-only three-CNN prediction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import GTM_Model6_three_cnn_data as data
import GTM_Model6_three_cnn_model as networks

SCHEMA = "GTM_Model6_three_cnn_runtime_v1"
STAGES = ("airborne", "emit", "fusion")
DEFAULTS = dict(seed=17, epochs=3, batch_size=1, base_channels=8,
                learning_rate=0.001, max_minutes=10.0, tile_size=128,
                halo=32, threshold=0.5)


def _tf():
    import tensorflow as tf
    return tf


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights_hash(model):
    h = hashlib.sha256()
    for weight in model.weights:
        value = np.asarray(weight.numpy())
        h.update(str(value.shape).encode())
        h.update(value.tobytes(order="C"))
    return h.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _source_hashes():
    folder = Path(__file__).resolve().parent
    names = ("GTM_Model6_three_cnn_model.py", "GTM_Model6_three_cnn_runtime.py",
             "GTM_Model6_three_cnn_data.py")
    return {name: _hash(folder / name) for name in names}


def _config(config):
    c = dict(DEFAULTS)
    if config:
        if set(config) - set(c):
            raise ValueError(f"Unknown configuration fields: {sorted(set(config)-set(c))}")
        c.update(config)
    for k in ("seed", "epochs", "batch_size", "base_channels", "tile_size", "halo"):
        if isinstance(c[k], bool) or int(c[k]) != c[k]:
            raise ValueError(f"{k} must be an integer")
        c[k] = int(c[k])
    for k in ("learning_rate", "max_minutes", "threshold"):
        c[k] = float(c[k])
        if not math.isfinite(c[k]):
            raise ValueError(f"{k} must be finite")
    if (c["epochs"] < 1 or c["batch_size"] < 1 or c["base_channels"] < 4
            or c["tile_size"] < 4 or c["halo"] < 0 or c["max_minutes"] <= 0
            or c["learning_rate"] <= 0 or not 0 < c["threshold"] < 1):
        raise ValueError("Invalid training configuration")
    return c


def runtime_info():
    try:
        tf = _tf()
        return dict(python=platform.python_version(), tensorflow=tf.__version__,
                    keras=tf.keras.__version__, numpy=np.__version__,
                    devices=[d.name for d in tf.config.list_physical_devices()],
                    gpus=[d.name for d in tf.config.list_physical_devices("GPU")])
    except ImportError as exc:
        return dict(python=platform.python_version(), numpy=np.__version__,
                    tensorflow=None, error=str(exc))


def _rows(manifest):
    return manifest["observations"]


def _fit_rows(manifest, stage):
    if stage == "fusion":
        return [r for r in _rows(manifest) if r["role"] == "fusion_fit" and r["target_kind"] == "emit"]
    return [r for r in _rows(manifest) if r["role"] == "branch_fit" and r["target_kind"] == stage]


def _validation_rows(manifest, stage):
    kind = "emit" if stage == "fusion" else stage
    return [r for r in _rows(manifest) if r["role"] == "validation" and r["target_kind"] == kind]


def _normalization(rows):
    total = np.zeros(5, dtype=np.float64)
    square = np.zeros(5, dtype=np.float64)
    count = 0
    for row in rows:
        arrays = data.load_arrays(row)
        image = np.asarray(arrays["image"], dtype=np.float64)
        valid = np.asarray(arrays["input_valid"], dtype=bool)
        values = image[valid]
        if values.size:
            total += values.sum(axis=0)
            square += (values * values).sum(axis=0)
            count += len(values)
    if count == 0:
        raise ValueError("No valid Sentinel pixels to fit normalization")
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 1e-12))
    return dict(mean=mean.tolist(), std=std.tolist(), count=int(count),
                fitted_ids=[r["id"] for r in rows])


def _normalized(image, valid, norm):
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    x = (np.asarray(image, dtype=np.float32) - mean) / std
    return np.where(np.asarray(valid, bool)[..., None], x, 0).astype(np.float32)


def _class_counts(rows):
    counts = {0: 0, 1: 0}
    for row in rows:
        a = data.load_arrays(row)
        valid = np.asarray(a["input_valid"], bool)
        target = np.asarray(a["target"])
        nh, nw = target.shape
        fy, fx = valid.shape[0] // nh, valid.shape[1] // nw
        supported = np.asarray(a["target_valid"], bool) & valid.reshape(nh, fy, nw, fx).all(axis=(1, 3))
        for cls in counts:
            counts[cls] += int(np.sum(supported & (target == cls)))
    return counts


def _sample_tile(arrays, tile_size, halo, rng, preferred_class):
    """Choose a bounded native-cell aligned tile with reviewed supervision."""
    target = arrays["target"]
    nh, nw = target.shape
    h, w = arrays["input_valid"].shape
    fy, fx = h // nh, w // nw
    cells_y = min(nh, max(1, (tile_size + 2 * halo) // fy))
    cells_x = min(nw, max(1, (tile_size + 2 * halo) // fx))
    support = np.asarray(arrays["target_valid"], bool) & np.asarray(arrays["input_valid"], bool).reshape(nh, fy, nw, fx).all(axis=(1, 3))
    candidates = np.argwhere(support & (target == preferred_class))
    if not len(candidates):
        candidates = np.argwhere(support)
    if not len(candidates):
        raise ValueError("No reviewed target cells with complete Sentinel support")
    cy, cx = candidates[int(rng.integers(len(candidates)))]
    sy = min(max(0, int(cy) - cells_y // 2), nh - cells_y)
    sx = min(max(0, int(cx) - cells_x // 2), nw - cells_x)
    fine = (slice(sy * fy, (sy + cells_y) * fy), slice(sx * fx, (sx + cells_x) * fx))
    native = (slice(sy, sy + cells_y), slice(sx, sx + cells_x))
    return {"image": arrays["image"][fine], "input_valid": arrays["input_valid"][fine],
            "target": target[native], "target_valid": arrays["target_valid"][native]}


def _d4(array, index):
    # Mirror then rotate gives eight distinct square-grid symmetries.
    out = np.flip(array, axis=1) if index >= 4 else array
    return np.rot90(out, index % 4, axes=(0, 1)).copy()


def _pad(x):
    h, w = x.shape[:2]
    return np.pad(x, ((0, (-h) % 4), (0, (-w) % 4), (0, 0))), h, w


def _probability(model, stage, image, valid, norms, training=False):
    tf = _tf()
    h, w = image.shape[:2]
    if stage == "fusion":
        a, _, _ = _pad(_normalized(image, valid, norms["airborne"]))
        e, _, _ = _pad(_normalized(image, valid, norms["emit"]))
        logits = model([tf.convert_to_tensor(a[None]), tf.convert_to_tensor(e[None])], training=training)
    else:
        x, _, _ = _pad(_normalized(image, valid, norms[stage]))
        _, logits = model(tf.convert_to_tensor(x[None]), training=training)
    return tf.math.sigmoid(logits[0, :h, :w, 0])


def _loss(model, stage, arrays, norms):
    tf = _tf()
    image = arrays["image"]
    valid = np.asarray(arrays["input_valid"], dtype=bool)
    target = np.asarray(arrays["target"], dtype=np.float32)
    target_valid = np.asarray(arrays["target_valid"], dtype=bool)
    h, w = valid.shape
    nh, nw = target.shape
    if h % nh or w % nw:
        raise ValueError("Invalid native-grid ratio")
    fy, fx = h // nh, w // nw
    support = valid.reshape(nh, fy, nw, fx).all(axis=(1, 3)) & target_valid
    prob = _probability(model, stage, image, valid, norms, training=True)
    native = tf.reduce_mean(tf.reshape(prob, (nh, fy, nw, fx)), axis=(1, 3))
    return networks.masked_bce_dice(native, tf.convert_to_tensor(np.nan_to_num(target)),
                                     tf.convert_to_tensor(support))


def _validation_loss(model, stage, arrays, norms, config):
    valid = np.asarray(arrays["input_valid"], bool)
    fine = _tiled(model, stage, arrays["image"], valid, norms,
                  config["tile_size"], config["halo"])
    native, complete = networks.native_probability(fine, valid, arrays["target"].shape)
    supported = np.asarray(arrays["target_valid"], bool) & complete
    if not supported.any():
        raise ValueError("Validation has no reviewed supported target cells")
    y = np.asarray(arrays["target"])[supported]
    p = np.clip(native[supported], 1e-6, 1 - 1e-6)
    bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    dice = 1 - (2 * np.sum(p * y) + 1) / (np.sum(p) + np.sum(y) + 1)
    return float(bce + dice)


def _checkpoint_path(run_dir, stage):
    return Path(run_dir) / f"GTM_Model6_three_cnn_{stage}.weights.h5"


def _receipt_path(run_dir, stage):
    return Path(run_dir) / f"GTM_Model6_three_cnn_{stage}_training.json"


def _read_parent(run_dir, stage, manifest, config, source_hashes):
    receipt_path = _receipt_path(run_dir, stage)
    if not receipt_path.is_file():
        raise ValueError(f"Missing {stage} training receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = Path(run_dir) / receipt["checkpoint"]["path"]
    if (receipt.get("schema") != SCHEMA or not receipt.get("usable")
            or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
            or receipt.get("config") != config
            or receipt.get("source_hashes") != source_hashes
            or not checkpoint.is_file() or _hash(checkpoint) != receipt["checkpoint"]["sha256"]):
        raise ValueError(f"Stale, altered, or incompatible {stage} parent checkpoint")
    return receipt


def _read_deployed_receipt(run_dir, stage, source_hashes, expected_manifest_sha256=None,
                           expected_config=None, expected_data_kind=None):
    """Validate saved lineage without opening the historical training manifest."""
    path = _receipt_path(run_dir, stage)
    if not path.is_file():
        raise ValueError(f"Missing {stage} training receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != SCHEMA or receipt.get("stage") != stage
            or receipt.get("usable") is not True
            or receipt.get("source_hashes") != source_hashes):
        raise ValueError(f"Invalid or stale {stage} training receipt")
    manifest_sha = receipt.get("manifest_sha256")
    if (not isinstance(manifest_sha, str) or len(manifest_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in manifest_sha)
            or (expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256)):
        raise ValueError(f"Invalid {stage} training manifest lineage")
    if (receipt.get("data_kind") not in ("synthetic", "measured")
            or (expected_data_kind is not None and receipt["data_kind"] != expected_data_kind)):
        raise ValueError(f"Invalid {stage} training data provenance")
    saved_config = receipt.get("config")
    if not isinstance(saved_config, dict) or _config(saved_config) != saved_config:
        raise ValueError(f"Invalid {stage} model configuration")
    if expected_config is not None and saved_config != expected_config:
        raise ValueError(f"Incompatible {stage} model configuration")
    checkpoint = receipt.get("checkpoint")
    expected_name = _checkpoint_path(run_dir, stage).name
    if (not isinstance(checkpoint, dict) or checkpoint.get("path") != expected_name
            or not isinstance(checkpoint.get("sha256"), str)):
        raise ValueError(f"Invalid {stage} checkpoint reference")
    checkpoint_path = _checkpoint_path(run_dir, stage)
    if not checkpoint_path.is_file() or _hash(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError(f"Altered or missing {stage} checkpoint")
    norms = receipt.get("normalization")
    needed = {"airborne", "emit"} if stage == "fusion" else {stage}
    if not isinstance(norms, dict) or set(norms) != needed:
        raise ValueError(f"Invalid {stage} normalization")
    for norm in norms.values():
        if not isinstance(norm, dict):
            raise ValueError(f"Invalid {stage} normalization")
        mean, std = np.asarray(norm.get("mean")), np.asarray(norm.get("std"))
        if (mean.shape != (5,) or std.shape != (5,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all()
                or not (std > 0).all()):
            raise ValueError(f"Invalid {stage} normalization")
    return receipt


def _build(stage, config, run_dir=None, parents=None):
    if stage != "fusion":
        return networks.build_branch(config["base_channels"], stage)
    a = networks.build_branch(config["base_channels"], "airborne")
    e = networks.build_branch(config["base_channels"], "emit")
    a.load_weights(str(Path(run_dir) / parents["airborne"]["checkpoint"]["path"]))
    e.load_weights(str(Path(run_dir) / parents["emit"]["checkpoint"]["path"]))
    return networks.build_fusion(a, e, config["base_channels"])


def train_stage(manifest_path, run_dir, stage, config=None):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    c = _config(config)
    manifest = data.load_manifest(manifest_path)
    folder = Path(run_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_path(folder, stage)
    receipt_path = _receipt_path(folder, stage)
    if checkpoint.exists() or receipt_path.exists():
        raise FileExistsError(f"Stage {stage} already has an artifact; choose a new run directory")
    fit = _fit_rows(manifest, stage)
    validation = _validation_rows(manifest, stage)
    if not fit or not validation:
        raise ValueError(f"Stage {stage} needs fitting and independent validation rows")
    fit_counts = _class_counts(fit)
    validation_counts = _class_counts(validation)
    if min([*fit_counts.values(), *validation_counts.values()]) < 1:
        raise ValueError("Fitting and validation each require reviewed positive and negative cells")
    hashes = _source_hashes()
    parents = {}
    if stage == "fusion":
        parents = {s: _read_parent(folder, s, manifest, c, hashes) for s in ("airborne", "emit")}
        norms = {s: parents[s]["normalization"][s] for s in parents}
        before = {s: parents[s]["checkpoint"]["sha256"] for s in parents}
    else:
        norms = {stage: _normalization(fit)}
        before = {}
    tf = _tf()
    tf.keras.utils.set_random_seed(c["seed"])
    model = _build(stage, c, folder, parents)
    parent_memory_before = {s: _weights_hash(model.get_layer(s)) for s in parents}
    optimizer = tf.keras.optimizers.Adam(c["learning_rate"])
    history = []
    best_loss = float("inf")
    best_weights = None
    stale_epochs = 0
    early_stopped = False
    start = time.monotonic()
    deadline = start + c["max_minutes"] * 60
    rng = np.random.default_rng(c["seed"])
    for epoch in range(c["epochs"]):
        if time.monotonic() >= deadline:
            break
        examples = [(r, aug) for r in fit for aug in range(8)]
        rng.shuffle(examples)
        losses = []
        for offset in range(0, len(examples), c["batch_size"]):
            if time.monotonic() >= deadline:
                break
            chunk = examples[offset:offset + c["batch_size"]]
            with tf.GradientTape() as tape:
                per = []
                for row, aug in chunk:
                    a = _sample_tile(data.load_arrays(row), c["tile_size"], c["halo"], rng, aug % 2)
                    a = {k: _d4(v, aug) for k, v in a.items()}
                    per.append(_loss(model, stage, a, norms))
                batch_loss = tf.reduce_mean(per)
            variables = model.trainable_variables
            grads = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients([(g, v) for g, v in zip(grads, variables) if g is not None])
            losses.append(float(batch_loss.numpy()))
        if not losses:
            break
        val_losses = [_validation_loss(model, stage, data.load_arrays(r), norms, c) for r in validation]
        current_val = float(np.mean(val_losses))
        history.append(dict(epoch=epoch + 1, fit_loss=float(np.mean(losses)),
                            validation_loss=current_val,
                            fit_batches=len(losses), planned_batches=(len(examples) + c["batch_size"] - 1)//c["batch_size"]))
        if current_val < best_loss - 1e-5:
            best_loss = current_val
            best_weights = [np.asarray(w).copy() for w in model.get_weights()]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 3:
            early_stopped = True
            break
        if time.monotonic() >= deadline:
            break
    if not history:
        raise TimeoutError("Time cap reached before a usable training epoch")
    if best_weights is not None:
        model.set_weights(best_weights)
    # An epoch interrupted mid-fit remains usable but is reported as incomplete.
    full_epochs = sum(h["fit_batches"] == h["planned_batches"] for h in history)
    temp = checkpoint.with_name(checkpoint.stem + f".tmp.{os.getpid()}.weights.h5")
    model.save_weights(str(temp))
    os.replace(temp, checkpoint)
    frozen = {s: (_hash(folder / parents[s]["checkpoint"]["path"]) == before[s]
                  and _weights_hash(model.get_layer(s)) == parent_memory_before[s]) for s in parents}
    if not all(frozen.values()):
        checkpoint.unlink(missing_ok=True)
        raise RuntimeError("Fusion training changed a branch checkpoint")
    result = dict(schema=SCHEMA, data_kind=manifest["data_kind"], stage=stage,
                  manifest_sha256=manifest["manifest_sha256"], manifest_path=str(Path(manifest_path).resolve()),
                  config=c, checkpoint=dict(path=checkpoint.name, sha256=_hash(checkpoint)),
                  source_hashes=hashes, normalization=norms,
                  fit_ids=[r["id"] for r in fit], fit_groups=sorted({r["group"] for r in fit}),
                  validation_ids=[r["id"] for r in validation],
                  validation_groups=sorted({r["group"] for r in validation}),
                  parents={s: dict(receipt_sha256=_hash(_receipt_path(folder, s)),
                                   checkpoint_sha256=before[s]) for s in parents},
                  frozen_weights_unchanged=frozen, requested_epochs=c["epochs"],
                  completed_epochs=full_epochs, history=history,
                  completed=full_epochs == c["epochs"] or early_stopped, usable=True,
                  early_stopped=early_stopped,
                  class_counts=dict(fit=fit_counts, validation=validation_counts),
                  elapsed_seconds=round(time.monotonic() - start, 3), runtime=runtime_info(),
                  interpretation="synthetic engineering check only" if manifest["data_kind"] == "synthetic" else "measured")
    _json(receipt_path, result)
    return result


def _tiled(model, stage, image, valid, norms, tile_size, halo):
    h, w = valid.shape
    summed = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    overlap = min(halo, max(0, (tile_size - 1) // 2))
    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = min(y + tile_size, h), min(x + tile_size, w)
            # Keep the stride-2 encoder on the same global pooling phase.
            ay, ax = max(0, ((y - halo) // 4) * 4), max(0, ((x - halo) // 4) * 4)
            by = min(h, ((y1 + halo + 3) // 4) * 4)
            bx = min(w, ((x1 + halo + 3) // 4) * 4)
            p = _probability(model, stage, image[ay:by, ax:bx],
                             valid[ay:by, ax:bx], norms).numpy()
            core = p[y-ay:y1-ay, x-ax:x1-ax]
            wy = np.ones(y1-y, dtype=np.float64)
            wx = np.ones(x1-x, dtype=np.float64)
            if overlap:
                if y > 0:
                    wy[:min(overlap, len(wy))] *= np.linspace(0, 1, min(overlap, len(wy)), endpoint=False) + 1 / overlap
                if y1 < h:
                    wy[-min(overlap, len(wy)):] *= np.linspace(1, 0, min(overlap, len(wy)), endpoint=False)
                if x > 0:
                    wx[:min(overlap, len(wx))] *= np.linspace(0, 1, min(overlap, len(wx)), endpoint=False) + 1 / overlap
                if x1 < w:
                    wx[-min(overlap, len(wx)):] *= np.linspace(1, 0, min(overlap, len(wx)), endpoint=False)
            weights = wy[:, None] * wx[None, :]
            summed[y:y1, x:x1] += core * weights
            weight_sum[y:y1, x:x1] += weights
    if not np.all(weight_sum > 0):
        raise RuntimeError("Tiling left uncovered pixels")
    output = (summed / weight_sum).astype(np.float32)
    output[~valid] = np.nan
    return output


def _metrics(native, target, support, threshold):
    known = support & np.isfinite(native)
    if not known.any():
        return dict(tp=0, fp=0, fn=0, tn=0, precision=None, recall=None, iou=None,
                    evaluated_cells=0)
    pred = native[known] >= threshold
    truth = target[known] > 0.5
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth)); tn = int(np.sum(~pred & ~truth))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, evaluated_cells=int(known.sum()),
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None)


def predict_run(manifest_path, run_dir, stage, roles=("test",)):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    manifest = data.load_manifest(manifest_path)
    folder = Path(run_dir).resolve()
    hashes = _source_hashes()
    receipt_path = _receipt_path(folder, stage)
    if not receipt_path.is_file():
        raise ValueError("Stage has no training receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    c = _config(receipt.get("config"))
    receipt = _read_parent(folder, stage, manifest, c, hashes)
    parents = {}
    if stage == "fusion":
        parents = {s: _read_parent(folder, s, manifest, c, hashes) for s in ("airborne", "emit")}
        for s in parents:
            if (receipt["parents"][s]["receipt_sha256"] != _hash(_receipt_path(folder, s))
                    or receipt["parents"][s]["checkpoint_sha256"] != parents[s]["checkpoint"]["sha256"]):
                raise ValueError("Fusion parent changed after training")
    model = _build(stage, c, folder, parents)
    model.load_weights(str(folder / receipt["checkpoint"]["path"]))
    result_path = folder / f"GTM_Model6_three_cnn_{stage}_results.json"
    if result_path.exists():
        raise FileExistsError("Prediction results already exist; choose a new run directory")
    role_set = set(roles)
    if not role_set or not role_set <= {"test", "validation", "branch_fit", "fusion_fit"}:
        raise ValueError("Unsupported prediction roles")
    kind = "emit" if stage == "fusion" else stage
    selected = [r for r in _rows(manifest) if r["role"] in role_set and r["target_kind"] == kind]
    if not selected:
        raise ValueError("No rows for requested roles")
    records = []
    for row in selected:
        a = data.load_arrays(row)
        valid = np.asarray(a["input_valid"], dtype=bool)
        probability = _tiled(model, stage, a["image"], valid, receipt["normalization"],
                             c["tile_size"], c["halo"])
        native, native_support = networks.native_probability(probability, valid, a["target"].shape)
        target_valid = np.asarray(a["target_valid"], dtype=bool) & native_support
        target = np.asarray(a["target"], dtype=np.float32)
        metrics = _metrics(native, target, target_valid, c["threshold"])
        finite = probability[np.isfinite(probability)]
        positive = finite >= c["threshold"]
        flags = []
        if not len(finite): flags.append("blank_prediction")
        if len(finite) and np.ptp(finite) < 1e-5: flags.append("constant_probability")
        if len(finite) and positive.all(): flags.append("all_positive")
        if len(finite) and not positive.any(): flags.append("empty_threshold_mask")
        if not metrics["evaluated_cells"]: flags.append("empty_native_support")
        if manifest["data_kind"] == "synthetic": flags.append("synthetic_engineering_only")
        name = f"GTM_Model6_three_cnn_{stage}_{row['id']}_prediction.npz"
        path = folder / name
        if path.exists():
            raise FileExistsError(f"Prediction already exists: {name}")
        temporary = folder / (name + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(temporary, probability=probability.astype(np.float32),
                            valid=valid.astype(np.uint8), native_probability=native,
                            target=target, target_valid=target_valid.astype(np.uint8))
        os.replace(temporary, path)
        records.append(dict(id=row["id"], role=row["role"], target_kind=row["target_kind"],
                            group=row["group"], prediction=dict(path=name, sha256=_hash(path)),
                            native_metrics=metrics, quality_flags=flags,
                            probability_min=float(finite.min()) if len(finite) else None,
                            probability_max=float(finite.max()) if len(finite) else None,
                            predicted_area_pixels=int(positive.sum()),
                            valid_pixels=int(valid.sum()), total_pixels=int(valid.size),
                            native_supported_cells=int(target_valid.sum())))
    result = dict(schema=SCHEMA, data_kind=manifest["data_kind"], stage=stage,
                  manifest_sha256=manifest["manifest_sha256"],
                  checkpoint=receipt["checkpoint"], threshold=c["threshold"],
                  config=c, source_hashes=hashes, training_receipt_sha256=_hash(receipt_path),
                  records=records, replay_atol=1e-5,
                  interpretation="synthetic engineering check only" if manifest["data_kind"] == "synthetic" else "measured")
    _json(result_path, result)
    return result


def infer_scenes(scene_manifest_path, run_dir, stage, output_dir):
    """Predict reviewed Sentinel scenes using saved weights, with no target access."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    scenes = data.load_scenes(scene_manifest_path)
    folder = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Inference output directory already exists")
    hashes = _source_hashes()
    training = _read_deployed_receipt(folder, stage, hashes)
    if scenes["data_kind"] != training["data_kind"]:
        raise ValueError("Scene provenance differs from the training data provenance")
    config = training["config"]
    parents = {}
    if stage == "fusion":
        lineage = training.get("parents")
        if not isinstance(lineage, dict) or set(lineage) != {"airborne", "emit"}:
            raise ValueError("Fusion parent lineage is incomplete")
        for parent_stage in ("airborne", "emit"):
            parent = _read_deployed_receipt(
                folder, parent_stage, hashes,
                expected_manifest_sha256=training["manifest_sha256"],
                expected_config=config, expected_data_kind=training["data_kind"])
            if (lineage[parent_stage].get("receipt_sha256") != _hash(_receipt_path(folder, parent_stage))
                    or lineage[parent_stage].get("checkpoint_sha256") != parent["checkpoint"]["sha256"]):
                raise ValueError("Fusion parent receipt or checkpoint changed")
            parents[parent_stage] = parent
        if training["normalization"] != {s: parents[s]["normalization"][s] for s in parents}:
            raise ValueError("Fusion branch normalization differs from its parents")
    elif training.get("parents") != {}:
        raise ValueError("Branch receipt has unexpected parent lineage")
    model = _build(stage, config, folder, parents)
    model.load_weights(str(_checkpoint_path(folder, stage)))
    # Only create an output location once all provenance and model loading passed.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in scenes["observations"]:
        arrays = data.load_scene_arrays(row)
        valid = np.asarray(arrays["input_valid"], dtype=bool)
        probability = _tiled(model, stage, arrays["image"], valid,
                             training["normalization"], config["tile_size"], config["halo"])
        finite = probability[np.isfinite(probability)]
        positive = finite >= config["threshold"]
        flags = []
        if not len(finite): flags.append("blank_prediction")
        if len(finite) and np.ptp(finite) < 1e-5: flags.append("constant_probability")
        if len(finite) and positive.all(): flags.append("all_positive")
        if len(finite) and not positive.any(): flags.append("empty_threshold_mask")
        if scenes["data_kind"] == "synthetic": flags.append("synthetic_engineering_only")
        digest = hashlib.sha256(row["id"].encode("utf-8")).hexdigest()[:20]
        name = f"GTM_Model6_three_cnn_{stage}_{digest}_inference.npz"
        path = output / name
        temp = output / (name + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(temp, probability=probability.astype(np.float32),
                            valid=valid.astype(np.uint8))
        os.replace(temp, path)
        records.append(dict(id=row["id"], prediction=dict(path=name, sha256=_hash(path)),
                            quality_flags=flags,
                            probability_min=float(finite.min()) if len(finite) else None,
                            probability_max=float(finite.max()) if len(finite) else None,
                            predicted_area_pixels=int(positive.sum()),
                            valid_pixels=int(valid.sum()), total_pixels=int(valid.size)))
    result = dict(schema="GTM_Model6_three_cnn_inference_v1",
                  data_kind=scenes["data_kind"], scene_manifest_sha256=scenes["manifest_sha256"],
                  training_manifest_sha256=training["manifest_sha256"], stage=stage,
                  checkpoint=training["checkpoint"], threshold=config["threshold"],
                  records=records, training_receipt_sha256=_hash(_receipt_path(folder, stage)),
                  source_hashes=hashes,
                  interpretation="synthetic engineering check only" if scenes["data_kind"] == "synthetic" else "measured")
    _json(output / f"GTM_Model6_three_cnn_{stage}_inference.json", result)
    return result


def verify_replay(manifest_path, run_dir, stage):
    """Rebuild the saved graph and compare every prediction without changing its arrays."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    manifest = data.load_manifest(manifest_path)
    folder = Path(run_dir).resolve()
    result_path = folder / f"GTM_Model6_three_cnn_{stage}_results.json"
    if not result_path.is_file():
        raise ValueError("Missing prediction result receipt")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    hashes = _source_hashes()
    receipt_path = _receipt_path(folder, stage)
    training = json.loads(receipt_path.read_text(encoding="utf-8"))
    c = _config(training.get("config"))
    training = _read_parent(folder, stage, manifest, c, hashes)
    if (result.get("schema") != SCHEMA or result.get("stage") != stage
            or result.get("manifest_sha256") != manifest["manifest_sha256"]
            or result.get("checkpoint") != training["checkpoint"]
            or result.get("training_receipt_sha256") != _hash(receipt_path)
            or result.get("source_hashes") != hashes):
        raise ValueError("Prediction provenance changed")
    parents = {}
    if stage == "fusion":
        parents = {s: _read_parent(folder, s, manifest, c, hashes) for s in ("airborne", "emit")}
        for s in parents:
            if (training["parents"][s]["receipt_sha256"] != _hash(_receipt_path(folder, s))
                    or training["parents"][s]["checkpoint_sha256"] != parents[s]["checkpoint"]["sha256"]):
                raise ValueError("Fusion parent changed after training")
    model = _build(stage, c, folder, parents)
    model.load_weights(str(folder / training["checkpoint"]["path"]))
    by_id = {r["id"]: r for r in _rows(manifest)}
    atol = float(result["replay_atol"])
    if not 0 <= atol <= 1e-4:
        raise ValueError("Invalid replay tolerance")
    checks = []
    for record in result["records"]:
        row = by_id.get(record["id"])
        if row is None or row["role"] != record["role"] or row["target_kind"] != record["target_kind"]:
            raise ValueError("Prediction row provenance mismatch")
        prediction_path = folder / record["prediction"]["path"]
        if not prediction_path.is_file() or _hash(prediction_path) != record["prediction"]["sha256"]:
            raise ValueError("Prediction array hash changed")
        arrays = data.load_arrays(row)
        valid = np.asarray(arrays["input_valid"], bool)
        again = _tiled(model, stage, arrays["image"], valid, training["normalization"],
                       c["tile_size"], c["halo"])
        native, native_support = networks.native_probability(again, valid, arrays["target"].shape)
        with np.load(prediction_path, allow_pickle=False) as saved:
            if set(saved.files) != {"probability", "valid", "native_probability", "target", "target_valid"}:
                raise ValueError("Prediction array schema changed")
            original = saved["probability"]
            original_native = saved["native_probability"]
            expected_target_valid = np.asarray(arrays["target_valid"], bool) & native_support
            if (not np.array_equal(saved["valid"].astype(bool), valid)
                    or not np.array_equal(saved["target_valid"].astype(bool), expected_target_valid)
                    or not np.array_equal(saved["target"], arrays["target"], equal_nan=True)
                    or not np.array_equal(np.isfinite(original), np.isfinite(again))
                    or not np.array_equal(np.isfinite(original_native), np.isfinite(native))):
                raise ValueError(f"Saved support or target changed for {row['id']}")
            if not np.allclose(original, again, atol=atol, rtol=0, equal_nan=True):
                raise ValueError(f"Checkpoint replay changed fine prediction for {row['id']}")
            if not np.allclose(original_native, native, atol=atol, rtol=0, equal_nan=True):
                raise ValueError(f"Checkpoint replay changed native prediction for {row['id']}")
            if not np.array_equal(original >= c["threshold"], again >= c["threshold"]):
                raise ValueError(f"Checkpoint replay changed threshold mask for {row['id']}")
            delta = np.abs(original - again)
            checks.append(dict(id=row["id"], array_sha256=_hash(prediction_path),
                               max_abs_difference=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0,
                               threshold_mask_equal=True))
    replay = dict(schema=SCHEMA, stage=stage, manifest_sha256=manifest["manifest_sha256"],
                  checkpoint=training["checkpoint"], results_sha256=_hash(result_path),
                  tolerance=atol, records=checks, verified=True,
                  interpretation="synthetic engineering check only" if manifest["data_kind"] == "synthetic" else "measured")
    replay_path = folder / f"GTM_Model6_three_cnn_{stage}_replay.json"
    if replay_path.exists():
        raise FileExistsError("Replay receipt already exists")
    _json(replay_path, replay)
    return replay
