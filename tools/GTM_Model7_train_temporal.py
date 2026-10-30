"""Bounded CPU comparison of two temporal CNNs on a sealed real S2 cohort.

The fixed protocol and E2E failure list live in docs/GTM_Model7_temporal_training_protocol.md.
No test labels are read until all training updates and the final checkpoint are sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

BASE = dict(seed=20261023, crop=48, draws_per_epoch=48, batch_size=8,
            learning_rate=0.001, weight_decay=0.0001, dice_weight=0.1,
            gradient_norm_cap=1.0,
            threshold=0.5, norm_clip=8.0, replay_atol=1e-6,
            epochs=60, max_minutes=15.0, dev_every=2)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor(value: np.ndarray) -> torch.Tensor:
    # Torch 2.4 in the local runtime has a broken NumPy 2 bridge.
    return torch.tensor(np.asarray(value).tolist(), dtype=torch.float64)


def array(value: torch.Tensor) -> np.ndarray:
    return np.asarray(value.detach().cpu().tolist(), dtype=np.float32)


def chw(images: np.ndarray) -> torch.Tensor:
    if images.ndim == 3:
        return tensor(np.moveaxis(images, -1, 0).copy()).unsqueeze(0)
    return tensor(np.moveaxis(images, -1, 1).copy())


def source_path(manifest_path: Path, raw: str) -> Path:
    source = Path(raw)
    if source.is_absolute():
        return source
    local = (manifest_path.parent / source).resolve()
    if local.exists():
        return local
    return (ROOT / source).resolve()


def array_path(manifest_path: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("Missing array_file")
    path = (manifest_path.parent / raw).resolve()
    if not path.is_relative_to(manifest_path.parent.resolve()) or path.suffix.lower() != ".npz":
        raise ValueError(f"Unsafe array_file: {raw}")
    return path


def load_manifest(path: Path, *, inspect_test_labels: bool = False) -> tuple[dict, list[dict]]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "GTM_Model7_temporal_cohort_v1":
        raise ValueError("Expected GTM_Model7_temporal_cohort_v1")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Manifest has no scenes")
    ids, groups, source_hash_cache, counts = set(), {}, {}, {"train": 0, "dev": 0, "test": 0}
    for scene in scenes:
        sid, split, group = scene.get("id"), scene.get("split"), scene.get("group_id")
        if not isinstance(sid, str) or not sid or sid in ids or "/" in sid or "\\" in sid:
            raise ValueError(f"Invalid or duplicate scene id: {sid}")
        if split not in counts or not isinstance(group, str) or not group:
            raise ValueError(f"Bad split/group for {sid}")
        if group in groups and groups[group] != split:
            raise ValueError(f"Group {group} crosses splits")
        ids.add(sid)
        groups[group] = split
        counts[split] += 1
        file = array_path(path, scene.get("array_file"))
        if sha(file) != scene.get("array_sha256"):
            raise ValueError(f"Array hash changed: {sid}")
        sources = scene.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Missing acquisition sources: {sid}")
        for source in sources:
            source_file = source_path(path, source["path"])
            if source_file not in source_hash_cache:
                source_hash_cache[source_file] = sha(source_file)
            if source_hash_cache[source_file] != source["sha256"]:
                raise ValueError(f"Acquisition source hash changed: {source_file}")
        if split != "test" or inspect_test_labels:
            load_scene(path, scene, labels=True)
        else:
            load_scene(path, scene, labels=False)
    if min(counts.values()) == 0:
        raise ValueError(f"All train/dev/test splits required: {counts}")
    return manifest, scenes


def load_scene(manifest_path: Path, scene: dict, *, labels: bool) -> dict:
    with np.load(array_path(manifest_path, scene["array_file"]), allow_pickle=False) as z:
        required = ["target_image", "reference_image", "valid"]
        if labels:
            required += ["target", "label_native", "valid_native"]
        data = {key: z[key] for key in required}
    if (data["target_image"].shape != (100, 100, 6) or
            data["reference_image"].shape != (100, 100, 6) or
            data["valid"].shape != (100, 100)):
        raise ValueError(f"Bad image grids: {scene['id']}")
    valid = data["valid"].astype(bool)
    if not valid.any() or not np.isfinite(data["target_image"][valid]).all() or not np.isfinite(data["reference_image"][valid]).all():
        raise ValueError(f"Nonfinite or empty supported input: {scene['id']}")
    if labels:
        target, native, support = data["target"], data["label_native"], data["valid_native"].astype(bool)
        if target.shape != (100, 100) or native.shape != (200, 200) or support.shape != (200, 200):
            raise ValueError(f"Bad target grids: {scene['id']}")
        if not support.any() or not np.isin(target, [0, 1]).all() or not np.isin(native, [0, 1]).all():
            raise ValueError(f"Nonbinary or empty targets: {scene['id']}")
        if not np.array_equal(valid, support.reshape(100, 2, 100, 2).all(axis=(1, 3))):
            raise ValueError(f"20m/native support aggregation differs: {scene['id']}")
        projected = (native.reshape(100, 2, 100, 2) == 1).any(axis=(1, 3))
        if not np.array_equal(target[valid].astype(bool), projected[valid]):
            raise ValueError(f"20m target disagrees with native label: {scene['id']}")
    return data


def training_fingerprint(scenes: list[dict]) -> str:
    selected = [{"id": s["id"], "split": s["split"], "group_id": s["group_id"],
                 "array_sha256": s["array_sha256"]} for s in scenes if s["split"] != "test"]
    payload = json.dumps({"training_scenes": selected, "schedule": {k: BASE[k] for k in
        ("seed", "crop", "draws_per_epoch", "batch_size")}}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def schedule_hash(draws: list[list[tuple]]) -> str:
    return hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest()


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.a = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.b = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.b(F.relu(self.a(x))))


class ResidualTemporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(12, 8, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(8, d) for d in (1, 2, 4)))
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current, reference = x[:, :6], x[:, 6:]
        feature = F.relu(self.stem(torch.cat((current, current - reference), dim=1)))
        return self.head(self.blocks(feature))


class ConvPair(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.a = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.b = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.b(F.relu(self.a(x))))


class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvPair(6, 8)
        self.enc2 = ConvPair(8, 16)
        self.enc3 = ConvPair(16, 32)
        self.fuse1 = nn.Conv2d(24, 8, 1)
        self.fuse2 = nn.Conv2d(48, 16, 1)
        self.fuse3 = nn.Conv2d(96, 32, 1)
        self.dec2 = ConvPair(48, 16)
        self.dec1 = ConvPair(24, 8)
        self.head = nn.Conv2d(8, 1, 1)

    @staticmethod
    def fuse(a: torch.Tensor, b: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        return F.relu(layer(torch.cat((a, a - b, (a - b).abs()), dim=1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        a, b = x[:, :6], x[:, 6:]
        a1, b1 = self.enc1(a), self.enc1(b)
        a2, b2 = self.enc2(F.avg_pool2d(a1, 2)), self.enc2(F.avg_pool2d(b1, 2))
        a3, b3 = self.enc3(F.avg_pool2d(a2, 2)), self.enc3(F.avg_pool2d(b2, 2))
        s1 = self.fuse(a1, b1, self.fuse1)
        s2 = self.fuse(a2, b2, self.fuse2)
        z = self.fuse(a3, b3, self.fuse3)
        z = self.dec2(torch.cat((F.interpolate(z, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2), 1))
        z = self.dec1(torch.cat((F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1), 1))
        return self.head(z)[..., :h, :w]


def new_model(name: str) -> nn.Module:
    return (ResidualTemporal() if name == "residual" else SiameseUNet()).double()


def normalization(path: Path, scenes: list[dict]) -> dict:
    count = 0
    sums = np.zeros(6, np.float64)
    squares = np.zeros(6, np.float64)
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = load_scene(path, scene, labels=True)
        valid = data["valid"].astype(bool)
        for key in ("target_image", "reference_image"):
            pixels = data[key][valid].astype(np.float64)
            count += len(pixels)
            sums += pixels.sum(axis=0)
            squares += np.square(pixels).sum(axis=0)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0))
    scale = np.maximum(std, 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite train normalization")
    return {"center": mean.tolist(), "scale": scale.tolist(), "train_date_pixels": count,
            "clip_standardized": BASE["norm_clip"], "invalid_fill": 0.0}


def prepare(data: dict, norm: dict) -> np.ndarray:
    valid = data["valid"].astype(bool)
    dates = []
    for key in ("target_image", "reference_image"):
        x = np.zeros((100, 100, 6), np.float32)
        x[valid] = np.clip((data[key][valid].astype(np.float64) - norm["center"]) / norm["scale"],
                           -BASE["norm_clip"], BASE["norm_clip"]).astype(np.float32)
        dates.append(x)
    return np.concatenate(dates, axis=-1)


def schedule(path: Path, scenes: list[dict], epochs: int) -> list[list[tuple[str, int, int, int, bool]]]:
    positives, negatives, labels = [], [], {}
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = load_scene(path, scene, labels=True)
        valid = data["valid"].astype(bool)
        positive = np.argwhere((data["target"] == 1) & valid)
        if len(positive):
            positives.append(scene["id"])
            labels[scene["id"]] = positive
        else:
            negatives.append(scene["id"])
            labels[scene["id"]] = np.argwhere(valid)
    if not positives or not negatives:
        raise ValueError("Training requires positive scenes and reviewed negative scenes")
    rng = np.random.default_rng(BASE["seed"])
    result = []
    for _ in range(epochs):
        draws = []
        for i in range(BASE["draws_per_epoch"]):
            positive = i % 2 == 0
            pool = positives if positive else negatives
            sid = pool[int(rng.integers(len(pool)))]
            points = labels[sid]
            cy, cx = points[int(rng.integers(len(points)))]
            top = int(np.clip(cy - BASE["crop"] // 2, 0, 100 - BASE["crop"]))
            left = int(np.clip(cx - BASE["crop"] // 2, 0, 100 - BASE["crop"]))
            draws.append((sid, top, left, int(rng.integers(8)), positive))
        result.append(draws)
    return result


def d4(x: np.ndarray, code: int) -> np.ndarray:
    if code & 1:
        x = np.flip(x, axis=1)
    return np.rot90(x, code >> 1, axes=(0, 1)).copy()


def batch(path: Path, by_id: dict, draws: list[tuple], norm: dict, cache: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, masks, positive = [], [], [], []
    for sid, top, left, transform, is_positive in draws:
        if sid not in cache:
            data = load_scene(path, by_id[sid], labels=True)
            cache[sid] = (prepare(data, norm), data["target"].astype(np.float32), data["valid"].astype(bool))
        x, y, valid = cache[sid]
        sl = np.s_[top:top + BASE["crop"], left:left + BASE["crop"]]
        xs.append(d4(x[sl], transform))
        ys.append(d4(y[sl], transform))
        masks.append(d4(valid[sl], transform))
        positive.append(is_positive)
    if not all(mask.any() for mask in masks):
        raise ValueError("Sampled crop has no support")
    return chw(np.stack(xs)), tensor(np.stack(ys))[:, None], tensor(np.stack(masks))[:, None], tensor(np.asarray(positive))


def fit_loss(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor,
             positive_crop: torch.Tensor, pos_weight: float) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=torch.tensor(pos_weight))
    bce = (raw * valid).sum() / valid.sum()
    positive = positive_crop[:, None, None, None] * valid
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y * positive).sum() + 1) / ((p * positive).sum() + (y * positive).sum() + 1)
    return bce + BASE["dice_weight"] * dice


def train_pos_weight(path: Path, scenes: list[dict]) -> float:
    positive = negative = 0
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = load_scene(path, scene, labels=True)
        y = data["target"][data["valid"].astype(bool)]
        positive += int((y == 1).sum())
        negative += int((y == 0).sum())
    if not positive or not negative:
        raise ValueError("Train lacks either class")
    return float(min(10.0, max(1.0, negative / positive)))


def infer(model: nn.Module, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        p = array(torch.sigmoid(model(chw(x))[0, 0]))
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite full-scene probability")
    return p


def confusion(probability: np.ndarray, target: np.ndarray, support: np.ndarray) -> dict:
    pred = probability[support] >= BASE["threshold"]
    truth = target[support] == 1
    return {"tp": int((pred & truth).sum()), "fp": int((pred & ~truth).sum()),
            "fn": int((~pred & truth).sum()), "tn": int((~pred & ~truth).sum())}


def score(c: dict) -> dict:
    tp, fp, fn, tn = (c[key] for key in ("tp", "fp", "fn", "tn"))
    return {"confusion": c, "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_area_rate": fp / (fp + tn) if fp + tn else None}


def dev_report(model: nn.Module, path: Path, scenes: list[dict], norm: dict) -> dict:
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    activated = 0
    for scene in scenes:
        if scene["split"] != "dev":
            continue
        data = load_scene(path, scene, labels=True)
        p = infer(model, prepare(data, norm), data["valid"])
        native = np.repeat(np.repeat(p, 2, axis=0), 2, axis=1)
        c = confusion(native, data["label_native"], data["valid_native"].astype(bool))
        for key in pooled:
            pooled[key] += c[key]
        activated += bool(c["fp"])
    return {**score(pooled), "activated_scenes": activated}


def checkpoint(path: Path, model: nn.Module, norm: dict, config: dict, epoch: int) -> None:
    torch.save({"model": model.state_dict(), "normalization": norm, "config": config,
                "epoch": epoch, "source_sha256": sha(Path(__file__))}, path)


def predictions(model: nn.Module, manifest_path: Path, scenes: list[dict], norm: dict, run: Path) -> tuple[dict, dict]:
    folder = run / "predictions"
    folder.mkdir()
    records = {}
    pooled = dict(tp=0, fp=0, fn=0, tn=0)
    for scene in scenes:
        if scene["split"] != "test":
            continue
        data = load_scene(manifest_path, scene, labels=True)
        valid = data["valid"].astype(bool)
        p20 = infer(model, prepare(data, norm), valid)
        native = np.repeat(np.repeat(p20, 2, axis=0), 2, axis=1)
        support = data["valid_native"].astype(bool)
        c = confusion(native, data["label_native"], support)
        for key in pooled:
            pooled[key] += c[key]
        negative = not np.any(data["label_native"][support] == 1)
        records[scene["id"]] = {**c, "scene_kind": scene.get("scene_kind"),
             "negative_control": negative,
             "negative_any_activation": bool(c["fp"]) if negative else None,
             "negative_fp_area_rate": c["fp"] / int(support.sum()) if negative else None,
             "supported_native_pixels": int(support.sum()),
             "prediction_sha256": None}
        file = folder / f"{scene['id']}.npz"
        np.savez_compressed(file, probability20=p20.astype(np.float32),
                            probability_native=native.astype(np.float32), valid20=valid)
        records[scene["id"]]["prediction_sha256"] = sha(file)
    return records, score(pooled)


def run_train(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    config = {**BASE, "epochs": args.epochs, "max_minutes": args.max_minutes}
    if args.epochs < 1 or args.max_minutes <= 0 or BASE["draws_per_epoch"] % BASE["batch_size"]:
        raise ValueError("Invalid fixed training schedule")
    _, scenes = load_manifest(manifest_path)
    run = args.output.resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run}")
    norm = normalization(manifest_path, scenes)
    draws = schedule(manifest_path, scenes, args.epochs)
    pos_weight = train_pos_weight(manifest_path, scenes)
    by_id = {scene["id"]: scene for scene in scenes}
    config["train_pos_weight"] = pos_weight
    torch.manual_seed(BASE["seed"])
    model = new_model(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE["learning_rate"], weight_decay=BASE["weight_decay"])
    run.mkdir(parents=True, exist_ok=True)
    snapshot = run / "source"
    snapshot.mkdir()
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    shutil.copyfile(manifest_path, snapshot / "manifest.json")
    write_json(snapshot / "config.json", config)
    source_start_hash = sha(snapshot / Path(__file__).name)
    checkpoint(run / "initial.pt", model, norm, config, 0)
    start = time.monotonic()
    curve = []
    cache = {}
    for epoch, epoch_draws in enumerate(draws, 1):
        model.train()
        weighted_loss = 0.0
        for offset in range(0, len(epoch_draws), BASE["batch_size"]):
            if time.monotonic() - start > args.max_minutes * 60:
                raise TimeoutError("Time cap reached before fixed update schedule completed; comparison invalid")
            x, y, valid, positive = batch(manifest_path, by_id,
                                          epoch_draws[offset:offset + BASE["batch_size"]], norm, cache)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = fit_loss(logits, y, valid, positive, pos_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), BASE["gradient_norm_cap"],
                                                            error_if_nonfinite=True)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Nonfinite gradient norm at epoch {epoch}")
            optimizer.step()
            weighted_loss += float(loss.item())
        row = {"epoch": epoch, "train_loss": weighted_loss / (len(epoch_draws) / BASE["batch_size"]),
               "elapsed_seconds": time.monotonic() - start}
        if epoch % BASE["dev_every"] == 0 or epoch == args.epochs:
            row["dev"] = dev_report(model, manifest_path, scenes, norm)
        if time.monotonic() - start > args.max_minutes * 60:
            raise TimeoutError("Time cap reached before fixed schedule and dev reports completed; comparison invalid")
        curve.append(row)
        write_json(run / "learning_curve.json", curve)
        print(json.dumps({"model": args.model, **row}), flush=True)
    runtime = time.monotonic() - start
    if sha(Path(__file__)) != source_start_hash or sha(manifest_path) != sha(snapshot / "manifest.json"):
        raise ValueError("Trainer source or manifest changed during training")
    checkpoint(run / "final.pt", model, norm, config, args.epochs)
    shutil.copyfile(run / "final.pt", run / "best.pt")
    records, pooled = predictions(model, manifest_path, scenes, norm, run)
    if not any(not record["negative_control"] for record in records.values()) or not any(
            record["negative_control"] for record in records.values()):
        raise ValueError("Test requires supported positive and negative control scenes")
    counts = {split: sum(s["split"] == split for s in scenes) for split in ("train", "dev", "test")}
    receipt = {"schema": "GTM_Model7_temporal_training_v1", "model": args.model,
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "training_fingerprint": training_fingerprint(scenes), "source_sha256": source_start_hash,
        "draw_schedule_sha256": schedule_hash(draws),
        "config": config, "normalization": norm, "split_counts": counts,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime_seconds_training": runtime, "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "checkpoint_selection": "none; final fixed epoch is primary; best.pt aliases final.pt",
        "checkpoint_sha256": {name: sha(run / f"{name}.pt") for name in ("initial", "best", "final")},
        "snapshot_sha256": {file.name: sha(file) for file in snapshot.iterdir()},
        "learning_curve_sha256": sha(run / "learning_curve.json"),
        "test_scenes": records, "pooled_test": pooled,
        "limits": "Delivered native 10m grid repeats each 20m probability into a 2x2 block; it is not a calibrated 10m prediction"}
    write_json(run / "receipt.json", receipt)
    print(json.dumps({"training": "complete", "receipt": str(run / "receipt.json")}), flush=True)


def replay(args: argparse.Namespace) -> None:
    manifest_path, run = args.manifest.resolve(), args.run.resolve()
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    _, scenes = load_manifest(manifest_path)
    if sha(manifest_path) != receipt["manifest_sha256"] or sha(Path(__file__)) != receipt["source_sha256"]:
        raise ValueError("Manifest or trainer source changed since run")
    path = run / f"{args.checkpoint}.pt"
    if sha(path) != receipt["checkpoint_sha256"][args.checkpoint]:
        raise ValueError("Checkpoint hash changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = new_model(receipt["model"])
    model.load_state_dict(state["model"])
    if state["normalization"] != receipt["normalization"]:
        raise ValueError("Checkpoint normalization differs from receipt")
    checked = 0
    for scene in scenes:
        if scene["split"] != "test":
            continue
        data = load_scene(manifest_path, scene, labels=False)
        actual = infer(model, prepare(data, state["normalization"]), data["valid"])
        with np.load(run / "predictions" / f"{scene['id']}.npz", allow_pickle=False) as z:
            expected = z["probability20"]
        if not np.allclose(actual, expected, atol=BASE["replay_atol"], rtol=0, equal_nan=True):
            raise AssertionError(f"Fresh-process replay differs: {scene['id']}")
        checked += 1
    print(json.dumps({"replay": "passed", "model": receipt["model"], "scenes": checked,
                      "atol": BASE["replay_atol"]}))


def benchmark(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    _, scenes = load_manifest(manifest_path)
    norm = normalization(manifest_path, scenes)
    draws = schedule(manifest_path, scenes, 1)[0][:BASE["batch_size"]]
    by_id = {scene["id"]: scene for scene in scenes}
    x, y, valid, positive = batch(manifest_path, by_id, draws, norm, {})
    pos_weight = train_pos_weight(manifest_path, scenes)
    output = {}
    for name in ("residual", "siamese"):
        torch.manual_seed(BASE["seed"])
        model = new_model(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=BASE["learning_rate"], weight_decay=BASE["weight_decay"])
        start = time.monotonic()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = fit_loss(model(x), y, valid, positive, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), BASE["gradient_norm_cap"],
                                           error_if_nonfinite=True)
            optimizer.step()
        output[name] = {"seconds_per_update_after_import": (time.monotonic() - start) / 3,
                        "parameters": sum(p.numel() for p in model.parameters()), "loss": float(loss.item())}
    report = {"schema": "GTM_Model7_temporal_cpu_benchmark_v1",
              "manifest_sha256": sha(manifest_path), "trainer_sha256": sha(Path(__file__)),
              "config": BASE, "benchmark": output,
              "updates_per_epoch": BASE["draws_per_epoch"] // BASE["batch_size"],
              "60_epoch_training_seconds_excluding_dev_and_io": {
                  name: values["seconds_per_update_after_import"] * 60 * BASE["draws_per_epoch"] / BASE["batch_size"]
                  for name, values in output.items()}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=("residual", "siamese"), required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=BASE["epochs"])
    train.add_argument("--max-minutes", type=float, default=BASE["max_minutes"])
    for command in ("validate", "fingerprint", "benchmark"):
        entry = sub.add_parser(command)
        entry.add_argument("--manifest", type=Path, required=True)
        if command == "benchmark":
            entry.add_argument("--output", type=Path)
    verify = sub.add_parser("replay")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--checkpoint", choices=("best", "final"), default="best")
    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "replay":
        replay(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "validate":
        _, scenes = load_manifest(args.manifest, inspect_test_labels=True)
        print(json.dumps({"valid": True, "scenes": len(scenes)}))
    elif args.command == "fingerprint":
        _, scenes = load_manifest(args.manifest)
        print(json.dumps({"training_fingerprint": training_fingerprint(scenes)}))


if __name__ == "__main__":
    main()
