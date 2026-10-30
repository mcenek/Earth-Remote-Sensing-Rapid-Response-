"""Fresh native-grid MARS transfer study. Never reads protected test imagery."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EarthRemoteSensingRapidResponse"))
from mars_s2l_adapter import compute_mbmp  # noqa: E402

PROTOCOL = ROOT / "docs/GTM_Model7_us_transfer_v13_protocol_20261026.md"
SOURCE = ROOT / "outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json"
SOURCE_SHA = "1f08f7cf0d0a610044b58a5de9eaff1b708359fd4d06cf8a88a00bc43708c63a"
OUT = ROOT / "outputs/GTM_Model7_us_transfer_v13_20261026"
SEED = 20261026
UPDATES = 4800
MILESTONES = (1200, 2400, 3600, 4800)
CUTOFFS = tuple(round(k / 100, 3) for k in range(5, 100, 5)) + (.975, .99, .995)
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
ARMS = ("native_zero_wind", "native_wind")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def manifest() -> list[dict]:
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Frozen source manifest hash mismatch")
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))["scenes"]
    counts = collections.Counter((r["split"], r["scene_kind"]) for r in rows)
    expected = {("train", "PLUME"): 268, ("train", "NO_PLUME"): 690,
                ("dev", "PLUME"): 46, ("dev", "NO_PLUME"): 152}
    if counts != expected or len({r["id"] for r in rows}) != 1156:
        raise ValueError(f"Frozen TRAIN/DEV inventory mismatch: {counts}")
    for row in rows:
        if not math.isfinite(float(row["official_csv_row"]["wind_u"])) or not math.isfinite(float(row["official_csv_row"]["wind_v"])):
            raise ValueError("Missing/nonfinite wind: " + row["id"])
    return rows


def source_path(item: dict) -> Path:
    path = Path(item["path"])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError("Changed source: " + str(path))
    return path


def reconstruct(row: dict) -> dict[str, np.ndarray]:
    sources = {item["role"]: source_path(item) for item in row["sources"]}
    required = {"target_image", "reference_image", "target_cloud", "reference_cloud"}
    if row["scene_kind"] == "PLUME":
        required.add("plume_mask")
    if not required <= sources.keys():
        raise ValueError("Missing pinned source role: " + row["id"])
    with rasterio.open(sources["target_image"]) as current_ds, rasterio.open(sources["reference_image"]) as prior_ds, \
            rasterio.open(sources["target_cloud"]) as current_cloud_ds, rasterio.open(sources["reference_cloud"]) as prior_cloud_ds:
        for ds in (current_ds, prior_ds):
            if (ds.count != 12 or ds.shape != (200, 200) or tuple(ds.res) != (10.0, 10.0)
                or tuple(ds.descriptions)[:6] != BANDS or set(ds.dtypes) != {"uint16"}):
                raise ValueError("Native band/grid contract failed: " + row["id"])
        if (current_ds.transform != prior_ds.transform or current_ds.crs != prior_ds.crs
            or str(current_ds.crs) != row["reference_grid"]["crs"]
            or tuple(current_ds.transform)[:6] != tuple(row["reference_grid"]["affine"])):
            raise ValueError("Native bidate geometry changed: " + row["id"])
        for ds in (current_cloud_ds, prior_cloud_ds):
            if ds.shape != (200, 200) or ds.transform != current_ds.transform or ds.crs != current_ds.crs or ds.count != 1 or ds.dtypes[0] != "uint8":
                raise ValueError("Cloud geometry changed: " + row["id"])
        a = current_ds.read(indexes=(1, 2, 3, 4, 5, 6))
        b = prior_ds.read(indexes=(1, 2, 3, 4, 5, 6))
        ca, cb = current_cloud_ds.read(1), prior_cloud_ds.read(1)
        if not np.isin(ca, (0, 1, 2, 3, 4)).all() or not np.isin(cb, (0, 1, 2, 3, 4)).all():
            raise ValueError("Cloud code changed: " + row["id"])
        valid = (ca == 0) & (cb == 0) & (a > 0).all(0) & (b > 0).all(0)
        if current_ds.nodata is not None:
            valid &= (a != current_ds.nodata).all(0)
        if prior_ds.nodata is not None:
            valid &= (b != prior_ds.nodata).all(0)
        valid20 = valid.reshape(100, 2, 100, 2).all(axis=(1, 3))
        support = np.repeat(np.repeat(valid20, 2, 0), 2, 1)
        label = np.zeros((200, 200), dtype=np.uint8)
        if row["scene_kind"] == "PLUME":
            with rasterio.open(sources["plume_mask"]) as ds:
                if ds.shape != (200, 200) or ds.transform != current_ds.transform or ds.crs != current_ds.crs or ds.count != 1 or ds.dtypes[0] != "uint8":
                    raise ValueError("Label geometry changed: " + row["id"])
                label = ds.read(1)
            if not np.isin(label, (0, 1)).all() or not label.any():
                raise ValueError("Producer native label changed: " + row["id"])
    original = (SOURCE.parent / row["array_file"]).resolve()
    if sha(original) != row["array_sha256"]:
        raise ValueError("Original cache hash changed: " + row["id"])
    with np.load(original, allow_pickle=False) as z:
        if not np.array_equal(z["label_native"], label) or not np.array_equal(z["valid_native"].astype(bool), support):
            raise ValueError("Native label/support changed: " + row["id"])
        if not np.array_equal(z["valid"].astype(bool), valid20):
            raise ValueError("20 m support changed: " + row["id"])
        aa = a.astype(np.float32).transpose(1, 2, 0).reshape(100, 2, 100, 2, 6).mean(axis=(1, 3))
        bb = b.astype(np.float32).transpose(1, 2, 0).reshape(100, 2, 100, 2, 6).mean(axis=(1, 3))
        if not np.array_equal(z["target_image"], aa) or not np.array_equal(z["reference_image"], bb):
            raise ValueError("Selected clear-reference reconstruction mismatch: " + row["id"])
    if row["scene_kind"] == "PLUME" and not (label & support).any():
        raise ValueError("Unsupported positive label: " + row["id"])
    aa = np.clip(a.astype(np.float32) / 5000.0, 0.0, 2.0)
    bb = np.clip(b.astype(np.float32) / 5000.0, 0.0, 2.0)
    mbmp = compute_mbmp(aa, bb, valid_mask=support)
    cloud = ((ca > 0) | (cb > 0)).astype(np.float32)
    wind = np.clip(np.array([float(row["official_csv_row"]["wind_u"]), float(row["official_csv_row"]["wind_v"])], np.float32), -20, 20) / 8.0
    x = np.concatenate((mbmp[None], aa, bb, np.broadcast_to(wind[:, None, None], (2, 200, 200)), cloud[None])).astype(np.float32)
    if x.shape != (16, 200, 200) or not np.isfinite(x).all() or not support.any():
        raise ValueError("Invalid native inputs: " + row["id"])
    return {"x": x, "y": label, "valid": support.astype(np.uint8), "wind": wind}


def prepare() -> None:
    rows = manifest()
    destination = OUT / "data"
    if (destination / "receipt.json").exists():
        raise FileExistsError("Prepared receipt already exists; refusing to overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    index = []
    for i, row in enumerate(rows):
        cache = destination / f"{row['id']}.npz"
        if cache.exists():
            raise FileExistsError("Unreceipted native cache: " + str(cache))
        data = reconstruct(row)
        np.savez_compressed(cache, **data)
        index.append({"id": row["id"], "split": row["split"], "scene_kind": row["scene_kind"],
                      "site": row["physical_location_id"], "cache": cache.name, "sha256": sha(cache),
                      "source_array_sha256": row["array_sha256"],
                      "sources": [{k: s[k] for k in ("role", "path", "sha256", "bytes")} for s in row["sources"]],
                      "wind_raw": [float(row["official_csv_row"][k]) for k in ("wind_u", "wind_v")],
                      "valid_pixels": int(data["valid"].sum()), "label_pixels": int((data["y"] * data["valid"]).sum())})
        if (i + 1) % 100 == 0:
            print(json.dumps({"prepared": i + 1, "total": len(rows)}), flush=True)
    save_json(destination / "index.json", index)
    save_json(destination / "receipt.json", {"source_manifest_sha256": SOURCE_SHA, "protocol_sha256": sha(PROTOCOL),
              "script_sha256": sha(Path(__file__)), "index_sha256": sha(destination / "index.json"),
              "counts": dict(collections.Counter(x["split"] for x in index)), "all_native_reconstructions_verified": True})


def index() -> list[dict]:
    receipt = json.loads((OUT / "data/receipt.json").read_text(encoding="utf-8"))
    if receipt["source_manifest_sha256"] != SOURCE_SHA or receipt["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Native cache source/protocol drift")
    path = OUT / "data/index.json"
    if sha(path) != receipt["index_sha256"]:
        raise ValueError("Native cache index drift")
    return json.loads(path.read_text(encoding="utf-8"))


def scene(item: dict, *, verify: bool = True) -> dict[str, np.ndarray]:
    path = OUT / "data" / item["cache"]
    if verify and sha(path) != item["sha256"]:
        raise ValueError("Native scene cache drift: " + item["id"])
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("x", "y", "valid", "wind")}


def transform(x: np.ndarray, y: np.ndarray, valid: np.ndarray, turns: int, mirror: bool, arm: str):
    x, y, valid = np.rot90(x, turns, axes=(-2, -1)).copy(), np.rot90(y, turns).copy(), np.rot90(valid, turns).copy()
    u, v = float(x[13, 0, 0]), float(x[14, 0, 0])
    for _ in range(turns):
        u, v = -v, u
    if mirror:
        x, y, valid = x[..., ::-1].copy(), y[:, ::-1].copy(), valid[:, ::-1].copy()
        u = -u
    x[13], x[14] = (0.0, 0.0) if arm == "native_zero_wind" else (u, v)
    return x, y, valid


def block(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU())


class Up(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.conv = block(ci, co)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat((skip, deep), dim=1))


class NativeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = block(16, 32)
        self.down1 = block(32, 64)
        self.down2 = block(64, 128)
        self.down3 = block(128, 256)
        self.down4 = block(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.inc(x)
        b = self.down1(F.max_pool2d(a, 2))
        c = self.down2(F.max_pool2d(b, 2))
        d = self.down3(F.max_pool2d(c, 2))
        e = self.down4(F.max_pool2d(d, 2))
        return self.out(self.up4(self.up3(self.up2(self.up1(e, d), c), b), a))


def new_model() -> NativeUNet:
    torch.manual_seed(SEED)
    model = NativeUNet().float()
    with torch.no_grad():
        model.inc[0].weight[:, 13:15].zero_()
    return model


def loss(logits: torch.Tensor, y: torch.Tensor, valid: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    if logits.shape != y.shape or y.shape != valid.shape or positive.shape != (len(y),):
        raise ValueError("Loss shape mismatch")
    if not all(torch.isfinite(z).all() for z in (logits, y, valid)) or valid.sum() <= 0:
        raise ValueError("Nonfinite or empty native loss")
    if (not torch.all((y == 0) | (y == 1)) or not torch.all((valid == 0) | (valid == 1))
        or not torch.equal(positive > .5, ((y * valid).sum(dim=(1, 2, 3)) > 0))):
        raise ValueError("Native binary targets, support or class flag mismatch")
    raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(2.0, device=logits.device), reduction="none")
    bce = (raw * valid).sum() / valid.sum()
    selected = valid * positive[:, None, None, None]
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y * selected).sum() + 1) / ((p * selected).sum() + (y * selected).sum() + 1)
    return bce + .1 * dice


def schedule(items: list[dict]) -> list[list[dict]]:
    rng = random.Random(SEED)
    pools = {}
    for kind in ("PLUME", "NO_PLUME"):
        groups = collections.defaultdict(list)
        for i, item in enumerate(items):
            if item["split"] == "train" and item["scene_kind"] == kind:
                groups[item["site"]].append(i)
        pools[kind] = groups
    keys = {k: sorted(v) for k, v in pools.items()}
    result = []
    for _ in range(UPDATES):
        batch = []
        for kind in ("PLUME", "NO_PLUME"):
            for _ in range(4):
                site = rng.choice(keys[kind])
                i = rng.choice(pools[kind][site])
                batch.append({"index": i, "turns": rng.randrange(4), "mirror": bool(rng.randrange(2))})
        rng.shuffle(batch)
        result.append(batch)
    return result


def settings() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def batch(items: list[dict], draw: list[dict], arm: str, device: torch.device, *, verify: bool = True):
    xlist, ylist, vlist, plist = [], [], [], []
    for d in draw:
        item = items[d["index"]]
        data = scene(item, verify=verify)
        x, y, v = transform(data["x"], data["y"], data["valid"], d["turns"], d["mirror"], arm)
        xlist.append(x)
        ylist.append(y[None])
        vlist.append(v[None])
        plist.append(item["scene_kind"] == "PLUME")
    return (torch.from_numpy(np.stack(xlist)).to(device), torch.from_numpy(np.stack(ylist).astype(np.float32)).to(device),
            torch.from_numpy(np.stack(vlist).astype(np.float32)).to(device), torch.tensor(plist, dtype=torch.float32, device=device))


def preflight() -> None:
    settings()
    items = index()
    draws = schedule(items)
    schedule_path = OUT / "preflight/draw_schedule.json"
    if schedule_path.exists():
        raise FileExistsError("Preflight already exists")
    save_json(schedule_path, draws)
    if len(draws) != UPDATES or any(len(x) != 8 or collections.Counter(items[d["index"]]["scene_kind"] for d in x) != {"PLUME": 4, "NO_PLUME": 4} for x in draws):
        raise ValueError("Unbalanced or incomplete frozen draw schedule")
    model = new_model()
    if any(model.inc[0].weight[:, 13:15].flatten()):
        raise ValueError("Wind slots are not zero initialized")
    if tuple(model(torch.zeros(1, 16, 200, 200)).shape) != (1, 1, 200, 200):
        raise ValueError("Native architecture output shape changed")
    for turns in range(4):
        for mirror in (False, True):
            x = np.zeros((16, 3, 3), np.float32)
            x[13], x[14] = 2, 3
            xt, _, _ = transform(x, np.zeros((3, 3)), np.ones((3, 3)), turns, mirror, "native_wind")
            u, v = 2, 3
            for _ in range(turns):
                u, v = -v, u
            if mirror:
                u = -u
            if (xt[13, 0, 0], xt[14, 0, 0]) != (u, v):
                raise ValueError("Wind D4 vector mismatch")
    # Independent NumPy loss on a real balanced batch.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, v, positive = batch(items, draws[0], "native_wind", device)
    model = new_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    model.train()
    logits = model(x)
    value = loss(logits, y, v, positive)
    z, truth, support, flags = [q.detach().cpu().numpy().astype(np.float64) for q in (logits, y, v, positive)]
    # Explicit stable weighted BCE, independent of torch's pos_weight branch.
    bce_np = ((1 - truth) * np.logaddexp(0, z) + 2 * truth * np.logaddexp(0, -z))
    bce_np = (bce_np * support).sum() / support.sum()
    selected = support * flags[:, None, None, None]
    prob = 1 / (1 + np.exp(-z))
    dice_np = 1 - (2 * (prob * truth * selected).sum() + 1) / ((prob * selected).sum() + (truth * selected).sum() + 1)
    independent = float(bce_np + .1 * dice_np)
    if abs(float(value.detach()) - independent) > 2e-5:
        raise ValueError(f"Independent native loss mismatch: {float(value.detach())} vs {independent}")
    before = model.inc[0].weight.detach().clone()
    value.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Nonfinite native gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    if torch.equal(before, model.inc[0].weight):
        raise ValueError("Real native step did not change weights")
    model.eval()
    with torch.no_grad():
        reference = model.cpu()(x[:1].cpu())
    path = OUT / "preflight/roundtrip.pt"
    torch.save(model.state_dict(), path)
    reloaded = new_model().cpu().eval()
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        delta = float((reference - reloaded(x[:1].cpu())).abs().max())
    if delta != 0:
        raise ValueError("CPU/GPU save-load drift")
    # Time the actual full-scene optimization, including compressed-cache batch loads.
    model = new_model().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    timings = []
    for draw in draws[:5]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        xx, yy, vv, pp = batch(items, draw, "native_wind", device)
        optimizer.zero_grad(set_to_none=True)
        objective = loss(model(xx), yy, vv, pp)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    steady = float(np.mean(timings[2:]))
    projected = steady * UPDATES
    if projected > 2400:
        raise RuntimeError(f"Native fit projected {projected:.0f}s > 40-minute bound")
    save_json(OUT / "preflight/receipt.json", {"script_sha256": sha(Path(__file__)), "prepared_with_script_sha256": json.loads((OUT / "data/receipt.json").read_text())["script_sha256"],
              "protocol_sha256": sha(PROTOCOL),
              "data_receipt_sha256": sha(OUT / "data/receipt.json"), "schedule_sha256": sha(schedule_path),
              "roundtrip_sha256": sha(path), "loss_torch": float(value.detach()), "loss_numpy": independent,
              "roundtrip_max_abs_delta": delta, "parameters": sum(p.numel() for p in model.parameters()),
              "device": str(device), "real_batch_shape": list(x.shape), "changed_weights": True,
              "timings_seconds": timings, "steady_seconds_per_update": steady,
              "projected_4800_seconds": projected, "under_40_minutes": projected <= 2400})


def preflight_receipt() -> dict:
    path = OUT / "preflight/receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt["script_sha256"] != sha(Path(__file__)) or receipt["protocol_sha256"] != sha(PROTOCOL)
        or receipt["data_receipt_sha256"] != sha(OUT / "data/receipt.json")
        or receipt["schedule_sha256"] != sha(OUT / "preflight/draw_schedule.json")
        or not receipt["under_40_minutes"]):
        raise ValueError("Preflight identity or timing changed")
    return receipt


def verify_all_cache(items: list[dict]) -> None:
    for item in items:
        path = OUT / "data" / item["cache"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("Native cache changed: " + item["id"])


def checkpoint_path(arm: str, step: int) -> Path:
    return OUT / f"fit_{arm}" / f"step_{step}.pt"


def score_model(model: NativeUNet, items: list[dict], split: str, destination: Path,
                device: torch.device, arm: str) -> dict[str, dict]:
    destination.mkdir(parents=True, exist_ok=False)
    model.eval()
    result = {}
    with torch.inference_mode():
        for item in items:
            if item["split"] != split:
                continue
            data = scene(item, verify=False)
            x = data["x"].copy()
            if arm == "native_zero_wind":
                x[13:15] = 0
            value = torch.from_numpy(x[None]).to(device)
            probability = torch.sigmoid(model(value))[0, 0].float().cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError("Nonfinite native score: " + item["id"])
            path = destination / f"{item['id']}.npz"
            np.savez_compressed(path, probability_native=probability,
                                probability20=probability.reshape(100, 2, 100, 2).mean(axis=(1, 3)),
                                valid_native=data["valid"], target_native=data["y"])
            result[item["id"]] = {"probability_sha256": sha(path), "valid_native_pixels": int(data["valid"].sum()),
                                  "positive_native_pixels": int((data["valid"] * data["y"]).sum())}
    return result


def train(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError("Unknown native arm")
    settings()
    preflight_receipt()
    items = index()
    verify_all_cache(items)
    draws = json.loads((OUT / "preflight/draw_schedule.json").read_text(encoding="utf-8"))
    folder = OUT / f"fit_{arm}"
    if folder.exists():
        raise FileExistsError("Native arm already exists: " + str(folder))
    folder.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = new_model().to(device)
    initial_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    history, milestones = [], {}
    start = time.monotonic()
    for step, draw in enumerate(draws, 1):
        if time.monotonic() - start > 2400:
            raise TimeoutError("Native fit exceeded 40 minutes before completion")
        model.train()
        x, y, valid, positive = batch(items, draw, arm, device, verify=False)
        optimizer.zero_grad(set_to_none=True)
        objective = loss(model(x), y, valid, positive)
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite fit loss")
        objective.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite fit gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 100 == 0:
            history.append({"step": step, "loss": float(objective.detach()), "seconds": time.monotonic() - start})
            print(json.dumps({"arm": arm, **history[-1]}), flush=True)
        if step in MILESTONES:
            path = checkpoint_path(arm, step)
            torch.save({"model_state_dict": model.state_dict(), "step": step, "arm": arm,
                        "initial_state_sha256": initial_hash, "protocol_sha256": sha(PROTOCOL),
                        "source_manifest_sha256": SOURCE_SHA}, path)
            predictions = score_model(model, items, "dev", folder / f"dev_step_{step}", device, arm)
            milestones[str(step)] = {"checkpoint_sha256": sha(path), "dev": {"scenes": predictions}}
    save_json(folder / "history.json", history)
    metadata = {"arm": arm, "initial_state_sha256": initial_hash, "updates": UPDATES, "milestones": milestones,
                "seed": SEED, "optimizer": {"name": "AdamW", "learning_rate": .0003, "weight_decay": .0001},
                "loss": "weighted_BCE_positive_2_plus_0.1_positive_Dice_native_valid",
                "schedule_sha256": sha(OUT / "preflight/draw_schedule.json"),
                "preflight_sha256": sha(OUT / "preflight/receipt.json"),
                "data_index_sha256": sha(OUT / "data/index.json"),
                "source_manifest_sha256": SOURCE_SHA, "protocol_sha256": sha(PROTOCOL),
                "script_sha256": sha(Path(__file__)), "duration_seconds": time.monotonic() - start,
                "device": str(device), "parameters": sum(p.numel() for p in model.parameters())}
    save_json(folder / "metadata.json", metadata)
    save_json(folder / "receipt.json", {"complete": True, "arm": arm,
              "metadata_sha256": sha(folder / "metadata.json"), "history_sha256": sha(folder / "history.json"),
              "checkpoint_steps": list(MILESTONES), "duration_seconds": metadata["duration_seconds"]})


def fit_metadata(arm: str) -> dict:
    folder = OUT / f"fit_{arm}"
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    if not receipt["complete"] or receipt["metadata_sha256"] != sha(folder / "metadata.json"):
        raise ValueError("Incomplete or changed native fit")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (meta["arm"] != arm or meta["script_sha256"] != sha(Path(__file__)) or
        meta["protocol_sha256"] != sha(PROTOCOL) or meta["source_manifest_sha256"] != SOURCE_SHA or
        meta["schedule_sha256"] != sha(OUT / "preflight/draw_schedule.json") or
        meta["data_index_sha256"] != sha(OUT / "data/index.json")):
        raise ValueError("Native fit source or implementation changed")
    return meta


def loaded_model(arm: str, step: int, device: torch.device, meta: dict | None = None) -> NativeUNet:
    meta = meta or fit_metadata(arm)
    path = checkpoint_path(arm, step)
    if sha(path) != meta["milestones"][str(step)]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["step"] != step or state["arm"] != arm or
        state["initial_state_sha256"] != meta["initial_state_sha256"] or
        state["protocol_sha256"] != sha(PROTOCOL)):
        raise ValueError("Checkpoint identity mismatch")
    model = NativeUNet().float()
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval()


def replay(arm: str) -> None:
    settings()
    meta = fit_metadata(arm)
    items = index()
    verify_all_cache(items)
    result = {}
    for step in MILESTONES:
        model = loaded_model(arm, step, torch.device("cpu"), meta)
        error, support_equal, count = 0.0, True, 0
        with torch.inference_mode():
            for item in items:
                if item["split"] != "dev":
                    continue
                path = OUT / f"fit_{arm}" / f"dev_step_{step}" / f"{item['id']}.npz"
                receipt = meta["milestones"][str(step)]["dev"]["scenes"][item["id"]]
                if sha(path) != receipt["probability_sha256"]:
                    raise ValueError("Saved DEV score changed")
                data = scene(item, verify=False)
                x = data["x"].copy()
                if arm == "native_zero_wind":
                    x[13:15] = 0
                fresh = torch.sigmoid(model(torch.from_numpy(x[None])))[0, 0].numpy()
                with np.load(path, allow_pickle=False) as z:
                    error = max(error, float(np.max(np.abs(fresh - z["probability_native"]))))
                    support_equal &= np.array_equal(data["valid"], z["valid_native"]) and np.array_equal(data["y"], z["target_native"])
                    support_equal &= int(data["valid"].sum()) == receipt["valid_native_pixels"]
                    support_equal &= int((data["valid"] * data["y"]).sum()) == receipt["positive_native_pixels"]
                count += 1
        if error > 1e-6 or not support_equal or count != 198:
            raise ValueError(f"Fresh CPU replay mismatch at {step}: error={error}, support={support_equal}, rows={count}")
        result[str(step)] = {"scene_count": count, "max_abs_probability_native_error": error,
                             "support_and_native_counts_exact": support_equal}
    save_json(OUT / f"replay_{arm}/receipt.json", {"passed": True, "arm": arm, "milestones": result,
              "fit_metadata_sha256": sha(OUT / f"fit_{arm}/metadata.json"),
              "script_sha256": sha(Path(__file__)), "protocol_sha256": sha(PROTOCOL)})


def native_confusion(probability: np.ndarray, target: np.ndarray, valid: np.ndarray, cutoff: float) -> dict[str, int]:
    good = valid.astype(bool)
    truth = target.astype(bool) & good
    predicted = (probability >= cutoff) & good
    return {"tp": int((predicted & truth).sum()), "fp": int((predicted & ~truth).sum()),
            "fn": int((~predicted & truth).sum()), "tn": int((~predicted & ~truth & good).sum())}


def measure(predictions: dict[str, dict], items: list[dict], cutoff: float, step: int) -> dict:
    pooled = collections.Counter()
    clear_pixels, clear_fp, clear_scenes, activated_clear = 0, 0, 0, 0
    positive_recall, positive_sites = [], collections.defaultdict(lambda: collections.Counter())
    misses, recovered, scene_counts = 0, 0, {}
    for item in items:
        if item["split"] != "dev":
            continue
        data = predictions[item["id"]]
        c = native_confusion(data["probability_native"], data["target_native"], data["valid_native"], cutoff)
        scene_counts[item["id"]] = c
        pooled.update(c)
        if item["scene_kind"] == "NO_PLUME":
            clear_scenes += 1
            clear_pixels += c["fp"] + c["tn"]
            clear_fp += c["fp"]
            activated_clear += int(c["fp"] > 0)
        else:
            recall = c["tp"] / max(c["tp"] + c["fn"], 1)
            positive_recall.append(recall)
            positive_sites[item["site"]].update({"tp": c["tp"], "truth": c["tp"] + c["fn"]})
            misses += int(c["tp"] == 0)
            recovered += int(recall >= .1)
    tp, fp, fn, tn = (pooled[k] for k in ("tp", "fp", "fn", "tn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    clear_fpr = clear_fp / max(clear_pixels, 1)
    activation = activated_clear / max(clear_scenes, 1)
    gates = {"precision_at_least_70": precision >= .7, "recall_at_least_30": recall >= .3,
             "clear_pixel_fpr_at_most_001": clear_fpr <= .001,
             "clear_activation_at_most_15": activation <= .15,
             "half_positive_observations_recover_10": recovered >= len(positive_recall) / 2}
    useful = {"f1_at_least_70": f1 >= .7, "precision_at_least_70": precision >= .7,
              "recall_at_least_65": recall >= .65,
              "clear_pixel_fpr_at_most_001": clear_fpr <= .001,
              "clear_activation_at_most_15": activation <= .15}
    return {"step": step, "cutoff": cutoff, "pooled": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1},
            "mean_positive_site_recall": float(np.mean([c["tp"] / max(c["truth"], 1) for c in positive_sites.values()])),
            "mean_positive_scene_recall": float(np.mean(positive_recall)),
            "positive_scenes": len(positive_recall), "positive_sites": len(positive_sites),
            "whole_plume_misses": misses, "recovered_positive_scenes": recovered,
            "marked_native_pixels": tp + fp, "clear_pixel_fpr": clear_fpr,
            "clear_scenes": clear_scenes, "activated_clear_scenes": activated_clear,
            "clear_activation_rate": activation, "selection_gates": gates,
            "selection_eligible": all(gates.values()), "failed_selection_gates": [k for k, v in gates.items() if not v],
            "useful_target_gates": useful, "useful_target_met": all(useful.values()),
            "scene_native_confusion": scene_counts}


def rank(row: dict) -> tuple:
    return (row["pooled"]["f1"], row["mean_positive_site_recall"], row["mean_positive_scene_recall"],
            -row["clear_pixel_fpr"], -row["step"], row["cutoff"])


def saved_predictions(arm: str, step: int, items: list[dict], meta: dict) -> dict[str, dict]:
    output = {}
    for item in items:
        if item["split"] != "dev":
            continue
        path = OUT / f"fit_{arm}" / f"dev_step_{step}" / f"{item['id']}.npz"
        expected = meta["milestones"][str(step)]["dev"]["scenes"][item["id"]]["probability_sha256"]
        if sha(path) != expected:
            raise ValueError("Saved DEV probability changed")
        with np.load(path, allow_pickle=False) as z:
            output[item["id"]] = {k: z[k] for k in ("probability_native", "target_native", "valid_native")}
    if len(output) != 198:
        raise ValueError("DEV prediction inventory changed")
    return output


def select() -> None:
    items = index()
    output = OUT / "selection"
    if output.exists():
        raise FileExistsError("Native selection already frozen")
    measurements, choices, initial = {}, {}, {}
    for arm in ARMS:
        meta = fit_metadata(arm)
        replay_path = OUT / f"replay_{arm}/receipt.json"
        replayed = json.loads(replay_path.read_text(encoding="utf-8"))
        if (not replayed["passed"] or replayed["fit_metadata_sha256"] != sha(OUT / f"fit_{arm}/metadata.json") or
            set(replayed["milestones"]) != {str(x) for x in MILESTONES}):
            raise ValueError("Fresh CPU replay required before selection")
        initial[arm] = meta["initial_state_sha256"]
        rows = []
        for step in MILESTONES:
            predictions = saved_predictions(arm, step, items, meta)
            rows.extend(measure(predictions, items, cutoff, step) for cutoff in CUTOFFS)
        if len(rows) != 88:
            raise ValueError("Expected 22 cutoffs at four checkpoints")
        eligible = [row for row in rows if row["selection_eligible"]]
        selected = max(eligible, key=rank) if eligible else next(row for row in rows if row["step"] == 1200 and row["cutoff"] == .7)
        best = max(rows, key=rank)
        choices[arm] = {"step": selected["step"], "cutoff": selected["cutoff"],
                        "checkpoint": str(checkpoint_path(arm, selected["step"])),
                        "checkpoint_sha256": meta["milestones"][str(selected["step"])]["checkpoint_sha256"],
                        "eligible": selected["selection_eligible"],
                        "fallback_failed_diagnostic": not bool(eligible)}
        measurements[arm] = {"candidates": rows, "eligible_rows": len(eligible),
                             "selected": selected, "unconstrained_best_rejected_diagnostic": best}
    if len(set(initial.values())) != 1:
        raise ValueError("Two arms did not share identical initialized weights")
    eligible_arms = [arm for arm in ARMS if choices[arm]["eligible"]]
    winner = max(eligible_arms, key=lambda arm: rank(measurements[arm]["selected"])) if eligible_arms else None
    output.mkdir(parents=True)
    save_json(output / "measurements.json", measurements)
    save_json(output / "all_arms_frozen.json", {"passed": True, "choices": choices, "selected_arm": winner,
              "retain_historical_v11": winner is None, "source_manifest_sha256": SOURCE_SHA,
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "data_receipt_sha256": sha(OUT / "data/receipt.json"),
              "schedule_sha256": sha(OUT / "preflight/draw_schedule.json"),
              "measurements_sha256": sha(output / "measurements.json"),
              "fit_metadata_sha256": {arm: sha(OUT / f"fit_{arm}/metadata.json") for arm in ARMS},
              "replay_receipt_sha256": {arm: sha(OUT / f"replay_{arm}/receipt.json") for arm in ARMS},
              "selection_kind": "historically exposed original DEV 198; no v12 ASSESS or protected final imagery",
              "prospective_assessment_opened": False})


def infer() -> None:
    frozen_path = OUT / "selection/all_arms_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (not frozen["passed"] or frozen["script_sha256"] != sha(Path(__file__)) or
        frozen["protocol_sha256"] != sha(PROTOCOL) or
        frozen["measurements_sha256"] != sha(OUT / "selection/measurements.json")):
        raise ValueError("Frozen native choices changed")
    destination = OUT / "evaluation"
    if destination.exists():
        raise FileExistsError("Native evaluation already published")
    items = index()
    destination.mkdir(parents=True)
    models = {}
    for arm in ARMS:
        choice = frozen["choices"][arm]
        meta = fit_metadata(arm)
        predictions = saved_predictions(arm, choice["step"], items, meta)
        scenes = {}
        folder = destination / arm
        folder.mkdir()
        for item in items:
            if item["split"] != "dev":
                continue
            sid = item["id"]
            source = OUT / f"fit_{arm}" / f"dev_step_{choice['step']}" / f"{sid}.npz"
            target = folder / f"{sid}.npz"
            shutil.copyfile(source, target)
            data = predictions[sid]
            scenes[sid] = {"probability_sha256": sha(target),
                           "native_confusion": native_confusion(data["probability_native"], data["target_native"], data["valid_native"], choice["cutoff"]),
                           "valid_native_pixels": int(data["valid_native"].sum()),
                           "positive_native_pixels": int((data["target_native"] * data["valid_native"]).sum())}
        models[arm] = {"step": choice["step"], "cutoff": choice["cutoff"], "eligible": choice["eligible"],
                       "scenes": scenes}
    measurement = json.loads((OUT / "selection/measurements.json").read_text(encoding="utf-8"))
    save_json(destination / "summary.json", {"models": models,
              "measurements": {arm: measurement[arm]["selected"] for arm in ARMS},
              "selection_freeze_sha256": sha(frozen_path), "source_manifest_sha256": SOURCE_SHA,
              "script_sha256": sha(Path(__file__)), "scene_count": 198,
              "grid": "native 200x200 10m annotation; Sentinel-2 SWIR physical resolution remains 20m"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train", "replay", "select", "infer"))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        preflight()
    elif args.command in ("train", "replay"):
        if not args.arm:
            parser.error("--arm is required for train/replay")
        (train if args.command == "train" else replay)(args.arm)
    elif args.command == "select":
        select()
    else:
        infer()


if __name__ == "__main__":
    main()
