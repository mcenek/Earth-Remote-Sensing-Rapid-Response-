"""Train-only stabilized residual probe and bounded fit; no test-data access.

The failure protocol was written first in docs/GTM_Model7_stable_residual_protocol.md.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
ORIGINAL = ROOT / "tools/GTM_Model7_train_temporal.py"
PROTOCOL = ROOT / "docs/GTM_Model7_stable_residual_protocol.md"
MANIFEST = ROOT / "outputs/GTM_Model7_temporal_20261023/data/manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/GTM_Model7_recovery_20261023/stable_residual"
spec = importlib.util.spec_from_file_location("original_temporal", ORIGINAL)
old = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(old)

SEED = old.BASE["seed"]
LR = 0.0005
WEIGHT_DECAY = old.BASE["weight_decay"]
PROBE_STEPS = 300
FIT_STEPS = 1800
MAX_SECONDS = 900
PROBE_RECALL_MIN = 0.8
PROBE_NEG_FPR_MAX = 0.05


class StableBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.a = nn.Conv2d(8, 8, 3, padding=dilation, dilation=dilation)
        self.na = nn.GroupNorm(4, 8)
        self.b = nn.Conv2d(8, 8, 3, padding=1)
        self.nb = nn.GroupNorm(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.nb(self.b(F.relu(self.na(self.a(x)))))
        return F.relu(x + 0.1 * branch)


class StableResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(12, 8, 3, padding=1)
        self.stem_norm = nn.GroupNorm(4, 8)
        self.blocks = nn.Sequential(*(StableBlock(d) for d in (1, 2, 4)))
        self.head_norm = nn.GroupNorm(4, 8)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current, reference = x[:, :6], x[:, 6:]
        features = F.relu(self.stem_norm(self.stem(torch.cat((current, current - reference), 1))))
        return self.head(F.relu(self.head_norm(self.blocks(features))))


def new_model() -> StableResidual:
    return StableResidual().double()


def checked_scenes() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != "GTM_Model7_temporal_cohort_v1":
        raise ValueError("Unexpected cohort schema")
    scenes = [s for s in manifest["scenes"] if s["split"] in ("train", "dev")]
    if len(scenes) != 176 or sum(s["split"] == "train" for s in scenes) != 152:
        raise ValueError("Train/development cohort changed")
    source_hashes = {}
    for scene in scenes:
        file = old.array_path(MANIFEST, scene["array_file"])
        if old.sha(file) != scene["array_sha256"]:
            raise ValueError("Train/dev array changed: " + scene["id"])
        for source in scene["sources"]:
            path = old.source_path(MANIFEST, source["path"])
            if path not in source_hashes:
                source_hashes[path] = old.sha(path)
            if source_hashes[path] != source["sha256"]:
                raise ValueError("Train/dev source changed: " + str(path))
        old.load_scene(MANIFEST, scene, labels=True)
    return scenes


def step(model, optimizer, sample, *, ordinary: bool):
    x, y, valid, positive = sample
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    if ordinary:
        raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (raw * valid).sum() / valid.sum()
    else:
        raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                                 pos_weight=torch.tensor(2.0))
        bce = (raw * valid).sum() / valid.sum()
        on_positive_crops = positive[:, None, None, None] * valid
        p = torch.sigmoid(logits)
        dice = 1 - (2 * (p * y * on_positive_crops).sum() + 1) / (
            (p * on_positive_crops).sum() + (y * on_positive_crops).sum() + 1)
        loss = bce + 0.1 * dice
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite loss")
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(gradient):
        raise FloatingPointError("Nonfinite gradient")
    optimizer.step()
    return float(loss.item()), float(gradient.item())


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def checkpoint(path: Path, model, norm, stage: str, steps: int):
    torch.save({"model": model.state_dict(), "normalization": norm,
                "stage": stage, "steps": steps, "learning_rate": LR,
                "architecture": "GroupNorm4 stem_blocks_head scaled_residual_0.1 width8 dilation1_2_4"}, path)
    return old.sha(path)


def probe(output: Path, scenes: list[dict], norm: dict, started: float) -> dict:
    draws = old.schedule(MANIFEST, scenes, 1)[0][:4]
    by_id = {s["id"]: s for s in scenes}
    sample = old.batch(MANIFEST, by_id, draws, norm, {})
    x, y, valid, _ = sample
    np.savez_compressed(output / "probe_inputs.npz", current=old.array(x[:, :6]),
                        reference=old.array(x[:, 6:]), target=old.array(y),
                        valid=old.array(valid))
    torch.manual_seed(SEED)
    model = new_model()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    initial_hash = checkpoint(output / "probe_initial.pt", model, norm, "probe", 0)
    with torch.no_grad():
        first = float(((F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * valid).sum()
                       / valid.sum()).item())
    optim = optimizer_for(model)
    curve = []
    loss = first
    for n in range(1, PROBE_STEPS + 1):
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("Stable residual exceeded 15-minute cap")
        loss, grad = step(model, optim, sample, ordinary=True)
        if n % 10 == 0 or n == 1:
            curve.append({"step": n, "loss": loss, "gradient_norm": grad})
    final_hash = checkpoint(output / "probe_final.pt", model, norm, "probe", PROBE_STEPS)
    model.eval()
    with torch.no_grad():
        p = old.array(torch.sigmoid(model(x))[:, 0])
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite probe probability")
    np.savez_compressed(output / "probe_probabilities.npz", probability=p)
    target = old.array(y[:, 0])
    support = old.array(valid[:, 0]).astype(bool)
    positive = old.score(old.confusion(p[[0, 2]], target[[0, 2]], support[[0, 2]]))
    negative = old.score(old.confusion(p[[1, 3]], target[[1, 3]], support[[1, 3]]))
    movement = sum(float((value - initial[key]).abs().sum().item())
                   for key, value in model.state_dict().items())
    qualified = (movement > 0 and (positive["recall"] or 0) >= PROBE_RECALL_MIN and
                 negative["false_positive_area_rate"] < PROBE_NEG_FPR_MAX)
    result = {"draws": draws, "initial_loss": first, "final_loss": loss,
              "positive_probe": positive, "negative_probe": negative,
              "parameter_l1_change": movement, "qualified": qualified,
              "curve": curve, "initial_checkpoint_sha256": initial_hash,
              "final_checkpoint_sha256": final_hash,
              "input_sha256": old.sha(output / "probe_inputs.npz"),
              "probability_sha256": old.sha(output / "probe_probabilities.npz")}
    old.write_json(output / "probe_receipt.json", result)
    return result


def evaluate(model, scenes: list[dict], norm: dict, split: str, folder: Path) -> dict:
    folder.mkdir()
    pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    per_scene = {}
    for scene in scenes:
        if scene["split"] != split:
            continue
        data = old.load_scene(MANIFEST, scene, labels=True)
        valid = data["valid"].astype(bool)
        p = old.infer(model, old.prepare(data, norm), valid)
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite scene probability")
        c = old.confusion(p, data["target"], valid)
        for key in pooled:
            pooled[key] += c[key]
        file = folder / (scene["id"] + ".npz")
        np.savez_compressed(file, probability20=p, valid20=valid)
        per_scene[scene["id"]] = {"confusion": c, "probability_sha256": old.sha(file)}
    return {**old.score(pooled), "scene_count": len(per_scene), "scenes": per_scene}


def fit(output: Path, scenes: list[dict], norm: dict, started: float) -> dict:
    torch.manual_seed(SEED)
    model = new_model()
    fit_dir = output / "fit"
    fit_dir.mkdir()
    initial_hash = checkpoint(fit_dir / "initial.pt", model, norm, "fit", 0)
    optim = optimizer_for(model)
    draws = old.schedule(MANIFEST, scenes, FIT_STEPS // 6)
    by_id = {s["id"]: s for s in scenes}
    cache = {}
    curve = []
    n = 0
    for epoch in draws:
        for offset in range(0, len(epoch), 8):
            if time.monotonic() - started > MAX_SECONDS:
                raise TimeoutError("Stable residual exceeded 15-minute cap")
            sample = old.batch(MANIFEST, by_id, epoch[offset:offset + 8], norm, cache)
            loss, grad = step(model, optim, sample, ordinary=False)
            n += 1
            if n % 25 == 0 or n == 1:
                curve.append({"step": n, "loss": loss, "gradient_norm": grad})
    if n != FIT_STEPS:
        raise ValueError("Unexpected fit update count")
    final_hash = checkpoint(fit_dir / "final.pt", model, norm, "fit", FIT_STEPS)
    train = evaluate(model, scenes, norm, "train", fit_dir / "train_probabilities")
    dev = evaluate(model, scenes, norm, "dev", fit_dir / "dev_probabilities")
    result = {"steps": n, "initial_checkpoint_sha256": initial_hash,
              "final_checkpoint_sha256": final_hash,
              "schedule_sha256": old.schedule_hash(draws),
              "curve": curve, "train": train, "dev": dev,
              "elapsed_seconds": time.monotonic() - started}
    old.write_json(fit_dir / "receipt.json", result)
    return result


def run(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing populated output: " + str(output))
    started = time.monotonic()
    scenes = checked_scenes()
    norm = old.normalization(MANIFEST, scenes)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "GTM_Model7_stable_residual_v1",
                "manifest_sha256": old.sha(MANIFEST), "original_source_sha256": old.sha(ORIGINAL),
                "source_sha256": old.sha(HERE), "protocol_sha256": old.sha(PROTOCOL),
                "normalization": norm, "seed": SEED, "learning_rate": LR,
                "probe_steps": PROBE_STEPS, "fit_steps_if_probe_qualified": FIT_STEPS,
                "test_access": "none", "train_scenes": 152, "dev_scenes": 24}
    old.write_json(output / "metadata.json", metadata)
    probe_result = probe(output, scenes, norm, started)
    print(json.dumps({"probe_qualified": probe_result["qualified"],
                      "positive_recall": probe_result["positive_probe"]["recall"],
                      "negative_fpr": probe_result["negative_probe"]["false_positive_area_rate"],
                      "final_loss": probe_result["final_loss"]}), flush=True)
    if not probe_result["qualified"]:
        return
    result = fit(output, scenes, norm, started)
    print(json.dumps({"train_iou": result["train"]["iou"],
                      "train_recall": result["train"]["recall"],
                      "dev_fpr": result["dev"]["false_positive_area_rate"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def replay(output: Path) -> None:
    scenes = checked_scenes()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if (old.sha(MANIFEST) != metadata["manifest_sha256"] or
            old.sha(ORIGINAL) != metadata["original_source_sha256"] or
            old.sha(HERE) != metadata["source_sha256"] or
            old.sha(PROTOCOL) != metadata["protocol_sha256"]):
        raise ValueError("Replay source fingerprint changed")
    norm = metadata["normalization"]
    probe_checkpoint = torch.load(output / "probe_final.pt", map_location="cpu", weights_only=False)
    probe_model = new_model()
    probe_model.load_state_dict(probe_checkpoint["model"])
    with np.load(output / "probe_inputs.npz", allow_pickle=False) as z:
        x = np.concatenate((z["current"], z["reference"]), axis=1)
    probe_model.eval()
    with torch.no_grad():
        actual = old.array(torch.sigmoid(probe_model(old.tensor(x))[:, 0]))
    with np.load(output / "probe_probabilities.npz", allow_pickle=False) as z:
        probe_error = float(np.max(abs(actual - z["probability"])))
    if probe_error > 1e-6:
        raise ValueError("Probe probability replay differed")
    checked = 0
    maximum_error = probe_error
    if (output / "fit/final.pt").exists():
        fit_checkpoint = torch.load(output / "fit/final.pt", map_location="cpu", weights_only=False)
        fit_model = new_model()
        fit_model.load_state_dict(fit_checkpoint["model"])
        receipt = json.loads((output / "fit/receipt.json").read_text(encoding="utf-8"))
        for split in ("train", "dev"):
            for scene in scenes:
                if scene["split"] != split:
                    continue
                data = old.load_scene(MANIFEST, scene, labels=True)
                actual = old.infer(fit_model, old.prepare(data, norm), data["valid"])
                file = output / "fit" / (split + "_probabilities") / (scene["id"] + ".npz")
                if old.sha(file) != receipt[split]["scenes"][scene["id"]]["probability_sha256"]:
                    raise ValueError("Saved probability file changed")
                with np.load(file, allow_pickle=False) as z:
                    error = float(np.max(abs(actual - z["probability20"])))
                maximum_error = max(maximum_error, error)
                if error > 1e-6:
                    raise ValueError("Full-scene probability replay differed: " + scene["id"])
                confusion = old.confusion(actual, data["target"], data["valid"].astype(bool))
                if confusion != receipt[split]["scenes"][scene["id"]]["confusion"]:
                    raise ValueError("Full-scene confusion replay differed: " + scene["id"])
                checked += 1
    result = {"schema": "GTM_Model7_stable_residual_replay_v1", "status": "passed",
              "probe_max_abs_error": probe_error, "full_scene_count": checked,
              "max_abs_error": maximum_error, "test_access": "none"}
    old.write_json(output / "replay_receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.output.resolve())
    else:
        run(args.output.resolve())
