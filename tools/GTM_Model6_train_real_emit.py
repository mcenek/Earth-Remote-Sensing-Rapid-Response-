"""Bounded CNN 2 fitting diagnostic on one real Sentinel/EMIT annotation pair.

Zeros are weak unannotated background, not confirmed methane-free observations.
All crops, including D4 transforms, are fitting examples from the same scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "research" / "GTM_Model6" / "GTM_Model6_three_cnn_torch.py"
sys.path.insert(0, str(REFERENCE.parent))
from GTM_Model6_three_cnn_torch import Branch  # noqa: E402

torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

CONFIG = dict(seed=20261023, epochs=160, max_minutes=5.0, base_channels=4,
              crop_fine=48, tiles_per_epoch=24, batch_size=8, background_fraction=0.5,
              learning_rate=0.001,
              inference_core=48, inference_halo=24,
              threshold=0.5, replay_atol=1e-7)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor(value):
    return torch.tensor(np.asarray(value).tolist(), dtype=torch.float64)


def array(value):
    return np.asarray(value.detach().cpu().tolist(), dtype=np.float32)


def chw(image):
    if image.ndim == 3:
        return tensor(np.moveaxis(image, -1, 0).copy()).unsqueeze(0)
    return tensor(np.moveaxis(image, -1, 1).copy())


def source_hashes():
    return {p.name: sha(p) for p in (Path(__file__), REFERENCE)}


def load_data(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "GTM_Model6_real_annotation_pilot_v1":
        raise ValueError("Expected real annotation pilot manifest")
    if (manifest.get("data_kind") != "measured_provider_annotation" or
            manifest.get("role") != "fit_only" or manifest.get("synthetic") is not False or
            manifest.get("independent_test") is not False):
        raise ValueError("Expected measured fit-only provider annotation, with no independent test")
    name = manifest.get("array_file")
    expected = manifest.get("array_sha256")
    if not isinstance(name, str) or not name or not isinstance(expected, str):
        raise ValueError("Manifest needs array_file and array_sha256")
    npz = (path.parent / name).resolve()
    if not npz.is_relative_to(path.parent.resolve()) or npz.suffix.lower() != ".npz":
        raise ValueError("Array file must be a relative NPZ inside the manifest directory")
    if sha(npz) != expected:
        raise ValueError("Array hash differs from manifest")
    for source in manifest.get("sources", []):
        if sha(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Original acquisition source changed: {source['path']}")
    with np.load(npz, allow_pickle=False) as z:
        a = {k: z[k] for k in ("image", "input_valid", "target", "target_valid")}
    image = a["image"].astype(np.float64)
    valid = a["input_valid"].astype(bool)
    target = a["target"].astype(np.float64)
    support = a["target_valid"].astype(bool)
    h, w = valid.shape
    nh, nw = target.shape
    if image.shape != (h, w, 5) or (h, w) != (nh * 3, nw * 3) or support.shape != target.shape:
        raise ValueError("Expected aligned exact 3:1 5-band image/target grids")
    if not np.isfinite(image[valid]).all() or not np.isfinite(target[support]).all():
        raise ValueError("Nonfinite supported data")
    if not np.isin(target[support], (0, 1)).all():
        raise ValueError("Target must be provider annotation 0/1")
    complete = valid.reshape(nh, 3, nw, 3).all(axis=(1, 3))
    if np.any(support & ~complete):
        raise ValueError("Native support includes partially invalid input")
    if not np.any(support & (target == 1)) or not np.any(support & (target == 0)):
        raise ValueError("Need supported annotated and weak background cells")
    return manifest, npz, image, valid, target, support


def normalize(image, valid):
    pixels = image[valid]
    center = pixels.mean(axis=0)
    scale = pixels.std(axis=0)
    scale = np.maximum(scale, 1.0)
    x = np.zeros_like(image, dtype=np.float64)
    x[valid] = (pixels - center) / scale
    return x, dict(center_dn=center.tolist(), scale_dn=scale.tolist(),
                   fill_value_standardized=0.0, input_units="raw Sentinel DN")


def apply_norm(image, valid, norm):
    x = np.zeros_like(image, dtype=np.float64)
    x[valid] = (image[valid] - norm["center_dn"]) / norm["scale_dn"]
    return x


def d4(a, code):
    if code & 1:
        a = np.flip(a, axis=1)
    return np.rot90(a, code >> 1, axes=(0, 1)).copy()


def centers(target, support, crop):
    nh, nw = target.shape
    side = crop // 3
    positive, background = [], []
    for y in range(0, nh - side + 1):
        for x in range(0, nw - side + 1):
            ss = support[y:y+side, x:x+side]
            if not ss.any():
                continue
            yy = target[y:y+side, x:x+side]
            (positive if np.any((yy == 1) & ss) else background).append((y, x))
    if not positive:
        raise ValueError("No fitting crop contains a supported annotation")
    return positive, background


def native_prob(fine, shape):
    nh, nw = shape
    return fine.reshape(nh, 3, nw, 3).mean(axis=(1, 3)).astype(np.float32)


def loss(model, x, y, support):
    _, logits = model(chw(x))
    p = torch.sigmoid(logits)
    batch, nh, nw = y.shape
    coarse = p.reshape(batch, 1, nh, 3, nw, 3).mean(dim=(3, 5))[:, 0]
    yy = tensor(np.where(support, y, 0.0))
    mm = tensor(support.astype(np.float64))
    q = coarse.clamp(1e-8, 1 - 1e-8)
    pos = (mm * yy).sum()
    neg = (mm * (1 - yy)).sum()
    if pos <= 0 or neg <= 0:
        raise ValueError("Batch lacks both annotation classes")
    positive_bce = -(mm * yy * q.log()).sum() / pos
    background_bce = -(mm * (1 - yy) * (1 - q).log()).sum() / neg
    bce = (positive_bce + background_bce) / 2
    dice = 1 - (2 * (mm * coarse * yy).sum() + 1) / ((mm * coarse).sum() + (mm * yy).sum() + 1)
    return bce + dice


def inference(model, x, valid, config):
    h, w = valid.shape
    core, halo = config["inference_core"], config["inference_halo"]
    summed = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.uint16)
    model.eval()
    with torch.no_grad():
        for y in range(0, h, core):
            for xx in range(0, w, core):
                y1, x1 = min(y + core, h), min(xx + core, w)
                ay, ax = max(0, y - halo), max(0, xx - halo)
                by, bx = min(h, y1 + halo), min(w, x1 + halo)
                block = chw(x[ay:by, ax:bx])
                ph, pw = (-block.shape[-2]) % 4, (-block.shape[-1]) % 4
                if ph or pw:
                    block = F.pad(block, (0, pw, 0, ph), mode="replicate")
                _, logits = model(block)
                p = array(torch.sigmoid(logits[0, 0, :by-ay, :bx-ax]))
                summed[y:y1, xx:x1] += p[y-ay:y1-ay, xx-ax:x1-ax]
                count[y:y1, xx:x1] += 1
    if np.any(count != 1):
        raise RuntimeError("Inference coverage failed")
    fine = (summed / count).astype(np.float32)
    fine[~valid] = np.nan
    return fine


def agreement(p, target, support, threshold):
    pred = p[support] >= threshold
    truth = target[support] == 1
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    return dict(tp=tp, fp_weak_background=fp, fn=fn, tn_weak_background=tn,
                annotated_recall=tp/(tp+fn) if tp+fn else None,
                annotation_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                supported_cells=int(support.sum()),
                predicted_positive_cells=int(pred.sum()))


def probability_summary(p, support):
    values = p[support]
    return dict(min=float(np.min(values)), max=float(np.max(values)),
                mean=float(np.mean(values)), threshold_positive_fraction=float(np.mean(values >= .5)))


def fit(args):
    manifest, npz, image, valid, target, support = load_data(args.manifest)
    run = args.run_dir
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Run folder must be new and empty")
    run.mkdir(parents=True, exist_ok=True)
    config = {**CONFIG, "epochs": args.epochs, "max_minutes": args.max_minutes,
              "tiles_per_epoch": args.tiles_per_epoch}
    if (config["epochs"] < 1 or config["max_minutes"] <= 0 or
            config["tiles_per_epoch"] % config["batch_size"] != 0 or
            config["background_fraction"] != 0.5):
        raise ValueError("Invalid bounded balanced fit configuration")
    x, norm = normalize(image, valid)
    positive, background = centers(target, support, config["crop_fine"])
    torch.manual_seed(config["seed"])
    model = Branch(config["base_channels"]).double()
    initial_path, final_path = run / "initial.pt", run / "final.pt"
    torch.save(model.state_dict(), initial_path)
    initial_fine = inference(model, x, valid, config)
    initial_native = native_prob(initial_fine, target.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    history, sample_counts = [], dict(positive=0, weak_background=0)
    d4_counts = Counter()
    unique_origins = dict(positive=set(), weak_background=set())
    start = time.monotonic()
    deadline = start + 60 * config["max_minutes"]
    for epoch in range(config["epochs"]):
        if time.monotonic() >= deadline:
            break
        before_epoch = {k: v.clone() for k, v in model.state_dict().items()}
        losses = []
        epoch_counts = dict(positive=0, weak_background=0)
        epoch_d4 = Counter()
        epoch_origins = dict(positive=set(), weak_background=set())
        for batch_index in range(config["tiles_per_epoch"] // config["batch_size"]):
            if time.monotonic() >= deadline:
                break
            images, labels, masks = [], [], []
            kinds = (["positive"] * (config["batch_size"] // 2) +
                     ["weak_background"] * (config["batch_size"] // 2))
            rng.shuffle(kinds)
            for kind in kinds:
                pool = background if kind == "weak_background" and background else positive
                actual_kind = kind if pool is background else "positive"
                cy, cx = pool[int(rng.integers(len(pool)))]
                fy, fx = cy * 3, cx * 3
                side = config["crop_fine"]
                code = int(rng.integers(8))
                epoch_d4[code] += 1
                epoch_origins[actual_kind].add((cy, cx))
                images.append(d4(x[fy:fy+side, fx:fx+side], code))
                labels.append(d4(target[cy:cy+side//3, cx:cx+side//3], code))
                masks.append(d4(support[cy:cy+side//3, cx:cx+side//3], code))
                epoch_counts[actual_kind] += 1
                sample_counts[actual_kind] += 1
            model.train()
            optimizer.zero_grad()
            objective = loss(model, np.stack(images), np.stack(labels), np.stack(masks))
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite fitting loss")
            objective.backward()
            optimizer.step()
            losses.append(float(objective.item()))
        if len(losses) != config["tiles_per_epoch"] // config["batch_size"]:
            model.load_state_dict(before_epoch)
            sample_counts = {key: sample_counts[key]-epoch_counts[key] for key in sample_counts}
            break
        item = dict(epoch=epoch+1, fitting_loss=float(np.mean(losses)),
                    fitting_tiles=sum(epoch_counts.values()), fitting_batches=len(losses),
                    sampled=epoch_counts)
        history.append(item)
        d4_counts.update(epoch_d4)
        for kind in unique_origins:
            unique_origins[kind].update(epoch_origins[kind])
        print(json.dumps(item), flush=True)
    if not history:
        raise TimeoutError("No complete fitting epoch")
    torch.save(model.state_dict(), final_path)
    final_fine = inference(model, x, valid, config)
    final_native = native_prob(final_fine, target.shape)
    predictions = run / "predictions.npz"
    np.savez_compressed(predictions, initial_fine=initial_fine, final_fine=final_fine,
                        initial_native=initial_native, final_native=final_native,
                        image_dn=image.astype(np.float32), input_valid=valid,
                        target=target.astype(np.float32), target_valid=support)
    snapshots = run / "source_snapshots"
    snapshots.mkdir()
    for source in (Path(__file__), REFERENCE):
        shutil.copy2(source, snapshots / source.name)
    controls = dict(always_negative=agreement(np.zeros(target.shape), target, support, .5),
                    always_positive=agreement(np.ones(target.shape), target, support, .5))
    receipt = dict(schema="GTM_Model6_real_cnn2_fit_v1", data_kind="real_cached_observation",
                   stage="CNN 2", claim="single-scene fitting and provider-annotation agreement only",
                   target_semantics="1=provider-annotated plume; 0=weak unannotated background; unsupported=unknown",
                   fine_output_semantics="3:1 fine prediction is not validated fine detail",
                   cohort="all crops and D4 transforms are fitting only; no train/test split",
                   manifest_path=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
                   array_path=str(npz), array_sha256=sha(npz), source_hashes=source_hashes(),
                   source_snapshots={p.name: sha(snapshots/p.name) for p in (Path(__file__), REFERENCE)},
                   config=config, normalization=norm,
                   objective="class-balanced native-grid BCE plus batch-global soft Dice",
                   scene_shape=list(image.shape), target_shape=list(target.shape),
                   native_support_cells=int(support.sum()), annotated_cells=int(np.sum((target==1)&support)),
                   weak_background_cells=int(np.sum((target==0)&support)),
                   candidate_crops=dict(positive=len(positive), weak_background=len(background)),
                   sampled_crops=sample_counts,
                   d4_counts={str(i): d4_counts[i] for i in range(8)},
                   unique_sampled_crop_origins={kind: len(origins) for kind, origins in unique_origins.items()},
                   history=history, requested_epochs=config["epochs"],
                   completed_epochs=len(history), elapsed_seconds=round(time.monotonic()-start,3),
                   initial=dict(path=initial_path.name, sha256=sha(initial_path),
                                agreement=agreement(initial_native,target,support,.5),
                                probability=probability_summary(initial_native,support)),
                   final=dict(path=final_path.name, sha256=sha(final_path),
                              agreement=agreement(final_native,target,support,.5),
                              probability=probability_summary(final_native,support)),
                   controls=controls, prediction=dict(path=predictions.name, sha256=sha(predictions)),
                   flags=dict(empty_threshold_mask=bool(np.sum((final_native>=.5)&support)==0),
                              full_supported_mask=bool(np.all(final_native[support]>=.5)),
                              constant_probability=bool(np.ptp(final_native[support])<1e-5)),
                   runtime=dict(python=platform.python_version(), pytorch=torch.__version__,
                                numpy=np.__version__, dtype="float64 computation / float32 predictions",
                                cpu_threads=torch.get_num_threads(), mkldnn=False,
                                deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
    write_json(run / "receipt.json", receipt)
    print(json.dumps({"completed_epochs":len(history),"initial":receipt["initial"]["agreement"],
                      "final":receipt["final"]["agreement"],"flags":receipt["flags"]}),flush=True)


def replay(args):
    run = args.run_dir
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    if receipt["schema"] != "GTM_Model6_real_cnn2_fit_v1":
        raise ValueError("Wrong receipt schema")
    manifest, npz, image, valid, target, support = load_data(args.manifest)
    if sha(args.manifest) != receipt["manifest_sha256"] or sha(npz) != receipt["array_sha256"]:
        raise ValueError("Changed real input")
    if source_hashes() != receipt["source_hashes"]:
        raise ValueError("Changed training source")
    for name, digest in receipt["source_snapshots"].items():
        if sha(run / "source_snapshots" / name) != digest:
            raise ValueError("Changed source snapshot")
    if sha(run / "predictions.npz") != receipt["prediction"]["sha256"]:
        raise ValueError("Changed prediction archive")
    x = apply_norm(image, valid, receipt["normalization"])
    diffs = {}
    with np.load(run / "predictions.npz", allow_pickle=False) as saved:
        for stage in ("initial", "final"):
            info = receipt[stage]
            path = run / info["path"]
            if sha(path) != info["sha256"]:
                raise ValueError("Changed checkpoint")
            model = Branch(receipt["config"]["base_channels"]).double()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            fine = inference(model, x, valid, receipt["config"])
            native = native_prob(fine, target.shape)
            diffs[stage] = dict(fine_max_abs=float(np.max(np.abs(fine[valid]-saved[stage+"_fine"][valid]))),
                                native_max_abs=float(np.max(np.abs(native[support]-saved[stage+"_native"][support]))))
            if max(diffs[stage].values()) > receipt["config"]["replay_atol"]:
                raise ValueError(f"{stage} replay difference: {diffs[stage]}")
            if agreement(native,target,support,.5) != info["agreement"]:
                raise ValueError("Recorded annotation agreement changed")
        if not np.array_equal(target.astype(np.float32), saved["target"], equal_nan=True) or not np.array_equal(support,saved["target_valid"]):
            raise ValueError("Saved target/support changed")
    result = dict(schema="GTM_Model6_real_cnn2_replay_v1", passed=True,
                  manifest_sha256=receipt["manifest_sha256"],
                  receipt_sha256=sha(run / "receipt.json"), max_abs_diff=diffs,
                  tolerance=receipt["config"]["replay_atol"])
    # Replay never changes training artifacts; write a separate verification artifact once.
    path = run / "replay.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing replay receipt differs")
    else:
        write_json(path, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "replay"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--max-minutes", type=float, default=CONFIG["max_minutes"])
    parser.add_argument("--tiles-per-epoch", type=int, default=CONFIG["tiles_per_epoch"])
    args = parser.parse_args()
    if args.command == "fit":
        fit(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
