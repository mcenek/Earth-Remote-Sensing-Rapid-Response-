#!/usr/bin/env python3
"""Train a fresh MARS-S2L U-Net on a frozen, site-disjoint local cohort.

The data sampler, plume injection, preprocessing and loss come from the cached
UNEP-IMEO-MARS/marss2l source. The 16-channel U-Net is the inference-equivalent
implementation in evaluate_released_marss2l.py. A smaller native batch changes
BatchNorm statistics, so every run records this hardware adaptation explicitly.

Manifest schema: {"fit_ids": [id_loc_image, ...], "val_ids": [...]}.
No test or held-out IDs belong in this file. The runner never reads test labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/marss2l-code"
SOURCE_REVISION = "f7d264c2c845dfba1cb27f76ef6026275f8d8758"
METADATA_REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"
SOURCE_HASHES = {
    "loaders.py": "db760531edd9b70c6c5b2679fb5a99dc0e492a2c766b83bc8eca37b0d5952b51",
    "loss.py": "fb662921052d55ac5fee0b207304cc74e9ea78da72c7a9456b4095197d464433",
    "models.py": "8845fdd1db083271e25c55ea7896355ca631e94e6898903bb6be0ddc4c2b7f8b",
    "dataframe_image_plumes.py": "17e6a7dd216509141cd401c4cc2dfbadce529e64f6ecd93cd900bb37e603af80",
    "mars_sentinel2/plumesimulation.py": "cf42119999c7676fdf7756b6e6532cbff8208d85c96a0de3fad3422102a338ca",
    "trainer.py": "479e13c0ff76f25d52af1bda9757581c0f7cb23cc792c612a921432917a55718",
}
SEED = 20261030
DEFAULT_DRAWS = 65536
CUDA_MEMORY_FRACTION = 0.72


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_source_hashes() -> dict[str, str]:
    # The cache's nested .git directory can be hidden by the Windows sandbox.
    # Pin executable content directly rather than trusting its HEAD label.
    found = {name: sha256(UPSTREAM / "marss2l" / name) for name in SOURCE_HASHES}
    if found != SOURCE_HASHES:
        raise ValueError("Cached MARS source files differ from pinned SHA256 hashes")
    return found


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def frozen_manifest(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fit = payload["fit_ids"]
    val = payload["val_ids"]
    for name, values in (("fit_ids", fit), ("val_ids", val)):
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"{name} must be a nonempty list of id_loc_image strings")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate IDs")
    if set(fit) & set(val):
        raise ValueError("FIT and validation IDs overlap")
    return fit, val


def raw_audit(data_root: Path, fit: list[str], val: list[str]) -> dict:
    """Validate paths and site separation before importing heavy source code."""
    metadata = data_root / "validated_images_all.csv"
    plume_csv = data_root / "validated_images_plumes.csv"
    if not metadata.is_file() or not plume_csv.is_file():
        raise FileNotFoundError("Both validated_images_all.csv and validated_images_plumes.csv are required")
    wanted = set(fit) | set(val)
    rows = {}
    with metadata.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            key = row["id_loc_image"]
            if key in wanted:
                if key in rows:
                    raise ValueError(f"Duplicate metadata ID {key}")
                rows[key] = row
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"{len(missing)} manifest IDs are absent from metadata; example: {sorted(missing)[0]}")
    fit_sites = {rows[key]["location_name"] for key in fit}
    val_sites = {rows[key]["location_name"] for key in val}
    if fit_sites & val_sites:
        raise ValueError(f"FIT and validation share sites: {sorted(fit_sites & val_sites)[:5]}")
    plume_ids = set()
    bad_plume_geometries = []
    with plume_csv.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["id_loc_image"] in fit:
                plume_ids.add(row["id_loc_image"])
                if not row.get("geometry", "").strip():
                    bad_plume_geometries.append(row["id_loc_image"])
    if not plume_ids or bad_plume_geometries:
        raise ValueError(f"No usable FIT plume simulation geometry; empty examples {bad_plume_geometries[:5]}")
    absent = []
    unsafe = []
    bad_image_geometries = []
    for key in wanted:
        row = rows[key]
        if row["isplume"].lower() == "true" and not row.get("plume", "").strip():
            bad_image_geometries.append(key)
        # Pinned DatasetPlumes rasterizes positive masks from `plume` WKT.
        # Its simulation path reads CH4 rasters only for FIT plume CSV sources.
        required_columns = ["s2path", "cloudmaskpath"]
        if key in plume_ids:
            required_columns.append("ch4path")
        for column in required_columns:
            value = row.get(column, "")
            candidate = (data_root / value).resolve() if value else data_root
            if not value or data_root.resolve() not in candidate.parents:
                unsafe.append((key, column, value))
            elif not candidate.is_file():
                absent.append((key, column, value))
    if unsafe:
        raise ValueError(f"Missing/unsafe raster paths: {unsafe[:5]}")
    if bad_image_geometries:
        raise ValueError(f"Positive images lack plume geometry; examples {bad_image_geometries[:5]}")
    fit_pos = sum(rows[key]["isplume"].lower() == "true" for key in fit)
    val_pos = sum(rows[key]["isplume"].lower() == "true" for key in val)
    if fit_pos == 0 or val_pos == 0 or val_pos == len(val):
        raise ValueError("FIT needs positives and validation needs both classes for AP")
    no_reference_tile = [key for key in wanted if not rows[key].get("background_image_tile", "").strip()]
    return {
        "fit_rows": len(fit), "fit_positive_rows": fit_pos, "fit_sites": len(fit_sites),
        "validation_rows": len(val), "validation_positive_rows": val_pos,
        "validation_sites": len(val_sites), "fit_plume_source_rows": len(plume_ids),
        "missing_background_tile_ids_fit": sum(key in fit for key in no_reference_tile),
        "missing_background_tile_ids_validation": sum(key in val for key in no_reference_tile),
        "missing_background_tile_id_examples": no_reference_tile[:10],
        "plume_mask_convention": "rasterize `plume` WKT; plumepath TIFF is not read",
        "missing_rasters": len(absent), "missing_raster_examples": absent[:10],
        "metadata_sha256": sha256(metadata), "plumes_sha256": sha256(plume_csv),
    }


def paired_raster_qa(data_root: Path, fit: list[str], val: list[str],
                     regular_limit: int = 32, missing_reference_limit: int = 1024) -> dict:
    """Inspect paired SWIR content; report odd references without excluding rows."""
    import rasterio

    wanted = set(fit) | set(val)
    regular, missing_ref = [], []
    with (data_root / "validated_images_all.csv").open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["id_loc_image"] not in wanted:
                continue
            if row.get("background_image_tile", "").strip():
                regular.append(row)
            else:
                missing_ref.append(row)
    # Deterministic spread across IDs; every missing-ID case is
    # inspected up to the explicit cap. No sample is dropped from training.
    regular.sort(key=lambda row: row["id_loc_image"])
    missing_ref.sort(key=lambda row: row["id_loc_image"])
    if len(regular) > regular_limit:
        locations = np.linspace(0, len(regular) - 1, regular_limit, dtype=int)
        regular = [regular[index] for index in locations]
    selected = missing_ref[:missing_reference_limit] + regular
    examples = []
    zeros_ref = same_swir = 0
    for row in selected:
        path = data_root / row["s2path"]
        with rasterio.open(path) as raster:
            if raster.count != 12 or raster.width != 200 or raster.height != 200:
                raise ValueError(f"Source requires 12x200x200 paired raster: {path} has {raster.count}x{raster.height}x{raster.width}")
            swir = raster.read([5, 6, 11, 12]).astype(np.float32)
        current, reference = swir[:2], swir[2:]
        reference_zero = bool(np.all(reference == 0))
        identical = bool(np.array_equal(current, reference))
        zeros_ref += reference_zero
        same_swir += identical
        if len(examples) < 20 and (reference_zero or identical or not row.get("background_image_tile", "").strip()):
            examples.append({"id_loc_image": row["id_loc_image"],
                             "role": "fit" if row["id_loc_image"] in fit else "validation",
                             "background_tile_missing": not bool(row.get("background_image_tile", "").strip()),
                             "reference_swir_all_zero": reference_zero,
                             "current_reference_swir_identical": identical,
                             "current_swir_zero_fraction": float(np.mean(current == 0)),
                             "reference_swir_zero_fraction": float(np.mean(reference == 0))})
    return {"inspected_rows": len(selected), "missing_background_tile_rows_total": len(missing_ref),
            "missing_background_tile_rows_inspected": min(len(missing_ref), missing_reference_limit),
            "reference_swir_all_zero_count": zeros_ref,
            "current_reference_swir_identical_count": same_swir,
            "examples": examples,
            "source_handling": "Source normalizes/clamps paired bands and computes MBMP; it does not reject missing background acquisition IDs, identical references, or all-zero references here",
            "acquisition_date_inference": "No background date inferred when tile ID is missing"}


def source_modules():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if str(UPSTREAM) not in existing_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(UPSTREAM) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    try:
        from marss2l.dataframe_image_plumes import (  # type: ignore
            process_images_and_plumes, read_csv_images, read_csv_plumes,
        )
        from marss2l.loaders import DatasetPlumes  # type: ignore
        from marss2l.loss import DeltaXCH4Loss  # type: ignore
    except ImportError as error:
        raise RuntimeError(f"Pinned MARS source dependency unavailable: {error}") from error
    from loguru import logger  # type: ignore
    if not getattr(source_modules, "_quiet_sampler", False):
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        source_modules._quiet_sampler = True
    from evaluate_released_marss2l import ReleasedUNet
    bridge_pinned_crop_argument()
    return process_images_and_plumes, read_csv_images, read_csv_plumes, DatasetPlumes, DeltaXCH4Loss, ReleasedUNet


def square_window_size(value: int | tuple[int, int]) -> int:
    if isinstance(value, tuple):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"Pinned source crop expects a square window, received {value}")
        return int(value[0])
    return int(value)


def bridge_pinned_crop_argument() -> None:
    """Bridge the pinned loader's tuple argument to sampling.py's scalar API."""
    from marss2l import loaders  # type: ignore
    if getattr(loaders.sample_window, "_gtm_tuple_bridge", False):
        return
    original = loaders.sample_window

    def compatible_sample_window(plume_window, window_size_training, window_size_data,
                                 add_jitter=True):
        return original(plume_window, window_size_training,
                        square_window_size(window_size_data), add_jitter=add_jitter)

    compatible_sample_window._gtm_tuple_bridge = True
    loaders.sample_window = compatible_sample_window


class SeededDraws(torch.utils.data.Dataset):
    """Frozen source sampler draws indexed independently for process replay."""

    def __init__(self, source_dataset, epoch: int, count: int):
        self.source_dataset = source_dataset
        self.epoch = epoch
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getstate__(self):
        # Loguru's parent-process stderr sink owns an unpicklable TextIOWrapper.
        # The source dataset itself remains unchanged in the parent process.
        state = self.__dict__.copy()
        worker_source = copy.copy(self.source_dataset)
        worker_source.logger = WorkerLogger()
        state["source_dataset"] = worker_source
        return state

    def __getitem__(self, index: int):
        digest = hashlib.blake2b(f"{SEED}:{self.epoch}:{index}".encode(), digest_size=8).digest()
        draw_seed = int.from_bytes(digest, "little") % (2**32)
        np.random.seed(draw_seed)
        random.seed(draw_seed)
        torch.manual_seed(draw_seed)
        return self.source_dataset[index]


class WorkerLogger:
    """Picklable source logger; preserve warnings/errors without per-draw DEBUG."""

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, message, *args, **kwargs):
        print(f"MARS source warning: {message}", file=sys.stderr, flush=True)

    def error(self, message, *args, **kwargs):
        print(f"MARS source error: {message}", file=sys.stderr, flush=True)

    def opt(self, *args, **kwargs):
        return self


def source_worker_init(worker_id: int) -> None:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    bridge_pinned_crop_argument()


def write_selected_csv(source: Path, destination: Path, ids: set[str]) -> int:
    """Expose only approved rows to the upstream dataframe parser."""
    count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as original, \
         destination.open("w", encoding="utf-8", newline="") as selected:
        reader = csv.DictReader(original)
        writer = csv.DictWriter(selected, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["id_loc_image"] in ids:
                writer.writerow(row)
                count += 1
    return count


def build_datasets(data_root: Path, fit: list[str], val: list[str], draws: int):
    process_plumes, read_images, read_plumes, DatasetPlumes, Loss, Model = source_modules()
    with tempfile.TemporaryDirectory(prefix="mars_fit_subset_") as temporary:
        subset = Path(temporary)
        n_images = write_selected_csv(data_root / "validated_images_all.csv",
                                      subset / "images.csv", set(fit) | set(val))
        n_plumes = write_selected_csv(data_root / "validated_images_plumes.csv",
                                      subset / "plumes.csv", set(fit))
        if n_images != len(fit) + len(val) or n_plumes == 0:
            raise ValueError("Frozen FIT/validation rows or FIT simulation plumes are missing")
        frame = read_images(str(subset / "images.csv"), path_prepend_data=str(data_root))
        plume_frame = read_plumes(str(subset / "plumes.csv"))
    fit_frame = frame[frame.id_loc_image.isin(fit)].copy()
    val_frame = frame[frame.id_loc_image.isin(val)].copy()
    if len(fit_frame) != len(fit) or len(val_frame) != len(val):
        raise ValueError("Pinned reader filtered one or more frozen IDs")
    plume_frame = process_plumes(plume_frame, fit_frame)
    if plume_frame.empty or set(plume_frame.id_loc_image) - set(fit):
        raise ValueError("Plume simulation source escaped FIT")
    common = dict(device=torch.device("cpu"), multipass=True, wind=True,
                  cloud_mask=True, cat_mbmp=True, norm_wind=True, bands_l8=True,
                  film_dict_mapping=None, cache=False)
    training = DatasetPlumes(mode="train", image_dataframe=fit_frame,
                             plume_dataframe=plume_frame, do_simulation=True,
                             n_samples_per_epoch_train=draws,
                             window_size_training=192, simulate_on_source_fraction=0,
                             **common)
    validation = DatasetPlumes(mode="test", image_dataframe=val_frame,
                               do_simulation=False, **common)
    controls_frame = fit_frame.sort_values("percentage_clear", ascending=False)
    fit_real = DatasetPlumes(mode="test", image_dataframe=controls_frame,
                             do_simulation=False, **common)
    if training.bands_out != validation.bands_out or len(training.bands_out) != 16:
        raise ValueError("Expected identical 16-channel MARS data contracts")
    return training, validation, fit_real, Loss, Model


def independent_loss(logits: np.ndarray, target: np.ndarray, ch4: np.ndarray) -> float:
    positive = np.maximum(logits, 0) - logits + np.log1p(np.exp(-np.abs(logits)))
    negative = np.maximum(logits, 0) + np.log1p(np.exp(-np.abs(logits)))
    enhancement = np.clip(ch4, 100, 2000) / 1000
    pixel = np.where(target > 0, 10 * positive * enhancement, negative)
    return float(pixel.sum(axis=(-2, -1)).mean())


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cap_cuda_memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": "cpu", "allocator_fraction": None}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION, device_index)
    total = torch.cuda.get_device_properties(device_index).total_memory
    return {"device": str(device), "allocator_fraction": CUDA_MEMORY_FRACTION,
            "total_device_bytes": total,
            "allocator_limit_approx_bytes": int(total * CUDA_MEMORY_FRACTION),
            "reason": "Avoid Windows WDDM GPU memory spill and retain desktop headroom"}


def is_cuda_allocation_error(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and
        ("out of memory" in message or "cudacachingallocator" in message))


def loss_controls(Loss, device: torch.device) -> dict:
    criterion = Loss(weight_by_ch4=True, pos_weight=10, class_head=False,
                     device=device, weight_by_noise=False)
    logits_np = np.array([[[0.2, -0.7], [1.1, -1.4]]], dtype=np.float32)
    target_np = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
    ch4_np = np.array([[[0, 50], [2500, 0]]], dtype=np.float32)
    logits = torch.tensor(logits_np, device=device, requires_grad=True)
    target = torch.tensor(target_np, device=device)
    ch4 = torch.tensor(ch4_np[:, None], device=device)
    value = criterion(target=target, pred=logits, ch4=ch4)
    expected = independent_loss(logits_np, target_np, ch4_np)
    if not math.isfinite(value.item()) or abs(value.item() - expected) > 2e-5:
        raise ValueError(f"Source/Numpy loss mismatch: {value.item()} vs {expected}")
    value.backward()
    grad = logits.grad.detach().cpu().numpy()
    if not np.isfinite(grad).all() or not (grad[0, 0, 1] < 0 and grad[0, 1, 0] < 0):
        raise ValueError("Positive label gradient is invalid")
    negative = criterion(target=torch.zeros_like(target), pred=logits.detach(),
                         ch4=torch.zeros_like(ch4)).item()
    if not math.isfinite(negative):
        raise ValueError("Negative label control failed")
    return {"source_loss": value.item(), "numpy_loss": expected,
            "absolute_error": abs(value.item() - expected), "negative_control_loss": negative}


def loader_batch(dataset, size: int):
    return next(iter(DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0)))


def real_label_controls(dataset) -> tuple[dict, dict, dict]:
    """Read one real positive and clear negative FIT raster with simulation off."""
    seed_all(SEED)
    frame = dataset.image_dataframe
    positive_index = np.flatnonzero(frame.isplume.to_numpy())
    negative_index = np.flatnonzero(~frame.isplume.to_numpy())
    if len(positive_index) == 0 or len(negative_index) == 0:
        raise ValueError("Both real FIT classes are required for label controls")
    positive = None
    for index in positive_index[:20]:
        candidate = dataset[int(index)]
        if int(candidate["y_target"].sum().item()) > 0:
            positive = candidate
            break
    if positive is None:
        raise ValueError("No nonempty real FIT positive label among 20 clear candidates")
    negative = dataset[int(negative_index[0])]
    pos_pixels = int(positive["y_target"].sum().item())
    neg_pixels = int(negative["y_target"].sum().item())
    if pos_pixels <= 0 or neg_pixels != 0:
        raise ValueError("Real FIT positive/negative label control failed")
    cloud_index = 15
    return positive, negative, {
        "positive_id": positive["id_loc_image"], "negative_id": negative["id_loc_image"],
        "positive_target_pixels": pos_pixels, "negative_target_pixels": neg_pixels,
        "positive_clear_fraction": float((positive["y_context_ls0_0"][cloud_index] == 0).float().mean()),
        "negative_clear_fraction": float((negative["y_context_ls0_0"][cloud_index] == 0).float().mean()),
        "simulation": False, "source_validmask_convention": "cloudmask == 0 for CH4 retrieval",
        "source_loss_valid_pixel_mask": False,
    }


def crop_real_positive(item: dict, window: int = 192) -> dict:
    """Keep a real plume in a native-size fixed batch for overfit control."""
    y = item["y_target"]
    coords = torch.nonzero(y > 0)
    if len(coords) == 0:
        raise ValueError("Real positive target has no plume pixels")
    center = coords[len(coords) // 2]
    height, width = y.shape
    row = min(max(int(center[0]) - window // 2, 0), height - window)
    col = min(max(int(center[1]) - window // 2, 0), width - window)
    cropped = item.copy()
    for key in ("y_context_ls0_0", "y_target", "ch4forweighting"):
        cropped[key] = item[key][..., row:row + window, col:col + window]
    if int(cropped["y_target"].sum()) <= 0:
        raise ValueError("Real plume was lost in 192-pixel overfit crop")
    return cropped


def real_label_counterfactual(dataset, positive_id: str) -> dict:
    """Prove true label geometry cannot change the 16 model input channels."""
    frame = dataset.image_dataframe
    matching = np.flatnonzero(frame.id_loc_image.to_numpy() == positive_id)
    if len(matching) != 1:
        raise ValueError("Real FIT positive is missing or duplicated in control dataset")
    index = int(matching[0])
    true_item = dataset[index]
    repeated = dataset[index]
    if not torch.equal(true_item["y_context_ls0_0"], repeated["y_context_ls0_0"]):
        raise ValueError("Repeated positive test-mode source input changed")
    counter_frame = frame.iloc[[index]].copy()
    counter_frame.loc[:, "isplume"] = False
    counter_frame.loc[:, "isplumeneg"] = True
    counter_frame.loc[:, "plume"] = None
    counterpart = type(dataset)(
        mode="test", image_dataframe=counter_frame, do_simulation=False,
        device=torch.device("cpu"), multipass=True, wind=True,
        cloud_mask=True, cat_mbmp=True, norm_wind=True, bands_l8=True,
        film_dict_mapping=None, cache=False,
    )[0]
    x_true = true_item["y_context_ls0_0"]
    x_false = counterpart["y_context_ls0_0"]
    max_abs = float((x_true - x_false).abs().max().item())
    target_true = int(true_item["y_target"].sum().item())
    target_false = int(counterpart["y_target"].sum().item())
    if target_true <= 0 or target_false != 0 or not torch.equal(x_true, x_false):
        raise ValueError(f"True-label input leakage: max input difference {max_abs}, targets {target_true}/{target_false}")
    return {"real_positive_id": positive_id, "true_target_pixels": target_true,
            "counterfactual_target_pixels": target_false,
            "input_max_abs_difference": max_abs, "input_bitwise_equal": True,
            "repeated_test_mode_read_bitwise_equal": True,
            "ch4_loss_weighting_may_use_true_label": True}


def source_model_parity(Model) -> dict:
    from marss2l.models import UnetOriginal  # type: ignore
    source = UnetOriginal(in_channels=16, out_channels=1, class_head=False,
                          batch_norm=True).eval()
    local = Model(16).eval()
    expected = 13_587_585
    source_count = sum(p.numel() for p in source.parameters())
    local_count = sum(p.numel() for p in local.parameters())
    if source_count != expected or local_count != expected:
        raise ValueError(f"U-Net parameter count drift: {source_count}, {local_count}")
    local.load_state_dict(source.state_dict(), strict=True)
    seed_all(SEED)
    fixture = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        delta = (source({"y_context_ls0_0": fixture}) - local(fixture)).abs().max().item()
    if delta > 1e-5:
        raise ValueError(f"Source/released U-Net output drift: {delta}")
    return {"source_parameters": source_count, "released_equivalent_parameters": local_count,
            "fixture_max_abs_error": delta, "state_dict_strict": True}


def tensors(batch: dict, device: torch.device):
    x = batch["y_context_ls0_0"].to(device)
    y = batch["y_target"].to(device)
    ch4 = batch["ch4forweighting"].to(device)
    if x.ndim != 4 or x.shape[1:] != (16, 192, 192):
        raise ValueError(f"Unexpected training input {tuple(x.shape)}")
    if y.shape != x.shape[:1] + x.shape[2:] or ch4.shape != x.shape[:1] + (1,) + x.shape[2:]:
        raise ValueError("Training label/CH4 shapes differ from source contract")
    if not all(torch.isfinite(value).all().item() for value in (x, y, ch4)):
        raise ValueError("Nonfinite input, label or CH4 weighting")
    if not torch.all((y == 0) | (y == 1)).item():
        raise ValueError("Nonbinary source target")
    return x, y, ch4


def benchmark(dataset, Loss, Model, device: torch.device, batch_size: int,
              precision: str = "fp32", workers: int = 0) -> dict:
    seed_all(SEED)
    model = Model(16).to(device).train()
    criterion = Loss(weight_by_ch4=True, pos_weight=10, class_head=False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    warmup_steps, measured_steps = 3, 8
    replay = SeededDraws(dataset, epoch=0, count=(warmup_steps + measured_steps) * batch_size)
    options = {"batch_size": batch_size, "shuffle": False, "num_workers": workers}
    if workers:
        options.update(prefetch_factor=2, persistent_workers=False,
                       worker_init_fn=source_worker_init)
    loader = iter(DataLoader(replay, **options))
    io_seconds, compute_seconds, kernel_seconds, losses = [], [], [], []
    for step in range(warmup_steps + measured_steps):
        io_start = time.perf_counter()
        batch = next(loader)
        io_elapsed = time.perf_counter() - io_start
        if device.type == "cuda":
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
        compute_start = time.perf_counter()
        x, y, ch4 = tensors(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=precision == "bf16"):
            out = model(x)
        loss = criterion(target=y, pred=out.float(), ch4=ch4)
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError("Nonfinite parameter gradient")
        optimizer.step()
        if device.type == "cuda":
            event_end.record()
            torch.cuda.synchronize(device)
        compute_elapsed = time.perf_counter() - compute_start
        if step >= warmup_steps:
            io_seconds.append(io_elapsed)
            compute_seconds.append(compute_elapsed)
            kernel_seconds.append(event_start.elapsed_time(event_end) / 1000 if device.type == "cuda" else 0)
            losses.append(float(loss.item()))
    elapsed = sum(io_seconds) + sum(compute_seconds)
    samples_per_second = measured_steps * batch_size / elapsed
    epoch_hours = 65536 / samples_per_second / 3600
    return {"batch_size": batch_size, "precision": precision, "workers": workers,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps, "measured_elapsed_seconds": elapsed,
            "mean_data_io_seconds_per_step": float(np.mean(io_seconds)),
            "mean_compute_wall_seconds_per_step": float(np.mean(compute_seconds)),
            "mean_cuda_event_seconds_per_step": float(np.mean(kernel_seconds)),
            "samples_per_second_including_loader": samples_per_second,
            "projected_65536_draw_epoch_hours": epoch_hours,
            "projected_170_epoch_training_only_hours": epoch_hours * 170,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "step_losses": losses}


def tiny_overfit_and_serialization(batch: dict, Loss, Model, device: torch.device,
                                   output: Path) -> dict:
    """Prove a real source example can learn and survive a checkpoint roundtrip."""
    seed_all(SEED)
    model = Model(16).to(device).train()
    criterion = Loss(weight_by_ch4=True, pos_weight=10, class_head=False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    x, y, ch4 = tensors(batch, device)
    def mask_metrics() -> dict:
        model.eval()
        with torch.no_grad():
            prediction = torch.sigmoid(model(x)) >= 0.5
        truth = y > 0
        tp = int((prediction & truth).sum().item())
        fp = int((prediction & ~truth).sum().item())
        fn = int((~prediction & truth).sum().item())
        model.train()
        return {"iou": tp / max(tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "true_positive_pixels": tp, "false_positive_pixels": fp,
                "false_negative_pixels": fn}
    initial = mask_metrics()
    losses = []
    for _ in range(64):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(target=y, pred=model(x), ch4=ch4)
        if not torch.isfinite(loss).item():
            raise ValueError("Nonfinite tiny-batch overfit loss")
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
        if len(losses) >= 32 and len(losses) % 8 == 0:
            latest = mask_metrics()
            if latest["iou"] > max(0.01, initial["iou"] + 0.01):
                break
    final = mask_metrics()
    if min(losses[-8:]) >= losses[0]:
        raise ValueError(f"Tiny batch did not improve: {losses}")
    if final["iou"] <= max(0.01, initial["iou"] + 0.01):
        raise ValueError(f"Real plume IoU did not improve: initial={initial}, final={final}")
    path = output / "preflight_roundtrip.pt"
    atomic_checkpoint(path, {"model_state_dict": model.state_dict()})
    copy = Model(16).to(device)
    copy.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state_dict"])
    model.eval()
    copy.eval()
    with torch.no_grad():
        delta = (model(x) - copy(x)).abs().max().item()
    if delta > 1e-6:
        raise ValueError(f"Checkpoint roundtrip changed output by {delta}")
    path.unlink()
    return {"steps": len(losses), "initial": initial, "final": final,
            "real_positive_target_pixels": int(y.sum().item()),
            "initial_loss": losses[0], "final_loss": losses[-1],
            "roundtrip_max_abs_error": delta}


def preflight(args, fit: list[str], val: list[str], audit: dict, fingerprint: dict) -> dict:
    receipt = {"status": "running", "phase": "asset_audit", "fingerprint": fingerprint,
               "audit": audit, "source_revision": SOURCE_REVISION,
               "metadata_revision": METADATA_REVISION}
    receipt_path = args.output / "preflight.json"

    def phase(name: str, **updates) -> None:
        receipt.update(updates)
        receipt["phase"] = name
        atomic_json(receipt_path, receipt)

    phase("asset_audit")
    try:
        if audit["missing_rasters"]:
            raise FileNotFoundError(f"{audit['missing_rasters']} FIT/validation rasters absent; examples {audit['missing_raster_examples']}")
        phase("paired_raster_qa")
        paired_qa = paired_raster_qa(args.data_root, fit, val)
        phase("source_dataset", paired_raster_qa=paired_qa)
        training, validation, fit_real, Loss, Model = build_datasets(args.data_root, fit, val, args.draws)
        device = torch.device(args.device)
        memory_cap = cap_cuda_memory(device)
        controls = loss_controls(Loss, device)
        parity = source_model_parity(Model)
        phase("real_fit_controls", loss_controls=controls, source_model_parity=parity,
              cuda_memory_cap=memory_cap)
        positive, negative, real_controls = real_label_controls(fit_real)
        counterfactual = real_label_counterfactual(fit_real, real_controls["positive_id"])
        positive = crop_real_positive(positive)
        train_batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
                       for key, value in positive.items()}
        x, y, ch4 = tensors(train_batch, device)
        if int(y.sum().item()) <= 0 or not torch.isfinite(negative["y_context_ls0_0"]).all().item():
            raise ValueError("Real FIT label controls failed")
        val_batch = loader_batch(validation, 1)
        vx = val_batch["y_context_ls0_0"]
        if vx.shape[1:] != (16, 200, 200) or not torch.isfinite(vx).all().item():
            raise ValueError("Validation sample is not finite 16x200x200")
        phase("tiny_real_plume_overfit", real_fit_label_controls=real_controls,
              real_label_counterfactual=counterfactual,
              sample_target_pixels=int(y.sum().item()))
        overfit = tiny_overfit_and_serialization(train_batch, Loss, Model, device, args.output)
        phase("batch_sweep", overfit_and_serialization=overfit)
        measurements = {}
        for size in args.benchmark_batches:
            try:
                measurements[str(size)] = benchmark(training, Loss, Model, device, size,
                                                    precision=args.precision, workers=args.num_workers)
            except RuntimeError as error:
                if not is_cuda_allocation_error(error):
                    raise
                measurements[str(size)] = {"batch_size": size, "status": "out_of_memory", "error": str(error)}
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            phase("batch_sweep", batch_sweep=measurements, last_benchmark_batch=size)
        if measurements[str(args.batch_size)].get("status") == "out_of_memory":
            raise ValueError(f"Native batch {args.batch_size} did not fit GPU memory")
        phase("complete", benchmark=measurements[str(args.batch_size)],
              benchmark_caveat="Three warmup and eight measured training steps per batch; projections exclude validation and checkpoint time",
              batchnorm_adaptation=f"native batch {args.batch_size} instead of released 96",
              training_precision=args.precision, training_num_workers=args.num_workers,
              indexed_draw_adaptation="Per-(epoch,index) deterministic seeds preserve each sampled example across Windows worker counts but change the original global RNG sequence",
              source_crop_api_bridge="Pinned loaders.py passes (H,W) to sampling.py's scalar window_size_data; bridge converts square (200,200) to 200 for 192 crops",
              simulation="pinned DatasetPlumes enabled; source plume rows restricted to FIT")
        receipt["status"] = "passed"
        atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        atomic_json(receipt_path, receipt)
        raise


def _step_model(model, criterion, optimizer, batch: dict, device: torch.device,
                precision: str) -> tuple[float, torch.Tensor]:
    x, y, ch4 = tensors(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=precision == "bf16"):
        logits = model(x)
    loss = criterion(target=y, pred=logits.float(), ch4=ch4)
    if not torch.isfinite(loss).item():
        raise ValueError(f"Nonfinite {precision} loss")
    loss.backward()
    return float(loss.item()), logits.detach().float()


def precision_parity(pool: list[dict], Loss, Model, device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is not supported by this GPU/runtime")
    batch = {key: torch.stack([item[key] for item in pool[:8]])
             for key in ("y_context_ls0_0", "y_target", "ch4forweighting")}
    criterion = Loss(weight_by_ch4=True, pos_weight=10, class_head=False, device=device)
    outcomes = {}
    for precision in ("fp32", "bf16"):
        seed_all(SEED)
        model = Model(16).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        loss, logits = _step_model(model, criterion, optimizer, batch, device, precision)
        gradients = [parameter.grad.detach().float().cpu().clone()
                     for parameter in model.parameters() if parameter.grad is not None]
        outcomes[precision] = (loss, logits.cpu(), gradients)
        del model, optimizer
        torch.cuda.empty_cache()
    fp_loss, fp_logits, fp_grad = outcomes["fp32"]
    bf_loss, bf_logits, bf_grad = outcomes["bf16"]
    dot = sum(float((a * b).sum()) for a, b in zip(fp_grad, bf_grad))
    norm_fp = math.sqrt(sum(float((a * a).sum()) for a in fp_grad))
    norm_bf = math.sqrt(sum(float((b * b).sum()) for b in bf_grad))
    metrics = {"fp32_loss": fp_loss, "bf16_loss": bf_loss,
               "relative_loss_error": abs(fp_loss - bf_loss) / max(abs(fp_loss), 1e-6),
               "logit_mean_abs_error": float((fp_logits - bf_logits).abs().mean()),
               "logit_max_abs_error": float((fp_logits - bf_logits).abs().max()),
               "gradient_cosine": dot / max(norm_fp * norm_bf, 1e-12),
               "gradient_norm_relative_error": abs(norm_fp - norm_bf) / max(norm_fp, 1e-12),
               "fixture_batch": 8, "loss_and_optimizer_dtype": "float32"}
    if metrics["relative_loss_error"] > 0.10 or metrics["gradient_cosine"] < 0.8:
        raise ValueError(f"BF16 parity control failed: {metrics}")
    return metrics


def replay_batch(pool: list[dict], start: int, size: int) -> dict:
    selected = [pool[(start + offset) % len(pool)] for offset in range(size)]
    return {key: torch.stack([item[key] for item in selected])
            for key in ("y_context_ls0_0", "y_target", "ch4forweighting")}


def replay_gpu_benchmark(pool: list[dict], Loss, Model, device: torch.device,
                         size: int, precision: str) -> dict:
    seed_all(SEED)
    model = Model(16).to(device).train()
    criterion = Loss(weight_by_ch4=True, pos_weight=10, class_head=False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    torch.cuda.reset_peak_memory_stats(device)
    warmup, measured = 3, 8
    io_seconds, compute_seconds, kernel_seconds, losses = [], [], [], []
    try:
        for step in range(warmup + measured):
            io_start = time.perf_counter()
            batch = replay_batch(pool, step * size, size)
            io_elapsed = time.perf_counter() - io_start
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            compute_start = time.perf_counter()
            loss, _ = _step_model(model, criterion, optimizer, batch, device, precision)
            optimizer.step()
            event_end.record()
            torch.cuda.synchronize(device)
            compute_elapsed = time.perf_counter() - compute_start
            if step >= warmup:
                io_seconds.append(io_elapsed)
                compute_seconds.append(compute_elapsed)
                kernel_seconds.append(event_start.elapsed_time(event_end) / 1000)
                losses.append(loss)
        elapsed = sum(io_seconds) + sum(compute_seconds)
        speed = measured * size / elapsed
        return {"status": "passed", "batch_size": size, "precision": precision,
                "warmup_steps": warmup, "measured_steps": measured,
                "samples_per_second_replay": speed,
                "mean_replay_batch_seconds": float(np.mean(io_seconds)),
                "mean_compute_wall_seconds": float(np.mean(compute_seconds)),
                "mean_cuda_event_seconds": float(np.mean(kernel_seconds)),
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "projected_65536_draw_epoch_training_only_hours": 65536 / speed / 3600,
                "measured_losses": losses}
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def replay_loader_audit(source_dataset, count: int, workers: int) -> dict:
    frozen = SeededDraws(source_dataset, epoch=0, count=count)
    options = {"batch_size": 8, "shuffle": False, "num_workers": workers}
    if workers:
        options.update(prefetch_factor=2, persistent_workers=False,
                       worker_init_fn=source_worker_init)
    started = time.perf_counter()
    digest = hashlib.sha256()
    first_batch_seconds = None
    rows = 0
    for batch in DataLoader(frozen, **options):
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - started
        for sample_id in batch["id_loc_image"]:
            digest.update(sample_id.encode())
        for key in ("y_context_ls0_0", "y_target", "ch4forweighting"):
            digest.update(batch[key].contiguous().numpy().tobytes())
        rows += len(batch["id_loc_image"])
    elapsed = time.perf_counter() - started
    return {"workers": workers, "rows": rows, "wall_seconds_including_startup": elapsed,
            "first_batch_seconds": first_batch_seconds,
            "samples_per_second": rows / elapsed, "content_sha256": digest.hexdigest()}


def performance(args, fit: list[str], val: list[str], audit: dict, fingerprint: dict) -> dict:
    proof_path = args.output / "preflight.json"
    if not proof_path.is_file():
        raise RuntimeError("A passed source/GPU preflight is required before performance audit")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    keys = ("manifest_sha256", "metadata_sha256", "plumes_sha256")
    old_source = proof.get("fingerprint", {}).get("source_file_sha256", {})
    source_overlap_matches = bool(old_source) and all(
        fingerprint["source_file_sha256"].get(name) == digest
        for name, digest in old_source.items())
    if proof.get("status") != "passed" or not source_overlap_matches or any(
        proof["fingerprint"].get(key) != fingerprint[key] for key in keys
    ):
        raise RuntimeError("Performance audit data/source fingerprint differs from passed preflight")
    path = args.output / "performance.json"
    receipt = {"status": "running", "phase": "source_replay", "fingerprint": fingerprint,
               "performance_samples": args.performance_samples,
               "precision_adaptation": "BF16 autocast only on U-Net forward; source preprocessing, weighted loss and Adam remain FP32"}
    atomic_json(path, receipt)
    try:
        training, _, _, Loss, Model = build_datasets(args.data_root, fit, val, args.performance_samples)
        device = torch.device(args.device)
        if device.type != "cuda":
            raise RuntimeError("Performance audit requires CUDA")
        receipt["cuda_memory_cap"] = cap_cuda_memory(device)
        atomic_json(path, receipt)
        seeded = SeededDraws(training, epoch=0, count=args.performance_samples)
        pool = [seeded[index] for index in range(args.performance_samples)]
        receipt["source_replay_ids_sha256"] = hashlib.sha256(
            "\n".join(item["id_loc_image"] for item in pool).encode()).hexdigest()
        receipt["phase"] = "bf16_parity"
        atomic_json(path, receipt)
        receipt["bf16_parity"] = precision_parity(pool, Loss, Model, device)
        receipt["phase"] = "gpu_sweep"
        atomic_json(path, receipt)
        receipt["gpu_sweep"] = {}
        for precision in ("fp32", "bf16"):
            for size in args.performance_batches:
                label = f"{precision}_{size}"
                if precision == "fp32" and size > 16:
                    receipt["gpu_sweep"][label] = {
                        "status": "skipped_measured_spill_risk",
                        "reason": "Prior uncapped FP32 batch32 reached 10.19 GB and throughput fell to 10.7 samples/s; batch48 reached 15.18 GB WDDM spill"}
                    receipt["last_gpu_case"] = label
                    atomic_json(path, receipt)
                    continue
                try:
                    receipt["gpu_sweep"][label] = replay_gpu_benchmark(pool, Loss, Model, device, size, precision)
                except RuntimeError as error:
                    if not is_cuda_allocation_error(error):
                        raise
                    receipt["gpu_sweep"][label] = {"status": "out_of_memory", "error": str(error)}
                    gc.collect()
                    torch.cuda.empty_cache()
                receipt["last_gpu_case"] = label
                atomic_json(path, receipt)
        receipt["phase"] = "loader_workers"
        atomic_json(path, receipt)
        loader0 = replay_loader_audit(training, args.performance_samples, 0)
        receipt["loader_0"] = loader0
        atomic_json(path, receipt)
        try:
            loader2 = replay_loader_audit(training, args.performance_samples, 2)
            receipt["loader_2"] = loader2
            receipt["loader_content_identical"] = loader0["content_sha256"] == loader2["content_sha256"]
        except Exception as error:
            receipt["loader_2"] = {"status": "failed", "error": repr(error)}
            receipt["loader_content_identical"] = False
        receipt["status"] = "passed"
        receipt["phase"] = "complete"
        atomic_json(path, receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        atomic_json(path, receipt)
        raise


def worker_audit(args, fit: list[str], val: list[str], fingerprint: dict) -> dict:
    """Compare exact 128 source draws across Windows 0/2-worker loaders only."""
    path = args.output / "worker_audit.json"
    receipt = {"status": "running", "phase": "source_dataset", "fingerprint": fingerprint}
    atomic_json(path, receipt)
    try:
        training, _, _, _, _ = build_datasets(args.data_root, fit, val, args.performance_samples)
        receipt["phase"] = "loader_0"
        atomic_json(path, receipt)
        loader0 = replay_loader_audit(training, args.performance_samples, 0)
        receipt["loader_0"] = loader0
        receipt["phase"] = "loader_2"
        atomic_json(path, receipt)
        loader2 = replay_loader_audit(training, args.performance_samples, 2)
        receipt["loader_2"] = loader2
        receipt["content_identical"] = loader0["content_sha256"] == loader2["content_sha256"]
        if not receipt["content_identical"]:
            raise ValueError("0/2 worker source draws changed content")
        receipt["status"] = "passed"
        receipt["phase"] = "complete"
        atomic_json(path, receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        atomic_json(path, receipt)
        raise


def source_threshold_cutoff(probabilities: torch.Tensor,
                            threshold_pixels: int = 100, tol: float = 1e-4) -> float:
    """Pinned trainer.threshold_cutoff binary search used for validation AP."""
    min_value = torch.min(probabilities).item()
    max_value = torch.max(probabilities).item()
    threshold = (min_value + max_value) / 2
    while max_value - min_value > tol:
        count = torch.sum(probabilities >= threshold).item()
        if count >= threshold_pixels:
            min_value = threshold
        else:
            max_value = threshold
        threshold = (min_value + max_value) / 2
    return float(threshold)


def validation_ap(model, dataset, device: torch.device, batch_size: int) -> dict:
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            x = batch["y_context_ls0_0"].to(device)
            probabilities = torch.sigmoid(model(x))
            scores.extend(source_threshold_cutoff(probabilities[index])
                          for index in range(len(probabilities)))
            labels.extend(batch["isplume"].cpu().int().tolist())
    return {"average_precision": float(average_precision_score(labels, scores)),
            "rows": len(labels), "positive_rows": sum(labels)}


def train(args, fit: list[str], val: list[str], audit: dict, fingerprint: dict) -> None:
    report_path = args.output / "preflight.json"
    if not report_path.is_file():
        raise RuntimeError("Run --mode preflight successfully before training")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("fingerprint") != fingerprint:
        raise RuntimeError("Preflight fingerprint differs from this run")
    seed_all(SEED)
    training, validation, _, Loss, Model = build_datasets(args.data_root, fit, val, args.draws)
    real_positive_ids = set(training.image_dataframe.loc[training.image_dataframe.isplume, "id_loc_image"])
    device = torch.device(args.device)
    memory_cap = cap_cuda_memory(device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 training requires a CUDA GPU with BF16 support")
    model = Model(16).to(device)
    criterion = Loss(weight_by_ch4=True, pos_weight=10, class_head=False, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    last_path = args.output / "last_epoch.pt"
    start_epoch, best_ap, best_epoch = 0, -1.0, -1
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["fingerprint"] != fingerprint:
            raise RuntimeError("Checkpoint fingerprint differs from frozen run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        start_epoch = checkpoint["epoch"] + 1
        best_ap, best_epoch = checkpoint["best_ap"], checkpoint["best_epoch"]
    atomic_json(args.output / "run_status.json", {"state": "running", "next_epoch": start_epoch,
                                                  "cuda_memory_cap": memory_cap,
                                                  "fingerprint": fingerprint})
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            losses = []
            started = time.perf_counter()
            counts = {"draws": 0, "real_positive_draws": 0, "simulated_positive_draws": 0,
                      "negative_draws": 0, "nonclear_pixels": 0, "total_pixels": 0,
                      "fully_nonclear_draws": 0}
            next_heartbeat = 100
            frozen_draws = SeededDraws(training, epoch=epoch, count=args.draws)
            options = {"batch_size": args.batch_size, "shuffle": False,
                       "num_workers": args.num_workers}
            if args.num_workers:
                options.update(prefetch_factor=2, persistent_workers=False,
                               worker_init_fn=source_worker_init)
            loader = DataLoader(frozen_draws, **options)
            for batch in loader:
                x, y, ch4 = tensors(batch, device)
                labels = batch["isplume"].int().tolist()
                for sample_id, is_positive in zip(batch["id_loc_image"], labels):
                    if is_positive:
                        key = "real_positive_draws" if sample_id in real_positive_ids else "simulated_positive_draws"
                    else:
                        key = "negative_draws"
                    counts[key] += 1
                counts["draws"] += len(labels)
                cloud = batch["y_context_ls0_0"][:, 15] > 0
                counts["nonclear_pixels"] += int(cloud.sum().item())
                counts["total_pixels"] += int(cloud.numel())
                counts["fully_nonclear_draws"] += int(cloud.flatten(1).all(dim=1).sum().item())
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=args.precision == "bf16"):
                    logits = model(x)
                loss = criterion(target=y, pred=logits.float(), ch4=ch4)
                if not torch.isfinite(loss).item():
                    raise ValueError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                if counts["draws"] >= next_heartbeat:
                    elapsed = time.perf_counter() - started
                    heartbeat = {"state": "running", "epoch": epoch, "draws_completed": counts["draws"],
                                 "draws_budget": args.draws, "elapsed_seconds": elapsed,
                                 "samples_per_second": counts["draws"] / max(elapsed, 1e-6),
                                 "recent_train_loss": losses[-1], "draw_counts": counts,
                                 "best_ap": best_ap, "best_epoch": best_epoch,
                                 "fingerprint": fingerprint}
                    atomic_json(args.output / "run_status.json", heartbeat)
                    print(json.dumps({"heartbeat": heartbeat}, sort_keys=True), flush=True)
                    next_heartbeat += 100
            metrics = validation_ap(model, validation, device, args.validation_batch_size)
            if metrics["average_precision"] >= best_ap:
                best_ap, best_epoch = metrics["average_precision"], epoch
                best_payload = {
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "fingerprint": fingerprint, "validation": metrics}
                immutable = args.output / ("epoch_000.pt" if epoch == 0 else f"best_epoch_{epoch:03d}.pt")
                if immutable.exists():
                    # An interrupted epoch can replay from the previous durable
                    # last_epoch.pt. Preserve the first published snapshot.
                    for retry in range(1, 1000):
                        candidate = immutable.with_name(f"{immutable.stem}_retry{retry}{immutable.suffix}")
                        if not candidate.exists():
                            immutable = candidate
                            break
                    else:
                        raise FileExistsError("Too many immutable checkpoint retries")
                atomic_checkpoint(immutable, best_payload)
                atomic_checkpoint(args.output / "best_epoch.pt", best_payload)
                atomic_json(args.output / "checkpoint_event.json", {
                    "status": "ready", "kind": "epoch0" if epoch == 0 else "best_update",
                    "epoch": epoch, "checkpoint": str(immutable),
                    "checkpoint_sha256": sha256(immutable),
                    "validation_ap": best_ap, "fingerprint": fingerprint,
                    "export_contract": "Output is a single 200x200 mask; external exporter owns independent raw-input reconstruction and holdout labels"})
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "torch_rng_state": torch.get_rng_state(),
                          "numpy_rng_state": np.random.get_state(),
                          "python_rng_state": random.getstate(),
                          "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                          "best_ap": best_ap, "best_epoch": best_epoch,
                          "fingerprint": fingerprint}
            epoch_report = {"epoch": epoch, "draws": args.draws, "native_batch_size": args.batch_size,
                            "precision": args.precision, "num_workers": args.num_workers,
                            "train_loss": float(np.mean(losses)), "validation": metrics,
                            "draw_counts": counts,
                            "seconds": time.perf_counter() - started,
                            "best_ap": best_ap, "best_epoch": best_epoch}
            atomic_json(args.output / f"epoch_{epoch:03d}.json", epoch_report)
            atomic_checkpoint(last_path, checkpoint)
            atomic_json(args.output / "run_status.json", {"state": "running", "next_epoch": epoch + 1,
                                                          "best_ap": best_ap, "best_epoch": best_epoch,
                                                          "fingerprint": fingerprint})
            print(json.dumps(epoch_report, sort_keys=True), flush=True)
            if epoch - best_epoch >= args.patience and epoch > args.patience:
                break
        atomic_json(args.output / "run_status.json", {"state": "completed", "best_ap": best_ap,
                                                      "best_epoch": best_epoch, "fingerprint": fingerprint})
    except BaseException as error:
        atomic_json(args.output / "run_status.json", {"state": "failed", "error": repr(error),
                                                      "fingerprint": fingerprint})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "preflight", "performance", "worker-audit", "train"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--num-workers", type=int, choices=(0, 2), default=0)
    parser.add_argument("--runtime-lock", type=Path,
                        default=ROOT / "tool-diagnostics/mars-baseline-20261030/requirements-minimal-locked.txt")
    parser.add_argument("--benchmark-batches", default="2,4,8,16")
    parser.add_argument("--performance-batches", default="8,16,32,48,64")
    parser.add_argument("--performance-samples", type=int, default=128)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--epochs", type=int, default=170)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    try:
        args.benchmark_batches = sorted({int(part.strip()) for part in args.benchmark_batches.split(",")}
                                        | {args.batch_size})
        args.performance_batches = sorted({int(part.strip()) for part in args.performance_batches.split(",")})
    except ValueError:
        parser.error("--benchmark-batches must be comma-separated positive integers")
    if min(args.batch_size, args.validation_batch_size, args.draws, args.epochs, args.patience) < 1:
        parser.error("batch, draws, epochs and patience must be positive")
    if not args.benchmark_batches or min(args.benchmark_batches) < 1:
        parser.error("benchmark batches must be positive")
    if not args.performance_batches or min(args.performance_batches) < 1 or args.performance_samples < 8:
        parser.error("performance batches must be positive and performance samples >= 8")
    fit, val = frozen_manifest(args.manifest)
    audit = raw_audit(args.data_root, fit, val)
    if not args.runtime_lock.is_file():
        raise FileNotFoundError(f"Runtime lock is missing: {args.runtime_lock}")
    fingerprint = {"manifest_sha256": sha256(args.manifest),
                   "metadata_sha256": audit["metadata_sha256"],
                   "plumes_sha256": audit["plumes_sha256"],
                   "draws": args.draws, "batch_size": args.batch_size,
                   "validation_batch_size": args.validation_batch_size,
                   "source_revision": SOURCE_REVISION,
                   "source_file_sha256": pinned_source_hashes(),
                   "runner_sha256": sha256(Path(__file__)),
                   "released_unet_script_sha256": sha256(ROOT / "tools/evaluate_released_marss2l.py"),
                   "runtime_lock_sha256": sha256(args.runtime_lock),
                   "torch_version": torch.__version__, "numpy_version": np.__version__,
                   "precision": args.precision, "num_workers": args.num_workers,
                   "cuda_allocator_fraction": CUDA_MEMORY_FRACTION}
    if args.mode == "audit":
        if not audit["missing_rasters"]:
            audit["paired_raster_qa"] = paired_raster_qa(args.data_root, fit, val)
        print(json.dumps(audit, indent=2))
    elif args.mode == "preflight":
        print(json.dumps(preflight(args, fit, val, audit, fingerprint), indent=2))
    elif args.mode == "performance":
        print(json.dumps(performance(args, fit, val, audit, fingerprint), indent=2))
    elif args.mode == "worker-audit":
        print(json.dumps(worker_audit(args, fit, val, fingerprint), indent=2))
    else:
        if audit["missing_rasters"]:
            raise FileNotFoundError(f"{audit['missing_rasters']} raster assets missing")
        train(args, fit, val, audit, fingerprint)


if __name__ == "__main__":
    main()
