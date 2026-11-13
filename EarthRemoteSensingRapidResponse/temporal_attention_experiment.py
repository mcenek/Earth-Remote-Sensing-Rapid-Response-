"""Bounded, two-stage training for temporal attention candidates.

The runner consumes already prepared batch dictionaries and never creates a test
partition or reports test metrics.  It restores the best validation checkpoint
for each stage and leaves the caller's model in ``eval`` mode with that selected
state installed.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import torch
from torch.nn import functional as F

from temporal_attention_training import annotation_support, dense_objective


_DENSE_EXCLUDED = ("scene_encoder.", "scene_head.")
_SCENE_PREFIXES = _DENSE_EXCLUDED


def _materialize_batches(batches: Iterable[Mapping[str, Any]], name: str) -> list[dict[str, Any]]:
    result = [dict(batch) for batch in batches]
    if not result:
        raise ValueError(f"{name} batches must be nonempty")
    ids: list[str] = []
    partition: str | None = None
    for batch in result:
        required = {"split", "sample_ids", "inputs", "plume", "pixel_truth_available", "scene_labels"}
        missing = required - set(batch)
        if missing:
            raise ValueError(f"batch missing required fields: {sorted(missing)}")
        split = batch["split"]
        if split not in ("train", "validation"):
            raise ValueError(f"unsupported split {split!r}; test data is not accepted")
        if partition is None:
            partition = split
        elif split != partition:
            raise ValueError(f"{name} batches contain mixed split labels")
        batch_ids = batch["sample_ids"]
        if not isinstance(batch_ids, (list, tuple)) or not batch_ids or any(not isinstance(x, str) or not x for x in batch_ids):
            raise ValueError("sample_ids must be a nonempty list of nonempty strings")
        b = len(batch_ids)
        plume, available, labels = batch["plume"], batch["pixel_truth_available"], batch["scene_labels"]
        if not isinstance(batch["inputs"], Mapping):
            raise ValueError("inputs must be a mapping of model keyword arguments")
        if not isinstance(plume, torch.Tensor) or plume.ndim != 4 or plume.shape[0] != b or plume.shape[1] != 1:
            raise ValueError("plume must have shape Bx1xHxW matching sample_ids")
        if not isinstance(available, torch.Tensor) or available.shape != (b,) or available.dtype != torch.bool:
            raise ValueError("pixel_truth_available must be boolean B")
        if not isinstance(labels, torch.Tensor) or labels.shape != (b,) or not labels.is_floating_point():
            raise ValueError("scene_labels must be floating point B")
        if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("scene_labels must be finite binary labels")
        ids.extend(batch_ids)
    if len(ids) != len(set(ids)):
        raise ValueError(f"sample_ids must be unique within {name}")
    return result


def _audit(train: Iterable[Mapping[str, Any]], validation: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tr, va = _materialize_batches(train, "train") , _materialize_batches(validation, "validation")
    if any(b["split"] != "train" for b in tr):
        raise ValueError("train_batches must have split='train'")
    if any(b["split"] != "validation" for b in va):
        raise ValueError("validation_batches must have split='validation'")
    train_ids = {x for b in tr for x in b["sample_ids"]}
    if train_ids & {x for b in va for x in b["sample_ids"]}:
        raise ValueError("train and validation sample_ids must be disjoint")
    return tr, va


def _dense_stats(logits: torch.Tensor, plume: torch.Tensor, available: torch.Tensor, support: torch.Tensor) -> dict[str, Any]:
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != plume.shape:
        raise ValueError("model mask_logits and plume must have identical Bx1xHxW shape")
    if not torch.isfinite(logits).all():
        raise ValueError("model mask_logits must be finite")
    if support.shape != logits.shape or support.dtype is not torch.bool:
        raise ValueError("model support must be boolean and match mask_logits")
    label_support = annotation_support(support, available)
    safe = torch.where(label_support, plume, torch.zeros_like(plume)).to(logits.dtype)
    active = plume[label_support]
    if active.numel() and (not torch.isfinite(active).all() or not ((active == 0) | (active == 1)).all()):
        raise ValueError("supported plume targets must be finite binary labels")
    counts = label_support.flatten(1).sum(1)
    supervised = counts > 0
    bce = (F.binary_cross_entropy_with_logits(logits, safe, reduction="none") * label_support).flatten(1).sum(1) / counts.clamp_min(1)
    prob = logits.sigmoid() * label_support
    target = safe * label_support
    positive = target.flatten(1).sum(1) > 0
    dice = 1 - (2 * (prob * target).flatten(1).sum(1) + 1) / ((prob + target).flatten(1).sum(1) + 1)
    return {"bce_values": bce[supervised], "dice_values": dice[positive], "supervised": int(supervised.sum()), "positive": int(positive.sum())}


def _forward(model: torch.nn.Module, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    outputs = model(**dict(batch["inputs"]))
    if not isinstance(outputs, Mapping) or "mask_logits" not in outputs or "scene_logits" not in outputs or "support" not in outputs:
        raise ValueError("model must return mask_logits, scene_logits, and support")
    if outputs["scene_logits"].shape != batch["scene_labels"].shape or not torch.isfinite(outputs["scene_logits"]).all():
        raise ValueError("model scene_logits must be finite and shaped B")
    return outputs


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _metric(stats: list[dict[str, Any]]) -> tuple[float, int, int]:
    bce = torch.cat([s["bce_values"].detach().cpu() for s in stats if s["bce_values"].numel()], 0) if any(s["bce_values"].numel() for s in stats) else torch.empty(0)
    dice = torch.cat([s["dice_values"].detach().cpu() for s in stats if s["dice_values"].numel()], 0) if any(s["dice_values"].numel() for s in stats) else torch.empty(0)
    if not bce.numel():
        return float("nan"), 0, int(dice.numel())
    value = 0.7 * bce.mean() + 0.3 * (dice.mean() if dice.numel() else 0.0)
    return float(value), int(bce.numel()), int(dice.numel())


def fit_two_stage(model: torch.nn.Module, train_batches: Iterable[Mapping[str, Any]], validation_batches: Iterable[Mapping[str, Any]], *, dense_epochs: int = 1, scene_epochs: int = 1, lr: float = 1e-3) -> dict[str, Any]:
    """Fit dense parameters, restore best validation dense state, then scene parameters."""
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (dense_epochs, scene_epochs)):
        raise ValueError("dense_epochs and scene_epochs must be positive integers")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a finite positive number")
    train, validation = _audit(train_batches, validation_batches)
    original_requires = {p: p.requires_grad for p in model.parameters()}
    dense_params = [p for n, p in model.named_parameters() if not n.startswith(_DENSE_EXCLUDED)]
    scene_params = [p for n, p in model.named_parameters() if n.startswith(_SCENE_PREFIXES)]
    if not dense_params or not scene_params:
        raise ValueError("model must expose both dense and scene parameters")
    model.eval()
    with torch.no_grad():
        train_dense_available = train_scene_available = validation_dense_available = validation_scene_available = 0
        for collection, is_train in ((train, True), (validation, False)):
            for b in collection:
                out = _forward(model, b)
                dense_count = _dense_stats(out["mask_logits"], b["plume"], b["pixel_truth_available"], out["support"])["supervised"]
                scene_count = int(out["support"].flatten(1).any(1).sum())
                if is_train:
                    train_dense_available += dense_count; train_scene_available += scene_count
                else:
                    validation_dense_available += dense_count; validation_scene_available += scene_count
    if not train_dense_available or not validation_dense_available:
        raise ValueError("no eligible train or validation dense observations")
    if not train_scene_available or not validation_scene_available:
        raise ValueError("no eligible train or validation scene observations")
    history: dict[str, Any] = {"dense": [], "scene": [], "counters": {"dense_train_scenes": 0, "dense_validation_scenes": 0, "scene_train_scenes": 0, "scene_validation_scenes": 0}}
    try:
        for p in model.parameters(): p.requires_grad = False
        for p in dense_params: p.requires_grad = True
        opt = torch.optim.Adam(dense_params, lr=lr)
        best_loss, best_state, best_epoch = float("inf"), None, 0
        for epoch in range(1, dense_epochs + 1):
            model.train()
            model.scene_encoder.eval()
            model.scene_head.eval()
            train_stats = []
            for b in train:
                opt.zero_grad(set_to_none=True); out = _forward(model, b); s = _dense_stats(out["mask_logits"], b["plume"], b["pixel_truth_available"], out["support"])
                train_stats.append({k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in s.items()})
                if s["supervised"]:
                    objective = dense_objective(
                        out["mask_logits"], b["plume"],
                        annotation_support(out["support"], b["pixel_truth_available"]))["loss"]
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(dense_params, 1.0, error_if_nonfinite=True)
                    opt.step()
            model.eval(); val_stats = []
            with torch.no_grad():
                for b in validation:
                    out = _forward(model, b)
                    val_stats.append(_dense_stats(out["mask_logits"], b["plume"], b["pixel_truth_available"], out["support"]))
            loss, n, pos = _metric(val_stats)
            if not n: raise ValueError("no eligible validation dense observations")
            history["dense"].append({"epoch": epoch, "train_loss": _metric(train_stats)[0], "validation_loss": loss})
            if loss < best_loss: best_loss, best_state, best_epoch = loss, _state_cpu(model), epoch
        if best_state is None: raise ValueError("no eligible training dense observations")
        model.load_state_dict(best_state); history["best_dense_epoch"] = best_epoch
        for p in model.parameters(): p.requires_grad = False
        for p in scene_params: p.requires_grad = True
        opt = torch.optim.Adam(scene_params, lr=lr); best_loss, best_state, best_epoch = float("inf"), None, 0
        for epoch in range(1, scene_epochs + 1):
            model.eval(); model.scene_encoder.train(); model.scene_head.train(); train_losses = []
            for b in train:
                opt.zero_grad(set_to_none=True); out = _forward(model, b); eligible = out["support"].flatten(1).any(1); labels = b["scene_labels"].to(out["scene_logits"].dtype)
                if eligible.any():
                    loss = F.binary_cross_entropy_with_logits(out["scene_logits"][eligible], labels[eligible])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(scene_params, 1.0, error_if_nonfinite=True)
                    opt.step()
                    train_losses.extend(F.binary_cross_entropy_with_logits(out["scene_logits"][eligible].detach(), labels[eligible], reduction="none").tolist())
            model.eval(); vals = []
            with torch.no_grad():
                for b in validation:
                    out = _forward(model, b); eligible = out["support"].flatten(1).any(1)
                    if eligible.any():
                        vals.append(F.binary_cross_entropy_with_logits(out["scene_logits"][eligible], b["scene_labels"][eligible].to(out["scene_logits"].dtype), reduction="none"))
            if not vals: raise ValueError("no eligible validation scene observations")
            loss = float(torch.cat(vals).mean())
            history["scene"].append({"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses) if train_losses else float("nan"), "validation_loss": loss})
            if loss < best_loss: best_loss, best_state, best_epoch = loss, _state_cpu(model), epoch
        if best_state is None: raise ValueError("no eligible training scene observations")
        model.load_state_dict(best_state); history["best_scene_epoch"] = best_epoch
        with torch.no_grad():
            dense_train = dense_validation = scene_train = scene_validation = 0
            for collection, is_train in ((train, True), (validation, False)):
                for b in collection:
                    out = _forward(model, b)
                    dense_count = _dense_stats(out["mask_logits"], b["plume"], b["pixel_truth_available"], out["support"])["supervised"]
                    eligible = int(out["support"].flatten(1).any(1).sum())
                    if is_train: dense_train += dense_count; scene_train += eligible
                    else: dense_validation += dense_count; scene_validation += eligible
            history["counters"] = {
                "train_samples": sum(len(b["sample_ids"]) for b in train),
                "validation_samples": sum(len(b["sample_ids"]) for b in validation),
                "dense_train_scenes": dense_train, "dense_validation_scenes": dense_validation,
                "scene_train_scenes": scene_train, "scene_validation_scenes": scene_validation,
                "train_missing_annotations": sum(int((~b["pixel_truth_available"]).sum()) for b in train),
                "validation_missing_annotations": sum(int((~b["pixel_truth_available"]).sum()) for b in validation),
                "train_invalid_scenes": sum(len(b["sample_ids"]) for b in train) - scene_train,
                "validation_invalid_scenes": sum(len(b["sample_ids"]) for b in validation) - scene_validation,
            }
        history["optimization"] = {
            "optimizer": "Adam", "learning_rate": lr, "weight_decay": 0,
            "gradient_clip_norm": 1.0, "dense_epochs": dense_epochs,
            "scene_epochs": scene_epochs, "batch_order": "supplied order, repeated each epoch",
            "selection_ties": "earliest epoch", "bce_weight": 0.7,
        }
        return history
    finally:
        for p, value in original_requires.items(): p.requires_grad = value
        model.eval()
