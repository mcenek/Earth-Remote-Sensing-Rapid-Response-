"""Export four deterministic visual examples from the frozen MARS dense cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "Earth-Remote-Sensing-Rapid-Response-"
sys.path.insert(0, str(SIBLING / "EarthRemoteSensingRapidResponse"))
sys.path.insert(0, str(SIBLING / "tools"))

from mars_s2l_adapter import iter_manifest, load_sample, safe_asset_path  # noqa: E402
from evaluate_released_marss2l import component_mask, load_released_model, released_input  # noqa: E402


def pick(values: dict[str, np.ndarray]) -> dict[str, int]:
    ids, labels = values["ids"], values["labels"]
    bs, cs = values["baseline_scores"], values["candidate_scores"]
    bp, cp = values["baseline_pixels"], values["candidate_pixels"]
    # Deterministic, score/count based strata. These are examples, not a sample
    # estimate: candidates are selected from the frozen cached receipt only.
    rules = {
        "successful_positive": (labels == 1) & (cs > 0.90) & (cp[:, 0] > 1000) & (cp[:, 2] < 5000),
        "missed_positive": (labels == 1) & (cs < 0.10),
        "suppressed_false_alarm": (labels == 0) & (bs > 0.50) & (cs < 0.75) & (cp[:, 1] > 0),
        "background": (labels == 0) & (bs < 0.02) & (cs < 0.02) & (bp[:, 1] == 0),
    }
    chosen: dict[str, int] = {}
    for name, mask in rules.items():
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise RuntimeError(f"No cached candidate for {name}")
        # Stable lexical sample-ID tie-break after the intended score ordering.
        if name == "successful_positive":
            order = sorted(candidates.tolist(), key=lambda i: (-float(cs[i]), int(cp[i, 2]), str(ids[i])))
        elif name == "missed_positive":
            order = sorted(candidates.tolist(), key=lambda i: (float(cs[i]), -int(cp[i, 2]), str(ids[i])))
        elif name == "suppressed_false_alarm":
            order = sorted(candidates.tolist(), key=lambda i: (-float(bs[i]), float(cs[i]), -int(bp[i, 1]), str(ids[i])))
        else:
            order = sorted(candidates.tolist(), key=lambda i: (float(bs[i]), float(cs[i]), str(ids[i])))
        chosen[name] = order[0]
    return chosen


def main() -> None:
    cache = SIBLING / "outputs/mars_paper_test_v3_diagnostic_cache.npz"
    manifest = SIBLING / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L/paper_v3_sealed_test_samples.jsonl"
    data_root = SIBLING / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
    out = ROOT / "reports/research/assets"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(cache, allow_pickle=False) as z:
        values = {key: z[key] for key in ("aligned_sample_ids", "labels", "sites", "sensors", "baseline_scores", "candidate_scores", "baseline_pixels", "candidate_pixels")}
        available = set(z["available_ids"].astype(str).tolist())
    aligned = values["aligned_sample_ids"].astype(str)
    keep = np.asarray([str(value) in available for value in aligned], dtype=bool)
    for key in ("aligned_sample_ids", "labels", "sites", "sensors", "baseline_scores", "candidate_scores", "baseline_pixels", "candidate_pixels"):
        values[key] = values[key][keep]
    values["ids"] = values.pop("aligned_sample_ids").astype(str)
    chosen = pick(values)
    wanted = {values["ids"][i] for i in chosen.values()}
    records: dict[str, dict] = {}
    for record in iter_manifest(manifest):
        if str(record.get("sample_id")) in wanted:
            records[str(record["sample_id"])] = record
    if set(records) != wanted:
        raise RuntimeError(f"Manifest missing selected IDs: {sorted(wanted - set(records))}")
    model_path = SIBLING / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L/trained_models/MARSS2L_20250326/best_epoch"
    model = load_released_model(model_path, torch.device("cpu"), 16)

    tiles: list[list[Image.Image]] = []
    receipt: dict = {
        "scope": "four deterministic native-image examples from frozen MARS v3 diagnostic cache",
        "dense_prediction": "inherited released MARS-S2L probability mask; scene gate is unchanged v3 stronger head",
        "mask_thresholds": {"Sentinel-2": 0.8, "Landsat": 0.7},
        "minimum_connected_pixels": 100,
        "selection_is_population_estimate": False,
        "examples": [],
    }
    font = ImageFont.load_default()
    checkpoint_sha256 = "be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246"
    gate_cutoff = 0.75
    for role, idx in chosen.items():
        sample_id = str(values["ids"][idx])
        record = records[sample_id]
        sample = load_sample(data_root, record, require_enhancement=False, allow_missing_positive_mask=True)
        native_input = released_input(sample, (float(record.get("wind_u", 0.0)), float(record.get("wind_v", 0.0))), "mars-s2l")
        with torch.no_grad():
            probability = torch.sigmoid(model(torch.from_numpy(native_input[None, ...]))[0]).numpy()
        mask_threshold = 0.8 if record["sensor_family"] == "Sentinel-2" else 0.7
        native_mask = component_mask((probability >= mask_threshold).astype(np.float32))
        gate_open = float(values["candidate_scores"][idx]) >= 0.75
        observable = sample.observable_mask
        truth_available = bool(record.get("pixel_truth_available", False))
        truth = sample.plume_mask & observable if truth_available else np.zeros_like(observable)
        native_mask &= observable
        gated_mask = native_mask if gate_open else np.zeros_like(native_mask)
        native_counts = [int(np.count_nonzero(native_mask & truth)), int(np.count_nonzero(native_mask & observable & ~truth)), int(np.count_nonzero((~native_mask) & truth))] if truth_available else [0, int(np.count_nonzero(native_mask)), 0]
        gated_counts = [int(np.count_nonzero(gated_mask & truth)), int(np.count_nonzero(gated_mask & observable & ~truth)), int(np.count_nonzero((~gated_mask) & truth))] if truth_available else [0, int(np.count_nonzero(gated_mask)), 0]
        # Correct RGB is B04/B03/B02 = array channels 2/1/0. Reference is shown
        # in the same channel order and explicitly labelled as reference.
        def stretch(values: np.ndarray) -> np.ndarray:
            out = np.zeros_like(values, dtype=np.float32)
            valid = observable
            for c in range(3):
                channel = values[..., c]
                lo, hi = np.nanpercentile(channel[valid], [2, 98]) if valid.any() else (0.0, 1.0)
                out[..., c] = np.clip((channel - lo) / max(hi - lo, 1e-6), 0, 1)
            return np.uint8(out * 255)
        rgb = stretch(np.moveaxis(sample.reflectance_pair[[2, 1, 0]], 0, -1))
        truth_rgb = np.zeros_like(rgb)
        truth_rgb[truth] = [0, 220, 255]
        def panel(base: np.ndarray, mask: np.ndarray, title: str, subtitle: str) -> Image.Image:
            view = base.copy()
            view[mask, 0] = 255
            view[mask, 1] = (view[mask, 1] * 0.2).astype(np.uint8)
            view[mask, 2] = (view[mask, 2] * 0.2).astype(np.uint8)
            view[~observable] = [128, 128, 128]
            image = Image.fromarray(view, mode="RGB").resize((300, 300), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 300, 35), fill=(0, 0, 0))
            draw.text((5, 4), title, fill=(255, 255, 255), font=font)
            draw.text((5, 18), subtitle, fill=(255, 255, 255), font=font)
            return image
        subtitle = f"{record['sensor_family']} {record['label_state']}"
        row = [
            panel(rgb, np.zeros_like(native_mask), role.replace('_', ' '), subtitle),
            panel(truth_rgb, np.zeros_like(native_mask), "Dataset reference mask (cyan)" if truth_available else "NO_PLUME scene label", "Gray = unobserved; not independent truth" if truth_available else "No separate pixel annotation available"),
            panel(rgb, native_mask, "released mask BEFORE gate", f"pixels={int(native_mask.sum())} thr={mask_threshold:.1f}"),
            panel(rgb, gated_mask, "mask AFTER v3 gate", f"pixels={int(gated_mask.sum())} gate={'open' if gate_open else 'closed'}"),
        ]
        tiles.append(row)
        receipt["examples"].append({
            "role": role,
            "sample_id": sample_id,
            "group_id": str(values["sites"][idx]),
            "label_state": record["label_state"],
            "sensor_family": record["sensor_family"],
            "native_image": str(record["assets"][0]["path"]),
            "baseline_scene_score": float(values["baseline_scores"][idx]),
            "candidate_scene_score": float(values["candidate_scores"][idx]),
            "cached_baseline_tp_fp_fn": values["baseline_pixels"][idx].astype(int).tolist(),
            "cached_candidate_tp_fp_fn": values["candidate_pixels"][idx].astype(int).tolist(),
            "native_released_mask_tp_fp_fn": native_counts,
            "native_gated_mask_tp_fp_fn": gated_counts,
            "native_pre_gate_predicted_pixels": int(native_mask.sum()),
            "cached_candidate_pre_gate_predicted_pixels": int(values["candidate_pixels"][idx, 0] + values["candidate_pixels"][idx, 1]),
            "native_minus_cached_candidate_pre_gate_pixels": int(native_mask.sum() - values["candidate_pixels"][idx, 0] - values["candidate_pixels"][idx, 1]),
            "native_released_mask_threshold": mask_threshold,
            "native_released_mask_pixels": int(native_mask.sum()),
            "original_v3_scene_gate_cutoff": 0.75,
            "original_v3_scene_gate_open_from_cached_score": gate_open,
            "pixel_truth_available": bool(record.get("pixel_truth_available", False)),
            "visual_layer": "native corrected RGB, dataset reference plume mask in cyan or explicit missing pixel annotation, inherited released-MARS mask before gate, and mask after unchanged v3 gate",
        })
    montage = Image.new("RGB", (1200, 4 * 300), (32, 32, 32))
    for row_index, row in enumerate(tiles):
        for column_index, tile in enumerate(row):
            montage.paste(tile, (column_index * 300, row_index * 300))
    montage.save(out / "GTM_Model0_mars_dense_examples.png")
    receipt["assets"] = {
        "montage": "reports/research/assets/GTM_Model0_mars_dense_examples.png",
        "cache": "../Earth-Remote-Sensing-Rapid-Response-/outputs/mars_paper_test_v3_diagnostic_cache.npz",
        "manifest": "../Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L/paper_v3_sealed_test_samples.jsonl",
    }
    receipt["checkpoint_sha256"] = checkpoint_sha256
    receipt["verification_note"] = "Native counts are compared to cached candidate pre-gate counts; nonzero deltas are retained as exact mismatches, not silently reconciled. Negative rows have no pixel truth and use the explicit scene-label diagnostic assumption."
    (out / "GTM_Model0_mars_dense_examples.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
