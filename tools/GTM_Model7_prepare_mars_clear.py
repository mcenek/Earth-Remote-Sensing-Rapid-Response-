"""Export a new MARS repaired cohort with the frozen bidate snow-like QA gate.

Read docs/GTM_Model7_mars_clear_protocol.md before changing this rule.
The base exporter and its sealed output are not modified.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "tools/GTM_Model7_prepare_mars_repaired.py"
BASE_MANIFEST = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_data/manifest.json"
CLEAR_OUT = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data"
CLEAR_PROTOCOL = ROOT / "docs/GTM_Model7_mars_clear_protocol.md"
spec = importlib.util.spec_from_file_location("mars_repair_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)
original_raster_pair = base.raster_pair
base.OUT = CLEAR_OUT
base.PROTOCOL = CLEAR_PROTOCOL
GATE = {"green_b03_gt_dn": 3000, "nir_b08_gt_dn": 1000,
        "green_minus_swir_b11_over_sum_gt": 0.4, "date_fraction_gt": 0.10,
        "support": "native pixels passing the strict both-date cloud and radiometry mask"}
audited = []


def snow_fraction(image_path: Path, valid_native: np.ndarray) -> float:
    with rasterio.open(image_path) as raster:
        green, nir, swir = raster.read([2, 4, 5]).astype(np.float32)
    score = (green - swir) / np.maximum(green + swir, 1)
    flagged = (green > GATE["green_b03_gt_dn"]) & (nir > GATE["nir_b08_gt_dn"]) & \
              (score > GATE["green_minus_swir_b11_over_sum_gt"]) & valid_native
    return float(flagged.sum() / valid_native.sum())


def raster_pair_with_clear_gate(sid, ref_sid, metadata, originals, assets):
    payload, sources, reason, valid_fraction = original_raster_pair(sid, ref_sid, metadata, originals, assets)
    if reason:
        return payload, sources, reason, valid_fraction
    arrays, _ = payload
    paths = {entry["role"]: Path(entry["path"]) for entry in sources}
    target_fraction = snow_fraction(paths["target_image"], arrays["valid_native"])
    reference_fraction = snow_fraction(paths["reference_image"], arrays["valid_native"])
    rejected = max(target_fraction, reference_fraction) > GATE["date_fraction_gt"]
    audited.append({"sample_id": sid, "reference_observation_id": ref_sid,
                    "split_candidate_label": metadata[sid]["label_state"],
                    "target_snow_like_fraction": target_fraction,
                    "reference_snow_like_fraction": reference_fraction,
                    "excluded": rejected})
    if rejected:
        return None, sources, "snow_like_fraction_gt_10pct_target_or_reference", valid_fraction
    return payload, sources, None, valid_fraction


def main():
    if CLEAR_OUT.exists():
        raise FileExistsError("Versioned clear cohort must be new: " + str(CLEAR_OUT))
    if not BASE_MANIFEST.is_file():
        raise FileNotFoundError("Sealed base manifest is unavailable")
    frozen_base_sha = base.sha(BASE_MANIFEST)
    base.raster_pair = raster_pair_with_clear_gate
    base.main()
    manifest_path = CLEAR_OUT / "manifest.json"
    receipt_path = CLEAR_OUT / "receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    base_positive = {s["id"] for s in base_manifest["scenes"] if s["scene_kind"] == "PLUME"}
    clear_positive = {s["id"] for s in manifest["scenes"] if s["scene_kind"] == "PLUME"}
    if clear_positive != base_positive:
        raise ValueError("Frozen snow-like QA would alter positive scene IDs")
    manifest["base_manifest"] = base.file_source(BASE_MANIFEST)
    if manifest["base_manifest"]["sha256"] != frozen_base_sha:
        raise ValueError("Sealed base changed while creating the variant")
    manifest["variant_source"] = base.file_source(Path(__file__))
    manifest["frozen_snow_like_gate"] = GATE
    manifest["snow_like_gate_attempts"] = audited
    manifest["quality_limitations"].append(
        "Snow-like image QA is a fixed heuristic, not a certified snow classification; producer cloud code 0 does not certify surface clearance.")
    base.save(manifest_path, manifest)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest"] = base.file_source(manifest_path)
    receipt["base_manifest"] = base.file_source(BASE_MANIFEST)
    receipt["variant_source"] = base.file_source(Path(__file__))
    receipt["frozen_snow_like_gate"] = GATE
    receipt["snow_like_gate_summary"] = {
        "evaluated_after_base_raster_QA": len(audited),
        "excluded": sum(item["excluded"] for item in audited),
        "excluded_by_label": {label: sum(item["excluded"] and item["split_candidate_label"] == label
                                  for item in audited) for label in ("PLUME", "NO_PLUME")},
        "positive_scene_ids_preserved": True}
    base.save(receipt_path, receipt)
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": receipt["manifest"]["sha256"],
                      "counts": manifest["split_counts"], "gate": receipt["snow_like_gate_summary"]}, indent=2))


if __name__ == "__main__":
    main()
