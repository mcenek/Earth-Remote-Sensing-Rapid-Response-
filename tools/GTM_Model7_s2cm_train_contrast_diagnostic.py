"""Describe TRAIN-only MethaneS2CM temporal SWIR contrast; fit nothing."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/s2cm_reference_repair_data/manifest.json"
OUT = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/other_sources"
PROTOCOL = ROOT / "docs/GTM_Model7_s2cm_train_contrast_diagnostic.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dilate8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return np.logical_or.reduce([padded[i:i + mask.shape[0], j:j + mask.shape[1]]
                                 for i in range(3) for j in range(3)])


def median_mad(values: np.ndarray):
    if not len(values):
        return None, None
    median = float(np.median(values))
    return median, float(np.median(np.abs(values - median)))


def describe(values):
    x = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return {"n": len(x), "q05": float(np.quantile(x, .05)) if len(x) else None,
            "q25": float(np.quantile(x, .25)) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "q75": float(np.quantile(x, .75)) if len(x) else None,
            "q95": float(np.quantile(x, .95)) if len(x) else None,
            "negative": int((x < 0).sum()), "zero": int((x == 0).sum()),
            "positive": int((x > 0).sum())}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["source_dataset"] != "H1deaki/MethaneS2CM" or manifest["bands"] != ["B02", "B03", "B04", "B08", "B11", "B12"]:
        raise ValueError("Unexpected source or band order")
    scenes = [s for s in manifest["scenes"] if s["split"] == "train"]
    if len(scenes) != sum(manifest["split_counts"]["train"].values()):
        raise ValueError("TRAIN count mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in scenes:
        path = MANIFEST.parent / scene["array_file"]
        if sha(path) != scene["array_sha256"]:
            raise ValueError(f"Sealed TRAIN array changed: {scene['id']}")
        with np.load(path, allow_pickle=False) as z:
            current = z["target_image"].astype(np.float64)
            reference = z["reference_image"].astype(np.float64)
            valid = z["valid"].astype(bool)
            label = z["target"].astype(bool)
        if current.shape != (32, 32, 6) or reference.shape != current.shape or valid.shape != (32, 32) or label.shape != valid.shape:
            raise ValueError(f"Unexpected TRAIN array shape: {scene['id']}")
        if (current[valid] < 0).any() or (reference[valid] < 0).any() or not np.isfinite(current[valid]).all() or not np.isfinite(reference[valid]).all():
            raise ValueError(f"Invalid supported DN: {scene['id']}")
        logs = np.log1p(current) - np.log1p(reference)
        d = logs[..., 5] - logs[..., 4]
        inside = valid & label
        outside = valid & ~label
        row = {"id": scene["id"], "site": scene["physical_location_id"],
               "state": scene["state"], "kind": scene["scene_kind"],
               "reference_dataset": scene.get("reference_dataset", "reference90"),
               "valid_pixels": int(valid.sum()), "inside_pixels": int(inside.sum()),
               "outside_pixels": int(outside.sum())}
        if scene["scene_kind"] == "PLUME":
            ring = label.copy()
            for _ in range(5):
                ring = dilate8(ring)
            ring = ring & ~label & valid
            inside_med, _ = median_mad(d[inside])
            outside_med, outside_mad = median_mad(d[outside])
            ring_med, ring_mad = median_mad(d[ring])
            row.update(ring_pixels=int(ring.sum()), inside_median=inside_med,
                       outside_median=outside_med, outside_mad=outside_mad,
                       ring_median=ring_med, ring_mad=ring_mad,
                       global_contrast=(inside_med - outside_med if inside_med is not None and outside_med is not None else None),
                       ring_contrast=(inside_med - ring_med if inside_med is not None and ring_med is not None else None),
                       unanalysable_reason=("no_supported_annotation" if inside_med is None else
                                            "no_supported_outside" if outside_med is None else
                                            "no_supported_1_to_5_pixel_ring" if ring_med is None else None))
        else:
            if inside.any():
                raise ValueError(f"Negative source has annotated pixels: {scene['id']}")
            row.update(abs_log_change_six_band_median=float(np.median(np.abs(logs[valid]).mean(axis=1))),
                       abs_log_change_b11_median=float(np.median(np.abs(logs[..., 4][valid]))),
                       abs_log_change_b12_median=float(np.median(np.abs(logs[..., 5][valid]))),
                       temporal_swir_ratio_median=float(np.median(d[valid])))
        rows.append(row)
    positives = [r for r in rows if r["kind"] == "PLUME"]
    negatives = [r for r in rows if r["kind"] == "NO_PLUME"]
    by_site = defaultdict(list)
    for row in positives:
        by_site[row["site"]].append(row)
    sites = []
    for site, members in sorted(by_site.items()):
        sites.append({"site": site, "state": members[0]["state"], "scenes": len(members),
                      "global_contrast": describe([r["global_contrast"] for r in members])["median"],
                      "ring_contrast": describe([r["ring_contrast"] for r in members])["median"]})
    summary = {"schema": "GTM_Model7_s2cm_train_contrast_diagnostic_v1",
               "manifest_sha256": sha(MANIFEST), "protocol_sha256": sha(PROTOCOL),
               "scope": "TRAIN arrays only; no model fitting, DEV/TEST array access, or methane causation claim",
               "counts": {"train_scenes": len(rows), "positive": len(positives), "negative": len(negatives),
                          "positive_sites": len(sites), "negative_sites": len({r["site"] for r in negatives}),
                          "positive_unanalysable_global": sum(r["global_contrast"] is None for r in positives),
                          "positive_unanalysable_ring": sum(r["ring_contrast"] is None for r in positives),
                          "positive_full_supported_mask": sum(r["outside_pixels"] == 0 for r in positives)},
               "scene_level": {"global_contrast": describe([r["global_contrast"] for r in positives]),
                               "ring_contrast": describe([r["ring_contrast"] for r in positives]),
                               "outside_mad": describe([r["outside_mad"] for r in positives]),
                               "ring_mad": describe([r["ring_mad"] for r in positives]),
                               "inside_pixels": describe([r["inside_pixels"] for r in positives]),
                               "ring_pixels": describe([r["ring_pixels"] for r in positives])},
               "equal_site": {"global_contrast": describe([s["global_contrast"] for s in sites]),
                              "ring_contrast": describe([s["ring_contrast"] for s in sites])},
               "negative_scene_photometric_change": {
                   "six_band_median_abs_log_change": describe([r["abs_log_change_six_band_median"] for r in negatives]),
                   "b11_median_abs_log_change": describe([r["abs_log_change_b11_median"] for r in negatives]),
                   "b12_median_abs_log_change": describe([r["abs_log_change_b12_median"] for r in negatives]),
                   "swir_ratio_median": describe([r["temporal_swir_ratio_median"] for r in negatives])}}
    with (OUT / "s2cm_train_contrast_scenes.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (OUT / "s2cm_train_contrast_sites.json").write_text(json.dumps(sites, indent=2), encoding="utf-8")
    (OUT / "s2cm_train_contrast_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
