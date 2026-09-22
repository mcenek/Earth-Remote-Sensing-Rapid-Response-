"""Rebuild the local GTM_Model4 v5.1 viewer bundle from frozen cached artifacts.

No model inference occurs. The NPZ supplies predictions/truth/validity; the sealed H5
supplies the three six-band temporal frames and point coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MethaneS2CM/l2a_location_split_32x32"
PRED = DATA / "v5_1_location_test_predictions.npz"
PACK = DATA / "v5_location_test_packed.h5"
CSV = DATA / "test.csv"
DEFAULT_OUT = ROOT / "outputs/GTM_viewer_bundles/ersrr_v5_1"
REPORT = ROOT / "reports/experiments/methanes2cm_v5_1_location_test.json"
PIXEL_THRESHOLD = 0.40
SCENE_THRESHOLD = 0.7335791607366197
DEFAULT_IDS = [86360, 83106, 82448, 97322, 98104, 87344, 83149, 83072]
OUTCOMES = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def png_rgb(frame: np.ndarray, out: Path) -> None:
    # H5/trainer six-band order is B02,B03,B04,B08,B11,B12; display R,G,B = B04,B03,B02.
    x = frame[[2, 1, 0]].astype(np.float32)
    lo = np.percentile(x, 2, axis=(1, 2), keepdims=True)
    hi = np.percentile(x, 98, axis=(1, 2), keepdims=True)
    x = np.clip((x - lo) / np.maximum(hi - lo, 1.0), 0, 1)
    Image.fromarray(np.rint(x.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB").save(out, optimize=True)


def choose_default(ids: np.ndarray, label: np.ndarray, decision: np.ndarray, groups: np.ndarray) -> list[int]:
    chosen: list[int] = []
    used: set[str] = set()
    for key in [(1, 1), (0, 1), (1, 0), (0, 0)]:
        candidates = np.flatnonzero((label == key[0]) & (decision == key[1]))
        for i in sorted(candidates.tolist(), key=lambda j: (str(groups[j]), int(ids[j]))):
            if str(groups[i]) in used:
                continue
            used.add(str(groups[i])); chosen.append(int(ids[i]))
            if sum(1 for j in chosen if (int(label[np.flatnonzero(ids == j)[0]]), int(decision[np.flatnonzero(ids == j)[0]])) == key) == 2:
                break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sample-id", type=int, action="append", dest="sample_ids")
    args = ap.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(PRED, allow_pickle=False)
    csv = pd.read_csv(CSV).set_index("id")
    pred_hash = digest(PRED)
    pack_hash = digest(PACK)
    ensemble = json.loads((ROOT / 'reports/experiments/methanes2cm_v5_1_ensemble_validation.json').read_text(encoding='utf-8'))
    checkpoint_identity = '; '.join(f"seed {entry['seed']}: {entry['checkpoint']['sha256']}" for entry in ensemble['seeds'])
    ids = z["sample_id"].astype(np.int64); labels = z["label"].astype(np.uint8); decisions = z["ersrr_v5_1_scene_decision"].astype(np.uint8)
    requested = args.sample_ids or DEFAULT_IDS
    pos = {int(v): i for i, v in enumerate(ids)}
    if any(v not in pos for v in requested):
        raise SystemExit("Requested sample ID is absent from the frozen NPZ")
    with h5py.File(PACK, "r") as h5:
        if not np.array_equal(h5["sample_id"][:], ids) or not np.array_equal(h5["label"][:], labels):
            raise SystemExit("H5/NPZ sample_id or label join failed")
        scenes = []
        for n, sid in enumerate(requested, 1):
            i = pos[sid]; row = csv.loc[sid]; lat, lon = float(row.latitude), float(row.longitude)
            outcome = OUTCOMES[(int(labels[i]), int(decisions[i]))]
            stem = f"{sid}_{outcome}"
            names = {"target": f"{stem}_target_rgb.png", "reference90": f"{stem}_reference90_rgb.png", "reference365": f"{stem}_reference365_rgb.png"}
            png_rgb(h5["target"][i], out / names["target"]); png_rgb(h5["reference90"][i], out / names["reference90"]); png_rgb(h5["reference365"][i], out / names["reference365"])
            scene = {
                "id": str(sid), "sampleId": sid, "sceneId": str(z["group_id"][i]), "exactLocationId": f"{lat},{lon}", "outcome": outcome,
                "label": int(labels[i]), "sceneDecision": int(decisions[i]), "sceneScore": float(z["ersrr_v5_1_scene_score"][i]), "sceneThreshold": SCENE_THRESHOLD,
                "pixelThreshold": PIXEL_THRESHOLD, "pixelDecisionDefinition": "probability >= 0.40 independently of scene gate", "split": "MethaneS2CM l2a_location_split_32x32 sealed location test",
                "latitude": lat, "longitude": lon, "coordinateSystem": "pixel", "imageSize": [32, 32], "locationPoint": [lat, lon], "georeference": None,
                "georeferenceNote": "H5 provides point coordinates only; no affine CRS or geographic bounds were invented.", "name": f"{outcome} · sample {sid}", "location": f"Benchmark point · {lat:.4f}, {lon:.4f}",
                "rgb": names["target"], "reference90Rgb": names["reference90"], "reference365Rgb": names["reference365"], "predictionAvailable": True,
                "truthLabel": "Dataset reference mask", "truthNote": "Cached binary benchmark mask, scored only on observable pixels.", "emitNote": "Benchmark pack contains no EMIT concentration image.",
                "rgbNote": "Target Sentinel-2 L2A; H5 order B02,B03,B04,B08,B11,B12; display RGB uses B04,B03,B02 with per-image percentile stretch.", "resolution": "32x32 native benchmark pixels; point location only",
                "predictionHash": pred_hash, "source": f"Sample {sid} · packed H5 SHA256 {pack_hash}", "checkpointHash": checkpoint_identity,
                "notes": "Scene decision uses the frozen calibrated scene rule. Pixel masks independently use probability >= 0.40. Temporal references are nominal -90/-365-day inputs; exact acquisition dates are unavailable.",
                "grid": {"width": 32, "height": 32, "probability": z["ersrr_v5_1_probability"][i].astype(np.float32).reshape(-1).tolist(), "truth": z["truth"][i].astype(np.uint8).reshape(-1).tolist(), "valid": z["observable"][i].astype(np.uint8).reshape(-1).tolist()},
            }
            scenes.append(scene)
    report_hash = digest(REPORT)
    manifest = {"id": "GTM_Model4", "name": "ERSRR v5.1 · frozen location test", "status": "Primary result · authentic cached predictions", "description": "MethaneS2CM location-held-out test: 20,789 L2A scenes across 816 locations.", "metrics": {"Scene AP": "0.8180", "Precision": "0.8630", "Recall": "0.3778", "FPR": "0.0607", "Pixel IoU": "0.1852"}, "decisionThreshold": PIXEL_THRESHOLD, "sceneThreshold": SCENE_THRESHOLD, "notes": "Scene score uses the frozen calibrated gate 0.7335791607366197. Dense pixel probability uses the independent threshold 0.40. The H5 contains point coordinates only; no affine CRS or geographic bounds are claimed.", "scenes": scenes, "source": {"predictionCache": str(PRED.relative_to(ROOT)).replace("\\", "/"), "predictionCacheSha256": pred_hash, "packedH5": str(PACK.relative_to(ROOT)).replace("\\", "/"), "packedH5Sha256": pack_hash, "sampleIdJoinValidated": True, "labelJoinValidated": True, "testCsvJoinValidated": True, "frameBandOrderVerified": ["B02", "B03", "B04", "B08", "B11", "B12"], "rgbIndicesVerified": [2, 1, 0], "checkpointIdentity": "Three-seed ERSRR v5.1 ensemble (seeds 1101/2202/3303); checkpoint files unavailable locally.", "frozenReport": str(REPORT.relative_to(ROOT)).replace("\\", "/"), "frozenReportSha256": report_hash, "reportUrl": str(REPORT.relative_to(ROOT)).replace("\\", "/")}}
    manifest["reportUrl"] = "GTM_Model51_frozen_test.md"
    shutil.copyfile(ROOT / "reports/experiments/METHANES2CM_V5_1_LOCATION_TEST.md", out / manifest["reportUrl"])
    manifest['status'] = 'Retired mapping candidate · authentic cached predictions'
    manifest["notes"] += " Scene AP is crop ranking, not segmentation accuracy. Full-cache spatial audit: 44.774% entirely positive masks and 51.571% blank masks. A spatially constant crop baseline reproduces 99.8289% of masks at the frozen threshold; pixel precision is 19.84%. This result supports crop classification signal but fails useful plume localization. Comparators were zero-shot L1C models evaluated on L2A data, not an architecture-controlled comparison."
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "sampleIds": requested, "bytes": sum(p.stat().st_size for p in out.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
