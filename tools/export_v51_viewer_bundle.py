"""Export small, authentic v5.1 cached examples for the local viewer."""
from __future__ import annotations

import hashlib
import json
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
OUT = ROOT / "outputs/GTM_viewer_bundles/ersrr_v5_1"
PIXEL_THRESHOLD = 0.4
SCENE_THRESHOLD = 0.7335791607366197


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rgb_png(values: np.ndarray, path: Path) -> None:
    # Six-band temporal frame contract is B02,B03,B04,B08,B11,B12. Use B04/B03/B02.
    x = values[[2, 1, 0]].astype(np.float32)
    lo = np.nanpercentile(x, 2, axis=(1, 2), keepdims=True)
    hi = np.nanpercentile(x, 98, axis=(1, 2), keepdims=True)
    x = np.clip((x - lo) / np.maximum(hi - lo, 1), 0, 1)
    Image.fromarray(np.rint(x.transpose(1, 2, 0) * 255).astype(np.uint8), "RGB").save(path, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(PRED, allow_pickle=False)
    with h5py.File(PACK, "r") as h5:
        ids_h5 = h5["sample_id"][:]
        ids_npz = z["sample_id"][:]
        if not np.array_equal(ids_h5, ids_npz):
            raise RuntimeError("H5/NPZ sample_id alignment failed")
        if not np.array_equal(h5["label"][:], z["label"][:]):
            raise RuntimeError("H5/NPZ label alignment failed")
        csv = pd.read_csv(CSV).set_index("id")
        if not set(ids_npz).issubset(csv.index):
            raise RuntimeError("NPZ sample IDs missing from test.csv")

        label = z["label"].astype(np.uint8)
        scene_decision = z["ersrr_v5_1_scene_decision"].astype(np.uint8)
        outcome_names = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}
        selected: list[tuple[int, str]] = []
        used_groups: set[str] = set()
        # Prefer US-coordinate rows when available, then deterministic group/sample order.
        for key in [(1, 1), (0, 1), (1, 0), (0, 0)]:
            candidates = np.flatnonzero((label == key[0]) & (scene_decision == key[1]))
            rows = []
            for i in candidates:
                sid = int(ids_npz[i]); r = csv.loc[sid]
                us = 24 <= float(r.latitude) <= 50 and -125 <= float(r.longitude) <= -66
                rows.append((not us, str(z["group_id"][i]), sid, int(i)))
            rows.sort()
            count = 0
            for _, group, sid, i in rows:
                if group in used_groups:
                    continue
                used_groups.add(group); selected.append((i, outcome_names[key])); count += 1
                if count == 2:
                    break
            if count != 2:
                raise RuntimeError(f"Could not select two distinct groups for {outcome_names[key]}")

        examples = []
        for n, (i, outcome) in enumerate(selected, 1):
            sid = int(ids_npz[i]); row = csv.loc[sid]
            stem = f"{n:02d}_{outcome}_{sid}"
            target_name = stem + "_target_rgb.png"
            ref90_name = stem + "_reference90_rgb.png"
            ref365_name = stem + "_reference365_rgb.png"
            rgb_png(h5["target"][i], OUT / target_name)
            rgb_png(h5["reference90"][i], OUT / ref90_name)
            rgb_png(h5["reference365"][i], OUT / ref365_name)
            prob = z["ersrr_v5_1_probability"][i].astype(np.float32)
            truth = z["truth"][i].astype(np.uint8)
            valid = z["observable"][i].astype(np.uint8)
            examples.append({
                "id": stem, "sampleId": sid, "sceneId": str(z["group_id"][i]), "exactLocationId": str(row.latitude) + "," + str(row.longitude),
                "outcome": outcome, "label": int(label[i]), "sceneDecision": int(scene_decision[i]),
                "sceneScore": float(z["ersrr_v5_1_scene_score"][i]), "sceneThreshold": SCENE_THRESHOLD,
                "pixelThreshold": PIXEL_THRESHOLD, "pixelDecisionDefinition": "probability >= 0.40 independently of scene gate",
                "split": "MethaneS2CM l2a_location_split_32x32 sealed location test", "latitude": float(row.latitude), "longitude": float(row.longitude),
                "coordinateSystem": "pixel", "imageSize": [32, 32], "locationPoint": [float(row.latitude), float(row.longitude)],
                "georeference": None, "georeferenceNote": "H5 has point coordinates only; no affine/CRS/bounds were invented.",
                "assets": {"targetRgb": target_name, "reference90Rgb": ref90_name, "reference365Rgb": ref365_name},
                "grid": {"width": 32, "height": 32, "probability": prob.reshape(-1).tolist(), "truth": truth.reshape(-1).tolist(), "valid": valid.reshape(-1).tolist()},
            })
    manifest = {"experimentId": "GTM_Model4", "name": "ERSRR v5.1 frozen location test", "model": "ERSRR v5.1", "examples": examples,
        "source": {"predictionCache": str(PRED.relative_to(ROOT)).replace("\\", "/"), "predictionCacheSha256": sha256(PRED), "packedInput": str(PACK.relative_to(ROOT)).replace("\\", "/"), "packedInputSha256": sha256(PACK), "checkpointSha256": "7b648548cc62ca3f6d428df2cf427e373fba5a7bdcf03aabada68bf6f1cfc446", "cacheAuthentic": True},
        "gating": {"denseProbabilityThreshold": PIXEL_THRESHOLD, "sceneScoreThreshold": SCENE_THRESHOLD, "sceneScoreField": "ersrr_v5_1_scene_score", "sceneAndPixelGatesIndependent": True},
        "joinValidation": {"sampleIdsH5EqualsNpZ": True, "labelsH5EqualsNpZ": True, "testCsvJoin": True, "selectedExamples": len(examples)}}
    (OUT / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "examples": [e["id"] for e in examples], "bytes": sum(p.stat().st_size for p in OUT.iterdir())}, indent=2))


if __name__ == "__main__":
    main()
