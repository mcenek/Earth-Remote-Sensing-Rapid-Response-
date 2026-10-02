"""Export the six saved Friday evidence cases without network or model execution.

Failure modes checked before writing: missing source assets, changed scene selection,
bad grid shapes, mismatched paired image sizes, missing native predictions, and
metric drift between saved native arrays and the viewer bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VOL = ROOT / "outputs/GTM_viewer_bundles/GTM_Model6_emit_2025_intake"
PILOT = ROOT / "outputs/GTM_viewer_bundles/GTM_Model6_mapper_mars_pilot_20260922_r5_mean"
RAW = ROOT / "outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r5"
READINESS = ROOT / "outputs/GTM_Model0_hf_pilot_20260925/emit_followup/swir_20260930/input_readiness.json"
OUT = ROOT / "GTM_Viewer/friday_20261002/assets"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, array: np.ndarray, records: dict) -> str:
    path = OUT / name
    Image.fromarray(array.astype(np.uint8), "RGB").save(path, optimize=False)
    records[name] = {"sha256": digest(path), "size": list(Image.open(path).size)}
    return f"assets/{name}"


def boundary(mask: np.ndarray) -> np.ndarray:
    near = np.zeros(mask.shape, bool)
    for axis in (0, 1):
        for step in (-1, 1):
            near |= np.roll(mask, step, axis=axis)
    return mask & ~(
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    ) | (~mask & near)


def metrics_from_npz(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as item:
        p = item["probability"] >= 0.5
        truth = item["target"] > 0.5
        valid = item["valid"].astype(bool) & item["context_support"].astype(bool)
        return {
            "tp": int(np.count_nonzero(p & truth & valid)),
            "fp": int(np.count_nonzero(p & ~truth & valid)),
            "fn": int(np.count_nonzero(~p & truth & valid)),
            "tn": int(np.count_nonzero(~p & ~truth & valid)),
            "evaluated_pixels": int(np.count_nonzero(valid)),
        }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    review = json.loads((VOL / "review.json").read_text(encoding="utf-8"))
    experiment = json.loads((PILOT / "experiment.json").read_text(encoding="utf-8"))
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert review["prediction"] is False and not review["ready_for_training"]
    assert not readiness["training_ready"] and len(readiness["five_target_bands_available"]) == 5
    assert experiment["decisionThreshold"] == 0.5 and len(experiment["scenes"]) == 65
    assert experiment["nativeAggregate"]["scenes"] == 65
    records: dict = {}
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (VOL / "review.json", PILOT / "experiment.json", READINESS)}

    rgb = np.asarray(Image.open(VOL / review["rgb"]).convert("RGB"))
    mf = np.asarray(Image.open(VOL / review["retrieval"]).convert("RGB"))
    vm = np.asarray(Image.open(VOL / review["mask"]).convert("RGBA"))[:, :, 3] > 0
    assert rgb.shape == mf.shape and vm.shape == rgb.shape[:2]
    assert int(vm.sum()) == 1075
    ref = mf.copy()
    # mask.png already stores the cyan boundary as nontransparent pixels.
    ref[vm] = (57, 236, 255)
    vol_case = {
        "id": "volusia_20250601", "label": "Volusia intake", "kind": "data", "date": "2025-06-01",
        "description": "Actual Sentinel-2 RGB and EMIT matched-filter retrieval from the 2025 Volusia intake. The cyan outline marks the saved provider mask displayed at 10 m; its annotation is native 60 m, not a 10 m prediction or verified fine-resolution truth.",
        "lesson": "A 12.74-minute overlap and five available target bands support an input review. Label-zero validity, retrieval units, radiometric scale/offset, a reference-date pair, and a reviewed negative still block training.",
        "panels": [
            {"title": "Sentinel-2 RGB", "image": save("case_volusia_rgb.png", rgb, records), "caption": "Saved display stretch of actual Sentinel-2 target RGB."},
            {"title": "EMIT retrieval + provider mask", "image": save("case_volusia_retrieval.png", ref, records), "caption": "Saved matched-filter display; cyan is the saved 60 m mask boundary shown on the 10 m display grid."},
            {"title": "Model prediction", "image": None, "emptyReason": "No model prediction exists for this intake.", "caption": "Input review only. No Volusia model score or enhancement result."},
        ],
        "metrics": [{"label": "Time offset", "value": "12.74 min"}, {"label": "Available target bands", "value": "5 / 5"}, {"label": "Training gate", "value": "Closed"}],
        "fullViewerUrl": "../GTM_Model6_emit_2025_intake.html",
    }
    for key in (review["rgb"], review["retrieval"], review["mask"]):
        source_hashes[str((VOL / key).relative_to(ROOT))] = digest(VOL / key)

    scenes = experiment["scenes"]
    selected = sorted(
        (s for s in scenes if s["label"] == "PLUME"),
        key=lambda s: {"0000": 0, "0001": 1, "0002": 2}[Path(s["rgb"]).stem.split("_")[0]],
    )
    selected += sorted((s for s in scenes if s["label"] == "NO_PLUME"), key=lambda s: -s["nativeMetrics"]["fp"])[:2]
    assert len(selected) == 5 and [s["label"] for s in selected] == ["PLUME"] * 3 + ["NO_PLUME"] * 2
    cases = [vol_case]
    native_checks = []
    for scene in selected:
        stem = Path(scene["rgb"]).stem.split("_")[0]
        grid_path = PILOT / scene["grid"]
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        assert (grid["width"], grid["height"]) == (160, 160)
        probability = np.asarray(grid["probability"], dtype=np.float32).reshape(160, 160)
        truth = np.asarray(grid["truth"], dtype=bool).reshape(160, 160)
        valid = np.asarray(grid["valid"], dtype=bool).reshape(160, 160)
        base = np.asarray(Image.open(PILOT / scene["rgb"]).convert("RGB"))
        assert base.shape == (160, 160, 3)
        # Mark unsupported pixels identically in all three aligned panels.
        yy, xx = np.indices(valid.shape)
        shaded = base.copy()
        outside = ~valid
        stripe = ((xx + yy) % 8 < 2)
        dark = np.where(stripe[..., None], [29, 42, 49], [45, 56, 62])
        shaded[outside] = (0.42 * base[outside] + 0.58 * dark[outside]).astype(np.uint8)
        outline = boundary(truth & valid) & valid
        truth_panel = shaded.copy()
        truth_panel[outline] = (55, 237, 255)
        pred_panel = shaded.copy()
        pred = (probability >= .5) & valid
        pred_panel[pred] = (0.42 * pred_panel[pred] + 0.58 * np.array([255, 74, 38])).astype(np.uint8)
        pred_panel[outline] = (55, 237, 255)
        raw_path = RAW / f"prediction_{stem}.npz"
        counts = metrics_from_npz(raw_path)
        for key, value in counts.items():
            assert value == scene["nativeMetrics"][key], (scene["id"], key, value, scene["nativeMetrics"][key])
        native_checks.append({"id": scene["id"], "rawPrediction": str(raw_path.relative_to(ROOT)), "sha256": digest(raw_path), "counts": counts})
        for key in (scene["rgb"], scene["grid"]):
            source_hashes[str((PILOT / key).relative_to(ROOT))] = digest(PILOT / key)
        is_positive = scene["label"] == "PLUME"
        review_label = {"0001": "Missed plume", "0000": "Localized result", "0002": "Partial result", "0034": "False alarm 869", "0012": "False alarm 717"}[stem]
        cases.append({
            "id": scene["id"], "label": review_label, "kind": "prediction", "date": scene["date"],
            "description": f"MARS reviewed {'plume' if is_positive else 'no-plume'} scene. Saved mean of four independent scale logits, displayed at the fixed 0.50 threshold. The model predicts only the central 88 × 88 pixels of the 200 × 200 source grid (19.36%); gray stripes mean no output, not a true negative.",
            "lesson": ("Compare the red model activation against the cyan reviewed reference within the scored center. This is a reused development fold and does not establish independent performance or source origin." if is_positive else "False activation on a reviewed negative is a core failure mode. Gray outside the supported center is unavailable and is excluded from scoring."),
            "panels": [
                {"title": "Sentinel-2 RGB", "image": save(f"case_{stem}_rgb.png", shaded, records), "caption": "Same full scene field; striped gray pixels lack valid model output."},
                {"title": "Reviewed reference", "image": save(f"case_{stem}_reference.png", truth_panel, records), "caption": "Cyan outline is the saved MARS reviewed plume mask within valid prediction support."},
                {"title": "Saved model output", "image": save(f"case_{stem}_prediction.png", pred_panel, records), "caption": "Orange-red is saved mean probability ≥ 0.50; cyan is the reference boundary. No output in gray stripes."},
            ],
            "metrics": [{"label": "Native IoU", "value": f"{scene['nativeMetrics']['iou'] * 100:.1f}%"}, {"label": "False positives", "value": str(counts["fp"])}, {"label": "True positives", "value": str(counts["tp"])}, {"label": "Support", "value": "19.36%"}],
            "fullViewerUrl": f"../index.html?experiment=GTM_Model6_mapper_r5_portable&scene={scene['id']}#compare",
        })
    agg = experiment["nativeAggregate"]
    assert (agg["tp"], agg["fp"], agg["fn"], agg["tn"]) == (492, 21387, 1021, 480144)
    data = {
        "title": "Methane research review", "meetingDate": "2026-10-02", "builtDate": "2026-09-30",
        "summary": "Six saved evidence cases: one EMIT/Sentinel input review and five MARS model scenes. The pilot fails its quality gate; use this review to choose the next measurable milestone.",
        "aggregate": {"scenes": 65, "positiveScenes": 3, "iou": agg["iou"], "precision": agg["precision"], "recall": agg["recall"], "status": "Failed quality gate", "scope": "Pooled native pixels within actual central support; reused development fold"},
        "cases": cases,
        "architectureImage": "assets/GTM_Model6_architecture.svg",
        "architectureCaption": "Current saved diagnostic: independent S16/S32/S64/S128 logits averaged at inference. No learned fusion, source-origin head, or quantitative EMIT enhancement was trained.",
        "decisions": [
            {"question": "What acceptance criteria and sequence serve the fixed US objective?", "why": "Agree measurable gates for plume regions, quantitative methane mapping, and emission origin. The current saved model addresses only plume segmentation and has failed its quality gate."},
            {"question": "What provider documentation confirms retrieval units and label-zero validity?", "why": "The Volusia annotation cannot support quantitative targets or verified negatives until those semantics are established."},
            {"question": "What source and wind labels will define an origin task?", "why": "Plume shape alone does not verify an emission origin; source and wind evidence must be specified before training that head."},
        ],
        "nextSteps": [
            {"title": "Set milestone criteria and sequence", "detail": "Keep the US methane objective fixed; agree held-out success criteria and order for plume regions, quantitative map, and emission origin before new model work.", "status": "Decision needed"},
            {"title": "Close Volusia data semantics", "detail": "Five target bands are local. Resolve exact-release label-zero validity, retrieval units, and target radiometric scale/offset.", "status": "Blocked"},
            {"title": "Build an independent paired cohort", "detail": "Acquire a matching reference-date crop and reviewed negative, then evaluate outside the reused development fold.", "status": "Blocked"},
        ],
        "pdf": "GTM_Model6_Friday_brief_20261002.pdf",
    }
    data_path = OUT / "data.js"
    data_path.write_text("window.GTM_FRIDAY_DATA = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n", encoding="utf-8", newline="\n")
    verification = {
        "status": "verified", "builtDate": "2026-09-30", "failureModes": ["missing source", "wrong case order", "grid/image shape drift", "native count drift", "panel mismatch"],
        "selection": [s["id"] for s in selected], "threshold": 0.5, "caseCount": len(cases),
        "nativeChecks": native_checks, "savedAggregate": agg, "sourceHashes": source_hashes,
        "outputImages": records, "dataJsSha256": digest(data_path),
        "pixelChecks": {"volusiaMaskPixels": int(vm.sum()), "selectedDisplayValidPixels": [int(np.count_nonzero(json.loads((PILOT / s["grid"]).read_text())["valid"])) for s in selected], "allPanelsAligned": True},
    }
    (OUT.parent / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported {len(cases)} cases, {len(records)} panels to {OUT}")


if __name__ == "__main__":
    main()
