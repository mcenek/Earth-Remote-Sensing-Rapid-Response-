"""Independent raw-array and fresh CPU replay for the one centered pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_centered_v5 as t


def raw_centered(data, key):
    valid = data["valid"].astype(bool)
    pixels = np.log1p(data[key][valid].astype(np.float64))
    return pixels - np.median(pixels, axis=0)


def verify(run: Path):
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    if meta["arm"] != t.ARM or receipt["arm"] != t.ARM:
        raise ValueError("Wrong centered pilot")
    for path, field in ((t.HERE, "trainer_sha256"), (t.V4, "v4_source_sha256"),
                        (t.V3, "v3_source_sha256"), (t.ORIGINAL, "original_sha256"),
                        (t.STABLE, "stable_source_sha256"), (t.PROTOCOL, "protocol_sha256"),
                        (t.BASE_MANIFEST, "baseline_manifest_sha256"),
                        (t.BASE_DRAWS, "baseline_schedule_file_sha256")):
        if t.sha(path) != meta[field]:
            raise ValueError(f"Versioned dependency changed: {path}")
    if t.sha(run / "metadata.json") != receipt["metadata_sha256"]:
        raise ValueError("Metadata changed")
    manifest_path = Path(meta["manifest"])
    if t.sha(manifest_path) != meta["manifest_sha256"]:
        raise ValueError("Manifest changed")
    t.set_branch(meta["config"]["branch"])
    _, scenes, counts, positives = t.v4.v3.checked_scenes(manifest_path)
    if counts != meta["split_counts"] or positives != meta["positive_counts"]:
        raise ValueError("Source, array, or split identity changed")
    draws = json.loads((run / "draw_schedule.json").read_text(encoding="utf-8"))
    control_file = run.parent / "balanced_full_scene" / "draw_schedule.json"
    if (t.sha(run / "draw_schedule.json") != meta["draw_schedule_file_sha256"] or
            t.sha(control_file) != meta["control_draw_schedule_sha256"] or
            draws != json.loads(control_file.read_text(encoding="utf-8"))):
        raise ValueError("Full-scene control draw schedule changed")
    if (len(draws) != 38400 or t.v4.draw_counts(draws, scenes) !=
            {"US_positive": 9600, "US_negative": 9600,
             "aux_positive": 9600, "aux_negative": 9600}):
        raise ValueError("Centered draw domain/class allocation changed")
    norm = meta["normalization"]
    if norm["radiometry"] != "log1p_dn_date_band_median_centered":
        raise ValueError("Wrong centered normalization")
    count, sums, squares = 0, np.zeros(6, np.float64), np.zeros(6, np.float64)
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = t.load_scene(manifest_path, scene)
        for key in ("target_image", "reference_image"):
            pixels = raw_centered(data, key)
            count += len(pixels)
            sums += pixels.sum(axis=0)
            squares += np.square(pixels).sum(axis=0)
    center = sums / count
    scale = np.maximum(np.sqrt(np.maximum(squares / count - center * center, 0)), 1e-6)
    if (count != norm["train_date_pixels"] or
            not np.allclose(center, norm["center"], rtol=0, atol=1e-12) or
            not np.allclose(scale, norm["scale"], rtol=0, atol=1e-12)):
        raise ValueError("Train-only centered normalizer differs from raw-array reconstruction")
    train = [s for s in scenes if s["split"] == "train"]
    dev = [s for s in scenes if s["split"] == "dev"]
    for scene in (train[0], train[-1], dev[0], dev[-1]):
        data = t.load_scene(manifest_path, scene)
        valid = data["valid"].astype(bool)
        actual = t.prepare(data, norm)
        manual = np.zeros_like(actual)
        for date, key in enumerate(("target_image", "reference_image")):
            centered = raw_centered(data, key)
            manual[valid, date*6:(date+1)*6] = np.clip(
                (centered - center) / scale, -8, 8).astype(np.float32)
        if not np.array_equal(actual, manual) or not np.all(actual[~valid] == 0):
            raise ValueError("Centered preprocessing differs from raw-array reconstruction")
    model_results = {}
    for name in ("stable_residual", "siamese"):
        report = receipt["models"][name]
        maximum, checked = 0.0, 0
        for step, milestone in report["milestones"].items():
            checkpoint = run / name / f"step_{step}.pt"
            if t.sha(checkpoint) != milestone["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["normalization"] != norm or saved["steps"] != int(step):
                raise ValueError("Checkpoint preprocessing or step differs")
            model = t.new_model(name)
            model.load_state_dict(saved["model"])
            model.eval()
            for scene in dev:
                data = t.load_scene(manifest_path, scene)
                probability = t.predict(model, t.prepare(data, norm))
                expected = milestone["dev"]["scenes"][scene["id"]]
                file = run / name / f"dev_step_{step}" / f"{scene['id']}.npz"
                if t.sha(file) != expected["probability_sha256"]:
                    raise ValueError("Saved DEV probability changed")
                with np.load(file, allow_pickle=False) as z:
                    error = float(np.max(np.abs(probability-z["probability20"])))
                    if error > t.REPLAY_ATOL or not np.array_equal(z["valid20"], data["valid"].astype(bool)):
                        raise ValueError("Fresh centered CPU replay differs")
                    maximum = max(maximum, error)
                native = np.repeat(np.repeat(probability, t.native_factor(), 0), t.native_factor(), 1)
                if t.confusion(native, data["label_native"], data["valid_native"].astype(bool)) != expected["native_confusion"]:
                    raise ValueError("Native confusion differs")
                checked += 1
        model_results[name] = {"milestones": len(report["milestones"]),
                               "dev_scene_replays": checked, "maximum_probability_error": maximum}
    result = {"schema": "GTM_Model7_centered_v5_e2e", "passed": True, "arm": t.ARM,
              "manifest_sha256": meta["manifest_sha256"], "trainer_sha256": meta["trainer_sha256"],
              "metadata_sha256": t.sha(run / "metadata.json"), "fresh_process": True,
              "source_and_array_identity": "verified", "independent_raw_preprocessing": "verified",
              "draw_domain_class_counts": meta["draw_domain_class_counts"],
              "models": model_results, "test_array_access": "none"}
    t.write_json(run / "e2e_receipt.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve())


if __name__ == "__main__":
    main()
