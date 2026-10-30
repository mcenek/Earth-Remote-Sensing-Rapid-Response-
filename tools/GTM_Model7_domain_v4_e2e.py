"""Fresh-process artifact replay for controlled domain v4 training arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_train_repaired_domain_v4 as t


def verify(run: Path):
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    arm = meta["arm"]
    if arm not in t.ARMS or receipt["arm"] != arm:
        raise ValueError("Unknown or inconsistent arm")
    for path, field in ((t.HERE, "trainer_sha256"), (t.V3, "v3_source_sha256"),
                        (t.ORIGINAL, "original_sha256"), (t.STABLE, "stable_source_sha256"),
                        (t.PROTOCOL, "protocol_sha256"), (t.BASE_MANIFEST, "baseline_manifest_sha256"),
                        (t.BASE_DRAWS, "baseline_schedule_file_sha256")):
        if t.sha(path) != meta[field]:
            raise ValueError(f"Source or frozen input changed: {path}")
    if t.sha(run / "metadata.json") != receipt["metadata_sha256"]:
        raise ValueError("Metadata changed")
    if t.sha(run / "draw_schedule.json") != meta["draw_schedule_file_sha256"]:
        raise ValueError("Draw schedule changed")
    manifest_path = Path(meta["manifest"])
    if t.sha(manifest_path) != meta["manifest_sha256"]:
        raise ValueError("Manifest changed")
    t.set_branch(meta["config"]["branch"])
    _, scenes, counts, positives = t.v3.checked_scenes(manifest_path)
    if counts != meta["split_counts"] or positives != meta["positive_counts"]:
        raise ValueError("Source/array identity or split counts changed")
    draws = json.loads((run / "draw_schedule.json").read_text(encoding="utf-8"))
    if len(draws) != t.ARMS[arm] * 8 or t.draw_counts(draws, scenes) != meta["draw_domain_class_counts"]:
        raise ValueError("Draw count or domain/class allocation changed")
    original = t._baseline_draws()
    if arm in ("us_original_norm", "us_combined_norm") and draws != original:
        raise ValueError("US control draw sequence changed")
    if arm in ("balanced_domains", "balanced_full_scene"):
        if [r for i, r in enumerate(draws) if i % 8 < 4] != original:
            raise ValueError("Baseline US draw order was not preserved")
        if any([r[4] for r in draws[i:i+8]] != [True, False, True, False]*2
               for i in range(0, len(draws), 8)):
            raise ValueError("Balanced batch class pattern changed")
    if arm == "expanded_long":
        prior = json.loads((t.ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/"
                            "diverse_training_log_v3_fit/draw_schedule.json").read_text(encoding="utf-8"))
        if draws[:len(prior)] != prior:
            raise ValueError("Expanded first 2400 updates changed")
    if arm == "balanced_full_scene":
        sibling = run.parent / "balanced_domains" / "draw_schedule.json"
        if sibling.exists() and draws != json.loads(sibling.read_text(encoding="utf-8")):
            raise ValueError("Full-scene arm changed the balanced scene/D4 schedule")
    if arm == "us_original_norm":
        base = json.loads((t.ROOT / "outputs/GTM_Model7_dataset_repair_20261024/"
                           "mars_clear_training_log_v3/metadata.json").read_text(encoding="utf-8"))
        if meta["normalization"] != base["normalization"]:
            raise ValueError("Original normalizer changed")
    dev = [s for s in scenes if s["split"] == "dev"]
    model_results = {}
    for name in ("stable_residual", "siamese"):
        report = receipt["models"][name]
        maximum = 0.0
        checked = 0
        for step, milestone in report["milestones"].items():
            checkpoint = run / name / f"step_{step}.pt"
            if t.sha(checkpoint) != milestone["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["normalization"] != meta["normalization"] or saved["steps"] != int(step):
                raise ValueError("Checkpoint configuration changed")
            torch.manual_seed(t.SEED)
            model = t.new_model(name)
            model.load_state_dict(saved["model"])
            model.eval()
            for scene in dev:
                data = t.load_scene(manifest_path, scene)
                probability = t.predict(model, t.prepare(data, meta["normalization"]))
                file = run / name / f"dev_step_{step}" / f"{scene['id']}.npz"
                expected = milestone["dev"]["scenes"][scene["id"]]
                if t.sha(file) != expected["probability_sha256"]:
                    raise ValueError("Saved DEV probability changed")
                with np.load(file, allow_pickle=False) as z:
                    error = float(np.max(np.abs(probability - z["probability20"])))
                    if error > t.REPLAY_ATOL or not np.array_equal(data["valid"].astype(bool), z["valid20"]):
                        raise ValueError("Fresh CPU probability replay differs")
                    maximum = max(maximum, error)
                native = np.repeat(np.repeat(probability, t.native_factor(), 0), t.native_factor(), 1)
                if t.confusion(native, data["label_native"], data["valid_native"].astype(bool)) != expected["native_confusion"]:
                    raise ValueError("Fresh native confusion differs")
                checked += 1
        model_results[name] = {"milestones": len(report["milestones"]),
                               "dev_scene_replays": checked, "maximum_probability_error": maximum}
    result = {"schema": "GTM_Model7_domain_v4_e2e", "passed": True,
              "arm": arm, "manifest_sha256": meta["manifest_sha256"],
              "trainer_sha256": meta["trainer_sha256"], "metadata_sha256": t.sha(run / "metadata.json"),
              "draw_domain_class_counts": meta["draw_domain_class_counts"],
              "source_and_array_identity": "verified", "fresh_process": True,
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
