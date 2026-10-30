"""Fresh-process E2E verification for train-only temporal recovery."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
RECOVERY = ROOT / "tools/GTM_Model7_recover_training.py"
spec = importlib.util.spec_from_file_location("temporal_recovery", RECOVERY)
r = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r)
old = r.old


def verify_checkpoint(out: Path, arm: str, step: int, scenes: list[dict], norm: dict) -> dict:
    folder = out / arm
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    checkpoint = folder / f"step_{step}.pt"
    recorded = receipt["checkpoints"][str(step)]
    assert old.sha(checkpoint) == recorded["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["normalization"] == norm
    assert state["original_source_sha256"] == old.sha(r.ORIGINAL)
    assert state["recovery_source_sha256"] == old.sha(RECOVERY)
    model = old.new_model(state["model_name"])
    model.load_state_dict(state["model"])
    pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    count = 0
    prediction_dir = folder / f"train_predictions_step_{step}"
    for scene in scenes:
        if scene["split"] != "train":
            continue
        data = old.load_scene(r.MANIFEST, scene, labels=True)
        valid = data["valid"].astype(bool)
        actual = old.infer(model, old.prepare(data, norm), valid)
        file = prediction_dir / f"{scene['id']}.npz"
        assert old.sha(file) == recorded["train"]["scenes"][scene["id"]]["prediction_sha256"]
        with np.load(file, allow_pickle=False) as z:
            expected, saved_valid = z["probability20"], z["valid20"]
        assert expected.shape == (100, 100) and np.array_equal(valid, saved_valid)
        assert np.isfinite(expected).all() and ((expected >= 0) & (expected <= 1)).all()
        assert np.allclose(actual, expected, rtol=0, atol=1e-6)
        confusion = old.confusion(actual, data["target"], valid)
        assert all(confusion[key] == recorded["train"]["scenes"][scene["id"]][key] for key in pooled)
        for key in pooled:
            pooled[key] += confusion[key]
        count += 1
    assert pooled == recorded["train"]["confusion"]
    dev = r.evaluate(model, scenes, norm, "dev")
    assert dev["confusion"] == recorded["dev"]["confusion"]
    assert count == recorded["train"]["scene_count"]
    return {"arm": arm, "step": step, "train_scenes": count,
            "train_confusion": pooled, "dev_confusion": dev["confusion"],
            "checkpoint_sha256": old.sha(checkpoint)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=r.DEFAULT_OUT)
    args = parser.parse_args()
    out = args.run.resolve()
    receipt = json.loads((out / "recovery_receipt.json").read_text(encoding="utf-8"))
    assert receipt["manifest_sha256"] == old.sha(r.MANIFEST)
    assert receipt["original_source_sha256"] == old.sha(r.ORIGINAL)
    assert receipt["recovery_source_sha256"] == old.sha(RECOVERY)
    assert receipt["protocol_sha256"] == old.sha(ROOT / "docs/GTM_Model7_recovery_protocol.md")
    assert receipt["test_selected"] is False
    _, scenes = old.load_manifest(r.MANIFEST)  # never opens test label keys
    assert receipt["splits"] == {split: sum(s["split"] == split for s in scenes)
                                  for split in ("train", "dev", "test")}
    checks = []
    for weight in (1, 10):
        for step in (360, 1800):
            arm = f"residual_w{weight}"
            if step == receipt["selected_steps"] and weight == receipt["selected_positive_weight"]:
                checks.append(verify_checkpoint(out, arm, step, scenes, receipt["normalization"]))
            else:
                path = out / arm / f"step_{step}.pt"
                assert old.sha(path) == receipt["residual_arms"][str(weight)]["checkpoints"][str(step)]["checkpoint_sha256"]
    checks.append(verify_checkpoint(out, f"siamese_w{receipt['selected_positive_weight']}",
                                    receipt["selected_steps"], scenes, receipt["normalization"]))
    artifact = {"schema": "GTM_Model7_recovery_e2e_v1", "passed": True,
                "manifest_sha256": receipt["manifest_sha256"],
                "original_source_sha256": receipt["original_source_sha256"],
                "recovery_source_sha256": receipt["recovery_source_sha256"],
                "checks": checks, "test_labels_read": False}
    old.write_json(out / "e2e_receipt.json", artifact)
    print(json.dumps({"e2e": "passed", "artifact": str(out / "e2e_receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
