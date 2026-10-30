"""Fresh-process E2E replay of balanced v2 train/development artifacts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
TRAINER = HERE.with_name("GTM_Model7_train_repaired_balanced_v2.py")
spec = importlib.util.spec_from_file_location("repaired_trainer", TRAINER)
trainer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trainer)


def replay_probe(folder: Path, report: dict, norm: dict):
    assert trainer.sha(folder / "initial.pt") == report["checkpoint_sha256"]["initial"]
    assert trainer.sha(folder / "final.pt") == report["checkpoint_sha256"]["final"]
    assert trainer.sha(folder / "probe_arrays.npz") == report["arrays_sha256"]
    state = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
    assert state["normalization"] == norm
    model = trainer.new_model(report["model"])
    model.load_state_dict(state["model"])
    with np.load(folder / "probe_arrays.npz", allow_pickle=False) as z:
        image = z["image12"]
        target = z["target"]
        valid = z["valid"]
        expected = z["probability_final"]
    model.eval()
    with torch.no_grad():
        actual = torch.sigmoid(model(torch.from_numpy(image).float()))[:, 0].cpu().numpy()
    assert actual.shape == expected.shape == (4, trainer.crop_size(), trainer.crop_size())
    assert np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
    error = float(np.max(np.abs(actual - expected)))
    assert error <= trainer.REPLAY_ATOL
    positive = trainer.metrics(trainer.confusion(actual[[0, 2]], target[[0, 2]], valid[[0, 2]]))
    negative = trainer.metrics(trainer.confusion(actual[[1, 3]], target[[1, 3]], valid[[1, 3]]))
    assert positive["confusion"] == report["positive"]["confusion"]
    assert negative["confusion"] == report["negative"]["confusion"]
    return {"model": report["model"], "max_abs_error": error,
            "positive_confusion": positive["confusion"],
            "negative_confusion": negative["confusion"]}


def replay_milestone(folder: Path, model_name: str, step: int, row: dict,
                     manifest_path: Path, scenes: list[dict], norm: dict):
    path = folder / f"step_{step}.pt"
    assert trainer.sha(path) == row["checkpoint_sha256"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["normalization"] == norm and state["steps"] == step
    model = trainer.new_model(model_name)
    model.load_state_dict(state["model"])
    errors = []
    count = 0
    for split in ("train", "dev"):
        pooled = {key: 0 for key in ("tp", "fp", "fn", "tn")}
        for scene in scenes:
            if scene["split"] != split:
                continue
            data = trainer.load_scene(manifest_path, scene)
            valid = data["valid"].astype(bool)
            actual = trainer.predict(model, trainer.prepare(data, norm))
            file = folder / f"{split}_step_{step}" / f"{scene['id']}.npz"
            record = row[split]["scenes"][scene["id"]]
            assert trainer.sha(file) == record["probability_sha256"]
            with np.load(file, allow_pickle=False) as z:
                expected, saved_valid = z["probability20"], z["valid20"]
                native, native_valid = z["probability_native"], z["valid_native"]
            assert expected.shape == (trainer.side(), trainer.side()) and np.array_equal(saved_valid, valid)
            factor = trainer.native_factor()
            assert native.shape == (trainer.side() * factor, trainer.side() * factor)
            assert np.array_equal(native_valid, data["valid_native"].astype(bool))
            assert np.array_equal(native, np.repeat(np.repeat(expected, factor, 0), factor, 1))
            assert np.isfinite(expected).all() and ((expected >= 0) & (expected <= 1)).all()
            error = float(np.max(np.abs(actual - expected)))
            assert error <= trainer.REPLAY_ATOL
            errors.append(error)
            c = trainer.confusion(actual, data["target"], valid)
            assert c == record["confusion"]
            native_c = trainer.confusion(np.repeat(np.repeat(actual, factor, 0), factor, 1),
                                         data["label_native"], data["valid_native"].astype(bool))
            assert native_c == record["native_confusion"]
            for key in pooled:
                pooled[key] += c[key]
            count += 1
        assert pooled == row[split]["pooled"]["confusion"]
    return {"model": model_name, "step": step, "scenes": count,
            "max_abs_error": max(errors), "checkpoint_sha256": trainer.sha(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("methanes2cm",), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.branch != "methanes2cm":
        raise ValueError("Balanced v2 replay is restricted to MethaneS2CM")
    trainer.set_branch(args.branch)
    manifest_path, run = args.manifest.resolve(), args.run.resolve()
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    assert trainer.sha(manifest_path) == metadata["manifest_sha256"]
    assert trainer.sha(TRAINER) == metadata["trainer_sha256"]
    assert metadata["trainer_path"] == str(TRAINER)
    assert trainer.sha(trainer.ORIGINAL) == metadata["original_sha256"]
    assert trainer.sha(trainer.STABLE) == metadata["stable_source_sha256"]
    assert trainer.sha(trainer.PROTOCOL) == metadata["protocol_sha256"]
    assert trainer.sha(run / "metadata.json") == receipt["metadata_sha256"]
    _, scenes, counts, positive_counts = trainer.checked_scenes(manifest_path)
    assert counts == metadata["split_counts"] and positive_counts == metadata["positive_counts"]
    assert metadata["config"]["branch"] == args.branch
    norm = trainer.normalization(manifest_path, scenes)
    assert norm == metadata["normalization"]
    draws = trainer.schedule(manifest_path, scenes, trainer.STEPS)
    assert trainer.sha(run / "draw_schedule.json") == metadata["draw_schedule_file_sha256"]
    assert json.loads((run / "draw_schedule.json").read_text(encoding="utf-8")) == [list(d) for d in draws]
    import hashlib
    schedule_hash = hashlib.sha256(json.dumps(draws, separators=(",", ":")).encode()).hexdigest()
    probes = []
    milestones = []
    for name in ("stable_residual", "siamese"):
        probe = receipt["probes"][name]
        assert probe["qualified"]
        probes.append(replay_probe(run / f"probe_{name}", probe, norm))
        model_receipt = receipt["models"][name]
        assert model_receipt["draw_schedule_sha256"] == schedule_hash
        assert trainer.sha(run / name / "initial.pt") == model_receipt["initial_checkpoint_sha256"]
        assert trainer.choose_pareto(model_receipt["milestones"]) == model_receipt["selection"]
        for step in model_receipt["selection"]["pareto_steps"]:
            milestones.append(replay_milestone(run / name, name, step,
                model_receipt["milestones"][str(step)], manifest_path, scenes, norm))
        for step, row in model_receipt["milestones"].items():
            assert trainer.sha(run / name / f"step_{step}.pt") == row["checkpoint_sha256"]
    artifact = {"schema": "GTM_Model7_repaired_balanced_v2_e2e_v1", "passed": True,
                "manifest_sha256": trainer.sha(manifest_path),
                "trainer_sha256": trainer.sha(TRAINER), "probe_replays": probes,
                "pareto_checkpoint_replays": milestones, "test_array_access": "none"}
    trainer.write_json(run / "e2e_receipt.json", artifact)
    print(json.dumps({"e2e": "passed", "artifact": str(run / "e2e_receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
