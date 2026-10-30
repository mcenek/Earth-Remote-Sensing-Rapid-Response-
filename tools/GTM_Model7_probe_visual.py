"""Reproduce the fixed four-crop TRAIN memorization probe for visual review."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SOURCE = ROOT / "tools/GTM_Model7_recover_training.py"
spec = importlib.util.spec_from_file_location("recovery", SOURCE)
r = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(r)
old = r.old
OUT = ROOT / "outputs/GTM_Model7_recovery_20261023/probe_visual"


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing populated output: {OUT}")
    _, scenes = old.load_manifest(r.MANIFEST)
    norm = old.normalization(r.MANIFEST, scenes)
    draws = old.schedule(r.MANIFEST, scenes, 1)[0][:4]
    sample = old.batch(r.MANIFEST, {s["id"]: s for s in scenes}, draws, norm, {})
    x, y, valid, _ = sample
    OUT.mkdir(parents=True)
    result = {"schema": "GTM_Model7_train_probe_visual_v1",
              "description": "TRAIN CROP MEMORIZATION ONLY; no held-out evidence",
              "draws": draws, "steps": 300, "seed": old.BASE["seed"],
              "manifest_sha256": old.sha(r.MANIFEST),
              "trainer_sha256": old.sha(r.ORIGINAL),
              "recovery_sha256": old.sha(SOURCE),
              "source_sha256": old.sha(HERE), "models": {}}
    for name in ("residual", "siamese"):
        torch.manual_seed(old.BASE["seed"])
        model = old.new_model(name)
        opt = r.optimizer(model)
        model.eval()
        with torch.no_grad():
            before = old.array(torch.sigmoid(model(x))[:, 0])
        for _ in range(300):
            loss, _ = r.update(model, opt, sample, 1, True)
        model.eval()
        with torch.no_grad():
            after = old.array(torch.sigmoid(model(x))[:, 0])
        file = OUT / f"{name}_probe.npz"
        # Array order is NCHW. Channels 0:6 and 6:12 are standardized target/reference.
        np.savez_compressed(file, image12=old.array(x), target=old.array(y[:, 0]),
                            valid=old.array(valid[:, 0]).astype(bool),
                            probability_initial=before, probability_final=after)
        checkpoint = OUT / f"{name}_step300.pt"
        torch.save({"model": model.state_dict(), "normalization": norm,
                    "model_name": name, "steps": 300, "probe_loss": "ordinary masked BCE",
                    "source_sha256": old.sha(HERE)}, checkpoint)
        result["models"][name] = {"loss_final": loss, "array_sha256": old.sha(file),
                                   "checkpoint_sha256": old.sha(checkpoint),
                                   "positive_draw_indices": [0, 2], "negative_draw_indices": [1, 3]}
    old.write_json(OUT / "receipt.json", result)
    print(json.dumps({"probe_visual": "complete", "receipt": str(OUT / "receipt.json")}))


if __name__ == "__main__":
    main()
