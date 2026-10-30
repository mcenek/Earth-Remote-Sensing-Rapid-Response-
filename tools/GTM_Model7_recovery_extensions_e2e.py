"""Fresh-process E2E replay of the train-only follow-up experiments."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


e2e = load("recovery_e2e", ROOT / "tools/GTM_Model7_recovery_e2e.py")
r = e2e.r
old = r.old
follow = load("recovery_followup", ROOT / "tools/GTM_Model7_recovery_followup.py")
hard = load("recovery_hard_negative", ROOT / "tools/GTM_Model7_hard_negative.py")


def main():
    _, scenes = old.load_manifest(r.MANIFEST)  # test inputs only
    checks = []
    for kind, module in (("stability_followup", follow), ("hard_negative", hard)):
        out = module.OUT
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        hashes = receipt["hashes"] if kind == "stability_followup" else {
            "manifest": receipt["manifest_sha256"],
            "original_trainer": receipt["original_trainer_sha256"],
            "first_recovery": receipt["recovery_source_sha256"],
            "followup": receipt["source_sha256"],
            "protocol": receipt["protocol_sha256"]}
        assert hashes["manifest"] == old.sha(r.MANIFEST)
        assert hashes["original_trainer"] == old.sha(r.ORIGINAL)
        assert hashes["first_recovery"] == old.sha(r.HERE)
        assert hashes["followup"] == old.sha(module.HERE)
        protocol = (ROOT / "docs/GTM_Model7_recovery_followup.md" if kind == "stability_followup"
                    else ROOT / "docs/GTM_Model7_hard_negative_protocol.md")
        assert hashes["protocol"] == old.sha(protocol)
        if kind == "hard_negative":
            schedule = hard.hard_schedule(r.MANIFEST, scenes, 300)
            assert old.schedule_hash(schedule) == receipt["schedule_sha256"]
            assert old.schedule_hash(hard.ORIGINAL_SCHEDULE(r.MANIFEST, scenes, 300)) == receipt["baseline_schedule_sha256"]
        for weight in (2, 3):
            arm = f"siamese_w{weight}"
            checked = e2e.verify_checkpoint(out, arm, 1800, scenes, receipt["normalization"])
            checks.append({"experiment": kind, **checked})
            if kind == "hard_negative":
                dev = receipt["arms"][str(weight)]["dev_predictions"]
                folder = out / arm / "dev_predictions_step_1800"
                for scene in scenes:
                    if scene["split"] != "dev":
                        continue
                    file = folder / f"{scene['id']}.npz"
                    assert old.sha(file) == dev["scenes"][scene["id"]]["prediction_sha256"]
                    with np.load(file, allow_pickle=False) as z:
                        p, valid = z["probability20"], z["valid20"]
                    assert p.shape == (100, 100) and np.isfinite(p).all()
                    assert ((p >= 0) & (p <= 1)).all() and valid.any()
        assert receipt["test_labels_read"] is False
    artifact = {"schema": "GTM_Model7_recovery_extensions_e2e_v1", "passed": True,
                "checks": checks, "test_labels_read": False,
                "original_trainer_sha256": old.sha(r.ORIGINAL)}
    target = ROOT / "outputs/GTM_Model7_recovery_20261023/extensions_e2e_receipt.json"
    old.write_json(target, artifact)
    print(json.dumps({"extensions_e2e": "passed", "artifact": str(target)}))


if __name__ == "__main__":
    main()
