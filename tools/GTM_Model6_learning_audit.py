"""Read-only model replay and random-initialization controls for a synthetic run.

Writes new evidence arrays and a JSON receipt. Does not fit or change weights.
Acceptance and failure cases are in the learning-review PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/GTM_Model6"))
import GTM_Model6_three_cnn_data as data
import GTM_Model6_three_cnn_runtime as rt
import GTM_Model6_three_cnn_model as networks


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rates(c):
    c = dict(c)
    for name, numerator, denominator in (
        ("iou", c["tp"], c["tp"] + c["fp"] + c["fn"]),
        ("precision", c["tp"], c["tp"] + c["fp"]),
        ("recall", c["tp"], c["tp"] + c["fn"]),
        ("pixel_false_positive_rate", c["fp"], c["fp"] + c["tn"]),
    ):
        c[name] = numerator / denominator if denominator else None
    return c


def total(rows, key):
    return rates({k: sum(row[key][k] for row in rows) for k in ("tp", "fp", "fn", "tn")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new audit output directory")
    manifest = data.load_manifest(args.manifest)
    if manifest["data_kind"] != "synthetic":
        raise ValueError("This audit is explicitly limited to synthetic engineering evidence")
    args.output.mkdir(parents=True)
    rows = manifest["observations"]
    cohort = Counter()
    for row in rows:
        a = data.load_arrays(row)
        positive = bool(np.any(a["target"][a["target_valid"].astype(bool)] == 1))
        cohort[(row["target_kind"], row["role"], "positive" if positive else "negative")] += 1
    result = {
        "schema": "GTM_Model6_learning_audit_v1", "data_kind": "synthetic",
        "claim": "Artificial spectral-cue learning and software evidence only; no methane detection accuracy",
        "manifest_sha256": manifest["manifest_sha256"], "source_hashes": rt._source_hashes(),
        "cohort": {"/".join(k): v for k, v in sorted(cohort.items())},
        "runtime": rt.runtime_info(), "stages": {}, "invocation": sys.argv,
        "random_control": "Same seeded architecture and fitted normalization before loading trained weights; CNN 3 already has trained frozen branches",
    }
    tf = rt._tf()
    for stage in ("airborne", "emit", "fusion"):
        path = rt._receipt_path(args.run_dir, stage)
        receipt = json.loads(path.read_text(encoding="utf-8"))
        config = receipt["config"]
        receipt = rt._read_parent(args.run_dir, stage, manifest, config, result["source_hashes"])
        parents = {s: rt._read_parent(args.run_dir, s, manifest, config, result["source_hashes"])
                   for s in ("airborne", "emit")} if stage == "fusion" else {}
        for parent in parents:
            if receipt["parents"][parent]["receipt_sha256"] != rt._hash(rt._receipt_path(args.run_dir, parent)):
                raise ValueError("Changed frozen-branch receipt")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config["seed"])
        model = rt._build(stage, config, args.run_dir, parents)
        initial_hash = rt._weights_hash(model)
        frozen_before = {s: rt._weights_hash(model.get_layer(s)) for s in parents}
        kind = "emit" if stage == "fusion" else stage
        tests = [r for r in rows if r["role"] == "test" and r["target_kind"] == kind]
        validation = rt._validation_rows(manifest, stage)
        norms = receipt["normalization"]
        untrained = {}
        for row in tests:
            a = data.load_arrays(row)
            untrained[row["id"]] = rt._tiled(model, stage, a["image"], a["input_valid"].astype(bool),
                                             norms, config["tile_size"], config["halo"])
        initial_loss = float(np.mean([rt._validation_loss(model, stage, data.load_arrays(r), norms, config)
                                      for r in validation]))
        model.load_weights(str(args.run_dir / receipt["checkpoint"]["path"]))
        trained_hash = rt._weights_hash(model)
        if initial_hash == trained_hash:
            raise ValueError("Trained model is identical to its random initialization")
        if any(rt._weights_hash(model.get_layer(s)) != frozen_before[s] for s in parents):
            raise ValueError("Frozen feature branch changed")
        fitting_examples = []
        for desired_positive in (True, False):
            for row in rt._fit_rows(manifest, stage):
                a = data.load_arrays(row)
                is_positive = bool(np.any(a["target"][a["target_valid"].astype(bool)] == 1))
                if is_positive != desired_positive:
                    continue
                valid = a["input_valid"].astype(bool)
                p = rt._tiled(model, stage, a["image"], valid, norms, config["tile_size"], config["halo"])
                filename = f"{stage}_fit_{'positive' if is_positive else 'negative'}.npz"
                payload = dict(a, probability=p)
                if stage == "fusion":
                    na, h, w = rt._pad(rt._normalized(a["image"], valid, norms["airborne"]))
                    ne, _, _ = rt._pad(rt._normalized(a["image"], valid, norms["emit"]))
                    fa, _ = model.get_layer("airborne")(tf.convert_to_tensor(na[None]), training=False)
                    fe, _ = model.get_layer("emit")(tf.convert_to_tensor(ne[None]), training=False)
                    payload.update(airborne_features=np.asarray(fa)[0, :h, :w],
                                   emit_features=np.asarray(fe)[0, :h, :w])
                np.savez_compressed(args.output / filename, **payload)
                fitting_examples.append(dict(id=row["id"], role=row["role"], positive=is_positive,
                                             file=filename, input_sha256=row["input"]["sha256"]))
                break
        saved_result = json.loads((args.run_dir / f"GTM_Model6_three_cnn_{stage}_results.json").read_text())
        saved = {r["id"]: r for r in saved_result["records"]}
        records = []
        arrays_to_keep = {}
        max_replay = 0.0
        for row in tests:
            a = data.load_arrays(row)
            valid = a["input_valid"].astype(bool)
            p = rt._tiled(model, stage, a["image"], valid, norms, config["tile_size"], config["halo"])
            native, complete = networks.native_probability(p, valid, a["target"].shape)
            before, _ = networks.native_probability(untrained[row["id"]], valid, a["target"].shape)
            support = complete & a["target_valid"].astype(bool)
            rec = saved[row["id"]]
            prediction = args.run_dir / rec["prediction"]["path"]
            if rt._hash(prediction) != rec["prediction"]["sha256"]:
                raise ValueError("Saved prediction hash changed")
            with np.load(prediction, allow_pickle=False) as z:
                diff = float(np.max(np.abs(p[valid] - z["probability"][valid])))
            max_replay = max(max_replay, diff)
            if diff > 1e-5:
                raise ValueError(f"Checkpoint replay differs: {row['id']} ({diff})")
            measured = rt._metrics(native, a["target"], support, config["threshold"])
            if any(measured[k] != rec["native_metrics"][k] for k in ("tp", "fp", "fn", "tn")):
                raise ValueError("Recomputed counts disagree with saved run")
            positive = bool(np.any(a["target"][support] == 1))
            item = dict(id=row["id"], positive=positive, group=row["group"], trained=measured,
                        untrained=rt._metrics(before, a["target"], support, config["threshold"]),
                        always_negative=rt._metrics(np.zeros_like(native), a["target"], support, .5),
                        always_positive=rt._metrics(np.ones_like(native), a["target"], support, .5),
                        probability_max=float(np.nanmax(native[support])), replay_max_abs=diff,
                        fine_threshold_cells=int(np.sum(valid & (p >= config["threshold"]))),
                        fine_supported_cells=int(valid.sum()))
            records.append(item)
            arrays_to_keep[row["id"]] = dict(image=a["image"], input_valid=a["input_valid"],
                target=a["target"], target_valid=a["target_valid"], native_probability=native,
                native_untrained=before, probability=p, untrained=untrained[row["id"]])
        positives = sorted([r for r in records if r["positive"]], key=lambda r: r["trained"]["iou"])
        negatives = sorted([r for r in records if not r["positive"]], key=lambda r: r["trained"]["fp"], reverse=True)
        selections = [("worst_positive", positives[0]), ("median_positive", positives[len(positives)//2]),
                      ("best_positive", positives[-1]), ("worst_negative", negatives[0]),
                      ("median_negative", negatives[len(negatives)//2])]
        for label, rec in selections:
            name = f"{stage}_{label}.npz"
            np.savez_compressed(args.output / name, **arrays_to_keep[rec["id"]])
            rec.setdefault("evidence_files", []).append(name)
        final_loss = float(np.mean([rt._validation_loss(model, stage, data.load_arrays(r), norms, config)
                                    for r in validation]))
        report = dict(history=receipt["history"], config=config, completed_epochs=receipt["completed_epochs"],
            requested_epochs=receipt["requested_epochs"], elapsed_seconds=receipt["elapsed_seconds"],
            initial_validation_loss=initial_loss, trained_validation_loss=final_loss,
            initial_weight_hash=initial_hash, trained_weight_hash=trained_hash,
            checkpoint_sha256=receipt["checkpoint"]["sha256"],
            frozen_weights_unchanged=receipt["frozen_weights_unchanged"], replay_max_abs=max_replay,
            test_scenes=len(records), positive_scenes=len(positives), negative_scenes=len(negatives),
            trained=total(records, "trained"), untrained=total(records, "untrained"),
            always_negative=total(records, "always_negative"), always_positive=total(records, "always_positive"),
            negative_scenes_with_any_false_activation=sum(r["trained"]["fp"] > 0 for r in negatives),
            negative_false_positive_cells=sum(r["trained"]["fp"] for r in negatives),
            negative_evaluated_cells=sum(r["trained"]["evaluated_cells"] for r in negatives),
            negative_scenes_with_any_fine_activation=sum(r["fine_threshold_cells"] > 0 for r in negatives),
            negative_fine_threshold_cells=sum(r["fine_threshold_cells"] for r in negatives),
            negative_fine_supported_cells=sum(r["fine_supported_cells"] for r in negatives),
            fitting_examples=fitting_examples,
            selections={label: rec["id"] for label, rec in selections}, records=records)
        result["stages"][stage] = report
        print(json.dumps({"stage": stage, "trained": report["trained"], "untrained": report["untrained"],
                          "negative_scenes_with_any_false_activation": report["negative_scenes_with_any_false_activation"]}), flush=True)
        if stage == "fusion":
            selected = positives[len(positives)//2]
            row = next(r for r in tests if r["id"] == selected["id"])
            a = data.load_arrays(row)
            valid = a["input_valid"].astype(bool)
            na, h, w = rt._pad(rt._normalized(a["image"], valid, norms["airborne"]))
            ne, _, _ = rt._pad(rt._normalized(a["image"], valid, norms["emit"]))
            inputs = [tf.convert_to_tensor(na[None]), tf.convert_to_tensor(ne[None])]
            fa, la = model.get_layer("airborne")(inputs[0], training=False)
            fe, le = model.get_layer("emit")(inputs[1], training=False)
            probe = tf.keras.Model(model.inputs, [model.get_layer("frozen_feature_join").output,
                                                  model.get_layer("fusion_decoder_act1").output,
                                                  model.output])
            joined, decoded, logits = probe(inputs, training=False)
            to_array = lambda x: np.asarray(x)[0, :h, :w]
            direct = np.asarray(tf.math.sigmoid(logits))[0, :h, :w, 0]
            payload = dict(image=a["image"], input_valid=a["input_valid"], target=a["target"],
                target_valid=a["target_valid"], normalized_airborne=na[:h, :w], normalized_emit=ne[:h, :w],
                airborne_features=to_array(fa), emit_features=to_array(fe), joined_features=to_array(joined),
                fusion_features=to_array(decoded), airborne_score=to_array(tf.math.sigmoid(la))[..., 0],
                emit_score=to_array(tf.math.sigmoid(le))[..., 0], fusion_probability=direct,
                saved_probability=arrays_to_keep[selected["id"]]["probability"])
            np.savez_compressed(args.output / "feed_forward.npz", **payload)
            result["feed_forward"] = {"id": selected["id"], "selection": "median CNN3 test-plume IoU",
                "shapes": {k: list(v.shape) for k, v in payload.items()},
                "full_scene_vs_tiled_max_abs": float(np.max(np.abs(direct[valid]-payload["saved_probability"][valid]))),
                "note": "CNN1/2 spatial features, not their masks, feed CNN3. Each branch uses its own fitted normalization. References never enter the forward pass."}
    write_json(args.output / "audit.json", result)
    print(str(args.output / "audit.json"), flush=True)


if __name__ == "__main__":
    main()
