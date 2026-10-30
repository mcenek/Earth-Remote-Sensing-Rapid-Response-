"""Apply the frozen image-only MethaneS2CM reference rule to a new cohort copy.

See docs/GTM_Model7_s2cm_reference_repair_protocol.md. The rule was written
before this script accesses the internal TEST images. No model outputs are read.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/s2cm_data"
DEFAULT = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/s2cm_reference_repair_data"
PROTOCOL = ROOT / "docs/GTM_Model7_s2cm_reference_repair_protocol.md"
BASE_SHA = "4cb3fc6d0e2ad547c3951d663193682ce00455c189aedc1f47f92bf81de9f0a5"
INDICES = (1, 2, 3, 7, 10, 11)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def visible_median(image: np.ndarray) -> float:
    if image.shape != (32, 32, 6):
        raise ValueError("Unexpected prepared image shape")
    return float(np.median(image[..., :3]))


def bright(reference_median: float, target_median: float) -> bool:
    return reference_median > 4500 and reference_median > 2 * target_median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    baseline, dest = args.baseline.resolve(), args.output.resolve()
    if dest.exists():
        raise FileExistsError(f"Immutable reference repair output exists: {dest}")
    base_manifest_path = baseline / "manifest.json"
    if sha(base_manifest_path) != BASE_SHA:
        raise ValueError("Baseline cohort changed after rule freeze")
    baseline_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest.get("source_dataset") != "H1deaki/MethaneS2CM":
        raise ValueError("Wrong baseline source")
    h5_path = Path(baseline_manifest["source_h5"]["path"])
    h5_sha = baseline_manifest["source_h5"]["sha256"]
    if sha(h5_path) != h5_sha:
        raise ValueError("Packed TRAIN HDF5 changed")
    scenes = baseline_manifest["scenes"]
    decisions: dict[str, dict] = {}
    test_image_reads = test_alternative_reads = 0
    with h5py.File(h5_path, "r") as source:
        if source.attrs.get("source_split") != "l2a_location_split_32x32/train.csv":
            raise ValueError("Packed source is not official TRAIN")
        indices = {str(value): index for index, value in enumerate(source["sample_id"][:])}
        for scene in scenes:
            path = baseline / scene["array_file"]
            if sha(path) != scene["array_sha256"]:
                raise ValueError(f"Baseline scene array changed: {scene['id']}")
            # Load imagery alone for the frozen decision. Source labels are not consulted.
            with np.load(path, allow_pickle=False) as old:
                target = old["target_image"]
                reference90 = old["reference_image"]
            if scene["split"] == "test":
                test_image_reads += 1
            target_median = visible_median(target)
            reference90_median = visible_median(reference90)
            record = {"reference_dataset": "reference90", "target_visible_median_dn": target_median,
                      "reference90_visible_median_dn": reference90_median,
                      "reference90_bright_shift": bright(reference90_median, target_median),
                      "reference365_visible_median_dn": None,
                      "reference365_support_fraction": None,
                      "reference365_bright_shift": None,
                      "unresolved_reference_quality": False}
            if record["reference90_bright_shift"]:
                index = indices[scene["source_row"]["id"]]
                raw365 = source["reference365"][index, INDICES, :, :]
                reference365 = np.moveaxis(raw365, 0, -1).astype(np.float32)
                if scene["split"] == "test":
                    test_alternative_reads += 1
                median365 = visible_median(reference365)
                valid = np.isfinite(target).all(-1) & np.isfinite(reference365).all(-1) & (target != 0).all(-1) & (reference365 != 0).all(-1)
                support = float(valid.mean())
                record.update(reference365_visible_median_dn=median365,
                              reference365_support_fraction=support,
                              reference365_bright_shift=bright(median365, target_median))
                if not record["reference365_bright_shift"] and support >= .90:
                    raw_target = source["target"][index, INDICES, :, :]
                    record.update(reference_dataset="reference365", reference_image=reference365,
                                  valid=valid, input_pair_sha256=hashlib.sha256(raw_target.tobytes() + raw365.tobytes()).hexdigest())
                else:
                    record["unresolved_reference_quality"] = True
            decisions[scene["id"]] = record
    # Check duplicate split identities after the choice, before writing any output.
    pairs = collections.defaultdict(set)
    for scene in scenes:
        record = decisions[scene["id"]]
        pairs[record.get("input_pair_sha256", scene["input_pair_sha256"])].add(scene["split"])
    if any(len(splits) > 1 for splits in pairs.values()):
        raise ValueError("Frozen reference choice created a cross-split identical input pair")
    dest.mkdir(parents=True)
    new_scenes = []
    choice_counts = {split: {"reference90": 0, "reference365": 0, "unresolved_bright90": 0}
                     for split in ("train", "dev", "test")}
    changed_replay = 0
    with h5py.File(h5_path, "r") as source:
        indices = {str(value): index for index, value in enumerate(source["sample_id"][:])}
        for scene in scenes:
            record = decisions[scene["id"]]
            chosen = record["reference_dataset"]
            source_file = baseline / scene["array_file"]
            destination_file = dest / scene["array_file"]
            updated = dict(scene)
            updated["reference_dataset"] = chosen
            updated["reference_quality"] = "image_only_bright_shift_repaired_with_reference365" if chosen == "reference365" else (
                "image_only_bright_shift_unresolved" if record["unresolved_reference_quality"] else "no_strict_bright_shift_detected; cloud_status_unknown")
            updated["reference_selection"] = {key: value for key, value in record.items()
                                              if key not in ("reference_image", "valid", "input_pair_sha256")}
            choice_counts[scene["split"]][chosen] += 1
            if record["unresolved_reference_quality"]:
                choice_counts[scene["split"]]["unresolved_bright90"] += 1
            if chosen == "reference365":
                # Copy existing label arrays unchanged. They did not affect the reference choice.
                with np.load(source_file, allow_pickle=False) as old:
                    arrays = {key: old[key] for key in old.files}
                arrays["reference_image"] = record["reference_image"]
                arrays["valid"] = record["valid"]
                arrays["valid_native"] = record["valid"].copy()
                np.savez_compressed(destination_file, **arrays)
                updated["input_pair_sha256"] = record["input_pair_sha256"]
                updated["valid_fraction"] = record["reference365_support_fraction"]
                index = indices[scene["source_row"]["id"]]
                expected_target = np.moveaxis(source["target"][index, INDICES, :, :], 0, -1).astype(np.float32)
                expected_reference = np.moveaxis(source["reference365"][index, INDICES, :, :], 0, -1).astype(np.float32)
                with np.load(destination_file, allow_pickle=False) as actual:
                    if (not np.array_equal(actual["target_image"], expected_target) or
                            not np.array_equal(actual["reference_image"], expected_reference) or
                            not np.array_equal(actual["valid"], record["valid"]) or
                            not np.array_equal(actual["valid_native"], record["valid"])):
                        raise ValueError(f"Source replay mismatch: {scene['id']}")
                changed_replay += 1
            else:
                shutil.copyfile(source_file, destination_file)
            updated["array_sha256"] = sha(destination_file)
            new_scenes.append(updated)
    manifest = {**baseline_manifest, "cohort_name": "GTM_Model7_s2cm_reference_repair_20261024",
                "reference_repair_protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha(PROTOCOL)},
                "source_baseline_manifest": {"path": str(base_manifest_path), "sha256": BASE_SHA},
                "reference_selection_rule": "default reference90; choose reference365 only when reference90 visible median DN >4500 and >2x target visible median, reference365 does not pass that gate, and joint six-band target/reference365 support >=90%; image-only; fixed before TEST image access",
                "reference_choices": choice_counts,
                "test_image_access": {"baseline_test_target_and_reference90_arrays_read_for_fixed_QA": test_image_reads,
                                      "test_reference365_arrays_read_only_after_bright_gate": test_alternative_reads,
                                      "test_labels_used_for_reference_choice": False,
                                      "test_model_outputs_read": False},
                "quality_limitations": baseline_manifest["quality_limitations"] + [
                    "Image-only reference choice does not prove cloud clearance or plume-free background.",
                    "The fixed reference rule was applied to TEST imagery after freeze, so TEST input imagery is no longer image-unseen."],
                "scenes": new_scenes}
    save(dest / "manifest.json", manifest)
    receipt = {"schema": "GTM_Model7_s2cm_reference_repair_receipt_v1", "baseline_manifest_sha256": BASE_SHA,
               "manifest_sha256": sha(dest / "manifest.json"), "protocol_sha256": sha(PROTOCOL),
               "exporter_sha256": sha(Path(__file__)), "source_h5_sha256": h5_sha,
               "scene_count": len(new_scenes), "reference_choices": choice_counts,
               "changed_source_arrays_replayed": changed_replay,
               "test_image_access": manifest["test_image_access"], "new_downloads": 0}
    save(dest / "receipt.json", receipt)
    print(json.dumps({"directory": str(dest), "manifest_sha256": receipt["manifest_sha256"],
                      "scene_count": len(new_scenes), "reference_choices": choice_counts,
                      "changed_source_arrays_replayed": changed_replay,
                      "test_image_access": receipt["test_image_access"]}, indent=2))


if __name__ == "__main__":
    main()
