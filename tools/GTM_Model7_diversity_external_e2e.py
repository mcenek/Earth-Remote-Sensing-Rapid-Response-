"""Independently reconstruct every frozen external array from original MARS TIFFs."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/external_data"
PREPARE = ROOT / "tools/GTM_Model7_prepare_diversity_external.py"
REPLAY = ROOT / "tools/GTM_Model7_verify_cached_diversity_e2e.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


prep = module("external_preparation_contract", PREPARE)
source_replay = module("external_independent_raster_replay", REPLAY)


def verify(dest: Path) -> dict:
    manifest_path = dest / "manifest.json"
    receipt_path = dest / "receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if prep.base.sha(prep.TRAIN) != prep.TRAIN_SHA or receipt["training_manifest_sha256"] != prep.TRAIN_SHA:
        raise ValueError("Frozen combined cohort identity changed")
    if manifest["training_manifest"]["sha256"] != prep.TRAIN_SHA or receipt["manifest"]["sha256"] != prep.base.sha(manifest_path):
        raise ValueError("External manifest identity failed")
    selection_path = Path(manifest["selection"]["path"])
    if prep.base.sha(selection_path) != manifest["selection"]["sha256"] or receipt["selection_sha256"] != prep.base.sha(selection_path):
        raise ValueError("Frozen metadata selection identity failed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = prep.selected_records(selection)
    if [s["id"].removeprefix("MARS_") for s in manifest["scenes"]] != [
            c["sample_id"] for c in candidates if any(s["id"] == "MARS_" + c["sample_id"] for s in manifest["scenes"])]:
        raise ValueError("External scene order differs from frozen selection")
    original, pinned = prep.load_inventory()
    train = json.loads(prep.TRAIN.read_text(encoding="utf-8"))
    forbidden = prep.forbidden(train)
    by_site = collections.defaultdict(collections.Counter)
    details = []
    for scene in manifest["scenes"]:
        sid, rid = scene["id"].removeprefix("MARS_"), scene["reference_observation_id"]
        prep.check_metadata(sid, original, pinned, forbidden, "target")
        prep.check_metadata(rid, original, pinned, forbidden, "reference")
        eligible = prep.references(sid, original, pinned, forbidden)
        candidate = next(c for c in candidates if c["sample_id"] == sid)
        frozen_refs = candidate.get("reference_ids", candidate.get("reference_candidates"))
        frozen_refs = [x.get("sample_id", x.get("id")) if isinstance(x, dict) else x for x in frozen_refs]
        if frozen_refs != [id_ for _, id_ in eligible] or rid not in frozen_refs:
            raise ValueError("External reference was not metadata-frozen: " + sid)
        hours = (datetime.fromisoformat(pinned[sid]["target_datetime"]) -
                 datetime.fromisoformat(pinned[rid]["target_datetime"])).total_seconds() / 3600
        if not 24 <= hours <= 1440 or abs(hours - scene["target_minus_reference_hours"]) > 1e-8:
            raise ValueError("Reference chronology failed: " + sid)
        if scene["split"] != "test" or scene["source_level"] != "L1C" or scene["scene_kind"] != pinned[sid]["label_state"]:
            raise ValueError("External scene role failed: " + sid)
        if scene["source_row"] != {k: v for k, v in pinned[sid].items() if k != "paper_baseline"} or \
                scene["reference_source_row"] != {k: v for k, v in pinned[rid].items() if k != "paper_baseline"}:
            raise ValueError("Source metadata replay failed: " + sid)
        if any(x["sha256"] in forbidden["hashes"] for x in scene["sources"]
               if x["role"] in ("target_image", "reference_image")):
            raise ValueError("External spectral hash overlaps combined cohort: " + sid)
        details.append(source_replay.check_new(scene, dest))
        by_site[scene["physical_location_id"]][scene["scene_kind"]] += 1
    if sum(counts["PLUME"] > 0 for counts in by_site.values()) > 10 or any(
            counts["PLUME"] > 2 or counts["NO_PLUME"] > 4 for counts in by_site.values()):
        raise ValueError("External site quota failed")
    if len(details) != receipt["accepted_count"]:
        raise ValueError("External accepted count failed")
    return {"schema": "GTM_Model7_diversity_external_e2e_v1", "passed": True,
            "evaluation_manifest_sha256": prep.base.sha(manifest_path),
            "training_manifest_sha256": prep.TRAIN_SHA,
            "selection_sha256": prep.base.sha(selection_path), "accepted_count": len(details),
            "positive_sites": sum(counts["PLUME"] > 0 for counts in by_site.values()),
            "counts": dict(collections.Counter(s["scene_kind"] for s in manifest["scenes"])),
            "all_source_arrays_reconstructed": details, "downloads": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    dest = args.output.resolve()
    result = verify(dest)
    path = dest / "e2e_receipt.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != result:
            raise ValueError("Saved independent replay differs")
    else:
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "all_source_arrays_reconstructed"}, indent=2))


if __name__ == "__main__":
    main()
