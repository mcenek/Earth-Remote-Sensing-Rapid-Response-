"""Repeat source reconstruction and train/DEV/TEST identity checks for broader DEV."""
from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

from GTM_Model7_broader_dev_v7_inventory import ROOT, OUT, TRAIN, ext
from GTM_Model7_broader_dev_v7_prepare import PROTOCOL

REPLAY = ROOT / "tools/GTM_Model7_verify_cached_diversity_e2e.py"
spec = importlib.util.spec_from_file_location("broader_source_replay", REPLAY)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def main():
    dest = OUT / "dev_data"
    combined_path = dest / "combined_manifest.json"
    dev_path = dest / "manifest.json"
    freeze_path = OUT / "frozen_selection.json"
    reservation_path = OUT / "test_reservation.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != ext.base.sha(PROTOCOL) or frozen["train_manifest_sha256"] != ext.base.sha(TRAIN) or frozen["test_reservation_sha256"] != ext.base.sha(reservation_path):
        raise ValueError("Frozen provenance identity changed")
    if reservation["reserved_sites"] or reservation["reserved_positive"] or reservation["reserved_no_plume"] or not reservation["metadata_only"]:
        raise ValueError("TEST reservation mutated")
    train_original = [r for r in json.loads(TRAIN.read_text(encoding="utf-8"))["scenes"] if r["split"] == "train"]
    train_new = [r for r in combined["scenes"] if r["split"] == "train"]
    dev_new = [r for r in combined["scenes"] if r["split"] == "dev"]
    if len(train_original) != 958 or len(train_new) != 958 or dev_new != dev["scenes"] or len(dev_new) != len(frozen["selection"]):
        raise ValueError("Split identity/count mismatch")
    for old, new in zip(train_original, train_new):
        restored = dict(new)
        restored["array_file"] = old["array_file"]
        if restored != old or ext.base.sha(dest / new["array_file"]) != old["array_sha256"]:
            raise ValueError("TRAIN row or array changed: " + old["id"])
    train_sites = {r["physical_location_id"] for r in train_original}
    train_points = [(r["latitude"],r["longitude"]) for r in train_original]
    train_ids = {r["id"] for r in train_original}
    train_acqs = {r[k] for r in train_original for k in ("target_scene_id","reference_scene_id")}
    train_hashes = {s["sha256"] for r in train_original for s in r.get("sources",[]) if s["role"] in ("target_image","reference_image")}
    source_lookup = {}
    for src in frozen["source_manifests"]:
        p = ROOT / src["path"]
        if ext.base.sha(p) != src["sha256"]:
            raise ValueError("Source manifest changed")
        source_lookup.update(((src["path"],r["id"]),r) for r in json.loads(p.read_text(encoding="utf-8"))["scenes"])
    source_replay = []
    sites, countries = set(), set()
    for item, scene in zip(frozen["selection"], dev_new):
        if scene["id"] != item["id"] or scene["split"] != "dev" or scene["physical_location_id"] != item["site"] or scene["scene_kind"] != item["scene_kind"] or scene["reference_observation_id"] != item["reference_id"]:
            raise ValueError("Selected DEV scene mismatch")
        original = source_lookup[(item["source_manifest"],item["id"])]
        restored = dict(scene)
        restored["split"] = original["split"]
        restored["array_file"] = original["array_file"]
        restored.pop("broader_dev_provenance")
        if "country" not in original: restored.pop("country")
        if restored != original:
            raise ValueError("DEV source scene modified beyond split/path/provenance/country")
        if scene["physical_location_id"] in train_sites or any(ext.base.kilometers((scene["latitude"],scene["longitude"]),p)<25 for p in train_points):
            raise ValueError("DEV site overlaps TRAIN")
        if scene["id"] in train_ids or scene["target_scene_id"] in train_acqs or scene["reference_scene_id"] in train_acqs or any(s["sha256"] in train_hashes for s in scene["sources"] if s["role"] in ("target_image","reference_image")):
            raise ValueError("DEV identity overlaps TRAIN")
        if scene["source_row"]["label_state"] != scene["scene_kind"] or scene["reference_source_row"]["label_state"] != "NO_PLUME":
            raise ValueError("Explicit source label/reference contract failed")
        source_replay.append(replay.check_new(scene,dest))
        sites.add(scene["physical_location_id"])
        countries.add(scene["country"])
    counts = collections.Counter(r["scene_kind"] for r in dev_new)
    if counts != {"PLUME":46,"NO_PLUME":152} or len(sites)!=55 or len(countries)!=12 or len(train_new)!=958:
        raise ValueError("Final cohort diversity/count changed")
    receipt = {"schema":"GTM_Model7_broader_dev_v7_source_replay_e2e_v1","passed":True,
               "downloads":0,"test_pixels_read":False,"test_prediction_results_read":False,
               "train_manifest_sha256":ext.base.sha(TRAIN),"combined_manifest_sha256":ext.base.sha(combined_path),
               "dev_manifest_sha256":ext.base.sha(dev_path),"frozen_selection_sha256":ext.base.sha(freeze_path),
               "test_reservation_sha256":ext.base.sha(reservation_path),"train_unchanged":len(train_new),
               "dev_source_replayed":len(source_replay),"dev_counts":dict(counts),"dev_sites":len(sites),
               "dev_countries":len(countries),"source_replays":source_replay}
    receipt_path = OUT / "e2e_receipt.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Prior E2E receipt differs")
    else:
        receipt_path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k!="source_replays"},indent=2))


if __name__ == "__main__": main()
