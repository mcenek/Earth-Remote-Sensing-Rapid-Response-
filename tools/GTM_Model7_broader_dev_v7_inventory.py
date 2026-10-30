"""Metadata-only inventory of available cached MARS development/test sites."""
from __future__ import annotations

import collections
import importlib.util
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/GTM_Model7_broader_dev_20261024"
EXT_PATH = ROOT / "tools/GTM_Model7_prepare_diversity_external.py"
TRAIN = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json"
POOL = ROOT / "outputs/GTM_Model7_diversity_retrain_20261024/unseen_audit/weight_unseen_positive_pool.json"
spec = importlib.util.spec_from_file_location("broader_ext", EXT_PATH)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)


def actual_manifests():
    for path in (ROOT / "outputs").rglob("*manifest.json"):
        if "GTM_Model7_broader_dev_20261024" in str(path):
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = obj.get("scenes", [])
        if rows and any("physical_location_id" in row and "id" in row for row in rows):
            yield path, rows


def inventory():
    original, pinned = ext.load_inventory()
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    train_rows = train["scenes"]
    train_rows = [r for r in train_rows if r.get("split") == "train"]
    train_sites = {r["physical_location_id"] for r in train_rows}
    train_points = [(r["latitude"], r["longitude"]) for r in train_rows]
    train_ids = {r["id"].removeprefix("MARS_") for r in train_rows} | {r.get("reference_observation_id") for r in train_rows}
    train_acq = {r.get(k) for r in train_rows for k in ("target_scene_id", "reference_scene_id")}
    train_hash = {s.get("sha256") for r in train_rows for s in r.get("sources", []) if s.get("role") in ("target_image", "reference_image")}
    exposure = {"sites": set(), "points": [], "ids": set(), "acquisitions": set(), "hashes": set()}
    receipts = []
    for path, rows in actual_manifests():
        count = 0
        for row in rows:
            site = row.get("physical_location_id")
            if not site:
                continue
            count += 1
            exposure["sites"].add(site)
            if row.get("latitude") is not None and row.get("longitude") is not None:
                exposure["points"].append((float(row["latitude"]), float(row["longitude"])))
            exposure["ids"].update(str(row[k]).removeprefix("MARS_") for k in ("id", "reference_observation_id") if row.get(k))
            exposure["acquisitions"].update(row[k] for k in ("target_scene_id", "reference_scene_id") if row.get(k))
            exposure["hashes"].update(s["sha256"] for s in row.get("sources", []) if s.get("role") in ("target_image", "reference_image") and s.get("sha256"))
        if count:
            receipts.append({"path": str(path.relative_to(ROOT)), "sha256": ext.base.sha(path), "rows": count})
    # Existing viewer exposure inventory has official MARS IDs, independent of manifest paths.
    viewer_ids = set()
    for path in (ROOT / "GTM_Viewer/GTM_Model0_registry.json",):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        def walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in ("id", "sample_id", "scene_id") and isinstance(v, str) and v.removeprefix("MARS_") in original:
                        viewer_ids.add(v.removeprefix("MARS_"))
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)
        walk(obj)
    for sid in viewer_ids:
        row = original[sid]
        exposure["sites"].add(row["id_location"])
        exposure["points"].append((float(row["lat"]), float(row["lon"])))
        exposure["ids"].add(sid)
        exposure["acquisitions"].add(pinned[sid]["target_scene_id"])
        exposure["hashes"].add(next((a["remote_oid"] for a in pinned[sid]["assets"] if a["role"] == "image"), ""))
    strong_paths = [r for r in receipts if any(x in r["path"] for x in
                    ("combined_manifest.json", "external_data\\manifest.json", "fresh_data\\manifest.json",
                     "confirmation_data\\manifest.json", "temporal_20261023\\data\\manifest.json",
                     "GTM_Model6_mapper_mars_pilot_manifest.json", "GTM_Model6_plume_manifest.json"))]
    strongly_exposed_sites = set()
    strong_points = []
    strong_ids, strong_acquisitions, strong_hashes = set(), set(), set()
    for receipt in strong_paths:
        obj = json.loads((ROOT / receipt["path"]).read_text(encoding="utf-8"))
        strongly_exposed_sites.update(r["physical_location_id"] for r in obj["scenes"] if r.get("physical_location_id"))
        strong_points.extend((float(r["latitude"]),float(r["longitude"])) for r in obj["scenes"] if r.get("latitude") is not None and r.get("longitude") is not None)
        for r in obj["scenes"]:
            strong_ids.update(str(r[k]).removeprefix("MARS_") for k in ("id","reference_observation_id") if r.get(k))
            strong_acquisitions.update(r[k] for k in ("target_scene_id","reference_scene_id") if r.get(k))
            strong_hashes.update(s["sha256"] for s in r.get("sources",[]) if s.get("role") in ("target_image","reference_image") and s.get("sha256"))
    for sid in viewer_ids:
        strong_ids.add(sid)
        strongly_exposed_sites.add(original[sid]["id_location"])
        strong_points.append((float(original[sid]["lat"]),float(original[sid]["lon"])))
        strong_acquisitions.add(pinned[sid]["target_scene_id"])
        strong_hashes.add(next((a["remote_oid"] for a in pinned[sid]["assets"] if a["role"] == "image"), ""))
    by_site = collections.defaultdict(list)
    for sid, row in original.items():
        by_site[row["id_location"]].append(sid)
    candidates = collections.defaultdict(list)
    counts = collections.Counter()
    for sid, row in original.items():
        if pinned[sid]["label_state"] != "PLUME":
            continue
        site = row["id_location"]
        point = (float(row["lat"]), float(row["lon"]))
        if site in train_sites or any(ext.base.kilometers(point, p) < 25 for p in train_points):
            counts["train_site_or_25km"] += 1
            continue
        if pinned[sid]["observability"] != "clear" or float(row["percentage_clear"]) < 90:
            counts["not_clear_90"] += 1
            continue
        if row["satellite"] not in ("S2A", "S2B", "S2C"):
            counts["not_s2"] += 1
            continue
        assets = {a["role"]: a for a in pinned[sid]["assets"]}
        if not all(ext.cached.asset_present(assets[k]) for k in ("image", "cloud_mask", "plume_mask")):
            counts["not_cached"] += 1
            continue
        refs = []
        for rid in by_site[site]:
            rr = original[rid]
            if rid == sid or rr["split_name"] != row["split_name"] or ext.base.grid_key(rr) != ext.base.grid_key(row) or pinned[rid]["label_state"] != "NO_PLUME":
                continue
            hours = (datetime.fromisoformat(pinned[sid]["target_datetime"]) - datetime.fromisoformat(pinned[rid]["target_datetime"])).total_seconds()/3600
            if not (24 <= hours <= 1440 and pinned[sid]["target_datetime"][:10] != pinned[rid]["target_datetime"][:10]):
                continue
            if pinned[rid]["observability"] != "clear" or float(rr["percentage_clear"]) < 90:
                continue
            aa = {a["role"]: a for a in pinned[rid]["assets"]}
            if not all(ext.cached.asset_present(aa[k]) for k in ("image", "cloud_mask")):
                continue
            if rid in train_ids or pinned[rid]["target_scene_id"] in train_acq or aa["image"].get("remote_oid") in train_hash:
                continue
            refs.append((hours, rid))
        if not refs:
            counts["no_ref"] += 1
            continue
        refs.sort()
        rid = refs[0][1]
        if sid in train_ids or pinned[sid]["target_scene_id"] in train_acq or assets["image"].get("remote_oid") in train_hash:
            counts["train_identity"] += 1
            continue
        fresh = site not in strongly_exposed_sites and all(ext.base.kilometers(point, p) >= 25 for p in strong_points)
        if fresh and (sid in strong_ids or rid in strong_ids or pinned[sid]["target_scene_id"] in strong_acquisitions or pinned[rid]["target_scene_id"] in strong_acquisitions or assets["image"].get("remote_oid") in strong_hashes or next((a["remote_oid"] for a in pinned[rid]["assets"] if a["role"] == "image"), "") in strong_hashes):
            fresh = False
        candidates[site].append({"sample_id": sid, "reference_id": rid, "country": row["country"], "latitude": point[0], "longitude": point[1], "official_role": row["split_name"], "fresh": fresh, "exposed_site": site in exposure["sites"], "target_acquisition": pinned[sid]["target_scene_id"], "reference_acquisition": pinned[rid]["target_scene_id"], "target_sha256": assets["image"].get("remote_oid"), "reference_sha256": next((a["remote_oid"] for a in pinned[rid]["assets"] if a["role"] == "image"), "")})
    return {"schema": "GTM_Model7_broader_dev_v7_inventory_v1", "metadata_only": True, "source_pixels_read": False, "downloads": 0, "train_manifest_sha256": ext.base.sha(TRAIN), "manifest_receipts": receipts, "strong_manifest_receipts": strong_paths, "viewer_ids_count": len(viewer_ids), "candidate_positive_count": sum(map(len,candidates.values())), "candidate_site_count": len(candidates), "fresh_site_count": sum(any(x["fresh"] for x in v) for v in candidates.values()), "fresh_positive_count": sum(x["fresh"] for v in candidates.values() for x in v), "strongly_exposed_candidate_sites": sum(s in strongly_exposed_sites for s in candidates), "prepared_only_or_unclassified_candidate_sites": sum(s not in strongly_exposed_sites for s in candidates), "rejections": dict(counts), "sites": [{"site": s, "country": v[0]["country"], "positive_count": len(v), "fresh": any(x["fresh"] for x in v), "strongly_exposed": s in strongly_exposed_sites, "nearest_strong_km": round(min(ext.base.kilometers((v[0]["latitude"],v[0]["longitude"]),p) for p in strong_points),3), "nearest_any_manifest_km": round(min(ext.base.kilometers((v[0]["latitude"],v[0]["longitude"]),p) for p in exposure["points"]),3), "candidates": v} for s,v in sorted(candidates.items())]}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    result = inventory()
    path = OUT / "inventory.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("manifest_receipts","sites")}, indent=2))
    print("Fresh sites:", [(s["country"],s["site"],s["positive_count"]) for s in result["sites"] if s["fresh"]])
