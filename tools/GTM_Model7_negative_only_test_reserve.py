"""Metadata-only negative TEST reservation, with actual fit/eval/viewer exclusions."""
from __future__ import annotations

import collections
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from math import radians, sin, cos, asin

import numpy as np
from scipy.spatial import cKDTree

from GTM_Model7_broader_dev_v7_inventory import ROOT, OUT, ext

PROTOCOL = ROOT / "docs/GTM_Model7_negative_only_test_reservation_20261024.md"
DEST = OUT / "negative_only_test_reservation.json"
MANIFESTS = [
    "outputs/GTM_Model7_temporal_20261023/data/manifest.json",
    "outputs/GTM_Model7_recovery_20261023/confirmation_data/manifest.json",
    "outputs/GTM_Model7_dataset_repair_20261024/mars_data/manifest.json",
    "outputs/GTM_Model7_dataset_repair_20261024/s2cm_data/manifest.json",
    "outputs/GTM_Model7_nontexas_readiness_20261024/diverse_training/combined_manifest.json",
    "outputs/GTM_Model7_diversity_retrain_20261024/external_data/manifest.json",
    "outputs/GTM_Model7_unet_recall_20261024/fresh_data/manifest.json",
    "outputs/GTM_Model7_broader_dev_20261024/dev_data/combined_manifest.json",
    "outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r5/GTM_Model6_mapper_mars_pilot_manifest.json",
    "outputs/GTM_Model6/GTM_Model6_plume_01/GTM_Model6_plume_manifest.json",
    "outputs/GTM_Model6/GTM_Model6_unseen_20261023/data/manifest.json",
]


def stable(raw):
    return hashlib.sha256(("negative-only-test-v7-20261024:"+raw).encode()).hexdigest()


def spectral(sid, pinned):
    return next((a.get("remote_oid", "").lower() for a in pinned[sid]["assets"]
                 if a.get("role") == "image" and a.get("remote_oid_type") == "sha256_lfs"), "")


def xyz(point):
    lat,lon=map(radians,point)
    return (cos(lat)*cos(lon),cos(lat)*sin(lon),sin(lat))


def exposure(original, pinned):
    forbidden = {"ids":set(),"sites":set(),"acquisitions":set(),"hashes":set(),"points":[]}
    receipts = []
    for name in MANIFESTS:
        path = ROOT / name
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = [r for k in ("scenes","observations","training_observations") for r in obj.get(k,[]) if isinstance(r,dict)]
        for row in rows:
            sid = str(row.get("id",row.get("sample_id",""))).removeprefix("MARS_")
            if sid in original:
                forbidden["ids"].add(sid)
            rid = row.get("reference_observation_id")
            if rid: forbidden["ids"].add(str(rid).removeprefix("MARS_"))
            site = row.get("physical_location_id",row.get("id_location",row.get("site_id")))
            if site: forbidden["sites"].add(site)
            lat,lon = row.get("latitude",row.get("lat")),row.get("longitude",row.get("lon"))
            if lat is not None and lon is not None: forbidden["points"].append((float(lat),float(lon)))
            forbidden["acquisitions"].update(row[k] for k in ("target_scene_id","reference_scene_id") if row.get(k))
            forbidden["hashes"].update(s["sha256"].lower() for s in row.get("sources",[]) if s.get("role") in ("image","target_image","reference_image") and s.get("sha256"))
        receipts.append({"path":name,"sha256":ext.base.sha(path),"rows":len(rows)})
    viewer_path = ROOT / "outputs/GTM_viewer_bundles"
    scan = subprocess.run(["rg","--no-ignore","--hidden","-o","-N","--no-filename",
                           r"MARS_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                           str(viewer_path)],capture_output=True,text=True,check=False)
    if scan.returncode not in (0,1): raise RuntimeError(scan.stderr)
    viewer_ids = {line.removeprefix("MARS_") for line in scan.stdout.splitlines()}
    if not viewer_ids or not viewer_ids.issubset(original):
        raise ValueError("Viewer MARS ID inventory missing or unmapped")
    for sid in viewer_ids:
        row = original[sid]
        forbidden["ids"].add(sid)
        forbidden["sites"].add(row["id_location"])
        forbidden["points"].append((float(row["lat"]),float(row["lon"])))
        forbidden["acquisitions"].add(pinned[sid]["target_scene_id"])
        forbidden["hashes"].add(spectral(sid,pinned))
    forbidden["points"] = sorted(set(forbidden["points"]))
    return forbidden, receipts, len(viewer_ids)


def main():
    if DEST.exists(): raise FileExistsError(DEST)
    original,pinned = ext.load_inventory()
    print(f"Loaded {len(original)} official cached observations",flush=True)
    forbidden,receipts,viewer_count = exposure(original,pinned)
    print(f"Audited {len(receipts)} manifests and {viewer_count} viewer MARS IDs",flush=True)
    exposure_tree=cKDTree(np.array([xyz(p) for p in forbidden["points"]]))
    site_rows = collections.defaultdict(list)
    for sid,row in original.items(): site_rows[row["id_location"]].append(sid)
    print(f"Checking {len(site_rows)} physical locations",flush=True)
    rejections = collections.Counter()
    candidate_sites=[]
    for site_index,(site,ids) in enumerate(site_rows.items()):
        if site_index and site_index%500==0: print(f"Checked {site_index} sites; {len(candidate_sites)} candidate negative-only sites",flush=True)
        if any(pinned[sid]["label_state"]=="PLUME" for sid in ids):
            rejections["site_has_official_PLUME_observation"]+=1
            continue
        no_plume=[sid for sid in ids if pinned[sid]["label_state"]=="NO_PLUME"]
        if not no_plume: continue
        point=(float(original[no_plume[0]]["lat"]),float(original[no_plume[0]]["lon"]))
        chord=float(exposure_tree.query(xyz(point),k=1)[0])
        nearest=12742*asin(min(1,chord/2))
        if site in forbidden["sites"] or nearest <25:
            rejections["site_or_25km_prior_fit_eval_viewer_DEV"]+=1
            continue
        eligible=[]
        for sid in no_plume:
            try: ext.check_metadata(sid,original,pinned,forbidden,"target")
            except (ValueError,KeyError) as e:
                rejections[str(e).split(":")[0]]+=1
                continue
            if not spectral(sid,pinned) or spectral(sid,pinned) in forbidden["hashes"]:
                rejections["target_spectral_hash_prior_exposure"]+=1
                continue
            target=original[sid]
            when=datetime.fromisoformat(pinned[sid]["target_datetime"])
            refs=[]
            for rid in no_plume:
                if rid==sid or original[rid]["split_name"]!=target["split_name"] or ext.base.grid_key(original[rid])!=ext.base.grid_key(target):
                    continue
                ref_time=datetime.fromisoformat(pinned[rid]["target_datetime"])
                hours=(when-ref_time).total_seconds()/3600
                if not (24<=hours<=1440 and when.date()!=ref_time.date()): continue
                try: ext.check_metadata(rid,original,pinned,forbidden,"reference")
                except (ValueError,KeyError): continue
                if not spectral(rid,pinned) or spectral(rid,pinned) in forbidden["hashes"]: continue
                refs.append((hours,rid))
            if refs:
                refs.sort()
                rid=refs[0][1]
                eligible.append({"sample_id":sid,"reference_id":rid,"target_acquisition_id":pinned[sid]["target_scene_id"],
                                 "reference_acquisition_id":pinned[rid]["target_scene_id"],
                                 "target_image_sha256":spectral(sid,pinned),"reference_image_sha256":spectral(rid,pinned),
                                 "target_datetime":pinned[sid]["target_datetime"],"reference_datetime":pinned[rid]["target_datetime"],
                                 "official_role":target["split_name"]})
            else: rejections["no_qualifying_preceding_explicit_clear_reference"]+=1
        if eligible:
            eligible.sort(key=lambda x:stable(x["sample_id"]))
            candidate_sites.append({"site":site,"country":original[eligible[0]["sample_id"]]["country"],
                                    "latitude":point[0],"longitude":point[1],"nearest_prior_km":round(nearest,3),
                                    "eligible_pairs":eligible})
    countries=collections.defaultdict(list)
    for entry in candidate_sites: countries[entry["country"]].append(entry)
    for entries in countries.values(): entries.sort(key=lambda x:stable(x["site"]))
    ordered=[countries[c][i] for i in range(max(map(len,countries.values()),default=0))
             for c in sorted(countries,key=stable) if i<len(countries[c])]
    selected=[]
    chosen_acqs,chosen_hashes=set(),set()
    for entry in ordered:
        if len(selected)>=10: break
        point=(entry["latitude"],entry["longitude"])
        if any(ext.base.kilometers(point,(s["latitude"],s["longitude"]))<25 for s in selected):
            rejections["within_25km_other_reserved_site"]+=1
            continue
        pairs=[]
        for rec in entry["eligible_pairs"]:
            if len(pairs)>=2: break
            acqs={rec["target_acquisition_id"],rec["reference_acquisition_id"]}
            hashes={rec["target_image_sha256"],rec["reference_image_sha256"]}
            if acqs & chosen_acqs or hashes & chosen_hashes:
                rejections["cross_site_acquisition_or_hash"]+=1
                continue
            pairs.append(rec)
        if pairs:
            selected.append({k:v for k,v in entry.items() if k!="eligible_pairs"}|{"no_plume":pairs})
            chosen_acqs.update(rec[k] for rec in pairs for k in ("target_acquisition_id","reference_acquisition_id"))
            chosen_hashes.update(rec[k] for rec in pairs for k in ("target_image_sha256","reference_image_sha256"))
    result={"schema":"GTM_Model7_negative_only_test_metadata_reservation_v1","metadata_only":True,
            "source_pixels_read":False,"prediction_results_read":False,"downloads":0,
            "protocol":{"path":str(PROTOCOL.relative_to(ROOT)),"sha256":ext.base.sha(PROTOCOL)},
            "source_catalog":{"csv":[{"path":str((ext.base.MARS/n).relative_to(ROOT.parent)),"sha256":ext.base.sha(ext.base.MARS/n)} for n in ("train.csv","val.csv","test.csv")],
                              "pinned_samples_sha256":ext.base.sha(ext.base.MARS/"paper_v3_mixed_samples.jsonl")},
            "excluded_fit_eval_manifests":receipts,"viewer_mars_ids_mapped":viewer_count,
            "counts":{"eligible_negative_only_sites_before_inter_site":len(candidate_sites),
                      "reserved_sites":len(selected),"reserved_no_plume":sum(len(s["no_plume"]) for s in selected),"reserved_plume":0},
            "rejections":dict(rejections),"sites":selected,
            "scope":"Negative-only final cohort can measure false alarms after model/threshold freeze; it cannot validate plume recall. No QA replacement or pixel read has occurred."}
    DEST.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"counts":result["counts"],"viewer_mars_ids_mapped":viewer_count,
                      "countries":collections.Counter(s["country"] for s in selected),
                      "minimum_prior_km":min((s["nearest_prior_km"] for s in selected),default=None),
                      "reservation_sha256":ext.base.sha(DEST)},indent=2))


if __name__=="__main__": main()
