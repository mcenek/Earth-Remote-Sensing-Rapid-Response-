"""After root freeze, prepare and replay the sealed negative-only TEST cohort."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from GTM_Model7_broader_dev_v7_inventory import ROOT, OUT, ext
from GTM_Model7_negative_only_test_reserve import DEST as RESERVATION, PROTOCOL as RESERVATION_PROTOCOL, MANIFESTS, exposure, spectral

BASE=ext.base
MODEL_PROTOCOL=ROOT/"docs/GTM_Model7_negative_only_test_protocol_20261024.md"
REPLAY_PATH=ROOT/"tools/GTM_Model7_verify_cached_diversity_e2e.py"
spec=importlib.util.spec_from_file_location("negative_source_replay",REPLAY_PATH)
replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def save_new(path,obj):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def gate_check(path,expected_sha,reservation_sha):
    if not path.is_file() or BASE.sha(path)!=expected_sha:
        raise ValueError("Root selection gate missing or SHA mismatch")
    gate=json.loads(path.read_text(encoding="utf-8"))
    if gate.get("negative_only_reservation_sha256")!=reservation_sha or gate.get("both_fits_complete") is not True or gate.get("fresh_process_dev_replay_passed") is not True or gate.get("eligible_improved_dev_selection") is not True:
        raise ValueError("Root gate lacks sealed reservation, completed fits, DEV replay, or eligible selection")
    for key in ("selected_model_freeze","baseline_model_freeze"):
        item=gate.get(key)
        if not isinstance(item,dict) or not item.get("path") or not item.get("sha256") or BASE.sha(Path(item["path"]))!=item["sha256"]:
            raise ValueError("Root gate model freeze identity missing/mismatched: "+key)
        freeze=json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        flat=json.dumps(freeze)
        if "checkpoint_sha256" not in flat or not any(k in flat for k in ("cutoff","threshold")):
            raise ValueError("Model freeze lacks checkpoint or cutoff: "+key)
    return gate


def metadata_plan(reservation,original,pinned,forbidden):
    if reservation.get("schema")!="GTM_Model7_negative_only_test_metadata_reservation_v1" or not reservation.get("metadata_only") or reservation["counts"]["reserved_plume"]!=0:
        raise ValueError("Wrong negative-only metadata reservation")
    if len(reservation["sites"])>10: raise ValueError("Too many reserved sites")
    planned=[]
    used=set()
    for site in reservation["sites"]:
        sid_site=site["site"]
        if len(site["no_plume"])>2 or not site["no_plume"]:
            raise ValueError("Negative-only site quota failed")
        if any(row["id_location"]==sid_site and pinned[sid]["label_state"]=="PLUME" for sid,row in original.items()):
            raise ValueError("Catalog PLUME at negative-only site")
        point=(site["latitude"],site["longitude"])
        if sid_site in forbidden["sites"] or any(BASE.kilometers(point,p)<25 for p in forbidden["points"]):
            raise ValueError("Reserved site overlaps prior fit/eval/viewer/DEV")
        for record in site["no_plume"]:
            sid,rid=record["sample_id"],record["reference_id"]
            if sid in used: raise ValueError("Duplicate reserved target")
            used.add(sid)
            if original[sid]["id_location"]!=sid_site or original[rid]["id_location"]!=sid_site:
                raise ValueError("Target/reference site mismatch")
            if pinned[sid]["label_state"]!="NO_PLUME" or pinned[rid]["label_state"]!="NO_PLUME":
                raise ValueError("Not explicit NO_PLUME target/reference")
            ext.check_metadata(sid,original,pinned,forbidden,"target")
            ext.check_metadata(rid,original,pinned,forbidden,"reference")
            if (record["target_acquisition_id"]!=pinned[sid]["target_scene_id"] or
                record["reference_acquisition_id"]!=pinned[rid]["target_scene_id"] or
                record["target_image_sha256"]!=spectral(sid,pinned) or
                record["reference_image_sha256"]!=spectral(rid,pinned)):
                raise ValueError("Frozen acquisition or spectral hash changed")
            if original[sid]["split_name"]!=original[rid]["split_name"] or BASE.grid_key(original[sid])!=BASE.grid_key(original[rid]):
                raise ValueError("Frozen reference grid/split changed")
            t=datetime.fromisoformat(pinned[sid]["target_datetime"])
            r=datetime.fromisoformat(pinned[rid]["target_datetime"])
            hours=(t-r).total_seconds()/3600
            if not(24<=hours<=1440 and t.date()!=r.date()):
                raise ValueError("Frozen reference chronology changed")
            planned.append({"sample_id":sid,"reference_id":rid,"physical_site_id":sid_site,"offset_hours":hours})
    if len(planned)!=reservation["counts"]["reserved_no_plume"]:
        raise ValueError("Reserved no-plume count changed")
    return planned


def verify_metadata_only():
    reservation=json.loads(RESERVATION.read_text(encoding="utf-8"))
    if reservation["protocol"]["sha256"]!=BASE.sha(RESERVATION_PROTOCOL):
        raise ValueError("Reservation protocol changed")
    for source in reservation["source_catalog"]["csv"]:
        name=Path(source["path"]).name
        if BASE.sha(BASE.MARS/name)!=source["sha256"]: raise ValueError("Source catalog CSV changed")
    if BASE.sha(BASE.MARS/"paper_v3_mixed_samples.jsonl")!=reservation["source_catalog"]["pinned_samples_sha256"]:
        raise ValueError("Pinned source metadata changed")
    original,pinned=ext.load_inventory()
    forbidden,receipts,viewer_count=exposure(original,pinned)
    if receipts!=reservation["excluded_fit_eval_manifests"] or viewer_count!=reservation["viewer_mars_ids_mapped"]:
        raise ValueError("Exposure receipts changed")
    planned=metadata_plan(reservation,original,pinned,forbidden)
    sites=reservation["sites"]
    for i,site in enumerate(sites):
        point=(site["latitude"],site["longitude"])
        nearest=min(BASE.kilometers(point,p) for p in forbidden["points"])
        if nearest<25 or abs(nearest-site["nearest_prior_km"])>0.002:
            raise ValueError("Reserved prior distance changed")
        for other in sites[:i]:
            if BASE.kilometers(point,(other["latitude"],other["longitude"]))<25:
                raise ValueError("Reserved sites overlap within 25 km")
    by_site=collections.defaultdict(lambda:{"acqs":set(),"hashes":set()})
    for site in sites:
        for rec in site["no_plume"]:
            by_site[site["site"]]["acqs"].update(rec[k] for k in ("target_acquisition_id","reference_acquisition_id"))
            by_site[site["site"]]["hashes"].update(rec[k] for k in ("target_image_sha256","reference_image_sha256"))
    for site,values in by_site.items():
        for other,other_values in by_site.items():
            if site!=other and (values["acqs"]&other_values["acqs"] or values["hashes"]&other_values["hashes"]):
                raise ValueError("Cross-site source identity overlap")
    result={"schema":"GTM_Model7_negative_only_metadata_e2e_v1","passed":True,"metadata_only":True,
            "source_pixels_read":False,"prediction_results_read":False,"downloads":0,
            "reservation_sha256":BASE.sha(RESERVATION),"viewer_mars_ids_mapped":viewer_count,
            "reserved_sites":len(sites),"reserved_no_plume":len(planned),"reserved_plume":0,
            "minimum_prior_distance_km":min(s["nearest_prior_km"] for s in sites)}
    save_new(OUT/"negative_only_metadata_e2e_receipt.json",result)
    return result


def prepare(dest,gate_path,gate_sha):
    if dest.exists(): raise FileExistsError(dest)
    reservation_sha=BASE.sha(RESERVATION)
    gate_check(gate_path,gate_sha,reservation_sha)
    reservation=json.loads(RESERVATION.read_text(encoding="utf-8"))
    if reservation["protocol"]["sha256"]!=BASE.sha(RESERVATION_PROTOCOL): raise ValueError("Reservation protocol changed")
    for item in reservation["excluded_fit_eval_manifests"]:
        if BASE.sha(ROOT/item["path"])!=item["sha256"]: raise ValueError("Prior fit/eval manifest changed")
    original,pinned=ext.load_inventory()
    forbidden,receipts,viewer_count=exposure(original,pinned)
    if receipts!=reservation["excluded_fit_eval_manifests"] or viewer_count!=reservation["viewer_mars_ids_mapped"]:
        raise ValueError("Exposure inventory changed since reservation")
    planned=metadata_plan(reservation,original,pinned,forbidden)
    # Complete metadata plan is persisted before any source TIFF or array is decoded.
    dest.mkdir(parents=True)
    save_new(dest/"selection_receipt.json",{"schema":"GTM_Model7_negative_only_plan_v1","metadata_plan":planned,
             "reservation_sha256":reservation_sha,"gate_sha256":gate_sha,"downloads":0})
    scenes,attempts=[],[]
    for item in planned:
        sid,rid=item["sample_id"],item["reference_id"]
        row,pin=original[sid],pinned[sid]
        reason=None
        payload=sources=fraction=None
        snow_target=snow_ref=None
        try:
            assets={key:{a["role"]:a for a in pinned[key]["assets"]} for key in (sid,rid)}
            payload,sources,reason,fraction=BASE.raster_pair(sid,rid,pinned,original,assets)
            if reason is None:
                arrays,(coarse_grid,native_grid)=payload
                paths={source["role"]:Path(source["path"]) for source in sources}
                ext.cached.verify_csv_grid(paths,row,original[rid])
                hashes={source["role"]:source["sha256"] for source in sources}
                if hashes["target_image"]!=spectral(sid,pinned) or hashes["reference_image"]!=spectral(rid,pinned) or hashes["target_image"] in forbidden["hashes"] or hashes["reference_image"] in forbidden["hashes"]:
                    reason="spectral_identity_or_prior_overlap"
                else:
                    snow_target=ext.clear.snow_fraction(paths["target_image"],arrays["valid_native"])
                    snow_ref=ext.clear.snow_fraction(paths["reference_image"],arrays["valid_native"])
                    if max(snow_target,snow_ref)>ext.clear.GATE["date_fraction_gt"]:
                        reason="snow_like_fraction_gt_10pct_target_or_reference"
        except (ValueError,KeyError,OSError) as error:
            reason="source_or_raster_contract:"+str(error)
        attempts.append({"sample_id":sid,"reference_observation_id":rid,"offset_hours":item["offset_hours"],
                         "reason":reason or "accepted","bidate_valid_fraction":fraction,
                         "target_snow_like_fraction":snow_target,"reference_snow_like_fraction":snow_ref})
        if reason: continue
        array_path=dest/"arrays"/f"MARS_{sid}.npz"
        array_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(array_path,**arrays)
        scenes.append({"id":f"MARS_{sid}","name":f"NO_PLUME | {pin['target_datetime'][:10]} | {sid[:8]}",
                       "split":"test","group_id":"negative_only_site_"+row["id_location"],
                       "physical_location_id":row["id_location"],"scene_kind":"NO_PLUME","source_level":"L1C",
                       "country":row["country"],"sensor_family":"Sentinel-2","original_release_role":pin["research_role"],
                       "latitude":float(row["lat"]),"longitude":float(row["lon"]),
                       "target_scene_id":pin["target_scene_id"],"reference_scene_id":pinned[rid]["target_scene_id"],
                       "reference_observation_id":rid,"original_release_reference_scene_id":pin["reference_scene_id"],
                       "satellite_id":pin["target_scene_id"],"satellite_time":pin["target_datetime"],
                       "reference_time":pinned[rid]["target_datetime"],"target_minus_reference_hours":item["offset_hours"],
                       "input_grid":coarse_grid,"reference_grid":native_grid,"reference_input_grid":coarse_grid,
                       "clear_fraction":fraction,"target_snow_like_fraction":snow_target,
                       "reference_snow_like_fraction":snow_ref,"array_file":str(array_path.relative_to(dest)).replace("\\","/"),
                       "array_sha256":BASE.sha(array_path),"sources":sources,
                       "source_row":{k:v for k,v in pin.items() if k!="paper_baseline"},"official_csv_row":row,
                       "reference_source_row":{k:v for k,v in pinned[rid].items() if k!="paper_baseline"},
                       "source_label":"NO_PLUME","reference_mapping":"repeat_2",
                       "reference_kind":"reviewed_binary_annotation",
                       "notes":"Metadata-reserved, previously unused negative-only MARS L1C test; QA failures retained without replacement."})
    if BASE.sha(RESERVATION)!=reservation_sha or BASE.sha(gate_path)!=gate_sha:
        raise ValueError("Reservation or gate identity changed during preparation")
    manifest={"schema":"GTM_Model7_temporal_cohort_v1","synthetic":False,
              "role":"metadata_reserved_negative_only_test","downloads":0,"bands":BASE.BANDS,
              "input_channels":{"target_image":BASE.BANDS,"reference_image":BASE.BANDS},
              "selection":BASE.file_source(RESERVATION),"freeze_gate":BASE.file_source(gate_path),
              "protocol":BASE.file_source(MODEL_PROTOCOL),"preparation_source":BASE.file_source(Path(__file__)),
              "source_csv":[BASE.file_source(BASE.MARS/n) for n in ("train.csv","val.csv","test.csv")],
              "source_manifest":BASE.file_source(BASE.MARS/"paper_v3_mixed_samples.jsonl"),
              "reference_selection":"single metadata-reserved same-grid preceding clear NO_PLUME reference; no substitution",
              "quality_limitations":["Negative-only scenes can measure false alarms, not plume recall.",
                                     "No-plume annotation does not prove zero emissions."],
              "split_counts":{"test":{"NO_PLUME":len(scenes)}},"attempts":attempts,"scenes":scenes}
    save_new(dest/"manifest.json",manifest)
    receipt={"schema":"GTM_Model7_negative_only_preparation_v1","passed":bool(scenes),
             "reservation_sha256":reservation_sha,"gate_sha256":gate_sha,
             "manifest_sha256":BASE.sha(dest/"manifest.json"),"candidate_count":len(planned),
             "accepted_count":len(scenes),"attempt_counts":dict(collections.Counter(x["reason"].split(":")[0] for x in attempts)),
             "downloads":0}
    save_new(dest/"receipt.json",receipt)
    return receipt


def verify(dest):
    manifest_path=dest/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt=json.loads((dest/"receipt.json").read_text(encoding="utf-8"))
    if receipt["reservation_sha256"]!=BASE.sha(RESERVATION) or receipt["manifest_sha256"]!=BASE.sha(manifest_path):
        raise ValueError("Prepared identity changed")
    gate_file=Path(manifest["freeze_gate"]["path"])
    gate_check(gate_file,receipt["gate_sha256"],receipt["reservation_sha256"])
    reservation=json.loads(RESERVATION.read_text(encoding="utf-8"))
    original,pinned=ext.load_inventory()
    forbidden,receipts,viewer_count=exposure(original,pinned)
    if receipts!=reservation["excluded_fit_eval_manifests"] or viewer_count!=reservation["viewer_mars_ids_mapped"]:
        raise ValueError("Source exposure receipts changed")
    planned=metadata_plan(reservation,original,pinned,forbidden)
    attempts=manifest["attempts"]
    if [x["sample_id"] for x in attempts]!=[x["sample_id"] for x in planned] or len(attempts)!=receipt["candidate_count"]:
        raise ValueError("Fixed QA attempt list differs")
    accepted=[x["sample_id"] for x in attempts if x["reason"]=="accepted"]
    scenes=manifest["scenes"]
    if [x["id"].removeprefix("MARS_") for x in scenes]!=accepted:
        raise ValueError("Accepted scenes differ from QA attempts")
    details=[]
    for scene in scenes:
        sid=scene["id"].removeprefix("MARS_")
        plan=next(x for x in planned if x["sample_id"]==sid)
        if scene["split"]!="test" or scene["scene_kind"]!="NO_PLUME" or scene["reference_observation_id"]!=plan["reference_id"] or scene["physical_location_id"]!=plan["physical_site_id"]:
            raise ValueError("Scene differs from frozen negative plan")
        details.append(replay.check_new(scene,dest))
    result={"schema":"GTM_Model7_negative_only_source_e2e_v1","passed":True,
            "reservation_sha256":receipt["reservation_sha256"],"manifest_sha256":receipt["manifest_sha256"],
            "gate_sha256":receipt["gate_sha256"],"candidate_count":len(planned),"accepted_count":len(scenes),
            "accepted_sites":len({s["physical_location_id"] for s in scenes}),"accepted_no_plume":len(scenes),
            "accepted_plume":0,"downloads":0,"source_replays":details}
    out=dest/"e2e_receipt.json"
    if out.exists():
        if json.loads(out.read_text(encoding="utf-8"))!=result: raise ValueError("Saved E2E differs")
    else: save_new(out,result)
    return {k:v for k,v in result.items() if k!="source_replays"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--gate-sha256",required=True)
    sub.add_parser("verify-metadata")
    v=sub.add_parser("verify")
    v.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.mode=="prepare": print(json.dumps(prepare(args.output.resolve(),args.gate.resolve(),args.gate_sha256),indent=2))
    elif args.mode=="verify-metadata": print(json.dumps(verify_metadata_only(),indent=2))
    else: print(json.dumps(verify(args.output.resolve()),indent=2))
