"""Inference-only released MARS-S2L and CH4Net comparison on exposed DEV."""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import rasterio
import torch

import GTM_Model7_native_transfer_v13 as v13
import evaluate_released_marss2l as released

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/GTM_Model7_released_comparison_v15_20261026"
PROTOCOL = ROOT / "docs/GTM_Model7_released_comparison_v15_protocol_20261026.md"
ARMS = ("mars-s2l", "ch4net")
EXPECTED_PARAMS = {"mars-s2l": 13_587_585, "ch4net": 13_581_825}
TEACHER_CHECKPOINT = ROOT / "outputs/GTM_Model7_hard_negative_v11_20261025/fit/siamese/step_2400.pt"
TEACHER_SHA = "8a36587380ea14b86501b5bc245b3b26f82f52f380aa5e5e156900491ae39483"


def sha(path: Path) -> str:
    return v13.sha(path)


def save(path: Path, value: object) -> None:
    v13.save_json(path, value)


def model_paths(arm: str) -> tuple[Path, Path]:
    spec = released.RELEASE_SPECS[arm]
    folder = ROOT / spec["directory"]
    return folder / "best_epoch", folder / "config_experiment.json"


def source_check() -> tuple[list[dict], dict[str, dict], dict]:
    if sha(TEACHER_CHECKPOINT) != TEACHER_SHA or sha(v13.SOURCE) != v13.SOURCE_SHA:
        raise ValueError("Frozen comparison source mismatch")
    source_rows = {r["id"]: r for r in json.loads(v13.SOURCE.read_text(encoding="utf-8"))["scenes"] if r["split"] == "dev"}
    native = [r for r in v13.index() if r["split"] == "dev"]
    if len(native) != 198 or len(source_rows) != 198 or {r["id"] for r in native} != set(source_rows):
        raise ValueError("DEV cohort changed")
    if collections.Counter(r["scene_kind"] for r in native) != {"PLUME": 46, "NO_PLUME": 152}:
        raise ValueError("DEV class inventory changed")
    configs = {}
    for arm in ARMS:
        checkpoint, config_path = model_paths(arm)
        spec = released.RELEASE_SPECS[arm]
        if (sha(checkpoint) != spec["checkpoint_sha256"] or checkpoint.stat().st_size != spec["checkpoint_bytes"] or
            sha(config_path) != spec["config_sha256"]):
            raise ValueError("Released checkpoint/config changed: " + arm)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if any(config.get(k) != v for k, v in spec["expected_config"].items()):
            raise ValueError("Released config contract changed: " + arm)
        configs[arm] = config
    return native, source_rows, configs


def settings() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def read_source(item: dict, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    sources = {s["role"]: s for s in item["sources"]}
    required = {"target_image", "reference_image", "target_cloud", "reference_cloud"}
    if row["scene_kind"] == "PLUME": required.add("plume_mask")
    if not required <= set(sources): raise ValueError("Missing native sources")
    hashes = {}
    for role, source in sources.items():
        path = Path(source["path"])
        if path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
            raise ValueError("Source drift: " + item["id"] + "/" + role)
        hashes[role] = source["sha256"]
    with rasterio.open(sources["target_image"]["path"]) as ds:
        if ds.shape != (200, 200) or tuple(ds.descriptions)[:6] != v13.BANDS or set(ds.dtypes) != {"uint16"}:
            raise ValueError("Current band/grid contract")
        grid = (ds.transform, ds.crs)
        a = ds.read(indexes=(1, 2, 3, 4, 5, 6))
        a_nodata = ds.nodata
    with rasterio.open(sources["reference_image"]["path"]) as ds:
        if ds.shape != (200, 200) or tuple(ds.descriptions)[:6] != v13.BANDS or (ds.transform, ds.crs) != grid:
            raise ValueError("Selected reference band/grid contract")
        b = ds.read(indexes=(1, 2, 3, 4, 5, 6))
        b_nodata = ds.nodata
    with rasterio.open(sources["target_cloud"]["path"]) as ds:
        if ds.shape != (200, 200) or (ds.transform, ds.crs) != grid: raise ValueError("Current cloud grid")
        ca = ds.read(1)
    with rasterio.open(sources["reference_cloud"]["path"]) as ds:
        if ds.shape != (200, 200) or (ds.transform, ds.crs) != grid: raise ValueError("Reference cloud grid")
        cb = ds.read(1)
    good = (ca == 0) & (cb == 0) & (a > 0).all(0) & (b > 0).all(0)
    if a_nodata is not None: good &= (a != a_nodata).all(0)
    if b_nodata is not None: good &= (b != b_nodata).all(0)
    valid20 = good.reshape(100, 2, 100, 2).all((1, 3))
    valid = np.repeat(np.repeat(valid20, 2, 0), 2, 1).astype(np.uint8)
    y = np.zeros((200, 200), np.uint8)
    if row["scene_kind"] == "PLUME":
        with rasterio.open(sources["plume_mask"]["path"]) as ds:
            if ds.shape != (200, 200) or (ds.transform, ds.crs) != grid: raise ValueError("Native label grid")
            y = ds.read(1)
    native = v13.scene(item)
    if not np.array_equal(y, native["y"]) or not np.array_equal(valid, native["valid"]):
        raise ValueError("Native label/support mismatch: " + item["id"])
    return a, b, ca, y, hashes


def builder_input(a: np.ndarray, b: np.ndarray, ca: np.ndarray, wind: tuple[float, float]) -> np.ndarray:
    sample = SimpleNamespace(raw_pair=np.concatenate((a, b)), cloud_classes=ca)
    return released.released_input(sample, wind, "mars-s2l")


def independent_input(a: np.ndarray, b: np.ndarray, ca: np.ndarray, wind: tuple[float, float]) -> np.ndarray:
    spectral = np.clip(np.concatenate((a, b)).astype(np.float32) / 5000.0, 0, 2)
    def normalized(date):
        b11, b12 = date[4], date[5]
        ratio = np.ones(b11.shape, np.float32)
        use = (b11 != 0) & np.isfinite(b11) & np.isfinite(b12)
        ratio[use] = b12[use] / b11[use]
        median = float(np.median(ratio[use])) if use.any() else 1.0
        if not np.isfinite(median) or abs(median) < 1e-8: median = 1.0
        ratio[use] /= median
        return np.clip(ratio, 0, 10)
    current = normalized(spectral[:6]); reference = normalized(spectral[6:])
    mbmp = np.ones((200, 200), np.float32)
    np.divide(current, reference, out=mbmp, where=reference != 0)
    mbmp[~np.isfinite(mbmp)] = 1
    mbmp = np.clip(mbmp, 0, 10)
    w = np.broadcast_to(np.array(wind, np.float32)[:, None, None] / 8, (2, 200, 200)).copy()
    return np.concatenate((mbmp[None], spectral, w, (ca > 0).astype(np.float32)[None])).astype(np.float32)


def prepare() -> None:
    items, rows, configs = source_check()
    dest = OUT / "data"
    if dest.exists(): raise FileExistsError(dest)
    dest.mkdir(parents=True)
    records = []
    for i, item in enumerate(items):
        row = rows[item["id"]]
        a, b, ca, y, hashes = read_source(item, row)
        wind = tuple(float(np.clip(float(row["official_csv_row"][k]), -20, 20)) for k in ("wind_u", "wind_v"))
        x = builder_input(a, b, ca, wind)
        if x.shape != (16, 200, 200) or not np.isfinite(x).all(): raise ValueError("Bad released input")
        valid = v13.scene(item)["valid"]
        path = dest / f"{item['id']}.npz"
        np.savez_compressed(path, x16=x, y=y, valid=valid)
        records.append({"id":item["id"],"split":"dev","scene_kind":item["scene_kind"],"site":item["site"],
                        "cache":path.name,"sha256":sha(path),"source_sha256":hashes,
                        "valid_pixels":int(valid.sum()),"label_pixels":int((y*valid).sum()),
                        "release_training_location":row["source_row"].get("physical_location_id"),
                        "country":row["official_csv_row"].get("country")})
        if (i + 1) % 50 == 0: print(json.dumps({"prepared":i+1,"total":198}),flush=True)
    if sum(r["valid_pixels"] for r in records) != 7_894_236: raise ValueError("Common native support drift")
    save(dest/"index.json", records)
    save(dest/"receipt.json", {"scene_count":198,"valid_pixels":7_894_236,"index_sha256":sha(dest/"index.json"),
         "source_manifest_sha256":v13.SOURCE_SHA,"v13_data_receipt_sha256":sha(v13.OUT/"data/receipt.json"),
         "protocol_sha256":sha(PROTOCOL),"script_sha256":sha(Path(__file__)),
         "released_helper_sha256":sha(Path(released.__file__)),
         "mbmp_helper_sha256":sha(ROOT/"EarthRemoteSensingRapidResponse/mars_s2l_adapter.py"),
         "checkpoints":{arm:sha(model_paths(arm)[0]) for arm in ARMS},
         "configs":{arm:sha(model_paths(arm)[1]) for arm in ARMS},
         "new_data_bytes":sum((dest/r["cache"]).stat().st_size for r in records)})


def index() -> list[dict]:
    receipt = json.loads((OUT/"data/receipt.json").read_text(encoding="utf-8"))
    if (receipt["script_sha256"] != sha(Path(__file__)) or receipt["protocol_sha256"] != sha(PROTOCOL) or
        receipt["index_sha256"] != sha(OUT/"data/index.json") or receipt["source_manifest_sha256"] != v13.SOURCE_SHA or
        receipt["released_helper_sha256"] != sha(Path(released.__file__)) or
        receipt["mbmp_helper_sha256"] != sha(ROOT/"EarthRemoteSensingRapidResponse/mars_s2l_adapter.py")):
        raise ValueError("v15 data source drift")
    return json.loads((OUT/"data/index.json").read_text(encoding="utf-8"))


def scene(item: dict, verify: bool = False) -> dict[str, np.ndarray]:
    path = OUT/"data"/item["cache"]
    if verify and sha(path) != item["sha256"]: raise ValueError("v15 cache drift")
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in ("x16", "y", "valid")}


def exposure(items: list[dict], configs: dict) -> dict:
    locations = {r["release_training_location"] for r in items}
    result = {}
    for arm in ARMS:
        config = configs[arm]
        train = set(config["all_locs_train"])
        val = set(config.get("all_locs_val", []))
        matched_train = locations & train
        matched_val = locations & val
        result[arm] = {"released_training_samples":config.get("n_samples_train"),
                       "released_positive_training_samples":config.get("n_pos_train"),
                       "released_train_location_count":len(train),
                       "dev_unique_release_locations":len(locations),
                       "dev_locations_in_released_train":sorted(matched_train),
                       "dev_locations_in_released_validation":sorted(matched_val),
                       "dev_observations_at_released_train_locations":sum(r["release_training_location"] in train for r in items),
                       "dev_observations_at_released_validation_locations":sum(r["release_training_location"] in val for r in items),
                       "release_location_field":"source_row.physical_location_id"}
    return result


def preflight() -> None:
    settings()
    items = index()
    native, rows, configs = source_check()
    if [r["id"] for r in items] != [r["id"] for r in native]: raise ValueError("Cache order drift")
    if sum(r["valid_pixels"] for r in items) != 7_894_236: raise ValueError("Support drift")
    # Real cases include both classes and invalid edge support.
    positives = [r for r in items if r["scene_kind"] == "PLUME"]
    clears = [r for r in items if r["scene_kind"] == "NO_PLUME"]
    examples = positives[:4] + clears[:4]
    parity = []
    for item in examples:
        native_item = next(r for r in native if r["id"] == item["id"])
        a, b, ca, y, _ = read_source(native_item, rows[item["id"]])
        wind = tuple(float(np.clip(float(rows[item["id"]]["official_csv_row"][k]), -20, 20)) for k in ("wind_u", "wind_v"))
        built = builder_input(a, b, ca, wind)
        independent = independent_input(a, b, ca, wind)
        data = scene(item, verify=True)
        if not np.array_equal(built, independent) or not np.array_equal(built, data["x16"]):
            raise ValueError("Released input independent parity: " + item["id"])
        if not np.array_equal(data["x16"][1:7], np.clip(a.astype(np.float32)/5000,0,2)):
            raise ValueError("CH4Net current-band parity")
        if not np.array_equal(data["y"], y) or not np.array_equal(data["valid"], v13.scene(native_item)["valid"]):
            raise ValueError("Sample label/support parity")
        parity.append({"id":item["id"],"invalid_edge_pixels":int((~data["valid"].astype(bool))[[0,-1],:].sum()),
                       "current_cloud_positive_pixels":int((ca>0).sum()),
                       "mbmp_differs_from_valid_aware_pixels":int(np.count_nonzero(data["x16"][0] != v13.scene(native_item)["x"][0]))})
    model_checks = {}
    for arm in ARMS:
        checkpoint, config_path = model_paths(arm)
        spec = released.RELEASE_SPECS[arm]
        model = released.load_released_model(checkpoint, torch.device("cpu"), spec["input_channels"])
        params = sum(p.numel() for p in model.parameters())
        if params != EXPECTED_PARAMS[arm] or model.training: raise ValueError("Released active model contract")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("_orig_mod.module.").removeprefix("module."):v for k,v in payload["model_state_dict"].items()}
        surplus = sorted(set(state) - set(model.state_dict()))
        missing = sorted(set(model.state_dict()) - set(state))
        if missing or any(not k.startswith("out_mlp.") for k in surplus): raise ValueError("Checkpoint state keys")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer); buffer.seek(0)
        copy = released.ReleasedUNet(spec["input_channels"])
        copy.load_state_dict(torch.load(buffer,map_location="cpu",weights_only=True),strict=True)
        if any(not torch.equal(v,copy.state_dict()[k]) for k,v in model.state_dict().items()): raise ValueError("Serialization mismatch")
        timings = []
        scores = []
        for item in examples:
            data = scene(item)
            x = data["x16"] if arm == "mars-s2l" else data["x16"][1:7]
            start = time.monotonic()
            with torch.inference_mode():
                p = torch.sigmoid(model(torch.from_numpy(x[None])))[0].numpy()
            timings.append(time.monotonic()-start)
            if p.shape != (200,200) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
                raise ValueError("Nonfinite released output")
            scores.append({"id":item["id"],"min":float(p.min()),"max":float(p.max()),
                           "valid_exact_half_scores":int(((p==.5)&data["valid"].astype(bool)).sum())})
        # Omit cold first case; estimate one full 198-scene pass with generous margin.
        warmed = float(np.median(timings[1:]))
        if warmed*198 >= 600: raise TimeoutError("Released CPU inference unsuitable")
        model_checks[arm] = {"active_parameters":params,"unexpected_unused_out_mlp_keys":surplus,
                             "missing_keys":missing,"checkpoint_sha256":sha(checkpoint),
                             "config_sha256":sha(config_path),"timings_seconds":timings,
                             "warmed_median_seconds":warmed,"projected_198_seconds":warmed*198,
                             "finite_score_examples":scores,"cpu_float32_eval":True}
    result = {"passed":True,"scene_count":198,"common_valid_native_pixels":7_894_236,
              "feature_channels":["MBMP_release_nonzero_B11", *["target_"+b for b in v13.BANDS],
                                  *["selected_reference_"+b for b in v13.BANDS],"wind_u_over_8","wind_v_over_8","current_cloud_nonzero"],
              "ch4net_channels":"x16[1:7] current bands only", "reference_choice":"separately selected earlier clear reference; differs from release benchmark",
              "source_reconstruction_cases":parity,"models":model_checks,"release_training_exposure":exposure(items, configs),
              "data_receipt_sha256":sha(OUT/"data/receipt.json"),"data_index_sha256":sha(OUT/"data/index.json"),
              "source_manifest_sha256":v13.SOURCE_SHA,"protocol_sha256":sha(PROTOCOL),
              "script_sha256":sha(Path(__file__)),"released_helper_sha256":sha(Path(released.__file__)),
              "mbmp_helper_sha256":sha(ROOT/"EarthRemoteSensingRapidResponse/mars_s2l_adapter.py"),
              "teacher_checkpoint_sha256":TEACHER_SHA,
              "new_artifact_bytes_at_preflight":sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
              "full_valid_exact_half_check":"required during infer; only examples checked at preflight"}
    save(OUT/"preflight/receipt.json", result)
    print(json.dumps({"passed":True,"scene_count":198,"common_valid_native_pixels":7_894_236,
                      "models":{a:{"active_parameters":v["active_parameters"],"projected_198_seconds":v["projected_198_seconds"]} for a,v in model_checks.items()},
                      "source_reconstruction_cases":len(parity),"new_artifact_bytes_at_preflight":result["new_artifact_bytes_at_preflight"],
                      "script_sha256":result["script_sha256"]}),flush=True)


def preflight_receipt() -> dict:
    receipt = json.loads((OUT/"preflight/receipt.json").read_text(encoding="utf-8"))
    if (not receipt["passed"] or receipt["script_sha256"] != sha(Path(__file__)) or
        receipt["protocol_sha256"] != sha(PROTOCOL) or receipt["data_receipt_sha256"] != sha(OUT/"data/receipt.json") or
        receipt["released_helper_sha256"] != sha(Path(released.__file__))):
        raise ValueError("v15 preflight drift")
    return receipt


def artifact_bytes() -> int:
    return sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())


def infer(arm: str) -> None:
    if arm not in ARMS: raise ValueError("Unknown model")
    settings(); preflight_receipt(); items=index(); source_check()
    destination=OUT/"inference"/arm
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir(parents=True)
    checkpoint,_=model_paths(arm)
    model=released.load_released_model(checkpoint,torch.device("cpu"),released.RELEASE_SPECS[arm]["input_channels"])
    start=time.monotonic(); records={}; pooled=collections.Counter(); clear_active=0; clear_fp=0; clear_valid=0; exact_half=0
    for item in items:
        if time.monotonic()-start>=600: raise TimeoutError("Released inference exceeded 10min")
        d=scene(item,verify=True)
        x=d["x16"] if arm=="mars-s2l" else d["x16"][1:7]
        with torch.inference_mode():
            p=torch.sigmoid(model(torch.from_numpy(x[None])))[0].numpy().astype(np.float32)
        if p.shape!=(200,200) or not np.isfinite(p).all(): raise ValueError("Invalid released probability")
        good=d["valid"].astype(bool); truth=d["y"].astype(bool)&good
        exact_half+=int(np.count_nonzero((p==.5)&good))
        if exact_half: raise ValueError("Exact valid 0.5 tie; >= differs from release >")
        mask=(p>=.5)&good
        c={"tp":int((mask&truth).sum()),"fp":int((mask&~truth).sum()),
           "fn":int((~mask&truth).sum()),"tn":int((~mask&~truth&good).sum())}
        pooled.update(c)
        if item["scene_kind"]=="NO_PLUME":
            clear_active+=int(c["fp"]>0); clear_fp+=c["fp"];clear_valid+=c["fp"]+c["tn"]
        path=destination/f"{item['id']}.npz"
        np.savez_compressed(path,probability_native=p,target_native=d["y"],valid_native=d["valid"],binary_native=mask.astype(np.uint8))
        records[item["id"]]={"probability_sha256":sha(path),"native_confusion":c,
                              "valid_native_pixels":int(good.sum()),"positive_native_pixels":int(truth.sum())}
        if artifact_bytes()>=2*1024**3: raise OSError("v15 artifact cap exceeded")
    tp,fp,fn=(pooled[k] for k in ("tp","fp","fn"))
    summary={"arm":arm,"cutoff":.5,"checkpoint_sha256":sha(checkpoint),"script_sha256":sha(Path(__file__)),
             "protocol_sha256":sha(PROTOCOL),"preflight_sha256":sha(OUT/"preflight/receipt.json"),
             "scene_count":198,"scenes":records,"native_pooled":{**dict(pooled),"precision":tp/max(tp+fp,1),
             "recall":tp/max(tp+fn,1),"f1":2*tp/max(2*tp+fp+fn,1)},
             "clear_activated_scenes":clear_active,"clear_pixel_fpr":clear_fp/max(clear_valid,1),
             "exact_valid_half_scores":exact_half,"duration_seconds":time.monotonic()-start}
    save(destination/"summary.json",summary)


def replay(arm: str) -> None:
    if arm not in ARMS: raise ValueError("Unknown model")
    settings(); preflight_receipt(); items=index()
    destination=OUT/"replay"/arm
    if destination.exists(): raise FileExistsError(destination)
    metadata=json.loads((OUT/"inference"/arm/"summary.json").read_text(encoding="utf-8"))
    checkpoint,_=model_paths(arm)
    if metadata["checkpoint_sha256"]!=sha(checkpoint) or metadata["script_sha256"]!=sha(Path(__file__)):
        raise ValueError("Changed inference source")
    model=released.load_released_model(checkpoint,torch.device("cpu"),released.RELEASE_SPECS[arm]["input_channels"])
    start=time.monotonic(); count=0
    for item in items:
        if time.monotonic()-start>=600: raise TimeoutError("Released replay exceeded 10min")
        d=scene(item,verify=True); x=d["x16"] if arm=="mars-s2l" else d["x16"][1:7]
        with torch.inference_mode():
            p=torch.sigmoid(model(torch.from_numpy(x[None])))[0].numpy().astype(np.float32)
        path=OUT/"inference"/arm/f"{item['id']}.npz"
        if sha(path)!=metadata["scenes"][item["id"]]["probability_sha256"]: raise ValueError("Saved probability hash drift")
        with np.load(path,allow_pickle=False) as z:
            if (not np.array_equal(p,z["probability_native"]) or not np.array_equal(d["y"],z["target_native"]) or
                not np.array_equal(d["valid"],z["valid_native"]) or not np.array_equal(((p>=.5)&d["valid"].astype(bool)).astype(np.uint8),z["binary_native"])):
                raise ValueError("Non-exact released CPU replay: "+item["id"])
        count+=1
    save(destination/"receipt.json",{"passed":True,"arm":arm,"scene_count":count,"exact_probability_label_support_binary":True,
         "duration_seconds":time.monotonic()-start,"inference_summary_sha256":sha(OUT/"inference"/arm/"summary.json"),
         "script_sha256":sha(Path(__file__)),"protocol_sha256":sha(PROTOCOL)})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","preflight","infer","replay"))
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.command=="prepare": prepare()
    elif args.command=="preflight": preflight()
    elif args.command=="infer":
        if not args.arm: parser.error("--arm required")
        infer(args.arm)
    elif args.command=="replay":
        if not args.arm: parser.error("--arm required")
        replay(args.arm)


if __name__=="__main__": main()
