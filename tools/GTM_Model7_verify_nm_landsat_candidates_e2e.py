"""Independent cached-TIFF and preview replay for the two New Mexico Landsat candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/GTM_Model7_nontexas_readiness_20261024/nm_landsat_candidates"
OLD = ROOT / "outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json"
EXPECTED = {"0f41981f-2751-4bba-ab02-3035ef8df2c8": "204a3ea0-15ec-45c1-8436-ba2cada2708b",
            "18e171df-18a2-4316-aee8-8064a2221042": "fb5870e5-7e4e-470e-89b9-c535ed9ab27c"}
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(source: dict) -> Path:
    path = Path(source["path"])
    if not path.is_file() or path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
        raise ValueError("Source hash/size mismatch: " + str(path))
    git = hashlib.sha1()
    git.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            git.update(chunk)
    remote = source["sha256"] if source["remote_oid_type"] == "sha256_lfs" else git.hexdigest()
    if remote != source["remote_oid"]:
        raise ValueError("Upstream identity mismatch: " + str(path))
    return path


def km(a, b):
    y1, y2 = math.radians(a[0]), math.radians(b[0])
    z = math.sin((y2-y1)/2)**2 + math.cos(y1)*math.cos(y2)*math.sin(math.radians(b[1]-a[1])/2)**2
    return 12742*math.asin(min(1, math.sqrt(z)))


def run(dest: Path) -> dict:
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old = json.loads(OLD.read_text(encoding="utf-8"))
    if manifest["schema"] != "GTM_Model7_landsat_candidate_v1" or \
            manifest["old_mars_manifest"]["sha256"] != sha(OLD) or len(old["scenes"]) != 565:
        raise ValueError("Source manifest contract failed")
    eval_scenes = [s for s in old["scenes"] if s["split"] in ("dev", "test")]
    if len(manifest["candidates"]) != 2:
        raise ValueError("Candidate count differs")
    results = []
    for item in manifest["candidates"]:
        sid = item["target_observation_id"]
        if EXPECTED.get(sid) != item["reference_observation_id"]:
            raise ValueError("Wrong target/reference pair")
        if item["sensor_family"] != "Landsat" or item["product_level"] != "Collection 2 TOA" or \
                item["band_order"] != BANDS or item["training_array"] is not None:
            raise ValueError("Sensor or blocked-array contract failed")
        p = {s["role"]: checked(s) for s in item["sources"]}
        with rasterio.open(p["target_image"]) as t, rasterio.open(p["reference_image"]) as r, \
                rasterio.open(p["target_cloud"]) as tc, rasterio.open(p["reference_cloud"]) as rc, \
                rasterio.open(p["plume_mask"]) as ld:
            row, ref_row = item["target_official_train_row"], item["reference_official_train_row"]
            for ds, csv_row in ((t, row), (r, ref_row)):
                expected_affine = Affine(*(float(csv_row["transform_" + key]) for key in "abcdef"))
                if ds.shape != (200, 200) or ds.count != 12 or ds.res != (10, 10) or \
                        ds.transform != expected_affine or str(ds.crs) != csv_row["crs"] or \
                        ds.dtypes != ("uint16",)*12 or list(ds.descriptions) != BANDS + [b+"_bg" for b in BANDS]:
                    raise ValueError("Raster/CSV grid or spectral contract failed: " + sid)
            if t.transform != r.transform or t.crs != r.crs:
                raise ValueError("Target/reference grid changed: " + sid)
            for ds in (tc, rc, ld):
                if ds.shape != t.shape or ds.transform != t.transform or ds.crs != t.crs or \
                        ds.count != 1 or ds.dtypes[0] != "uint8":
                    raise ValueError("Cloud/label grid failed: " + sid)
            target = np.moveaxis(t.read([1, 2, 3, 4, 5, 6]), 0, -1).astype(np.float32)
            reference = np.moveaxis(r.read([1, 2, 3, 4, 5, 6]), 0, -1).astype(np.float32)
            ct, cr, label = tc.read(1), rc.read(1), ld.read(1)
            if not np.isin(ct, [0, 1, 2, 3, 4]).all() or not np.isin(cr, [0, 1, 2, 3, 4]).all() or \
                    not np.isin(label, [0, 1]).all() or not label.any():
                raise ValueError("Cloud/label values failed: " + sid)
            support = (ct == 0) & (cr == 0) & np.isfinite(target).all(-1) & \
                      np.isfinite(reference).all(-1) & (target > 0).all(-1) & (reference > 0).all(-1)
            for ds, image in ((t, target), (r, reference)):
                if ds.nodata is not None:
                    support &= (image != ds.nodata).all(-1)
            if float(support.mean()) < .9 or not np.any(label & support):
                raise ValueError("Support QA failed: " + sid)
            snow = {}
            for name, image in (("target", target), ("reference", reference)):
                g, nir, swir = image[:, :, 1], image[:, :, 3], image[:, :, 4]
                snow[name] = float(((g > 3000) & (nir > 1000) & ((g-swir)/np.maximum(g+swir, 1) > .4) & support).sum() / support.sum())
                if snow[name] > .10:
                    raise ValueError("Snow-like QA failed: " + sid)
            qa = item["raster_quality"]
            if abs(qa["valid_fraction"]-float(support.mean())) > 1e-12 or \
                    qa["supported_positive_pixels"] != int(np.count_nonzero(label & support)) or \
                    any(abs(qa["snow_like_fraction"][k]-snow[k]) > 1e-12 for k in snow):
                raise ValueError("Saved QA differs: " + sid)
            point = (float(row["lat"]), float(row["lon"]))
            nearest = min((km(point, (s["latitude"], s["longitude"])), s["id"], s["split"])
                          for s in eval_scenes)
            block = item["old_evaluation_exclusion"]
            if nearest[0] >= 25 or "within_25km_old_dev_test" not in block["reasons"] or \
                    abs(nearest[0]-block["nearest_old_dev_test_km"]) > 1e-9 or \
                    nearest[1] != block["nearest_old_dev_test_scene_id"]:
                raise ValueError("Old DEV/TEST exclusion differs: " + sid)
            a = datetime.fromisoformat(item["target_time"])
            b = datetime.fromisoformat(item["reference_time"])
            if a.date() == b.date() or not 24 <= (a-b).total_seconds()/3600 <= 1440:
                raise ValueError("Chronology failed: " + sid)
            colors = [2, 1, 0]
            low = np.percentile(np.concatenate((target[:, :, colors], reference[:, :, colors]), axis=0), 2, axis=(0, 1))
            high = np.percentile(np.concatenate((target[:, :, colors], reference[:, :, colors]), axis=0), 98, axis=(0, 1))
            for name, image in (("target", target), ("reference", reference)):
                rgb = (np.clip((image[:, :, colors]-low)/np.maximum(high-low, 1), 0, 1)*255).astype(np.uint8)
                info = item["previews"][name]
                if sha(Path(info["path"])) != info["sha256"] or not np.array_equal(np.asarray(Image.open(info["path"]).convert("RGB")), rgb):
                    raise ValueError("Image preview replay differs: " + sid + ":" + name)
            preview = np.zeros((200, 200, 3), dtype=np.uint8)
            preview[support] = (35, 45, 52)
            preview[label.astype(bool) & support] = (255, 115, 62)
            preview[label.astype(bool) & ~support] = (150, 55, 45)
            info = item["previews"]["annotation_support"]
            if sha(Path(info["path"])) != info["sha256"] or not np.array_equal(np.asarray(Image.open(info["path"]).convert("RGB")), preview):
                raise ValueError("Annotation preview replay differs: " + sid)
            results.append({"target_id": sid, "reference_id": item["reference_observation_id"],
                            "source_hashes": {k: sha(v) for k, v in p.items()},
                            "valid_fraction": float(support.mean()),
                            "supported_positive_pixels": int(np.count_nonzero(label & support)),
                            "nearest_old_dev_test_km": nearest[0], "blocked_from_fitting": True})
    if (dest / "arrays").exists() and list((dest / "arrays").glob("*.npz")):
        raise ValueError("Blocked cases have fitting arrays")
    return {"schema": "GTM_Model7_landsat_candidate_e2e_v1", "manifest_sha256": sha(manifest_path),
            "old_manifest_sha256": sha(OLD), "source_replayed": len(results),
            "preview_pngs_replayed": 3*len(results), "training_arrays": 0,
            "candidates": results, "downloads": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    dest = args.output.resolve()
    receipt = run(dest)
    path = dest / "e2e_receipt.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Repeat replay differs from existing E2E receipt")
    else:
        path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
