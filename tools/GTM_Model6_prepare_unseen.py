"""Seal cached real unseen scenes before frozen CNN 2 inference. No network I/O.

Protocol and failure cases: docs/GTM_Model6_unseen_evaluation_20261023.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
MARS = ROOT.parent / ("Earth-Remote-Sensing-Rapid-Response-/EarthRemoteSensingRapidResponse/"
                     "Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L")
FIT = ROOT / "outputs/GTM_Model6/GTM_Model6_real_emit_20261023/cnn2_fit_ready"
GEORGIA = ROOT / "outputs/GTM_Model6/GTM_Model6_georgia_crop_repair_20260922"
BANDS = ["B02", "B03", "B04", "B11", "B12"]
INDICES = [1, 2, 3, 5, 6]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source(path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError("Changed source: " + str(path))
    return dict(path=str(path), sha256=actual, bytes=path.stat().st_size)


def asset(row, role):
    item = next(a for a in row["assets"] if a["role"] == role)
    path = (MARS / item["path"]).resolve()
    if not path.is_relative_to(MARS.resolve()):
        raise ValueError("Asset escapes source directory")
    body = path.read_bytes()
    if len(body) != item["size"]:
        raise ValueError("Cached source size differs from pinned upstream")
    digest = hashlib.sha256(body).hexdigest()
    if item["remote_oid_type"] == "sha256_lfs":
        upstream = digest
    elif item["remote_oid_type"] == "git_blob_sha1":
        upstream = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
    else:
        raise ValueError("Unknown upstream identity type")
    if upstream != item["remote_oid"]:
        raise ValueError("Cached source differs from pinned upstream hash")
    return path, dict(path=str(path), sha256=digest, bytes=len(body),
                      upstream_identity_verified=True, role=role,
                      source_url=item["source_url"])


def km(a, b):
    y1, y2 = math.radians(a[0]), math.radians(b[0])
    dy, dx = y2-y1, math.radians(b[1]-a[1])
    return 12742 * math.asin(min(1, math.sqrt(math.sin(dy/2)**2 +
                    math.cos(y1)*math.cos(y2)*math.sin(dx/2)**2)))


def grid(shape, affine, crs):
    return dict(shape=list(shape), affine=list(affine)[:6], crs=str(crs))


def mars_arrays(row):
    img_path, img_source = asset(row, "image")
    cloud_path, cloud_source = asset(row, "cloud_mask")
    sources = [img_source, cloud_source]
    with rasterio.open(img_path) as src, rasterio.open(cloud_path) as clouds:
        expected = ["B02", "B03", "B04", "B08", "B11", "B12"]
        if (src.shape != (200, 200) or src.count != 12 or src.res != (10, 10) or
                row["band_order"][:6] != expected or list(src.descriptions[:6]) != expected or
                row["product_level"] != "L1C" or str(src.crs) != row["crs"] or
                clouds.shape != src.shape or clouds.transform != src.transform or clouds.crs != src.crs):
            raise ValueError("Changed observed Sentinel/cloud grid or band contract")
        raw = np.moveaxis(src.read(INDICES), 0, -1).astype(np.float32)
        cloud = clouds.read(1)
        if not np.isin(cloud, [0, 1, 2, 3]).all():
            raise ValueError("Unknown CloudSEN12 code")
        clear = (cloud == 0) & np.isfinite(raw).all(-1) & (raw != src.nodata).all(-1)
        image = raw.reshape(100, 2, 100, 2, 5).mean(axis=(1, 3))
        valid = clear.reshape(100, 2, 100, 2).all(axis=(1, 3))
        support = np.repeat(np.repeat(valid, 2, 0), 2, 1)
        transform10, crs = src.transform, src.crs
        if row["label_state"] == "PLUME":
            label_path, label_source = asset(row, "plume_mask")
            sources.append(label_source)
            with rasterio.open(label_path) as labels:
                if labels.shape != src.shape or labels.transform != src.transform or labels.crs != src.crs:
                    raise ValueError("Source annotation is not on the input footprint")
                target = labels.read(1).astype(np.float32)
                if not np.isin(target, [0, 1]).all() or not np.any(target == 1):
                    raise ValueError("Positive record has no binary producer plume annotation")
        else:
            # Whole observed scene is labeled NO_PLUME by the reviewed release.
            # A missing mask alone never creates this class.
            if row["label_state"] != "NO_PLUME":
                raise ValueError("Unknown producer scene class")
            target = np.zeros((200, 200), np.float32)
    return dict(image=image, input_valid=valid, target=target, target_valid=support,
                cloud_mask_native=cloud), sources, grid(valid.shape, transform10*Affine.scale(2), crs), grid(target.shape, transform10, crs)


def prepare_georgia(dest):
    qa_path = GEORGIA / "qa_v2/GTM_Model6_georgia_repaired_pair_qa.json"
    qa = json.loads(qa_path.read_text())
    sources = [source(qa_path), source(GEORGIA / "report.json", qa["acquisition_receipt_sha256"]),
               source(GEORGIA / "GTM_Model6_stac_item.json", qa["stac_sha256"])]
    with rasterio.open(GEORGIA / "GTM_Model6_scl.tif") as scl:
        affine, crs, shape = scl.transform, scl.crs, scl.shape
        image = np.zeros((*shape, 5), np.float32)
        valid = np.isin(scl.read(1), [4, 5, 6])
        sources.append(source(GEORGIA / "GTM_Model6_scl.tif", qa["bands"]["scl"]["sha256"]))
        for i, name in enumerate(["blue", "green", "red", "swir16", "swir22"]):
            path = GEORGIA / f"GTM_Model6_{name}.tif"
            sources.append(source(path, qa["bands"][name]["sha256"]))
            with rasterio.open(path) as src:
                support = np.zeros(shape, np.float32)
                raw = src.read(1)
                reproject(raw, image[..., i], src_transform=src.transform, src_crs=src.crs,
                          dst_transform=affine, dst_crs=crs, src_nodata=src.nodata,
                          dst_nodata=0, resampling=Resampling.average)
                reproject((np.isfinite(raw) & (raw != src.nodata)).astype(np.float32), support,
                          src_transform=src.transform, src_crs=src.crs, dst_transform=affine,
                          dst_crs=crs, resampling=Resampling.average)
                valid &= support > .99999
    info = qa["native"]["plm"]
    sources.append(source(info["path"], info["sha256"]))
    with rasterio.open(info["path"]) as ds:
        r, c, h, w = info["window"]
        window = Window(c, r, w, h)
        retrieval = ds.read(1, window=window).astype(np.float32)
        retrieval[~np.isfinite(retrieval) | (retrieval == ds.nodata)] = np.nan
        retrieval_grid = grid(retrieval.shape, ds.window_transform(window), ds.crs)
        display_retrieval = np.full(shape, np.nan, np.float32)
        reproject(retrieval, display_retrieval, src_transform=ds.window_transform(window),
                  src_crs=ds.crs, dst_transform=affine, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    path = dest / "Georgia-20241020.npz"
    np.savez_compressed(path, image=image, input_valid=valid, target=np.full(shape, np.nan, np.float32),
                        target_valid=np.zeros(shape, bool), retrieval_native=retrieval,
                        retrieval_display=display_retrieval)
    stac = json.loads((GEORGIA / "GTM_Model6_stac_item.json").read_text())
    return dict(id="Georgia-20241020", name="Georgia | unseen L2A and measured EMIT", group_id="Georgia-20241020",
                scene_kind="retrieval_context", source_level="L2A", satellite_id=qa["scene_id"],
                satellite_time=stac["properties"]["datetime"], reference_time="2024-10-20T17:05:04+00:00",
                offset_minutes=qa["sentinel_minus_emit_hours"]*60, input_grid=grid(shape, affine, crs),
                reference_grid=grid(shape, affine, crs), retrieval_grid=retrieval_grid,
                reference_mapping="none", reference_kind="continuous_retrieval_context",
                retrieval_units="ppm m", array_file=path.name, array_sha256=sha(path), sources=sources,
                independent_of_training=True, notes="Unseen L2A input. EMIT is a signed continuous retrieval, not a binary reference. No segmentation accuracy is available. Raw DN retained; BOA offset metadata remains ambiguous. Sentinel precedes EMIT by 31 minutes. The published peak is inside this repaired crop.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError("Prepared data directory must be new")
    receipt_path = FIT / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    fit_manifest = json.loads(Path(receipt["manifest_path"]).read_text())
    if sha(FIT / "final.pt") != receipt["final"]["sha256"]:
        raise ValueError("Accepted checkpoint changed")
    fit_bounds = array_bounds(*fit_manifest["input_grid"]["shape"], Affine(*fit_manifest["input_grid"]["affine"]))
    fit_bounds = transform_bounds(fit_manifest["input_grid"]["crs"], "EPSG:4326", *fit_bounds)
    manifest_path = MARS / "publication_v3_training_samples.jsonl"
    rows = [json.loads(line) for line in manifest_path.open(encoding="utf-8")]
    candidates = [r for r in rows if r.get("country") == "United States of America" and
                  r.get("sensor") == "Sentinel-2 MSI" and r.get("product_level") == "L1C"]
    audit, prepared, points, sites = [], [], [], set()
    dest.mkdir(parents=True)
    for label in ("PLUME", "NO_PLUME"):
        accepted = 0
        for row in sorted((r for r in candidates if r["label_state"] == label), key=lambda r:r["sample_id"]):
            if accepted == 8:
                break
            point = row["latitude"], row["longitude"]
            nearest_fit = (min(max(point[0], fit_bounds[1]), fit_bounds[3]),
                           min(max(point[1], fit_bounds[0]), fit_bounds[2]))
            why = None
            if row["observability"] != "clear":
                why = "producer_observability_not_clear"
            elif row["physical_location_id"] in sites:
                why = "already_selected_site"
            elif km(point, nearest_fit) < 25:
                why = "within_25_km_of_fitting_footprint"
            elif any(km(point, p) < 25 for p in points):
                why = "within_25_km_of_selected_site"
            elif row["target_scene_id"] == fit_manifest["satellite"]["id"]:
                why = "fitting_satellite_identity"
            if why:
                audit.append(dict(id=row["sample_id"], label=label, accepted=False, reason=why))
                continue
            try:
                arrays, sources, input_grid, reference_grid = mars_arrays(row)
            except FileNotFoundError:
                audit.append(dict(id=row["sample_id"], label=label, accepted=False, reason="missing_cached_asset"))
                continue
            clear_fraction = float(arrays["input_valid"].mean())
            if clear_fraction < .9 or (label == "PLUME" and not np.any(arrays["target_valid"] & (arrays["target"] == 1))):
                audit.append(dict(id=row["sample_id"], label=label, accepted=False, reason="less_than_90pct_clear_or_no_supported_plume", clear_fraction=clear_fraction))
                continue
            identifier = "MARS_" + row["sample_id"]
            array_path = dest / (identifier + ".npz")
            np.savez_compressed(array_path, **arrays)
            record = dict(id=identifier, name=("Plume" if label == "PLUME" else "No plume") + " | " + row["target_datetime"][:10] + " | " + row["sample_id"][:8],
                          group_id=row["physical_location_id"], scene_kind=label,
                          source_level="L1C", satellite_id=row["target_scene_id"], satellite_time=row["target_datetime"],
                          reference_time=row["target_datetime"], input_grid=input_grid, reference_grid=reference_grid,
                          array_file=array_path.name, array_sha256=sha(array_path), sources=sources,
                          reference_mapping="repeat_2", reference_kind="reviewed_binary_annotation",
                          independent_of_training=True, latitude=point[0], longitude=point[1],
                          clear_fraction=clear_fraction, fit_footprint_distance_km=km(point, nearest_fit),
                          source_row=row, source_label=label, label_source=row["label_source"],
                          notes="Unseen relative to the Volusia-only CNN 2 fit. Observed target-date Sentinel L1C DN, unlike fitting L2A DN. Reviewed MARS source annotation, not EMIT truth. Fixed fitting normalization and 0.50 cutoff. No-plume means no identified plume in this observation. Original 10 m annotation retained; 20 m predictions repeat exactly over each 2 by 2 source block. Other project models have previously used MARS.")
            prepared.append(record)
            points.append(point)
            sites.add(row["physical_location_id"])
            accepted += 1
            audit.append(dict(id=row["sample_id"], label=label, accepted=True, clear_fraction=clear_fraction))
    prepared.append(prepare_georgia(dest))
    for a in prepared[:-1]:
        for b in prepared[:-1]:
            if a is not b and km((a["latitude"],a["longitude"]), (b["latitude"],b["longitude"])) < 25:
                raise ValueError("Selection sites are too close")
    manifest = dict(schema="GTM_Model6_real_unseen_cohort_v1", synthetic=False, role="unseen_diagnostic",
                    downloads=0, bands=BANDS, training_receipt_sha256=sha(receipt_path),
                    training_checkpoint_sha256=receipt["final"]["sha256"],
                    protocol=source(ROOT / "docs/GTM_Model6_unseen_evaluation_20261023.md"),
                    preparation_source=source(__file__), source_manifest=source(manifest_path),
                    dataset_readme=source(MARS / "README.md"),
                    selection_rule="PLUME then NO_PLUME, lexicographic sample ID, up to 8 per class, >=90% clear valid cells, >=25km between sites and from Volusia fitting footprint; all before model inference",
                    selection_audit=audit, scenes=prepared,
                    source_revision="c26b1d7e31a0c5241fa37c9140802622c215eb32",
                    imagery_ledger_before=source(ROOT / "outputs/cafo_pilot_2026-09-11/range_cache/budget.json"))
    save(dest / "manifest.json", manifest)
    save(dest / "selection.json", dict(selected=[{k:r.get(k) for k in ("id","name","scene_kind","latitude","longitude","satellite_id","clear_fraction")} for r in prepared], audit=audit))
    print(json.dumps(dict(manifest=str(dest/"manifest.json"), selected=len(prepared),
                         classes={k:sum(r["scene_kind"]==k for r in prepared) for k in ["PLUME","NO_PLUME","retrieval_context"]}, downloads=0), indent=2))


if __name__ == "__main__":
    main()
