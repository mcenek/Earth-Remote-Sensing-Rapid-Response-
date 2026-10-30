"""Independently check and publish every frozen real-scene prediction.

Protocol: docs/GTM_Model6_unseen_evaluation_20261023.md. No training or network.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from GTM_Model6_prepare_unseen import sha, save, km
from GTM_Model0_local_viewer import load_bundle
from GTM_Model0_publish_viewer import publish

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "GTM_Model6_real_unseen_20261023_cnn2"


def counts(p, t, v):
    yes, truth = p[v] >= .5, t[v] == 1
    tp, fp, fn, tn = (int(np.count_nonzero(x)) for x in
                       (yes & truth, yes & ~truth, ~yes & truth, ~yes & ~truth))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def ratios(c):
    tp, fp, fn, tn = (c[k] for k in ["tp", "fp", "fn", "tn"])
    return dict(**c, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                evaluated_pixels=tp+fp+fn+tn)


def percent(x, digits=2):
    return f"{x:.{digits}%}" if x is not None else "n/a"


def encode(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def rgb_stretch(image, valid):
    rgb = image[..., [2, 1, 0]].copy()
    ranges = []
    for b in range(3):
        lo, hi = np.percentile(rgb[..., b][valid], [2, 98])
        ranges.append([float(lo), float(hi)])
        rgb[..., b] = np.clip((rgb[..., b]-lo)/max(hi-lo, 1), 0, 1)
    rgb[~valid] = .18
    return (rgb*255).astype(np.uint8), ranges


def paint(rgb, p, t, valid, known):
    reference = rgb.copy()
    positive = known & (t == 1)
    reference[positive] = (reference[positive]*.35 + np.array([0,225,230])*.65).astype(np.uint8)
    overlay = rgb.copy()
    predicted = valid & (p >= .5)
    overlay[predicted] = (overlay[predicted]*.55 + np.array([255,135,35])*.45).astype(np.uint8)
    # A one-cell cyan boundary leaves the input and predicted footprint visible.
    interior = positive.copy()
    for axis in (0, 1):
        interior &= np.roll(positive, 1, axis) & np.roll(positive, -1, axis)
    interior[[0,-1],:] = False
    interior[:,[0,-1]] = False
    overlay[positive & ~interior] = [0,225,230]
    error = rgb.copy()
    error[known & predicted & (t == 1)] = [0,165,151]
    error[known & predicted & (t == 0)] = [247,71,122]
    error[known & ~predicted & (t == 1)] = [152,121,240]
    error[~valid] = [70,79,90]
    return reference, overlay, error


def verify_original(row, prepared, source_rows):
    """Rebuild expected input/support/labels directly from acquisition rasters."""
    if row["reference_kind"] != "reviewed_binary_annotation":
        if prepared["target_valid"].any() or not np.isnan(prepared["target"]).all():
            raise ValueError("Continuous observation was converted to a binary reference")
        return
    original = source_rows[row["source_row"]["sample_id"]]
    if original != row["source_row"]:
        raise ValueError("Selected row differs from frozen publication manifest")
    files = {s["role"]: Path(s["path"]) for s in row["sources"]}
    with rasterio.open(files["image"]) as ds, rasterio.open(files["cloud_mask"]) as clouds:
        raw = np.moveaxis(ds.read([1,2,3,5,6]), 0, -1).astype(np.float32)
        h, w = ds.shape
        expected = (raw[0::2,0::2]+raw[1::2,0::2]+raw[0::2,1::2]+raw[1::2,1::2])/4
        clear = (clouds.read(1) == 0) & np.all(raw != ds.nodata, -1) & np.isfinite(raw).all(-1)
        valid = clear[0::2,0::2] & clear[1::2,0::2] & clear[0::2,1::2] & clear[1::2,1::2]
        if not np.array_equal(expected, prepared["image"]) or not np.array_equal(valid, prepared["input_valid"]):
            raise ValueError("Prepared input differs from original target-date bands")
        if row["input_grid"]["affine"] != list(ds.transform*Affine.scale(2))[:6] or row["reference_grid"]["affine"] != list(ds.transform)[:6]:
            raise ValueError("Prepared grid differs from source raster")
        if not np.array_equal(prepared["target_valid"], np.kron(valid, np.ones((2,2),bool))):
            raise ValueError("Prepared annotation support differs from source cloud/input coverage")
    if original["label_state"] == "NO_PLUME":
        expected_target = np.zeros((h,w), np.float32)
    else:
        with rasterio.open(files["plume_mask"]) as ds:
            expected_target = ds.read(1)
    if not np.array_equal(expected_target, prepared["target"]):
        raise ValueError("Prepared reference differs from original reviewed labels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    receipt = json.loads((args.predictions/"receipt.json").read_text())
    replay = json.loads(args.replay.read_text())
    if not replay["passed"] or replay["inference_receipt_sha256"] != sha(args.predictions/"receipt.json") or receipt["manifest_sha256"] != sha(args.manifest):
        raise ValueError("No matching successful fresh-process replay")
    if len(manifest["scenes"]) != len(receipt["scenes"]):
        raise ValueError("Not all selected scenes were predicted")
    source_doc = manifest["source_manifest"]
    if sha(source_doc["path"]) != source_doc["sha256"]:
        raise ValueError("Original publication manifest changed")
    source_rows = {r["sample_id"]: r for r in (json.loads(l) for l in Path(source_doc["path"]).open())}
    for item in [manifest["protocol"], manifest["dataset_readme"], manifest["preparation_source"], manifest["imagery_ledger_before"]]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Changed preparation evidence or download ledger")
    immutable = {str(p):sha(p) for folder in [args.predictions,Path(receipt["training_run"])] for p in folder.rglob("*") if p.is_file()}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scenes, rows, articles, image_rows, independent = [], [], [], [], []
    totals = {k:0 for k in ["tp","fp","fn","tn"]}
    negative = dict(fp=0, tn=0, scenes=0, activated=0)
    control = {k:0 for k in totals}
    for index, (row, record) in enumerate(zip(manifest["scenes"], receipt["scenes"])):
        if row["id"] != record["id"] or sha(args.predictions/record["array_file"]) != record["array_sha256"]:
            raise ValueError("Changed prediction identity")
        for s in row["sources"]:
            if sha(s["path"]) != s["sha256"]:
                raise ValueError("Changed original observation")
        if sha(args.manifest.parent/row["array_file"]) != row["array_sha256"]:
            raise ValueError("Changed prepared scene")
        with np.load(args.predictions/record["array_file"], allow_pickle=False) as z:
            prediction = {k:z[k] for k in z.files}
        with np.load(args.manifest.parent/row["array_file"], allow_pickle=False) as z:
            prepared = {k:z[k] for k in z.files}
        verify_original(row, prepared, source_rows)
        for key in ["input_valid","target","target_valid"]:
            if not np.array_equal(prediction[key], prepared[key], equal_nan=True):
                raise ValueError("Saved input support or labels changed")
        fine, valid = prediction["fine_probability"], prediction["input_valid"]
        p, t, known = prediction["reference_probability"], prediction["target"], prediction["target_valid"]
        if not np.isfinite(fine[valid]).all() or np.any((fine[valid]<0)|(fine[valid]>1)):
            raise ValueError("Invalid model scores")
        binary = row["reference_kind"] == "reviewed_binary_annotation"
        if binary and not np.array_equal(p, np.repeat(np.repeat(fine,2,0),2,1), equal_nan=True):
            raise ValueError("Hidden interpolation between output and source annotation")
        measured = ratios(counts(p,t,known)) if binary else None
        if measured:
            if any(measured[k] != record["summary"]["native_reference_metrics"][k] for k in totals):
                raise ValueError("Independent counts differ from inference report")
            for k in totals:
                totals[k] += measured[k]
                control[k] += counts(np.ones_like(p),t,known)[k]
            if row["scene_kind"] == "NO_PLUME":
                negative["scenes"] += 1
                negative["activated"] += int(measured["fp"]>0)
                negative["fp"] += measured["fp"]
                negative["tn"] += measured["tn"]
        rgb, stretch = rgb_stretch(prepared["image"],valid)
        display_valid = np.repeat(np.repeat(valid,2,0),2,1) if binary else valid
        if binary:
            rgb = np.repeat(np.repeat(rgb,2,0),2,1)
        reference, overlay, errors = paint(rgb,p,t,display_valid,known)
        folder = out / f"scene_{index:03d}"
        folder.mkdir()
        assets = {"input":rgb,"reference_figure":reference,"overlay_figure":overlay,"errors_figure":errors}
        heat = (colormaps["viridis"](np.nan_to_num(p))*255).astype(np.uint8)
        heat[~display_valid] = [70,79,90,255]
        assets["score_figure"] = heat
        if not binary:
            retrieval = prepared["retrieval_display"]
            finite = np.isfinite(retrieval)
            hi = float(np.nanpercentile(retrieval,98))
            rgba = (colormaps["magma"](np.clip(np.nan_to_num(retrieval)/max(hi,1),0,1))*255).astype(np.uint8)
            rgba[~finite] = [70,79,90,255]
            assets["reference_figure"] = rgba
            reference = rgba[...,:3]
        for name, arr in assets.items():
            Image.fromarray(arr).save(folder/(name+".png"))
        image_rows.append((row["name"], [rgb,reference,overlay,errors]))
        # Shared Mercator display only; native-source counts above remain frozen.
        g = row["reference_grid"] if binary else row["input_grid"]
        affine, crs = Affine(*g["affine"]),g["crs"]
        h,w = p.shape
        mercator = transform_bounds(crs,"EPSG:3857",*array_bounds(h,w,affine),densify_pts=21)
        destination = from_bounds(*mercator,w,h)
        def project(a, smooth=False):
            result=np.full((h,w),np.nan,np.float32)
            reproject(a.astype(np.float32),result,src_transform=affine,src_crs=crs,dst_transform=destination,dst_crs="EPSG:3857",src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.bilinear if smooth else Resampling.nearest)
            return result
        pp=project(p)
        tt=project(np.where(known,t,np.nan)) if binary else None
        vv=np.isfinite(pp)
        rg=np.stack([project(rgb[...,b],True) for b in range(3)],axis=-1)
        Image.fromarray(np.clip(np.nan_to_num(rg),0,255).astype(np.uint8)).save(folder/"satellite.png")
        rgba=np.zeros((h,w,4),np.uint8)
        rgba[vv & (pp>=.5)]=[255,135,35,210]
        Image.fromarray(rgba).save(folder/"prediction.png")
        if binary:
            rgba[:]=0
            rgba[np.isfinite(tt)&(tt==1)]=[0,225,230,190]
            Image.fromarray(rgba).save(folder/"reference.png")
        values=lambda a:[float(v) if np.isfinite(v) else None for v in a.ravel()]
        payload=dict(width=w,height=h,probability=values(pp),valid=vv.astype(np.uint8).ravel().tolist(),crs="EPSG:3857")
        if binary:
            payload["truth"]=values(tt)
        save(folder/"grid.json",payload)
        west,south,east,north=transform_bounds("EPSG:3857","EPSG:4326",*mercator)
        frac=record["summary"]["valid_input_positive_fraction"]
        note=("FAILED UNSEEN-SCENE TEST. " if binary else "UNSCORED UNSEEN EMIT CONTEXT. ")+row["notes"]
        if row["scene_kind"]=="NO_PLUME":
            note += f" False alarm: {percent(frac)} of valid input area is marked positive on a reviewed no-plume observation."
        scene=dict(id=row["id"],name=row["name"],task="plume_segmentation",bounds=[[south,west],[north,east]],
                   date=row["satellite_time"][:10],referenceDate=row["reference_time"][:10],
                   rgb=folder.name+"/satellite.png",predictionImage=folder.name+"/prediction.png",grid=folder.name+"/grid.json",
                   predictionAvailable=True,decisionThreshold=.5,dataKind="measured",dataset="MARS" if binary else "EMIT",
                   label=row["scene_kind"],source=manifest["source_revision"] if binary else row["satellite_id"],
                   checkpointHash=receipt["training_checkpoint_sha256"],predictionHash=record["array_sha256"],
                   notes=note,split="Unseen to Volusia checkpoint | fixed diagnostic selection",
                   rgbNote=f"Observed Sentinel {row['source_level']} target bands | 20 m model input | display-only percentile RGB",
                   truthLabel=("Reviewed no plume | no cyan plume expected" if row["scene_kind"]=="NO_PLUME" else "MARS reviewed plume annotation") if binary else "No binary annotation | EMIT retrieval in evidence report",
                   truthNote="Original 10 m delivered annotation; repeated 20 m scores create no extra measured detail." if binary else "Signed continuous EMIT enhancement is not a binary plume mask. No pixel accuracy calculated.",
                   resolution="20 m model output; native 10 m annotation comparison" if binary else "20 m output; native continuous EMIT context unscored",
                   qualityFlags=["unseen_to_checkpoint","input_and_target_domain_shift"] if binary else ["unseen_to_checkpoint","continuous_reference_unscored"],
                   outcome="False positive" if row["scene_kind"]=="NO_PLUME" else "Full-box prediction" if binary else "Unscored")
        if binary:
            scene["truthImage"]=folder.name+"/reference.png"
            scene["nativeMetrics"]={**measured,"support_fraction":float(known.mean()),"source_valid_pixels":int(known.sum()),"claim":"Frozen checkpoint; reviewed MARS L1C domain shift diagnostic"}
        scenes.append(scene)
        link=f"/?catalog=all&experiment={IDENTIFIER}&scene={row['id']}#compare"
        native_iou=measured["iou"] if measured else None
        rows.append(f'<tr><td><a href="{link}">{html.escape(row["name"])}</a></td><td>{row["source_level"]}</td><td>{html.escape(row["scene_kind"])}</td><td>{percent(frac)}</td><td>{percent(native_iou)}</td><td>{measured["fp"]:,}</td></tr>' if measured else f'<tr><td><a href="{link}">{html.escape(row["name"])}</a></td><td>L2A</td><td>Continuous EMIT</td><td>{percent(frac)}</td><td>Not scored</td><td>Unknown</td></tr>')
        tiles="".join(f'<figure><img src="{encode(folder/(key+".png"))}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>' for key,label in [("input","Actual model input"),("reference_figure","Reviewed annotation" if binary else "Measured EMIT enhancement, ppm m"),("overlay_figure","Frozen prediction at 0.50"),("errors_figure","Errors: pink extra / teal overlap / purple missed" if binary else "No binary reference; errors unavailable")])
        articles.append(f'<article id="{row["id"]}"><div class="case-title"><h2>{html.escape(row["name"])}</h2><a href="{link}">Open on map</a></div><p>{html.escape(note)}</p><div class="figures">{tiles}</div><details><summary>Fixed 0 to 1 score heatmap and provenance</summary><img class="score" src="{encode(folder/"score_figure.png")}" alt="Model scores on a fixed 0 to 1 color scale"><p class="mono">{html.escape(row["satellite_id"])}<br>Source timestamp: {html.escape(row["satellite_time"])}<br>Reference timestamp: {html.escape(row["reference_time"])}<br>Prediction SHA-256: {record["array_sha256"]}</p></details></article>')
        independent.append(dict(id=row["id"],counts=measured,positive_fraction=frac,rgb_display_stretch=stretch))
    pooled=ratios(totals)
    always_positive=ratios(control)
    fpr=negative["fp"]/(negative["fp"]+negative["tn"])
    negative_scene=next(s["id"] for s in scenes if s["label"]=="NO_PLUME")
    positives=sum(s["label"]=="PLUME" for s in scenes)
    # An overview preserves every selected observation, including failures.
    sheet=Image.new("RGB",(1040,len(image_rows)*250),(16,32,42))
    draw=ImageDraw.Draw(sheet)
    font_path=Path("C:/Windows/Fonts/segoeui.ttf")
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    for index,(name,images) in enumerate(image_rows):
        y=index*250
        draw.text((12,y+5),name,fill=(235,241,245),font=font)
        for col,(image,label) in enumerate(zip(images,["Sentinel input","Reviewed reference / EMIT","Frozen prediction","Errors (where known)"])):
            sheet.paste(Image.fromarray(image).resize((200,200),Image.Resampling.NEAREST),(col*260+15,y+28))
            draw.text((col*260+15,y+228),label,fill=(170,193,204),font=font)
    sheet.save(out/"all_scenes.png")
    conclusion=(f"This checkpoint fails this unseen-scene test. All {negative['scenes']} reviewed no-plume observations trigger strong false alarms. "
                f"It marks {percent(fpr)} of their evaluated background pixels as plume. Pooled annotation IoU across {positives} plume and {negative['scenes']} no-plume L1C scenes is {percent(pooled['iou'])}.")
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real unseen scenes | frozen CNN 2</title><style>
body{{margin:0;background:#101e28;color:#e4eef3;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1420px;margin:auto;padding:36px 24px 64px}}h1{{font-size:36px;line-height:1.15;margin:12px 0 20px}}h2{{font-size:20px;margin:0}}a{{color:#58d5d5}}.eyebrow{{letter-spacing:.12em;font-size:12px;color:#73c7ce;font-weight:700}}.verdict{{background:#39252c;border:1px solid #a15b68;padding:20px;border-radius:12px;font-size:18px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}}.stats div,article{{border:1px solid #314853;border-radius:12px;background:#152731;padding:18px}}.stats strong{{display:block;font-size:28px}}.stats span{{color:#abc0ca}}.note{{max-width:1100px;color:#b3c6d0}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #314853}}.table-wrap{{overflow-x:auto;margin:22px 0}}article{{margin-top:24px}}.case-title{{display:flex;justify-content:space-between;gap:16px}}article p{{color:#b3c6d0;font-size:13px}}.figures{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}figure{{margin:0;background:#0c1c25;border-radius:8px;overflow:hidden}}figure img{{display:block;width:100%;image-rendering:pixelated;cursor:zoom-in}}figcaption{{padding:8px 10px;font-size:12px}}details{{margin-top:16px}}summary{{cursor:pointer;color:#7cdae0}}.score{{width:240px;image-rendering:pixelated}}.mono{{font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}}.large{{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:#061019ef;z-index:10}}.large img{{max-width:94vw;max-height:94vh;image-rendering:pixelated}}@media(max-width:900px){{.figures,.stats{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:520px){{main{{padding:24px 12px}}h1{{font-size:28px}}.figures{{grid-template-columns:1fr 1fr}}.case-title{{display:block}}}}</style></head><body><main>
<div class="eyebrow">GTM MODEL 6 / REAL UNSEEN INPUT / FROZEN CNN 2</div><h1>It fits one scene.<br>It does not pass this transfer test.</h1><div class="verdict">{html.escape(conclusion)}</div>
<div class="stats"><div><strong>{positives}</strong><span>Reviewed plume scenes</span></div><div><strong>{negative['activated']} / {negative['scenes']}</strong><span>No-plume scenes with false alarms</span></div><div><strong>{percent(pooled['iou'])}</strong><span>Pooled source-annotation IoU</span></div><div><strong>0 bytes</strong><span>New imagery downloaded</span></div></div>
<p><a href="/?catalog=all&experiment={IDENTIFIER}&scene={negative_scene}#compare">Open a reviewed no-plume case in the viewer</a> | <a href="#cases">Inspect all saved results</a></p>
<p class="note">The weights, Volusia fitting normalization, tiling and 0.50 cutoff were fixed before inference. These sites did not enter this checkpoint's training. This is a diagnostic selection, not nationwide detection accuracy. Other project models previously used MARS.</p>
<p class="note"><strong>Input and reference differences matter.</strong> The 15 scored MARS observations contain real Sentinel L1C target imagery and reviewed MARS labels. Training used Sentinel L2A with an EMIT provider annotation. Large differences in input values may contribute to this failure. This test cannot isolate their effect from single-site overfitting or other model limits. The model still fails the pipeline we tested. The separate Georgia scene uses real L2A and continuous EMIT, but has no binary annotation and receives no segmentation score.</p>
<p class="note">The scene list was sealed before prediction: first available US sample IDs, positives first, at least 90% clear valid input, at least 25 km between sites and from the fitting footprint. Only seven plume sites met these limits. All eight selected no-plume sites remain in this report. NO_PLUME means the producer identified no plume in that observation, not proof that the facility never emits methane.</p>
<p class="note">Scores use the original delivered 10 m MARS labels. Each 20 m prediction repeats over four annotation cells without smoothing. Counts are from the original grid, not a map screenshot. This does not validate fine methane detail. All-positive control IoU is {percent(always_positive['iou'])}, essentially the same as this model. The apparent 100% plume recall occurs because it paints almost the whole box.</p>
<div class="table-wrap"><table><thead><tr><th>Observation</th><th>Input</th><th>Reference</th><th>Area predicted plume</th><th>Annotation IoU</th><th>Extra positive cells</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2 id="cases">Every selected observation</h2><p class="note">Cyan marks the reviewed plume. Orange is the prediction at 0.50. Pink marks extra predictions, teal overlap and purple missed annotation. Gray cells are excluded. Click a figure to enlarge it. Georgia's EMIT panel is a continuous measurement with a display range from zero to its 98th percentile; negative values are retained in the source arrays but clip in this display.</p>{''.join(articles)}
<h2 style="margin-top:32px">What this changes</h2><p class="note">The previous Volusia score established fitting only. It did not establish a working detector. Do not present this checkpoint as ready for US screening or source detection. The next experiment needs consistent input radiometry, several independent training sites, reviewed no-plume examples, and a separate evaluation set fixed before fitting. Keep this failed transfer result as a regression check, not as training feedback for this same reported test.</p>
<h2>Repeatable evidence</h2><p class="note">Fresh-process inference reproduced all 16 saved results with a maximum difference of zero. An independent raster check rebuilt the MARS inputs, clear support and original labels, checked exact score repetition and recomputed every reported count. Training and prediction artifacts remained unchanged. No training, threshold tuning, packages or imagery downloads occurred.</p>
<p class="mono">Manifest SHA-256: {sha(args.manifest)}<br>Frozen weights SHA-256: {receipt['training_checkpoint_sha256']}<br>Inference receipt SHA-256: {sha(args.predictions/'receipt.json')}<br>Source dataset: UNEP-IMEO/MARS-S2L, revision {manifest['source_revision']}, CC-BY-NC-SA-4.0, research use.</p>
</main><div class="large" id="large" role="dialog" aria-label="Expanded result"><img alt="Expanded scene result"></div><script>const box=document.getElementById('large');document.querySelectorAll('figure img').forEach(img=>img.addEventListener('click',()=>{{box.querySelector('img').src=img.src;box.style.display='flex';}}));box.addEventListener('click',()=>box.style.display='none');document.addEventListener('keydown',e=>{{if(e.key==='Escape')box.style.display='none';}});</script></body></html>'''
    (out/"index.html").write_text(body,encoding="utf-8")
    experiment=dict(id=IDENTIFIER,name="Unseen real scenes | frozen CNN 2",lifecycle="experimental",priority=False,
                    dataKind="measured",architectureFamily="GTM_Model6_three_cnn",stage="emit",runGroup="GTM_Model6_unseen_20261023",
                    status="FAILED TRANSFER | 7 plume + 8 no-plume + 1 unscored EMIT",
                    description=conclusion,notes="Frozen Volusia CNN 2. Real L1C and reviewed MARS labels differ from L2A/EMIT fitting. Every selected scene is included. Full-box false alarms are real saved outputs, not a display substitute.",
                    decisionThreshold=.5,nativeAggregate={**pooled,"scenes":15},
                    metrics={"Annotation IoU":percent(pooled["iou"]),"No-plume false alarms":f"{negative['activated']}/{negative['scenes']}","New downloads":"0 bytes"},scenes=scenes,reportUrl="index.html")
    save(out/"experiment.json",experiment)
    load_bundle(out/"experiment.json","bundle-assets/")
    after={str(p):sha(p) for folder in [args.predictions,Path(receipt["training_run"])] for p in folder.rglob("*") if p.is_file()}
    if after != immutable:
        raise ValueError("Rendering changed accepted inference or training files")
    verification=dict(passed=True,manifest_sha256=sha(args.manifest),inference_receipt_sha256=sha(args.predictions/"receipt.json"),
                      immutable_artifacts_unchanged=True,independent_counts=independent,pooled=pooled,always_positive_control=always_positive,
                      negatives={**negative,"pixel_false_positive_rate":fpr},downloads=0,replay_max_abs=max(max(r["max_abs_diff"].values()) for r in replay["scenes"]),
                      renderer_sha256=sha(__file__),html_sha256=sha(out/"index.html"))
    if args.publish:
        verification["publication"]=publish(out,ROOT/"outputs/GTM_viewer_bundles")
    save(out/"verification.json",verification)
    print(json.dumps({k:v for k,v in verification.items() if k not in ["independent_counts"]},indent=2))


if __name__ == "__main__":
    main()
