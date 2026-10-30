"""Verify and publish the real, grouped temporal-CNN comparison.

Failure cases were written first in docs/GTM_Model7_temporal_viewer_acceptance.md.
Rendering never fits a model or changes the sealed cohort or checkpoints.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from GTM_Model0_local_viewer import load_bundle
from GTM_Model0_publish_viewer import publish

ROOT = Path(__file__).resolve().parents[1]
MARS = ROOT.parent / "Earth-Remote-Sensing-Rapid-Response-" / "EarthRemoteSensingRapidResponse" / "Data Collection" / "s2_emit_pairs" / "publication-v1" / "external" / "MARS-S2L"
MODELS = ("residual", "siamese")
NAMES = {"residual": "Option 1 | Temporal residual CNN", "siamese": "Option 2 | Temporal U-Net"}
IDS = {m: "GTM_Model7_temporal_20261023_" + m for m in MODELS}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ratios(c):
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    return dict(**c, iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                evaluated_pixels=tp+fp+fn+tn)


def counts(p, truth, support):
    predicted, positive = p[support] >= .5, truth[support] == 1
    return dict(tp=int(np.sum(predicted & positive)), fp=int(np.sum(predicted & ~positive)),
                fn=int(np.sum(~predicted & positive)), tn=int(np.sum(~predicted & ~positive)))


def percent(value):
    return f"{value:.2%}" if value is not None else "Unavailable"


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def paired_rgb(current, reference, valid):
    ranges, rendered = [], []
    for i in (2, 1, 0):
        lo, hi = np.percentile(current[..., i][valid], (2, 98))
        ranges.append((float(lo), float(hi)))
    for image in (current, reference):
        rgb = np.stack([np.clip((image[..., band]-lo)/max(hi-lo, 1e-8), 0, 1)
                        for band, (lo, hi) in zip((2, 1, 0), ranges)], -1)
        rgb[~valid] = .2
        rendered.append(np.repeat(np.repeat((rgb*255).astype(np.uint8), 2, 0), 2, 1))
    return *rendered, ranges


def overlay(rgb, probability, truth, valid):
    painted = rgb.copy()
    positive, predicted = (truth == 1) & valid, (probability >= .5) & valid
    painted[predicted] = (painted[predicted]*.50 + np.array((255, 151, 56))*.50).astype(np.uint8)
    interior = positive.copy()
    interior[1:-1, 1:-1] &= positive[:-2, 1:-1] & positive[2:, 1:-1] & positive[1:-1, :-2] & positive[1:-1, 2:]
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    painted[positive & ~interior] = (46, 224, 219)
    painted[~valid] = (58, 69, 78)
    error = rgb.copy()
    error[predicted & positive] = (32, 179, 157)
    error[predicted & ~positive] = (244, 92, 129)
    error[~predicted & positive] = (168, 140, 240)
    error[~valid] = (58, 69, 78)
    heat = (colormaps["viridis"](np.nan_to_num(probability))*255).astype(np.uint8)
    heat[~valid] = (58, 69, 78, 255)
    return painted, error, heat


def original_arrays(scene, data):
    """Rebuild test inputs/labels independently from original released TIFFs."""
    row = scene["source_row"]
    assets = {a["role"]: MARS / a["path"] for a in row["assets"]}
    with rasterio.open(assets["image"]) as image, rasterio.open(assets["cloud_mask"]) as clouds:
        if image.shape != (200, 200) or image.count != 12 or clouds.shape != image.shape:
            raise ValueError("Original grid changed")
        if image.transform != clouds.transform or image.crs != clouds.crs:
            raise ValueError("Original input/cloud grids differ")
        raw = np.moveaxis(image.read(), 0, -1).astype(np.float32)
        clear = (clouds.read(1) == 0) & np.isfinite(raw).all(-1)
        if image.nodata is not None:
            clear &= (raw != image.nodata).all(-1)
        valid = clear.reshape(100, 2, 100, 2).all((1, 3))
        coarse = raw.reshape(100, 2, 100, 2, 12).mean((1, 3))
        for key, expected in (("target_image", coarse[..., :6]), ("reference_image", coarse[..., 6:]), ("valid", valid)):
            if not np.array_equal(data[key], expected, equal_nan=True):
                raise ValueError("Prepared data differs from raw source: " + key)
        native_support = np.repeat(np.repeat(valid, 2, 0), 2, 1)
        if not np.array_equal(native_support, data["valid_native"]):
            raise ValueError("Original/native support mismatch")
        grid = dict(shape=[200, 200], affine=list(image.transform)[:6], crs=str(image.crs))
    if row["label_state"] == "PLUME":
        with rasterio.open(assets["plume_mask"]) as labels:
            if list(labels.transform)[:6] != grid["affine"] or str(labels.crs) != grid["crs"]:
                raise ValueError("Original annotation alignment mismatch")
            label = labels.read(1)
    elif row["label_state"] == "NO_PLUME":
        label = np.zeros((200, 200), np.uint8)
    else:
        raise ValueError("Unknown reviewed class")
    if not np.array_equal(label, data["label_native"]):
        raise ValueError("Native annotation changed")
    if not np.array_equal(data["target"], label.reshape(100, 2, 100, 2).max((1, 3))):
        raise ValueError("Training annotation coarsening changed")
    return grid


def map_assets(folder, rgb, probability, truth, valid, grid, threshold=.5):
    affine, crs = Affine(*grid["affine"]), grid["crs"]
    h, w = probability.shape
    box = transform_bounds(crs, "EPSG:3857", *array_bounds(h, w, affine), densify_pts=21)
    dest = from_bounds(*box, w, h)

    def project(a, smooth=False):
        b = np.full((h, w), np.nan, np.float32)
        reproject(a.astype(np.float32), b, src_transform=affine, src_crs=crs,
                  dst_transform=dest, dst_crs="EPSG:3857", src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear if smooth else Resampling.nearest)
        return b

    p = project(np.where(valid, probability, np.nan))
    t = project(np.where(valid, truth, np.nan))
    satellite = np.stack([project(rgb[..., i], True) for i in range(3)], -1)
    Image.fromarray(np.clip(np.nan_to_num(satellite), 0, 255).astype(np.uint8)).save(folder/"satellite.png")
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[np.isfinite(p) & (p >= threshold)] = (255, 151, 56, 210)
    Image.fromarray(rgba).save(folder/"prediction.png")
    rgba[:] = 0
    rgba[np.isfinite(t) & (t == 1)] = (46, 224, 219, 200)
    Image.fromarray(rgba).save(folder/"truth.png")
    values = lambda a: [float(x) if np.isfinite(x) else None for x in a.ravel()]
    save(folder/"grid.json", dict(width=w, height=h, probability=values(p), truth=values(t),
                                valid=np.isfinite(p).astype(np.uint8).ravel().tolist(), crs="EPSG:3857"))
    west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", *box)
    return [[south, west], [north, east]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--siamese", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    manifest_path = args.data/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != "GTM_Model7_temporal_cohort_v1":
        raise ValueError("Unexpected cohort")
    tests = [s for s in manifest["scenes"] if s["split"] == "test"]
    tests.sort(key=lambda s: (s["source_row"]["label_state"] != "PLUME", s["id"]))
    if not tests:
        raise ValueError("No test observations")
    args.output.mkdir(parents=True, exist_ok=False)
    runs = {m: getattr(args, m) for m in MODELS}
    immutable = {str(p): sha(p) for run in runs.values() for p in run.rglob("*") if p.is_file()}
    receipts = {m: json.loads((runs[m]/"receipt.json").read_text()) for m in MODELS}
    for m in MODELS:
        if receipts[m]["model"] != m or receipts[m]["schema"] != "GTM_Model7_temporal_training_v1":
            raise ValueError("Run folder contains a different model")
        if receipts[m]["manifest_sha256"] != sha(manifest_path) or receipts[m]["config"]["threshold"] != .5:
            raise ValueError("Run does not use the sealed cohort and fixed threshold")
        if sha(runs[m]/"final.pt") != receipts[m]["checkpoint_sha256"]["final"]:
            raise ValueError("Accepted checkpoint changed")
    if receipts["residual"]["config"] != receipts["siamese"]["config"] or receipts["residual"]["draw_schedule_sha256"] != receipts["siamese"]["draw_schedule_sha256"]:
        raise ValueError("Models did not receive an identical training schedule")
    totals = {m: dict(tp=0, fp=0, fn=0, tn=0) for m in MODELS}
    negatives = {m: dict(scenes=0, activated=0, fp=0, tn=0) for m in MODELS}
    controls = {"all_positive": dict(tp=0, fp=0, fn=0, tn=0), "all_negative": dict(tp=0, fp=0, fn=0, tn=0)}
    scenes, cases, table, sheet_rows, independent = {m: [] for m in MODELS}, [], [], [], []
    for m in MODELS:
        (args.output/m).mkdir()
    for index, scene in enumerate(tests):
        file = args.data/scene["array_file"]
        if sha(file) != scene["array_sha256"]:
            raise ValueError("Sealed test array changed")
        with np.load(file, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        for source in scene["sources"]:
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Original source changed")
        grid = original_arrays(scene, data)
        rgb, background, stretch = paired_rgb(data["target_image"], data["reference_image"], data["valid"])
        label, valid = data["label_native"], data["valid_native"]
        reference = rgb.copy()
        reference[(label == 1) & valid] = (reference[(label == 1) & valid]*.35 + np.array((46, 224, 219))*.65).astype(np.uint8)
        for control in controls:
            c = counts(np.ones_like(label) if control == "all_positive" else np.zeros_like(label), label, valid)
            for key in c:
                controls[control][key] += c[key]
        figures, metrics, probabilities = {}, {}, {}
        kind = scene["source_row"]["label_state"]
        title = ("Plume" if kind == "PLUME" else "No plume") + " | " + scene["source_row"]["target_datetime"][:10] + " | " + scene["id"].replace("MARS_", "")[:8]
        record = dict(id=scene["id"], label=kind, group_id=scene["group_id"], models={})
        for m in MODELS:
            pred_file = runs[m]/"predictions"/(scene["id"] + ".npz")
            with np.load(pred_file, allow_pickle=False) as z:
                p, coarse = z["probability_native"], z["probability20"]
                if not np.array_equal(z["valid20"], data["valid"]):
                    raise ValueError("Saved prediction support changed")
            if not np.array_equal(p, np.repeat(np.repeat(coarse, 2, 0), 2, 1), equal_nan=True):
                raise ValueError("Fine display differs from real model output")
            if not np.isfinite(p[valid]).all() or np.any((p[valid] < 0) | (p[valid] > 1)):
                raise ValueError("Invalid saved scores")
            measured = ratios(counts(p, label, valid))
            accepted = receipts[m]["test_scenes"][scene["id"]]
            if sha(pred_file) != accepted["prediction_sha256"] or any(measured[k] != accepted[k] for k in ("tp", "fp", "fn", "tn")):
                raise ValueError("Independent native counts differ from training receipt")
            metrics[m], probabilities[m] = measured, p
            for key in totals[m]:
                totals[m][key] += measured[key]
            if kind == "NO_PLUME":
                negatives[m]["scenes"] += 1
                negatives[m]["activated"] += int(measured["fp"] > 0)
                negatives[m]["fp"] += measured["fp"]
                negatives[m]["tn"] += measured["tn"]
            folder = args.output/m/f"scene_{index:03d}"
            folder.mkdir()
            painted, errors, heat = overlay(rgb, p, label, valid)
            assets = dict(current=rgb, background=background, annotation=reference, overlay=painted, errors=errors, score=heat)
            for name, arr in assets.items():
                Image.fromarray(arr).save(folder/(name + ".png"))
            figures[m] = {k: image_uri(folder/(k + ".png")) for k in ("overlay", "errors", "score")}
            bounds = map_assets(folder, rgb, p, label, valid, grid)
            fraction = float(np.mean(p[valid] >= .5))
            notes = "Held out by connected 25 km geography and target/reference acquisition. Fixed final checkpoint and 0.50 threshold. Real MARS L1C input and reviewed release labels. The three positive observations share one geographic component; prior project exposure is possible."
            outcome = "False positive" if kind == "NO_PLUME" and measured["fp"] else "Correct negative" if kind == "NO_PLUME" else "Missed plume" if measured["tp"] == 0 else "Partial overlap"
            scene_info = dict(id=scene["id"], name=title, task="plume_segmentation", bounds=bounds,
                              date=scene["source_row"]["target_datetime"][:10],
                              referenceDate=scene["source_row"]["reference_scene_id"].split("_")[2][:8],
                              rgb=folder.name+"/satellite.png", predictionImage=folder.name+"/prediction.png",
                              truthImage=folder.name+"/truth.png", grid=folder.name+"/grid.json",
                              predictionAvailable=True, decisionThreshold=.5, dataKind="measured", dataset="MARS",
                              label=kind, split="Geographic holdout | unseen to these models", notes=notes,
                              source=scene["source_row"]["target_scene_id"], predictionHash=sha(pred_file),
                              checkpointHash=sha(runs[m]/"final.pt"), outcome=outcome,
                              truthLabel="MARS reviewed plume annotation" if kind == "PLUME" else "Reviewed no plume | no cyan plume expected",
                              truthNote="Original 10 m delivered mask. Predictions originate on a 20 m grid and repeat without smoothing.",
                              rgbNote="Real Sentinel-2 L1C target | display-only contrast stretch",
                              resolution="20 m model output; source annotation comparison on delivered 10 m grid",
                              qualityFlags=["held_out_from_this_model", "one_positive_test_geography", "reference_cloud_mask_unavailable"],
                              nativeMetrics={**measured, "source_valid_pixels":int(valid.sum()), "support_fraction":float(valid.mean()), "claim":"Fixed geographic pilot holdout"})
            scenes[m].append(scene_info)
            record["models"][m] = dict(**measured, predicted_area=fraction, prediction_sha256=sha(pred_file))
        independent.append(record)
        common = args.output/"residual"/f"scene_{index:03d}"
        case = dict(id=scene["id"], title=title, kind=kind, current=image_uri(common/"current.png"),
                    background=image_uri(common/"background.png"), annotation=image_uri(common/"annotation.png"),
                    models=figures, metrics=metrics, group=scene["group_id"],
                    source=scene["source_row"]["target_scene_id"], reference=scene["source_row"]["reference_scene_id"],
                    target_time=scene["satellite_time"], reference_time=scene["reference_time"],
                    later_reference=scene["target_minus_reference_hours"] < 0,
                    rgb_display_ranges=stretch)
        cases.append(case)
        table.append("<tr><td>"+html.escape(title)+"</td>"+"".join(f"<td>{percent(metrics[m]['iou'])}</td><td>{metrics[m]['fp']:,}</td>" for m in MODELS)+"</tr>")
        sheet_rows.append((title, [rgb, reference] + [overlay(rgb, probabilities[m], label, valid)[0] for m in MODELS]))
    pooled = {m: ratios(totals[m]) for m in MODELS}
    for m in MODELS:
        n = negatives[m]
        n["pixel_fpr"] = n["fp"]/(n["fp"]+n["tn"]) if n["fp"]+n["tn"] else None
    summary = dict(manifest_sha256=sha(manifest_path), pooled=pooled, negatives=negatives,
                   controls={k:ratios(v) for k,v in controls.items()}, scenes=independent,
                   test_scenes=len(tests), positive_test_scenes=sum(c["kind"] == "PLUME" for c in cases),
                   positive_test_groups=len({c["group"] for c in cases if c["kind"] == "PLUME"}),
                   zero_new_imagery_downloads=True)
    save(args.output/"comparison.json", summary)
    figure = Figure(figsize=(10, 3.5), facecolor="#0d191f", constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    ax.set_facecolor("#0d191f")
    for m, color in (("residual", "#70d5d0"), ("siamese", "#e6b56d")):
        curve = json.loads((runs[m]/"learning_curve.json").read_text())
        ax.plot([r["epoch"] for r in curve], [r["train_loss"] for r in curve], color=color, label=NAMES[m])
    ax.set_xlabel("Fixed training epoch", color="#b4c6cd")
    ax.set_ylabel("Training loss", color="#b4c6cd")
    ax.tick_params(colors="#b4c6cd")
    for spine in ax.spines.values():
        spine.set_color("#425a64")
    ax.grid(alpha=.15)
    ax.legend(facecolor="#182a33", labelcolor="#e7f0f3", edgecolor="#425a64")
    figure.savefig(args.output/"learning_curves.png", dpi=140)
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    sheet = Image.new("RGB", (1040, len(sheet_rows)*254), (15, 28, 37))
    draw = ImageDraw.Draw(sheet)
    for i, (name, images) in enumerate(sheet_rows):
        y = i*254
        draw.text((12, y+5), name, font=font, fill=(230, 240, 244))
        for j, (image, label) in enumerate(zip(images, ("Current Sentinel", "Reviewed annotation", "1: Residual CNN", "2: Temporal U-Net"))):
            sheet.paste(Image.fromarray(image).convert("RGB").resize((200, 200), Image.Resampling.NEAREST), (j*260+12, y+28))
            draw.text((j*260+12, y+230), label, font=font, fill=(160, 187, 199))
    sheet.save(args.output/"all_scenes.png")
    sheet.crop((0, 0, 1040, min(6, len(sheet_rows))*254)).save(args.output/"comparison_preview.png")
    options = "".join(f'<option value="{i}">{html.escape(c["title"])}</option>' for i,c in enumerate(cases))
    stats = "".join(f'<tr><td>{html.escape(NAMES[m])}</td><td>{percent(pooled[m]["iou"])}</td><td>{percent(pooled[m]["precision"])}</td><td>{percent(pooled[m]["recall"])}</td><td>{negatives[m]["activated"]}/{negatives[m]["scenes"]}</td><td>{percent(negatives[m]["pixel_fpr"])}</td></tr>' for m in MODELS)
    training = "; ".join(f"{NAMES[m]}: {receipts[m]['parameter_count']:,} parameters, {receipts[m]['runtime_seconds_training']:.1f} training seconds" for m in MODELS)
    body = HTML.replace("__OPTIONS__", options).replace("__STATS__", stats).replace("__ROWS__", "".join(table)).replace("__CASES__", json.dumps(cases).replace("</", "<\\/")).replace("__SUMMARY__", json.dumps(summary).replace("</", "<\\/")).replace("__CURVES__", image_uri(args.output/"learning_curves.png")).replace("__TRAINING__", html.escape(training))
    for m in MODELS:
        (args.output/m/"index.html").write_text(body, encoding="utf-8")
        for scene in scenes[m]:
            scene["referenceDate"] = scene["referenceDate"][:4]+"-"+scene["referenceDate"][4:6]+"-"+scene["referenceDate"][6:8]
        experiment = dict(id=IDS[m], name=NAMES[m], lifecycle="experimental", priority=False,
                          dataKind="measured", architectureFamily="GTM_Model7_temporal", runGroup="GTM_Model7_temporal_20261023",
                          status="REAL GEOGRAPHIC HOLDOUT | fixed local pilot",
                          description=f"Native annotation IoU {percent(pooled[m]['iou'])}; {negatives[m]['activated']}/{negatives[m]['scenes']} no-plume scenes activate at 0.50.",
                          notes="Compare options 1 and 2 on identical scenes. Three positive observations in one held-out geographic component; no nationwide claim. Reference image quality remains a limitation.",
                          decisionThreshold=.5, nativeAggregate={**pooled[m], "scenes":len(tests)},
                          metrics={"Held-out IoU":percent(pooled[m]["iou"]), "No-plume false alarms":f"{negatives[m]['activated']}/{negatives[m]['scenes']}", "Threshold":"0.50 fixed"},
                          scenes=scenes[m], reportUrl="index.html")
        save(args.output/m/"experiment.json", experiment)
        load_bundle(args.output/m/"experiment.json", "bundle-assets/")
    (args.output/"index.html").write_text(body, encoding="utf-8")
    after = {str(p): sha(p) for run in runs.values() for p in run.rglob("*") if p.is_file()}
    if after != immutable:
        raise ValueError("Rendering mutated accepted model artifacts")
    verification = dict(passed=True, original_test_rasters_rebuilt=len(tests), native_scores_independently_recomputed=True,
                        prediction_repeat_verified=True, model_artifacts_unchanged=True,
                        manifest_sha256=sha(manifest_path), renderer_sha256=sha(__file__), comparison_sha256=sha(args.output/"comparison.json"))
    if args.publish:
        verification["publication"] = {m:publish(args.output/m, ROOT/"outputs/GTM_viewer_bundles") for m in MODELS}
    save(args.output/"verification.json", verification)
    print(json.dumps(dict(pooled=pooled, negatives=negatives, verification=verification), indent=2))


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real temporal models | options 1 and 2</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d191f;color:#e7f0f3;font:15px/1.55 system-ui,sans-serif}main{max-width:1540px;margin:auto;padding:38px 28px 70px}.eyebrow{font-size:11px;letter-spacing:.15em;color:#71c9c7;font-weight:750}h1{font-size:clamp(28px,4vw,44px);letter-spacing:-.035em;line-height:1.08;margin:13px 0}h2{font-size:22px;letter-spacing:-.02em;margin:36px 0 12px}p{max-width:1050px;color:#b4c6cd}.scope{border-left:3px solid #e6b56d;padding:10px 18px;background:#253033;margin:22px 0}.scope strong{color:#f4d7a7}.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;border-block:1px solid #30434c;padding:16px 0;margin:24px 0 18px}label{font-size:12px;color:#a8c0c9;display:grid;gap:5px}select,button{font:inherit;border:1px solid #425a64;border-radius:6px;background:#182a33;color:#eef5f5;padding:9px 12px}button,a{cursor:pointer}button:hover{background:#25414b}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid #6bd8cf;outline-offset:3px}select{max-width:550px}a{color:#6bd8cf}#counter{color:#8ca8b5;font-size:12px}.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}figure{margin:0;background:#13252e;overflow:hidden;border-radius:7px}figure img{display:block;width:100%;aspect-ratio:1;object-fit:contain;image-rendering:pixelated;cursor:zoom-in}figcaption{padding:10px 11px;font-size:12px;color:#bdd0d7}figcaption strong{display:block;color:#eef5f5;margin-bottom:4px}.scene-title{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}.scene-title h2{margin:0;font-size:20px}.links{display:flex;gap:14px;font-size:12px}.legend{font-size:12px;color:#9fb7c1;margin:14px 0}.legend b{color:#e8f3f6}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:12px 14px;border-bottom:1px solid #2c414a}th{color:#9fb9c4;font-weight:550}td:first-child,th:first-child{padding-left:0}.mini{font:12px/1.65 ui-monospace,monospace;color:#9bb1be;overflow-wrap:anywhere}details{margin:26px 0}summary{cursor:pointer;color:#70d5d0}.dialog{position:fixed;inset:0;z-index:9;background:#061116ed;display:none;place-items:center;cursor:zoom-out}.dialog img{max-width:94vw;max-height:94vh;image-rendering:pixelated}.footer{margin-top:35px;padding-top:20px;border-top:1px solid #30434c;font-size:12px;color:#8da8b5}@media(max-width:1100px){.images{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){main{padding:24px 14px}.images{grid-template-columns:repeat(2,minmax(0,1fr))}.scene-title{display:block}.links{margin-top:9px}.controls{align-items:stretch}select{max-width:90vw;width:100%}}
</style></head><body><main><div class="eyebrow">ERSRR / LOCAL REAL-DATA COMPARISON / OCTOBER 23, 2026</div><h1>Two models.<br>The same unseen scenes.</h1><p>Option 1 learns spatial patterns in spectral changes between dates. Option 2 compares learned features at several scales. Both receive the current Sentinel image and its temporal reference. The masks below are actual saved predictions at the fixed 0.50 cutoff.</p><div class="scope"><strong>Small geographic pilot.</strong> The positive examples form only two broad geographic groups. One was used for fitting; the other supplies three held-out plume observations. Additional held-out no-plume scenes test false alarms. Development scenes contain negatives only, so no checkpoint or threshold was selected using development or test scores. These scenes are unseen to these two models; earlier project work may have examined them.</div>
<div class="controls"><label>Held-out scene<select id="scene" aria-label="Held-out scene">__OPTIONS__</select></label><button id="previous" type="button">Previous</button><button id="next" type="button">Next</button><label>Prediction display<select id="mode" aria-label="Prediction display"><option value="overlay">Mask and reference boundary</option><option value="score">Model score: fixed zero to one</option><option value="errors">Errors against reviewed annotation</option></select></label><span id="counter"></span></div>
<div class="scene-title"><h2 id="title"></h2><div class="links"><a id="map-residual">Open option 1 on map</a><a id="map-siamese">Open option 2 on map</a></div></div><div class="images"><figure><img id="current" alt="Current real Sentinel image"><figcaption><strong>Current observation</strong>Observed six-band input; RGB display</figcaption></figure><figure><img id="background" alt="Temporal reference Sentinel image"><figcaption><strong>Temporal reference</strong>Same RGB stretch as the current image</figcaption></figure><figure><img id="annotation" alt="Reviewed source plume annotation"><figcaption><strong>Reviewed annotation</strong>Cyan is the delivered plume label</figcaption></figure><figure><img id="residual" alt="Temporal residual CNN saved prediction"><figcaption><strong>1 / Residual CNN</strong><span id="metric-residual"></span></figcaption></figure><figure><img id="siamese" alt="Temporal U-Net saved prediction"><figcaption><strong>2 / Temporal U-Net</strong><span id="metric-siamese"></span></figcaption></figure></div><p class="legend"><b>Mask:</b> orange prediction, cyan annotation boundary. <b>Errors:</b> teal overlap, pink extra prediction, purple missed plume. <b>Score:</b> purple zero, teal middle, yellow one. Scores have not been calibrated into probabilities. Gray cells are excluded.</p><p id="source" class="mini"></p>
<h2>Whole held-out comparison</h2><div class="table-scroll"><table><thead><tr><th>Model</th><th>Plume IoU</th><th>Precision</th><th>Recall</th><th>No-plume scenes with any activation</th><th>No-plume pixel false-positive rate</th></tr></thead><tbody>__STATS__</tbody></table></div><p>Scores use the original delivered 10 m annotations. The models predict on a 20 m grid; each score repeats across four annotation cells without smoothing. This comparison measures plume segmentation. It does not establish finer methane reconstruction, an emission rate, CAFO recovery, or national detection accuracy.</p><p id="controls" class="mini"></p>
<details><summary>All selected observations and native scores</summary><div class="table-scroll"><table><thead><tr><th>Observation</th><th>Residual IoU</th><th>Residual FP cells</th><th>U-Net IoU</th><th>U-Net FP cells</th></tr></thead><tbody>__ROWS__</tbody></table></div></details><h2>Fitting progress</h2><p>__TRAINING__. Both curves show training loss only. A falling curve does not establish success on unseen scenes.</p><img src="__CURVES__" alt="Training loss by fixed epoch for both temporal models" style="display:block;width:100%;max-width:1000px"><details><summary>Training and evidence limits</summary><p>Only US Sentinel-2 L1C observations from the release's official training pool enter this new experiment. The official test split remains unused. Split boundaries include connected 25 km geographic groups and shared target or reference acquisitions. The prepared cohort is sealed before training. Normalization uses training imagery only. Both models use the same seeded crop schedule, real negative scenes, D4 augmentation, fixed update count and loss. Final checkpoint selection and the 0.50 cutoff were fixed in advance.</p><p>This is a retrospective comparison: some temporal references were acquired after the target date. A live detector would need a separate test using only references available at the target time. The release supplies a current-date cloud mask, but no separately verified reference-date cloud mask. Clear current imagery does not guarantee a cloud-free reference. Geographic labels and reviewed release annotations are supervision; a no-plume observation does not prove a facility never emits. No actual EMIT measurement is an input or a target in this comparison. MARS enhancement products and NASA EMIT measurements are separate products.</p><p>Original-image reconstruction, source hashes, independent native confusion counts, checkpoint replay and browser checks are saved with the run. Test positives come from only one broad area. The experiment compares these particular training runs and cannot establish which architecture is generally best.</p></details><div class="footer">Source: UNEP-IMEO/MARS-S2L, CC-BY-NC-SA-4.0. No new imagery was downloaded. Training checkpoints, saved arrays, source receipts and repeat commands remain in the local research repository.</div></main><div class="dialog" id="large" role="dialog" aria-label="Expanded scene figure"><img alt="Expanded research result"></div><script>
const cases=__CASES__,summary=__SUMMARY__,q=id=>document.getElementById(id),fmt=x=>x===null?'n/a':(100*x).toFixed(2)+'%';
function update(){const i=Number(q('scene').value),c=cases[i],mode=q('mode').value;q('title').textContent=c.title;q('counter').textContent=(i+1)+' / '+cases.length;for(const k of ['current','background','annotation'])q(k).src=c[k];for(const m of ['residual','siamese']){q(m).src=c.models[m][mode];const z=c.metrics[m];q('metric-'+m).textContent='IoU '+fmt(z.iou)+' · '+z.fp.toLocaleString()+' extra cells';q('map-'+m).href='/?catalog=all&experiment=GTM_Model7_temporal_20261023_'+m+'&scene='+encodeURIComponent(c.id)+'#compare';}q('source').textContent='Current: '+c.target_time+' | Reference: '+c.reference_time+(c.later_reference?' (later acquisition; retrospective comparison)':' (earlier acquisition)')+' | Holdout group: '+c.group+' | '+c.source+' | '+c.reference;q('previous').disabled=i===0;q('next').disabled=i===cases.length-1;}
q('scene').addEventListener('change',update);q('mode').addEventListener('change',update);q('previous').onclick=()=>{q('scene').value=Number(q('scene').value)-1;update()};q('next').onclick=()=>{q('scene').value=Number(q('scene').value)+1;update()};q('controls').textContent='Controls on identical pixels: all-positive IoU '+fmt(summary.controls.all_positive.iou)+'; all-negative recall '+fmt(summary.controls.all_negative.recall)+'.';for(const im of document.querySelectorAll('figure img'))im.onclick=()=>{q('large').querySelector('img').src=im.src;q('large').style.display='grid'};q('large').onclick=()=>q('large').style.display='none';document.addEventListener('keydown',e=>{if(e.key==='Escape')q('large').style.display='none'});update();
</script></body></html>'''


if __name__ == "__main__":
    main()
