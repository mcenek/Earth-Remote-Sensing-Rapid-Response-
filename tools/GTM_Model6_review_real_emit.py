"""Verify and show the real CNN 2 fitting diagnostic, offline.

This publishes a local saved-result bundle, not an independent accuracy claim.
The failure protocol is docs/GTM_Model6_real_emit_pilot_20261023.md.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling

from GTM_Model0_local_viewer import load_bundle
from GTM_Model0_publish_viewer import publish

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "GTM_Model6_real_emit_20261023_cnn2_fit"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def counts(p, truth, support):
    yes, actual = p[support] >= .5, truth[support] == 1
    tp, fp, fn, tn = [int(x.sum()) for x in
                       (yes & actual, yes & ~actual, ~yes & actual, ~yes & ~actual)]
    return dict(tp=tp, fp_weak_background=fp, fn=fn, tn_weak_background=tn,
                annotated_recall=tp / (tp + fn), annotation_iou=tp / (tp + fp + fn),
                supported_cells=int(support.sum()), predicted_positive_cells=int(yes.sum()))


def embed(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    receipt = json.loads((run / "receipt.json").read_text())
    replay = json.loads((run / "replay.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    if not replay["passed"] or replay["receipt_sha256"] != sha(run / "receipt.json"):
        raise ValueError("No matching successful fresh-process replay")
    if manifest["role"] != "fit_only" or manifest["synthetic"] is not False:
        raise ValueError("Expected real fitting diagnostic")
    if sha(args.manifest) != receipt["manifest_sha256"]:
        raise ValueError("Changed prepared manifest")
    original_array = args.manifest.parent / manifest["array_file"]
    if sha(original_array) != manifest["array_sha256"] or sha(original_array) != receipt["array_sha256"]:
        raise ValueError("Changed prepared array")
    for source in manifest["sources"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Changed original cached source")
    for name, digest in receipt["source_hashes"].items():
        if sha(run / "source_snapshots" / name) != digest:
            raise ValueError("Changed saved source snapshot")
    for key in ("initial", "final", "prediction"):
        if sha(run / receipt[key]["path"]) != receipt[key]["sha256"]:
            raise ValueError("Changed saved model output")
    immutable_before = {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}
    with np.load(run / "predictions.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    with np.load(original_array, allow_pickle=False) as z:
        measured = {k: z[k] for k in z.files}
    for name in ("target", "target_valid", "input_valid"):
        if not np.array_equal(a[name], measured[name]):
            raise ValueError("Saved reference/support differs from prepared source")
    if not np.array_equal(a["image_dn"], measured["image"]):
        raise ValueError("Saved input differs from prepared source")
    t, support = a["target"], a["target_valid"].astype(bool)
    h, w = t.shape
    agreements = {}
    for stage in ("initial", "final"):
        aggregate = a[stage + "_fine"].reshape(h, 3, w, 3).mean(axis=(1, 3))
        if not np.allclose(aggregate[support], a[stage + "_native"][support], atol=1e-7, rtol=0):
            raise ValueError("Native prediction differs from exact 3:1 aggregation")
        agreements[stage] = counts(a[stage + "_native"], t, support)
        if agreements[stage] != receipt[stage]["agreement"]:
            raise ValueError("Independent annotation counts disagree")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    figures = out / "figures"
    figures.mkdir()
    rgb = measured["image"][..., [2, 1, 0]].copy()
    valid = measured["input_valid"].astype(bool)
    stretch = []
    for band in range(3):
        lo, hi = np.percentile(rgb[..., band][valid], (2, 98))
        stretch.append([float(lo), float(hi)])
        rgb[..., band] = np.clip((rgb[..., band] - lo) / (hi - lo), 0, 1)
    rgb[~valid] = .12
    native_rgb = rgb.reshape(h, 3, w, 3, 3).mean(axis=(1, 3))
    initial, final = a["initial_native"], a["final_native"]
    extent = [0, w * .06, h * .06, 0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "figure.facecolor": "#f5f7fa", "axes.facecolor": "#e6ebf0"})

    def picture(name, draw, title, note=""):
        fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=140)
        draw(ax)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("km east from crop edge")
        ax.set_ylabel("km south from crop edge")
        if note:
            fig.text(.08, .025, note, fontsize=8, color="#42526a")
        fig.tight_layout(rect=[0, .05, 1, 1])
        path = figures / (name + ".png")
        fig.savefig(path)
        plt.close(fig)
        return path

    paths = {}
    paths["rgb"] = picture("input", lambda ax: ax.imshow(native_rgb, extent=extent),
                           "Real Sentinel input | 2025-06-01", "RGB stretch from the same fitting input; no model output in this panel.")
    reference = np.where(support, t, np.nan)
    paths["annotation"] = picture("annotation", lambda ax: ax.imshow(reference, extent=extent, interpolation="nearest",
                  cmap=ListedColormap(["#183046", "#00c8d4"]), vmin=0, vmax=1),
                  "EMIT provider annotation | native 60 m", "Cyan = annotated plume. Dark = weak background. Gray = unknown.")
    for stage, values in (("initial", initial), ("final", final)):
        def draw(ax, values=values):
            im = ax.imshow(np.where(support, values, np.nan), extent=extent, interpolation="nearest", cmap="magma", vmin=0, vmax=1)
            ax.figure.colorbar(im, ax=ax, shrink=.72, label="Model score, not calibrated probability")
        paths[stage] = picture(stage, draw, "Before fitting" if stage == "initial" else "After fitting | CNN 2",
                               "Fixed 0 to 1 color scale; mean of each 3 by 3 output block.")

    def overlay(ax):
        ax.imshow(native_rgb, extent=extent)
        ax.contour(np.where(support, t, np.nan), levels=[.5], extent=extent, origin="upper", colors=["#00e7f4"], linewidths=1.5)
        ax.contour(np.where(support, final >= .5, np.nan), levels=[.5], extent=extent, origin="upper", colors=["#ff941d"], linewidths=1.3)
    paths["overlay"] = picture("overlay", overlay, "Actual prediction over Sentinel", "Cyan = provider outline. Orange = CNN 2 outline at score 0.5.")
    errors = np.full(t.shape, np.nan)
    errors[support] = 0
    errors[support & (t == 1) & (final >= .5)] = 1
    errors[support & (t == 0) & (final >= .5)] = 2
    errors[support & (t == 1) & (final < .5)] = 3
    paths["errors"] = picture("errors", lambda ax: ax.imshow(errors, extent=extent, interpolation="nearest", vmin=0, vmax=3,
            cmap=ListedColormap(["#183046", "#00bba6", "#fb9b28", "#d55ee8"])),
            "Errors against the fitting annotation", "Teal = match. Orange = extra prediction. Purple = missed annotation.")
    paths["retrieval"] = picture("retrieval", lambda ax: ax.imshow(measured["retrieval"], extent=extent, cmap="magma", interpolation="nearest"),
                                "EMIT matched-filter retrieval", "Stored values only; physical units unconfirmed. This is not a model prediction.")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), dpi=130)
    for band, (ax, name) in enumerate(zip(axes, manifest["bands"])):
        band_values = measured["image"][..., band]
        lo, hi = np.percentile(band_values[valid], (2, 98))
        ax.imshow(np.where(valid, band_values, np.nan), extent=extent, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(name + " | raw DN")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("All five real inputs to CNN 2 | each band has a separate display stretch")
    fig.tight_layout()
    paths["bands"] = figures / "bands.png"
    fig.savefig(paths["bands"])
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 2.8), dpi=140)
    ax.plot([r["epoch"] for r in receipt["history"]], [r["fitting_loss"] for r in receipt["history"]], color="#207c8e", lw=1.2)
    ax.set(xlabel="Complete fitting epoch", ylabel="Fitting objective", title="Loss on sampled fitting crops; no held-out curve")
    ax.grid(alpha=.15)
    fig.tight_layout()
    paths["loss"] = figures / "loss.png"
    fig.savefig(paths["loss"])
    plt.close(fig)
    # One review sheet is also useful outside the browser.
    sheet = Image.new("RGB", (1512, 822), "#f5f7fa")
    for k, key in enumerate(("rgb", "annotation", "final", "initial", "overlay", "errors")):
        img = Image.open(paths[key]).convert("RGB").resize((504, 411), Image.Resampling.LANCZOS)
        sheet.paste(img, ((k % 3) * 504, (k // 3) * 411))
    sheet.save(figures / "real_prediction_review.png")
    # Bundle geometry uses a shared Mercator display grid. Scoring stays native.
    affine = Affine(*manifest["target_grid"]["affine"])
    crs = manifest["target_grid"]["crs"]
    bounds = array_bounds(h, w, affine)
    mercator = transform_bounds(crs, "EPSG:3857", *bounds, densify_pts=21)
    dst = from_bounds(*mercator, w, h)
    def project(values, nearest=True):
        result = np.full((h, w), np.nan, np.float32)
        reproject(values.astype(np.float32), result, src_transform=affine, src_crs=crs,
                  dst_transform=dst, dst_crs="EPSG:3857", src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear)
        return result
    pr = project(np.where(support, final, np.nan))
    tr = project(reference)
    vr = np.isfinite(pr) & np.isfinite(tr)
    rg = np.stack([project(native_rgb[..., k], False) for k in range(3)], axis=-1)
    folder = out / "scene"
    folder.mkdir()
    Image.fromarray((np.nan_to_num(rg) * 255).astype(np.uint8)).save(folder / "satellite.png")
    for name, condition, color in (("reference", vr & (tr == 1), [0, 225, 230, 190]),
                                   ("prediction", vr & (pr >= .5), [255, 135, 35, 210])):
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[condition] = color
        Image.fromarray(rgba).save(folder / (name + ".png"))
    values = lambda arr: [float(v) if np.isfinite(v) else None for v in arr.ravel()]
    save(folder / "grid.json", dict(width=w, height=h, probability=values(pr), truth=values(tr),
                                    valid=vr.astype(np.uint8).ravel().tolist(), crs="EPSG:3857"))
    west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", *mercator)
    fit = agreements["final"]
    nmetrics = dict(tp=fit["tp"], fp=fit["fp_weak_background"], fn=fit["fn"], tn=fit["tn_weak_background"],
                    precision=fit["tp"] / (fit["tp"] + fit["fp_weak_background"]), recall=fit["annotated_recall"],
                    iou=fit["annotation_iou"], evaluated_pixels=fit["supported_cells"],
                    support_fraction=float(support.mean()), source_valid_pixels=int(support.sum()),
                    claim="Fitting annotation agreement only; weak background")
    viewer_link = f"/?catalog=all&experiment={IDENTIFIER}&scene=Volusia-20250601-fit#compare"
    tiles = "".join(f'<figure><img src="{embed(paths[k])}" alt="{html.escape(label)}"><figcaption>{label}</figcaption></figure>' for k, label in
                    (("rgb", "Real satellite input"), ("annotation", "Provider annotation, 60 m"), ("final", "Actual CNN 2 output"),
                     ("initial", "Same model before fitting"), ("overlay", "Aligned prediction and annotation"), ("errors", "Where the fit is wrong")))
    metrics = f'<b>{fit["annotation_iou"]:.1%}</b> annotation IoU on the fitting scene. The full-box control gets {receipt["controls"]["always_positive"]["annotation_iou"]:.1%}.'
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real EMIT fitting review | CNN 2</title>
<style>body{{margin:0;background:#eef2f6;color:#182d42;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:36px 24px 64px}}h1{{font-size:34px;line-height:1.2;margin:8px 0 16px}}h2{{font-size:21px;margin-top:32px}}p{{max-width:950px}}.label{{font-size:12px;font-weight:800;letter-spacing:.13em;color:#28687b}}.notice{{padding:18px 22px;border:1px solid #dac593;border-radius:12px;background:#fff9e9}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}}figure{{margin:0;background:white;border:1px solid #dce4eb;border-radius:12px;overflow:hidden}}figure img{{display:block;width:100%;height:auto}}figcaption{{padding:10px 18px;font-size:13px;font-weight:650}}a{{color:#086b86}}.flow{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:20px;background:white;border:1px solid #dce4eb;border-radius:12px}}.flow span{{padding:12px;background:#e7f2f4;border-radius:8px}}table{{border-collapse:collapse;width:100%;max-width:900px;background:white}}th,td{{text-align:left;padding:10px 14px;border-bottom:1px solid #dce4eb}}.wide{{max-width:900px}}.meta{{font-size:13px;color:#526778;overflow-wrap:anywhere}}button{{font:inherit;cursor:pointer;border:1px solid #9fb8c9;background:white;color:#17485d;border-radius:8px;padding:9px 14px}}.large{{position:fixed;inset:0;background:#142333ec;display:none;align-items:center;justify-content:center;z-index:10;padding:30px}}.large img{{max-width:95vw;max-height:90vh;object-fit:contain}}@media(max-width:850px){{.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:550px){{main{{padding:24px 12px}}.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}}}</style></head><body><main>
<div class="label">GTM MODEL 6 / MEASURED INPUT / CNN 2</div><h1>Real data. A fitting check we can inspect.</h1>
<div class="notice"><strong>One site, fitting only.</strong> These predictions use real Sentinel imagery and a real EMIT provider annotation. The model saw this scene during fitting. This is not an independent accuracy result, a nationwide detector, or validated methane upscaling.</div>
<p>Volusia landfill, Florida, June 1, 2025. Sentinel followed EMIT by <strong>12.742 minutes</strong>. {metrics} Threshold is fixed at 0.5. No image or model weights were downloaded for this run.</p>
<p><a href="{viewer_link}">Open these saved predictions in Map &amp; Evidence</a> · Click any image below to inspect it.</p><div class="grid">{tiles}</div>
<h2>What goes through CNN 2</h2><div class="flow"><span>Five real Sentinel bands<br>165 × 183 × 5, raw DN</span>→<span>Fitting-only normalization<br>Small U-Net branch</span>→<span>20 m output scores<br>Mean of each 3 × 3 block</span>→<span>60 m annotation comparison<br>Masked BCE + Dice</span></div>
<figure><img src="{embed(paths["bands"])}" alt="B2, B3, B4, B11, B12 real input bands"><figcaption>Visible and SWIR input bands. Separate display stretches do not change the training arrays.</figcaption></figure>
<p>Only Sentinel bands enter the model. EMIT supplies the training reference. CNN 1 still needs airborne measurements, so this run does not train CNN 1 or the fusion CNN 3. Fine output pixels do not establish new measured detail.</p>
<h2>What the labels mean</h2><p>Cyan cells are provider-annotated plume. Zero is <strong>weak unannotated background</strong>, not proof that methane was absent. The fitting loss uses {fit["supported_cells"]:,} native cells: {manifest["counts"]["annotated_positive"]:,} annotated and {manifest["counts"]["unannotated_background"]:,} weak background. The {manifest["counts"]["unknown"]} unknown cells are excluded. Sentinel cloud and invalid cells are also excluded.</p>
<table><thead><tr><th>Fitting annotation comparison</th><th>Cells</th></tr></thead><tbody><tr><td>Annotated cells found</td><td>{fit["tp"]:,}</td></tr><tr><td>Annotated cells missed</td><td>{fit["fn"]:,}</td></tr><tr><td>Extra predictions in weak background</td><td>{fit["fp_weak_background"]:,}</td></tr><tr><td>Weak background left unmarked</td><td>{fit["tn_weak_background"]:,}</td></tr></tbody></table>
<h2>How it was fitted</h2><p>{receipt["completed_epochs"]} complete epochs, {sum(receipt["sampled_crops"].values()):,} sampled real crops, with {receipt["sampled_crops"]["weak_background"]:,} all-background crops and {receipt["sampled_crops"]["positive"]:,} crops containing annotation. Exact rotations and mirrors are applied to input, label, and support together. Every crop belongs to this same fitting scene. There is no held-out curve.</p>
<figure class="wide"><img src="{embed(paths["loss"])}" alt="Fitting loss by complete epoch"><figcaption>Fitting objective, not independent prediction quality</figcaption></figure>
<details><summary>Inspect the measured EMIT retrieval</summary><figure class="wide"><img src="{embed(paths["retrieval"])}" alt="EMIT matched-filter retrieval"><figcaption>This stored retrieval is supplied data. It is neither a CNN output nor a confirmed physical concentration map.</figcaption></figure></details>
<h2>What is missing next</h2><p>We need more independent close-time pairs, a documented background-label policy, and examples where no plume was annotated or detected within known valid observation coverage. The older catalog-outline cases cannot establish precision because their exterior is unknown. The older continuous EMIT TIFFs need a separate retrieval regression path. Airborne labels are needed to train CNN 1 and test whether fusion helps.</p>
<p>The first 20-epoch attempt collapsed to background. The balanced run shown here produces a spatial prediction and still makes substantial errors. Its improvement shows that the branch can fit this real annotation. It does not show that Sentinel alone can detect unseen methane plumes.</p>
<h2>Repeatable evidence</h2><p>Initial and final checkpoint predictions matched a fresh-process replay. A separate export check verified original source hashes, reference support, exact native aggregation, and every count above.</p><p class="meta">Run: {html.escape(str(run.relative_to(ROOT)))}<br>Manifest SHA-256: {receipt["manifest_sha256"]}<br>Final weights SHA-256: {receipt["final"]["sha256"]}<br>Saved predictions SHA-256: {receipt["prediction"]["sha256"]}</p>
</main><div class="large" id="large" role="dialog" aria-label="Expanded image"><img alt="Expanded research figure"></div><script>const box=document.getElementById('large');document.querySelectorAll('figure img').forEach(img=>img.addEventListener('click',()=>{{box.querySelector('img').src=img.src;box.style.display='flex';}}));box.addEventListener('click',()=>box.style.display='none');document.addEventListener('keydown',e=>{{if(e.key==='Escape')box.style.display='none';}});</script></body></html>'''
    (out / "index.html").write_text(body, encoding="utf-8")
    scene = dict(id="Volusia-20250601-fit", name="Volusia | real fitting sample", task="plume_segmentation",
                 bounds=[[south, west], [north, east]], date="2025-06-01", referenceDate="2025-06-01",
                 rgb="scene/satellite.png", truthImage="scene/reference.png", predictionImage="scene/prediction.png",
                 grid="scene/grid.json", predictionAvailable=True, decisionThreshold=.5,
                 truthLabel="EMIT provider annotation | fitting only",
                 truthNote="Zero is weak unannotated background. Unknown cells excluded. Not independent ground truth.",
                 rgbNote="Real Sentinel L2A DN | 12.742 min after EMIT | 2-98 percentile RGB stretch",
                 notes="FIT ONLY: this scene was used for training. Cyan is provider annotation; orange is CNN 2. Scores are not calibrated methane probabilities. Zero labels are weak background. No independent accuracy or fine-resolution recovery is established.",
                 split="Fitting only | Volusia-20250601", source=receipt["manifest_sha256"],
                 checkpointHash=receipt["final"]["sha256"], predictionHash=receipt["prediction"]["sha256"],
                 nativeMetrics=nmetrics, resolution="60 m native annotation and aggregated prediction", dataKind="measured",
                 qualityFlags=["fit_only", "weak_background", "single_site", "no_independent_evaluation"])
    experiment = dict(id=IDENTIFIER, name="Real data fit | CNN 2 | Volusia", lifecycle="experimental", priority=False,
                      dataKind="measured", architectureFamily="GTM_Model6_three_cnn", runGroup="GTM_Model6_real_emit_20261023",
                      stage="emit", status="FIT ONLY | one real Sentinel-EMIT scene",
                      description="Real-data fitting diagnostic with weak background labels. Not independent detection accuracy.",
                      notes=scene["notes"], decisionThreshold=.5, scenes=[scene], reportUrl="index.html")
    save(out / "experiment.json", experiment)
    load_bundle(out / "experiment.json", "bundle-assets/")
    immutable_after = {str(p.relative_to(run)): sha(p) for p in run.rglob("*") if p.is_file()}
    if immutable_before != immutable_after:
        raise ValueError("Review altered original training artifacts")
    verification = dict(passed=True, source_rasters_verified=len(manifest["sources"]),
                        manifest_sha256=sha(args.manifest), run_artifacts_unchanged=immutable_before == immutable_after,
                        independent_counts=agreements, full_scene_replay=replay,
                        html_sha256=sha(out / "index.html"), renderer_sha256=sha(__file__), rgb_stretch=stretch)
    if args.publish:
        verification["publication"] = publish(out, ROOT / "outputs/GTM_viewer_bundles")
    save(out / "verification.json", verification)
    print(json.dumps(dict(output=str(out), viewer=viewer_link, agreements=agreements,
                          publication=verification.get("publication")), indent=2))


if __name__ == "__main__":
    main()
