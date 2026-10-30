"""Build a portable HTML review from saved model replay evidence.

No training and no network access. Failure cases and acceptance precede this
implementation in the learning review's PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
from urllib.parse import urlencode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

STAGES = (("airborne", "CNN 1", "Airborne branch"),
          ("emit", "CNN 2", "EMIT branch"), ("fusion", "CNN 3", "Feature fusion"))
BANDS = ("B2 · blue", "B3 · green", "B4 · red", "B11 · SWIR 1", "B12 · SWIR 2")
INK = "#162c39"
ACCENT = "#147d76"


def pct(value):
    return "Not defined" if value is None else f"{100 * value:.1f}%"


def png(array, cmap="magma", vmin=0, vmax=1):
    array = np.asarray(array)
    if array.ndim == 2:
        norm = np.clip((array - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        rgba = (matplotlib.colormaps[cmap](np.nan_to_num(norm)) * 255).astype(np.uint8)
        rgba[~np.isfinite(array)] = (202, 212, 217, 255)
    else:
        rgba = (np.nan_to_num(np.clip(array, 0, 1), nan=.8) * 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgba).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def fig_png(fig, path):
    fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def picture(url, title, note="", cls=""):
    return (f'<figure class="{cls}"><img src="{url}" alt="{html.escape(title)}">'
            f'<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(note)}</span>'
            '</figcaption></figure>')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def native(a, key="probability"):
    p = a[key]
    nh, nw = a["target"].shape
    fy, fx = p.shape[0] // nh, p.shape[1] // nw
    valid = a["input_valid"].astype(bool).reshape(nh, fy, nw, fx).all(axis=(1, 3))
    valid &= a["target_valid"].astype(bool)
    result = np.nan_to_num(p).reshape(nh, fy, nw, fx).mean(axis=(1, 3))
    return np.where(valid, result, np.nan)


def error_image(a, probability):
    target = a["target"]
    valid = np.isfinite(probability) & a["target_valid"].astype(bool)
    prediction = probability >= .5
    truth = target == 1
    result = np.full((*truth.shape, 3), .94)
    result[valid & truth & prediction] = (.08, .51, .45)
    result[valid & ~truth & prediction] = (.9, .34, .14)
    result[valid & truth & ~prediction] = (.39, .37, .71)
    result[~valid] = (.78, .83, .85)
    return result


def overlay(a):
    image = np.nan_to_num(a["image"][:, :, [2, 1, 0]], nan=.8)
    rgb = np.clip(image * 2.1, 0, 1)
    fine = a["probability"]
    mask = np.isfinite(fine) & (fine >= .5)
    rgb[mask] = .45 * rgb[mask] + .55 * np.array([1, .3, .05])
    t = a["target"] == 1
    support = a["target_valid"].astype(bool)
    t &= support
    t = np.repeat(np.repeat(t, fine.shape[0]//t.shape[0], axis=0), fine.shape[1]//t.shape[1], axis=1)
    padded = np.pad(t, 1, constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge = t & ~interior
    rgb[edge] = (0, 1, .8)
    rgb[~a["input_valid"].astype(bool)] = (.78, .83, .85)
    return rgb


def activation_fig(array, path, title):
    channels = array.shape[-1]
    fig, axes = plt.subplots(1, channels, figsize=(max(6, channels * 1.65), 2.15), squeeze=False)
    finite = array[np.isfinite(array)]
    low, high = float(finite.min()), float(finite.max())
    for i, ax in enumerate(axes[0]):
        ax.imshow(array[..., i], cmap="viridis", vmin=low, vmax=high, interpolation="nearest")
        ax.set_title(f"Channel {i+1}", fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    return fig_png(fig, path), f"{low:.3f} to {high:.3f}; shared scale within this tensor; arbitrary units"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.audit / "audit.json").read_text(encoding="utf-8"))
    if report["data_kind"] != "synthetic":
        raise ValueError("This handout is limited to synthetic engineering evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "figures"
    assets.mkdir(exist_ok=True)
    payload = {"stages": {}, "steps": []}
    metric_rows = []
    for stage, number, name in STAGES:
        s = report["stages"][stage]
        metric_rows.append(f'<tr><th>{number}<small>{name}</small></th><td>{pct(s["untrained"]["iou"])}</td>'
            f'<td><strong>{pct(s["trained"]["iou"])}</strong></td><td>{pct(s["trained"]["precision"])}</td>'
            f'<td>{pct(s["trained"]["recall"])}</td><td>{s["negative_scenes_with_any_false_activation"]} / {s["negative_scenes"]}</td></tr>')
        curve = plt.figure(figsize=(9.8, 3.25))
        ax = curve.add_subplot(111)
        history = s["history"]
        epochs = [0] + [r["epoch"] for r in history]
        values = [s["initial_validation_loss"]] + [r["validation_loss"] for r in history]
        ax.plot(epochs, values, "o-", color=ACCENT, lw=2, label="Held-out validation")
        ax.plot([r["epoch"] for r in history], [r["fit_loss"] for r in history], "o--", color="#b66e32", label="Fitting examples")
        ax.axhline(s["trained_validation_loss"], color=ACCENT, lw=.8, alpha=.5, linestyle=":")
        ax.set(xlabel="Epoch (0 = seeded random control)", ylabel="BCE + soft Dice loss")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e7e7")
        ax.legend(frameon=False, fontsize=9)
        curve.tight_layout()
        curve_url = fig_png(curve, assets / f"{stage}_learning_curve.png")
        examples = []
        for label, scene_id in s["selections"].items():
            a = load(args.audit / f"{stage}_{label}.npz")
            rec = next(r for r in s["records"] if r["id"] == scene_id)
            title = label.replace("_", " ").capitalize()
            before = a["native_untrained"]
            after = a["native_probability"]
            panels = ''.join((
                picture(png(np.clip(a["image"][:, :, [2, 1, 0]] * 2.1, 0, 1)), "Artificial RGB input", "Display stretch only; 48 × 48"),
                picture(png(np.where(a["target_valid"], a["target"], np.nan), "Greys", 0, 1), "Reference", "Artificial binary labels; 16 × 16"),
                picture(png(after), "After training", "Native score; fixed 0 to 1 scale"),
                picture(png(before), "Before training", "Native score; fixed 0 to 1 scale"),
                picture(png(error_image(a, after)), "After training: errors", "Green = hit; orange = false alarm; purple = miss"),
                picture(png(overlay(a)), "Fine-grid overlay", "Orange = model ≥ 0.50; mint = coarse reference edge")))
            examples.append(dict(key=label, title=title, id=scene_id, images=panels,
                caption=f'Plume overlap: {pct(rec["trained"]["iou"])}. False-positive cells: {rec["trained"]["fp"]}. Missed plume cells: {rec["trained"]["fn"]}.',
                positive=rec["positive"]))
        fittings = []
        for sample in s["fitting_examples"]:
            a = load(args.audit / sample["file"])
            panels = []
            if stage != "fusion":
                for band, title in enumerate(BANDS):
                    panels.append(picture(png(a["image"][..., band], "gray", 0, .6), title, "Artificial reflectance; common 0 to 0.60 scale"))
            else:
                for key, title in (("airborne_features", "CNN 1 spatial features"), ("emit_features", "CNN 2 spatial features")):
                    url, note = activation_fig(a[key], assets / f'{stage}_{sample["positive"]}_{key}.png', title)
                    panels.append(picture(url, title, note, "wide"))
            panels.append(picture(png(np.where(a["target_valid"], a["target"], np.nan), "Greys"), "Training reference", "Artificial labels; loss only, never a CNN input"))
            panels.append(picture(png(native(a)), "Trained model output", "Training example; this is not held-out evidence"))
            fittings.append(dict(title="Plume training example" if sample["positive"] else "No-plume training example",
                                 id=sample["id"], images=''.join(panels)))
        payload["stages"][stage] = dict(number=number, name=name, curve=curve_url, examples=examples, fitting=fittings,
            epochs=f'{s["completed_epochs"]} full / {s["requested_epochs"]} requested epochs',
            loss=f'{s["initial_validation_loss"]:.4f} → {s["trained_validation_loss"]:.4f}',
            duration=f'{s["elapsed_seconds"]:.1f} seconds fitting and validation',
            fp=f'{s["negative_false_positive_cells"]} native false-positive cells / {s["negative_evaluated_cells"]} evaluated cells on {s["negative_scenes"]} no-plume test scenes. '
               f'Fine-grid activations: {s["negative_fine_threshold_cells"]} cells in {s["negative_scenes_with_any_fine_activation"]} no-plume scenes. Fine-grid truth is not independently available.',
            changed=s["initial_weight_hash"] != s["trained_weight_hash"],
            replay=s["replay_max_abs"], control=s["always_negative"], records=s["records"])
        if stage == "fusion":
            fig, axes = plt.subplots(2, 5, figsize=(14.5, 6.1))
            for index, label in enumerate(("median_positive", "worst_negative")):
                a = load(args.audit / f"{stage}_{label}.npz")
                panels = ((np.clip(a["image"][:, :, [2, 1, 0]] * 2.1, 0, 1), "Artificial RGB input", None),
                          (np.where(a["target_valid"], a["target"], np.nan), "Artificial reference", "Greys"),
                          (a["native_untrained"], "Before fitting CNN 3", "magma"),
                          (a["native_probability"], "After fitting CNN 3", "magma"),
                          (error_image(a, a["native_probability"]), "Hits / false alarms / misses", None))
                for ax, (array, title, cmap) in zip(axes[index], panels):
                    plot_cmap = matplotlib.colormaps[cmap].with_extremes(bad="#cad4d9") if cmap else None
                    ax.imshow(array, cmap=plot_cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                              interpolation="nearest")
                    if index == 0:
                        ax.set_title(title, fontsize=10, color=INK)
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_color("#d7e2e4")
                axes[index, 0].set_ylabel("Held-out plume" if index == 0 else "Held-out no-plume", color=INK, fontsize=11)
            fig.suptitle("CNN 3 · Synthetic learning review · Not measured methane accuracy", fontsize=15, color=INK)
            fig.text(.5, .015, "All score maps: 0 to 1. Native grid: 16 × 16. Green: correct plume. Orange: false alarm. Purple: missed plume.",
                     ha="center", fontsize=10, color="#56707b")
            fig.tight_layout(rect=[0, .035, 1, .95], h_pad=2, w_pad=1.4)
            fig_png(fig, assets / "fusion_prediction_review.png")
    ff = load(args.audit / "feed_forward.npz")
    input_panels = ''.join(picture(png(ff["image"][..., i], "gray", 0, .6), name, "Artificial reflectance; 0 to 0.60") for i, name in enumerate(BANDS))
    payload["steps"].append(dict(title="1. Read the five-band image", shape="48 × 48 × 5", active="source",
        text="Read B2, B3, B4, B11 and B12 from the same artificial scene. Grey pixels have no valid input. The reference is not an input.", images=input_panels))
    payload["steps"].append(dict(title="2. Apply each branch's saved normalization", shape="Two tensors, each 48 × 48 × 5", active="normalize",
        text="Subtract the fitting-set mean and divide by its standard deviation. Each branch has its own values. Invalid pixels become zero. No test data fits these values.",
        images=''.join(picture(png(ff[key][..., 4], "coolwarm", -3, 3), title, "Normalized B12 shown; fixed −3 to +3 scale")
                       for key, title in (("normalized_airborne", "CNN 1 input"), ("normalized_emit", "CNN 2 input")))))
    for key, title, active, score in (("airborne_features", "3. Run CNN 1", "cnn1", "airborne_score"), ("emit_features", "4. Run CNN 2", "cnn2", "emit_score")):
        url, note = activation_fig(ff[key], assets / f"feed_{key}.png", title)
        payload["steps"].append(dict(title=title, shape="48 × 48 × 4 features + 48 × 48 score", active=active,
            text="The U-Net reads the normalized image. It combines three internal spatial scales and skip connections. Its four-channel spatial features go to CNN 3. The score below is a separate branch output.",
            images=picture(url, "Actual spatial features", note, "wide") + picture(png(ff[score]), "Standalone branch score", "Fixed 0 to 1 scale; not the CNN 3 input")))
    url, note = activation_fig(ff["joined_features"], assets / "feed_joined_features.png", "Projected CNN 1 and CNN 2 features")
    payload["steps"].append(dict(title="5. Project and join both feature grids", shape="48 × 48 × 8", active="join",
        text="Two trainable 1 × 1 convolutions project the feature grids. Concatenation puts them side by side in the channel dimension. CNN 1 and CNN 2 weights stay frozen.",
        images=picture(url, "Actual joined features", note, "wide")))
    url, note = activation_fig(ff["fusion_features"], assets / "feed_fusion_features.png", "CNN 3 decoder features")
    payload["steps"].append(dict(title="6. Run CNN 3 and apply sigmoid", shape="48 × 48 × 4 → 48 × 48 × 1", active="cnn3",
        text="CNN 3 applies two spatial blocks, then a 1 × 1 output convolution. Sigmoid maps its logits to a score from zero to one. This is a plume-mask score, not methane concentration or a calibrated probability.",
        images=picture(url, "Actual CNN 3 decoder features", note, "wide") + picture(png(ff["fusion_probability"]), "Full-scene forward-pass score", "48 × 48; fixed 0 to 1 scale")))
    forward_arrays = dict(ff, probability=ff["fusion_probability"])
    payload["steps"].append(dict(title="7. Average to the reference grid and compare", shape="48 × 48 → 16 × 16", active="output",
        text="Average each 3 × 3 block. Score only reference cells with complete valid input and known labels. Apply the fixed 0.50 cutoff. Fine-grid texture has no independent fine-resolution truth in this experiment.",
        images=picture(png(native(forward_arrays)), "Native-grid model score", "16 × 16; the grid used for metrics") + picture(png(np.where(ff["target_valid"], ff["target"], np.nan), "Greys"), "Held-out artificial reference", "16 × 16; scoring only") + picture(png(error_image(ff, native(forward_arrays))), "Native-grid agreement", "Green = hit; orange = false alarm; purple = miss")))
    fusion = report["stages"]["fusion"]
    emit = report["stages"]["emit"]
    improvement = 100 * (fusion["trained"]["iou"] - emit["trained"]["iou"])
    verdict = (f'CNN 3 changes plume overlap by {improvement:+.1f} percentage points versus CNN 2 on the same artificial test scenes. '
               'This small change does not establish a useful benefit from fusion. '
               'This rehearsal tests optimization and data flow. It does not establish real methane detection or useful upscaling.')
    if report["stages"]["airborne"]["trained"]["recall"] == 0:
        verdict += (' CNN 1 misses every held-out plume at the fixed cutoff. Its lower loss is not successful plume detection. '
                    'The three-CNN design is not validated as a complete methane model.')
    all_rows = []
    for stage, number, _ in STAGES:
        for r in report["stages"][stage]["records"]:
            m = r["trained"]
            all_rows.append(f'<tr data-stage="{stage}"><td>{number}</td><th>{html.escape(r["id"])}</th><td>{"Plume" if r["positive"] else "No plume"}</td>'
                f'<td>{pct(r["untrained"]["iou"])}</td><td>{pct(m["iou"])}</td><td>{m["tp"]}</td><td>{m["fp"]}</td><td>{m["fn"]}</td><td>{m["tn"]}</td></tr>')
    table = ''.join(metric_rows)
    rows_html = ''.join(all_rows)
    json_payload = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    audit_payload = json.dumps(report, indent=2, allow_nan=False).replace("<", "\\u003c")
    protocol = html.escape(args.protocol.read_text(encoding="utf-8"))
    template = TEMPLATE.replace("__METRICS__", table).replace("__VERDICT__", html.escape(verdict))
    template = template.replace("__ROWS__", rows_html).replace("__DATA__", json_payload).replace("__AUDIT__", audit_payload)
    template = template.replace("__PROTOCOL__", protocol).replace("__MANIFEST_HASH__", report["manifest_sha256"])
    template = template.replace("__FORWARD_ID__", html.escape(report["feed_forward"]["id"]))
    template = template.replace("__DIRECT_DIFF__", f'{report["feed_forward"]["full_scene_vs_tiled_max_abs"]:.8g}')
    template = template.replace("__BACKEND__", html.escape(report.get("backend", "TensorFlow")))
    template = template.replace("__RUNTIME_JSON__", html.escape(json.dumps(report["runtime"], indent=2)))
    viewer_url = "http://127.0.0.1:8766/?" + urlencode(dict(catalog="all", experiment=report["run_id"] + "_fusion",
                                                          scene=report["feed_forward"]["id"])) + "#compare"
    template = template.replace("__VIEWER_URL__", html.escape(viewer_url, quote=True))
    target = args.output / "index.html"
    target.write_text(template, encoding="utf-8")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verification = dict(html_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        bytes=target.stat().st_size, embedded_images=template.count('data:image/png;base64,'),
        data_kind="synthetic", stages=list(payload["stages"]), steps=len(payload["steps"]),
        held_out_evaluations=len(all_rows), note="Build receipt only; inspect the actual browser output separately")
    (args.output / "build_receipt.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(path=str(target), **verification), indent=2))


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three-CNN learning review | GTM Model 6</title>
<style>
:root{--ink:#162c39;--muted:#56707b;--line:#d7e2e4;--accent:#147d76;--warm:#fff7e7;--paper:#f7faf9;--orange:#bd6127}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:90px}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 'Segoe UI',Arial,sans-serif}button,select{font:inherit}button,a,select{touch-action:manipulation}a{color:var(--accent)}button{cursor:pointer}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #e0b44a;outline-offset:3px}
header{position:sticky;top:0;z-index:5;background:#122e39;color:#ecf5f3;display:flex;align-items:center;justify-content:space-between;padding:17px 4vw;gap:20px}header strong{font-size:16px;letter-spacing:.04em}header span{font-size:12px;color:#b4cbd0}nav{display:flex;gap:20px}nav a{color:#dceae9;text-decoration:none;font-size:13px}nav a:hover{color:white;text-decoration:underline}
main{max-width:1420px;margin:auto;padding:36px 4vw 80px}h1{font-size:38px;line-height:1.2;letter-spacing:-.03em;margin:14px 0}h2{font-size:25px;line-height:1.3;margin:0 0 10px;letter-spacing:-.015em}h3{font-size:18px;margin:0 0 12px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:700}.lede{font-size:17px;max-width:850px;color:var(--muted)}.notice{padding:16px 20px;border-left:4px solid #c28c35;background:var(--warm);margin:22px 0}.notice strong{display:block}.notice p{margin:4px 0}.summary{display:grid;grid-template-columns:1.5fr 1fr;gap:50px;align-items:start}.facts{border-top:1px solid var(--line);padding-top:14px;margin-top:20px}.facts div{display:flex;justify-content:space-between;gap:20px;padding:8px 0;border-bottom:1px solid var(--line)}.facts strong{white-space:nowrap}.muted{color:var(--muted)}section{padding:35px 0;border-top:1px solid var(--line);margin-top:28px}p{max-width:1000px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:12px 16px;border-bottom:1px solid var(--line);font-size:14px}thead{background:#eaf1f0}th small{display:block;color:var(--muted);font-weight:400}td strong{color:var(--accent)}.comparison td:nth-child(3){background:#edf5f2}.caption{font-size:13px;color:var(--muted)}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:14px;margin:22px 0}label{font-size:13px;font-weight:600}select{padding:9px 36px 9px 12px;border:1px solid #a8bdc2;border-radius:5px;background:white;color:var(--ink);max-width:100%}.tabs{display:inline-flex;gap:3px;background:#e5eeed;padding:4px;border-radius:7px}.tabs button{padding:9px 22px;border:0;color:var(--muted);background:transparent;border-radius:4px}.tabs button[aria-pressed="true"]{color:white;background:var(--accent)}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}figure{margin:0;min-width:0}figure img{display:block;width:100%;aspect-ratio:1;object-fit:contain;image-rendering:pixelated;background:#e7eded;border:1px solid var(--line)}figcaption{font-size:13px;line-height:1.5;padding-top:9px}figcaption strong,figcaption span{display:block}figcaption span{color:var(--muted);font-size:12px}.wide{grid-column:1/-1}.wide img{aspect-ratio:auto;max-height:280px;image-rendering:auto}code,.mono{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.scene-id{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.curve{width:100%;max-height:310px;object-fit:contain;background:white}.learning{display:grid;grid-template-columns:2fr 1fr;gap:28px;align-items:center}.proof{padding:12px 0;border-bottom:1px solid var(--line)}.proof strong{display:block}.proof span{font-size:13px;color:var(--muted)}.flow{background:#edf3f1;padding:20px;margin:22px 0;overflow:auto}.flow svg{display:block;width:100%;min-width:700px;height:auto}.flow rect{fill:white;stroke:#b6cccd;stroke-width:1.5}.flow .active rect{fill:#d7eeE7;stroke:var(--accent);stroke-width:3}.flow text{font-family:'Segoe UI',Arial,sans-serif;fill:var(--ink);font-size:15px}.flow .sub{font-size:12px;fill:#58717b}.flow path{stroke:#6a9396;stroke-width:2;fill:none;marker-end:url(#arrow)}.step-layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:35px}.step-buttons{display:flex;flex-direction:column;gap:5px}.step-buttons button{text-align:left;border:0;border-left:3px solid transparent;background:transparent;padding:11px 14px;color:var(--muted);font-size:14px}.step-buttons button[aria-current="step"]{border-left-color:var(--accent);background:#e6f0ed;color:var(--ink);font-weight:600}.step-controls{display:flex;align-items:center;gap:12px;margin:20px 0}.step-controls button,.button{border:1px solid #9bb7b9;background:white;color:var(--ink);border-radius:5px;padding:8px 17px;text-decoration:none;font-size:13px}.step-controls button:disabled{opacity:.4;cursor:default}#step-shape{font-family:Consolas,monospace;font-size:14px;color:var(--accent)}details{margin:18px 0}summary{cursor:pointer;font-weight:600;padding:10px 0}pre{white-space:pre-wrap;font-size:12px;background:#edf2f1;padding:20px;max-height:420px;overflow:auto}.all-records{max-height:440px;overflow:auto}.all-records th{font-size:11px;overflow-wrap:anywhere}.all-records td{font-size:12px}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:15px 0}.legend i{display:inline-block;width:10px;height:10px;margin-right:6px;background:var(--c)}.colorbar{height:7px;max-width:260px;background:linear-gradient(90deg,#000004,#51127c,#b63679,#fb8761,#fcfdbf);margin-top:8px}.colorbar-label{display:flex;justify-content:space-between;max-width:260px;font-size:11px;color:var(--muted)}.hidden{display:none!important}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:20px}.jump{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}#selection-images,#step-images,#training-images{animation:appear .18s ease-out}@keyframes appear{from{opacity:.5}to{opacity:1}}@media(prefers-reduced-motion:reduce){*{animation:none!important;scroll-behavior:auto!important}}@media(max-width:900px){header{flex-wrap:wrap;padding:12px 5vw}header span{display:none}nav{gap:15px}main{padding:25px 5vw}.summary,.learning,.step-layout{grid-template-columns:1fr;gap:20px}.step-buttons{display:grid;grid-template-columns:1fr 1fr}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:30px}td,th{padding:10px;font-size:12px}}@media(max-width:480px){.gallery{grid-template-columns:1fr 1fr;gap:14px}.tabs button{padding:8px 15px}.step-buttons{grid-template-columns:1fr}nav{font-size:12px}figcaption strong{font-size:12px}}@media print{header{position:static}nav,.step-buttons,.step-controls{display:none}section{break-inside:avoid}.step-layout{display:block}body{background:white}}
</style></head><body>
<header><div><strong>GTM / MODEL 6</strong> <span>LOCAL RESEARCH REVIEW · 23 OCT 2026</span></div><nav aria-label="Report sections"><a href="#results">Results</a><a href="#examples">Predictions</a><a href="#training">CNN inputs</a><a href="#forward">Forward pass</a><a href="#evidence">Evidence</a></nav></header>
<main><div class="summary"><div><div class="eyebrow">Three-CNN architecture</div><h1>Training and prediction review</h1><p class="lede">Inspect what each network receives, what it learns, and where its predictions fail.</p><div class="notice"><strong>Artificial data only</strong><p>All images, plume labels and no-plume examples in this report are synthetic. “Airborne” and “EMIT” describe simulated supervision roles. This is not a measured methane result.</p></div></div><div class="facts"><div><span>Original artificial scenes</span><strong>240</strong></div><div><span>No plume / plume</span><strong>192 / 48</strong></div><div><span>Held-out test per CNN</span><strong>64 no plume + 16 plume</strong></div><div><span>Mask cutoff</span><strong>0.50, fixed before fitting</strong></div><div><span>Reference scoring grid</span><strong>16 × 16</strong></div></div></div>
<div class="jump"><a class="button" href="#examples">Inspect predictions</a><a class="button" href="#forward">Step through the network</a><a class="button" id="viewer-link" href="__VIEWER_URL__">Open in local experiment viewer</a></div>
<section id="results"><div class="eyebrow">01 / Held-out evidence</div><h2>Does the model learn the artificial signal?</h2><p>Overlap is intersection over union (IoU) on the native reference grid. Precision measures how many predicted plume cells are correct. Recall measures how many plume cells are found. No-plume scenes remain in every score.</p><div class="table-wrap"><table class="comparison"><thead><tr><th>Network</th><th>Before: overlap</th><th>After: overlap</th><th>After: precision</th><th>After: recall</th><th>No-plume scenes with any false alarm</th></tr></thead><tbody>__METRICS__</tbody></table></div><p class="caption">The native backend uses the actual saved initial weights and the same fitted normalization for its control. For CNN 3, both feature branches are already trained and frozen; the combiner is random. CNN 2 and CNN 3 use the same 80 test scenes. CNN 1 has a separate 80-scene test set. These are 240 model evaluations on 160 unique artificial test scenes. An always-negative control has zero plume recall and zero aggregate IoU; an always-positive control is also included in the audit.</p><p><strong>__VERDICT__</strong></p><div class="toolbar"><span class="muted">Inspect network</span><div class="tabs" aria-label="Network selection"><button type="button" data-network="airborne" aria-pressed="false">CNN 1</button><button type="button" data-network="emit" aria-pressed="false">CNN 2</button><button type="button" data-network="fusion" aria-pressed="true">CNN 3</button></div></div><div class="learning"><div><img id="loss-curve" class="curve" alt="Fitting and validation loss across epochs"></div><div><div class="proof"><strong id="loss-change"></strong><span>Validation loss, random to saved checkpoint</span></div><div class="proof"><strong id="epoch-count"></strong><span id="duration"></span></div><div class="proof"><strong id="weight-change"></strong><span>Saved parameters differ from the seeded random model</span></div><div class="proof"><strong id="replay-check"></strong><span>Maximum saved-output replay difference</span></div></div></div></section>
<section id="examples"><div class="eyebrow">02 / Inspect predictions</div><h2>Before and after, on unseen examples</h2><p>Each example was kept out of fitting. Selection uses the worst, median and best positive overlap, plus the worst and median negative false-alarm count. The full table below contains every test result.</p><div class="toolbar"><label for="case-select">Test example</label><select id="case-select"></select><strong id="example-network"></strong></div><p id="scene-id" class="scene-id"></p><div id="selection-images" class="gallery"></div><div class="colorbar"></div><div class="colorbar-label"><span>Model score 0</span><span>1</span></div><div class="legend"><span><i style="--c:#148273"></i>Correct plume cell</span><span><i style="--c:#e65724"></i>False alarm</span><span><i style="--c:#635eb5"></i>Missed plume cell</span><span><i style="--c:#c7d4d9"></i>Unknown, not scored</span></div><p id="example-caption"></p><p id="negative-count" class="caption"></p><p class="caption">The fine-grid overlay is a display, not the metric grid. A fine prediction can activate part of a no-plume reference cell and still average below 0.50. Reported false-alarm counts use the 16 × 16 reference grid. RGB has a display stretch; score maps all use a fixed 0 to 1 scale.</p></section>
<section id="training"><div class="eyebrow">03 / Data entering each CNN</div><h2>Inspect an actual fitting example</h2><p>CNN 1 and CNN 2 receive the five-band satellite tensor. Their intended future targets are measured airborne and EMIT labels. CNN 3 receives their spatial features. In this run, both label sources are artificial. Switch networks above to inspect all three.</p><div class="toolbar"><label for="training-select">Fitting example</label><select id="training-select"><option value="0">Plume</option><option value="1">No plume</option></select><strong id="training-network"></strong></div><p class="scene-id" id="training-id"></p><div id="training-images" class="gallery"></div><p class="caption">These examples show the data contract and fitted output. Use the held-out examples above to assess generalization.</p><details><summary>Fitting and validation split</summary><div class="table-wrap"><table><thead><tr><th>Stage</th><th>Fitting originals</th><th>Validation originals</th><th>Test originals</th></tr></thead><tbody><tr><th>CNN 1</th><td>4 plume + 16 no plume</td><td>2 plume + 8 no plume</td><td>16 plume + 64 no plume</td></tr><tr><th>CNN 2</th><td>4 plume + 16 no plume</td><td>2 plume + 8 no plume</td><td>16 plume + 64 no plume</td></tr><tr><th>CNN 3</th><td>4 plume + 16 no plume, separate fusion set</td><td>Same as CNN 2</td><td>Same as CNN 2</td></tr></tbody></table></div><p>Split before augmentation. Apply eight right-angle rotation and mirror variants to fitting examples only. There are 60 unique fitting originals and 480 fitting variants across all stages. Variants are not independent observations. Validation selects the saved checkpoint. Test results do not set the cutoff or training settings.</p></details></section>
<section id="forward"><div class="eyebrow">04 / One actual forward pass</div><h2>Follow the data through all three CNNs</h2><p class="scene-id">Held-out scene: __FORWARD_ID__. Selected by median CNN 3 test-plume overlap.</p><div class="flow"><svg viewBox="0 0 1070 255" role="img" aria-label="Satellite input goes through two normalized branches, then their frozen feature grids enter a trainable third CNN. The reference is only used after prediction."><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" style="fill:#6a9396;stroke:none;marker-end:none"/></marker></defs><g id="node-source"><rect x="5" y="83" width="150" height="72" rx="6"/><text x="22" y="113">Five-band image</text><text x="22" y="136" class="sub">48 × 48 × 5</text></g><g id="node-normalize"><rect x="194" y="83" width="145" height="72" rx="6"/><text x="210" y="113">Normalize × 2</text><text x="210" y="136" class="sub">Saved fitting values</text></g><g id="node-cnn1"><rect x="382" y="18" width="150" height="78" rx="6"/><text x="400" y="48">CNN 1</text><text x="400" y="70" class="sub">Frozen spatial features</text><text x="400" y="87" class="sub">48 × 48 × 4</text></g><g id="node-cnn2"><rect x="382" y="151" width="150" height="78" rx="6"/><text x="400" y="181">CNN 2</text><text x="400" y="203" class="sub">Frozen spatial features</text><text x="400" y="220" class="sub">48 × 48 × 4</text></g><g id="node-join"><rect x="576" y="83" width="132" height="72" rx="6"/><text x="592" y="113">Project + join</text><text x="592" y="136" class="sub">48 × 48 × 8</text></g><g id="node-cnn3"><rect x="750" y="83" width="125" height="72" rx="6"/><text x="768" y="113">CNN 3</text><text x="768" y="136" class="sub">Trainable fusion</text></g><g id="node-output"><rect x="918" y="83" width="145" height="72" rx="6"/><text x="935" y="113">Plume score</text><text x="935" y="136" class="sub">48 × 48 → 16 × 16</text></g><path d="M155 119H194"/><path d="M339 105L382 60"/><path d="M339 136L382 185"/><path d="M532 60L576 103"/><path d="M532 185L576 139"/><path d="M708 119H750"/><path d="M875 119H918"/><text x="918" y="201" class="sub">Reference → compare only</text><text x="918" y="220" class="sub">No target enters a CNN</text></svg></div><div class="step-layout"><div id="step-buttons" class="step-buttons" aria-label="Forward-pass steps"></div><div><div class="eyebrow" id="step-progress"></div><h3 id="step-title"></h3><div id="step-shape"></div><p id="step-text"></p><div id="step-images" class="gallery"></div><div class="step-controls"><button id="prev-step" type="button">Previous</button><button id="next-step" type="button">Next step</button></div></div></div><p class="caption">Feature maps show actual saved-model activations. Their units are arbitrary. The full-scene score and the tiled export differ by at most __DIRECT_DIFF__ on valid input pixels. This difference is separate from saved-checkpoint replay.</p><details><summary>What this implementation does and does not test</summary><p>The branches are small U-Nets with three internal spatial scales. They are not four independent 16, 32, 64 and 128 pixel context models. CNN 3 fuses spatial features from the two separately trained branches. There is no Gaussian source head, wind model, emission-rate estimate or CAFO detector in this run.</p><p>A 48 × 48 score map is fitted against 16 × 16 supervision by averaging 3 × 3 cells. Many fine maps can fit the same coarse reference. This is not proof of physical super-resolution. The artificial data adds a strong spectral cue with shared noise distributions in plume and no-plume cases. It is easier than real Sentinel methane inference.</p></details></section>
<section id="evidence"><div class="eyebrow">05 / Repeatable evidence</div><h2>Keep the evidence with the result</h2><p>All model inputs, checkpoints, predictions and hashes stay in the local run folder. This HTML embeds its images and numerical audit. You can copy and open it without this server. The separate viewer link requires a running local viewer and its data.</p><div class="jump"><button type="button" class="button" id="download-audit">Download complete numerical audit</button><button type="button" class="button" id="download-html">Save this HTML</button></div><p class="caption">Manifest SHA-256: <code>__MANIFEST_HASH__</code></p><details><summary>All 240 held-out model evaluations</summary><p class="caption">A no-plume scene with no predicted plume has no defined IoU because the union is empty. This is reported as “Not defined”, not 100% overlap. Its false-positive count remains available.</p><div class="all-records"><table><thead><tr><th>CNN</th><th>Scene</th><th>Reference</th><th>Before IoU</th><th>After IoU</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr></thead><tbody>__ROWS__</tbody></table></div></details><details><summary>Recorded protocol and exact settings</summary><pre>__PROTOCOL__</pre></details><details><summary>Training runtime</summary><p>Backend: __BACKEND__. Native Windows PyTorch was used after the local WSL TensorFlow runtime repeatedly stopped before it saved a checkpoint. The layer design is equivalent at the architecture level. Its initialization and numerical implementation differ from TensorFlow. Do not treat this run as a direct backend benchmark. Earlier native attempts failed numerical replay and were rejected. This fresh run uses float64 computation, one CPU thread and deterministic algorithms. Its original prediction files and receipts are retained for replay.</p><pre>__RUNTIME_JSON__</pre></details><details><summary>Repeat the local workflow</summary><pre>python tools/GTM_Model6_negative_rich_fixture.py --output &lt;new-data-folder&gt; --seed 20261023

# Run with the installed native Windows PyTorch environment.
python tools/GTM_Model6_three_cnn_torch.py pipeline --manifest &lt;new-data-folder&gt;/manifest.json --run-dir &lt;new-run-folder&gt; --audit-dir &lt;new-audit-folder&gt;
python tools/GTM_Model6_three_cnn_torch.py replay --manifest &lt;new-data-folder&gt;/manifest.json --run-dir &lt;new-run-folder&gt; --audit-dir &lt;second-audit-folder&gt;

# The training CLI has fixed settings recorded in the protocol.
# Use the existing Windows environment with NumPy, Pillow and Matplotlib.
python tools/GTM_Model6_three_cnn.py export --manifest &lt;new-data-folder&gt;/manifest.json --run-dir &lt;new-run-folder&gt; --stage all --export-dir &lt;new-exports-folder&gt; --publish
python tools/GTM_Model6_build_learning_walkthrough.py --audit &lt;new-audit-folder&gt; --protocol &lt;protocol-file&gt; --output &lt;new-html-folder&gt;
python tools/GTM_Model0_local_viewer.py --port 8766 --name Research-PC</pre></details><div class="notice"><strong>What is still needed for a methane result</strong><p>Verified, aligned Sentinel and airborne/EMIT observations, including reviewed no-plume scenes, matching dates, resolved radiometry and independent sites. The old EMIT cache has no verified negative-scene cohort. Unknown or unobserved pixels cannot supply negative labels.</p></div></section>
<footer>Local engineering experiment · No new imagery downloads · Native-grid metrics at a fixed 0.50 cutoff · Real methane accuracy and useful upscaling remain unverified.</footer></main>
<script id="review-data" type="application/json">__DATA__</script><script id="full-audit" type="application/json">__AUDIT__</script>
<script>
const review=JSON.parse(document.getElementById('review-data').textContent);let network='fusion',step=0;
const el=id=>document.getElementById(id);
function showCase(){const s=review.stages[network],v=s.examples[Number(el('case-select').value)];el('scene-id').textContent=v.id;el('selection-images').innerHTML=v.images;el('example-caption').textContent=v.caption;el('negative-count').textContent=s.fp;el('example-network').textContent=s.number+' · '+s.name;}
function showTraining(){const s=review.stages[network],v=s.fitting[Number(el('training-select').value)];el('training-network').textContent=s.number+' · '+s.name;el('training-id').textContent=v.id;el('training-images').innerHTML=v.images;}
function chooseNetwork(value){network=value;const s=review.stages[network];document.querySelectorAll('[data-network]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.network===network)));el('case-select').replaceChildren(...s.examples.map((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.title;return o}));el('case-select').value='1';el('loss-curve').src=s.curve;el('loss-change').textContent=s.loss;el('epoch-count').textContent=s.epochs;el('duration').textContent=s.duration;el('weight-change').textContent=s.changed?'Parameters changed':'No parameter change';el('replay-check').textContent=String(s.replay);showCase();showTraining();}
function showStep(index){step=Math.max(0,Math.min(review.steps.length-1,index));const s=review.steps[step];el('step-progress').textContent='Step '+(step+1)+' of '+review.steps.length;el('step-title').textContent=s.title;el('step-shape').textContent=s.shape;el('step-text').textContent=s.text;el('step-images').innerHTML=s.images;document.querySelectorAll('[data-step]').forEach(b=>b.setAttribute('aria-current',Number(b.dataset.step)===step?'step':'false'));document.querySelectorAll('.flow g[id^="node-"]').forEach(n=>n.classList.toggle('active',n.id==='node-'+s.active));el('prev-step').disabled=step===0;el('next-step').disabled=step===review.steps.length-1;}
document.querySelectorAll('[data-network]').forEach(b=>b.addEventListener('click',()=>chooseNetwork(b.dataset.network)));el('case-select').addEventListener('change',showCase);el('training-select').addEventListener('change',showTraining);
el('step-buttons').replaceChildren();review.steps.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.dataset.step=i;b.textContent=s.title;b.addEventListener('click',()=>showStep(i));el('step-buttons').append(b)});el('prev-step').addEventListener('click',()=>showStep(step-1));el('next-step').addEventListener('click',()=>showStep(step+1));
function save(content,type,name){const u=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
el('download-audit').addEventListener('click',()=>save(el('full-audit').textContent,'application/json','GTM_Model6_learning_audit_20261023.json'));el('download-html').addEventListener('click',()=>save('<!doctype html>\n'+document.documentElement.outerHTML,'text/html','GTM_Model6_learning_review_20261023.html'));
chooseNetwork('fusion');showStep(0);
</script></body></html>'''


if __name__ == "__main__":
    main()
