"""Render actual new TRAIN inputs and reference checks. No inference or test images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from GTM_Model7_review_repaired import source_rebuild

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def country(scene):
    return scene.get('country') or scene.get('source_row', {}).get('country') or scene.get('official_csv_row', {}).get('country', 'Unknown')


def rgb_pair(data, factor):
    """Pooled per-band DN mapping shared by both dates; display only."""
    valid = data['valid'].astype(bool)
    pooled = np.concatenate([data[key][..., (2, 1, 0)][valid]
                             for key in ('target_image', 'reference_image')])
    lo, hi = np.percentile(pooled, (2, 98), axis=0)
    center = (lo + hi) / 2
    width = np.maximum(hi - lo, 1000.)
    lo, hi = center - width / 2, center + width / 2
    images = []
    for key in ('target_image', 'reference_image'):
        rgb = np.clip((data[key][..., (2, 1, 0)] - lo) / (hi - lo), 0, 1)
        rgb[~valid] = .22
        images.append(np.repeat(np.repeat((rgb*255).astype('uint8'), factor, 0), factor, 1))
    return images


def pair_shift(data):
    """Image-only edge difference diagnostic, not a validated registration estimate."""
    images = []
    for key in ('target_image', 'reference_image'):
        x = np.log1p(data[key][..., 3].astype(np.float64))
        dy, dx = np.gradient(x)
        edges = np.hypot(dx, dy)
        center = np.median(edges[data['valid']])
        scale = max(float(np.median(np.abs(edges[data['valid']]-center))), .0001)
        images.append(np.clip((edges-center)/scale, -8, 8))
    scores = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y0, y1 = max(0, dy), min(100, 100+dy)
            x0, x1 = max(0, dx), min(100, 100+dx)
            support = data['valid'][y0:y1, x0:x1] & data['valid'][y0-dy:y1-dy, x0-dx:x1-dx]
            a, b = images[0][y0:y1, x0:x1], images[1][y0-dy:y1-dy, x0-dx:x1-dx]
            scores.append((float(np.median(np.abs(a[support]-b[support]))), abs(dy)+abs(dx), dy, dx))
    best = min(scores)
    zero = next(row[0] for row in scores if row[2:] == (0, 0))
    return dict(best_offset_pixels=[best[2], best[3]], zero_shift_error=zero,
                best_shift_error=best[0], improvement_fraction=(zero-best[0])/max(zero, 1e-8),
                interpretation='Image-change warning only; no automatic shift or label-based ranking.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--combined', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_hash, combined_hash = sha(args.manifest), sha(args.combined)
    manifest = json.loads(args.manifest.read_text())
    combined = json.loads(args.combined.read_text())
    scenes = manifest['scenes']
    if any(s['split'] != 'train' for s in scenes):
        raise ValueError('Input review accepts new training supplement only')
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 14)
    sites = defaultdict(list)
    diagnostics, images = [], {}
    for s in scenes:
        path = args.manifest.parent/s['array_file']
        if not path.resolve().is_relative_to(args.manifest.parent.resolve()) or sha(path) != s['array_sha256']:
            raise ValueError('Unsafe or changed prepared array')
        with np.load(path, allow_pickle=False) as z:
            data = {k:z[k] for k in z.files}
        source_rebuild(s, data)
        check = pair_shift(data)
        diagnostics.append(dict(id=s['id'], country=country(s), site=s['physical_location_id'],
                                scene_kind=s['scene_kind'], **check))
        sites[s['physical_location_id']].append(s)
        current, reference = rgb_pair(data, 2)
        label = current.copy()
        hit = (data['label_native'] == 1) & data['valid_native']
        label[hit] = (label[hit]*.35 + np.array((48, 224, 217))*.65).astype('uint8')
        images[s['id']] = (current, reference, label)
    selected = []
    for _, items in sorted(sites.items(), key=lambda item:(country(item[1][0]), item[0])):
        for kind in ('PLUME', 'NO_PLUME'):
            options = sorted((s for s in items if s['scene_kind'] == kind), key=lambda s:s['id'])
            if options:
                selected.append(options[0])
    cards, contacts = [], []
    positives = [s for s in selected if s['scene_kind'] == 'PLUME']
    for index, s in enumerate(selected):
        panel = Image.new('RGB', (660, 234), '#14262e')
        draw = ImageDraw.Draw(panel)
        for col, (array, label) in enumerate(zip(images[s['id']], ('Current Sentinel-2', 'Earlier clear reference', 'Released plume label'))):
            panel.paste(Image.fromarray(array), (col*220, 0))
            draw.text((col*220, 208), label, font=font, fill='#c8dbe1')
        image_name = f'input_{index:03d}.png'
        panel.save(args.output/image_name)
        gap = float(s['target_minus_reference_hours'])/24
        text = f"{country(s)} | {s['physical_location_id']} | {s['scene_kind']}"
        details = f"{s['satellite_time'][:10]} / reference {s['reference_time'][:10]} | {gap:.1f} days earlier | {s.get('clear_fraction',1):.1%} valid"
        cards.append(f'<article data-country="{html.escape(country(s),quote=True)}" data-kind="{s["scene_kind"]}"><h3>{html.escape(text)}</h3><p>{html.escape(details)}</p><img loading="lazy" src="{image_name}" alt="{html.escape(text,quote=True)}"><details><summary>Source identifiers</summary><pre>{html.escape(json.dumps({k:s.get(k) for k in ("id","target_scene_id","reference_scene_id","reference_observation_id","array_sha256")},indent=2))}</pre></details></article>')
    for page in range((len(positives)+5)//6):
        rows = positives[page*6:(page+1)*6]
        sheet = Image.new('RGB', (1100, len(rows)*235), '#101f27')
        draw = ImageDraw.Draw(sheet)
        for row, s in enumerate(rows):
            y = row*235
            draw.text((8,y+3), f"{country(s)} | {s['physical_location_id']} | {s['satellite_time'][:10]} | ref {s['target_minus_reference_hours']/24:.1f}d earlier",font=font,fill='#e5f2f4')
            for col, array in enumerate(images[s['id']]):
                sheet.paste(Image.fromarray(array), (8+col*366, y+28))
        name = f'positive_sites_{page:02d}.png'
        sheet.save(args.output/name)
        contacts.append(name)
    old_counts = Counter((s['split'],s['scene_kind']) for s in combined['scenes'])
    countries = Counter(country(s) for s in scenes if s['scene_kind']=='PLUME')
    proximity_path = ROOT/'outputs/GTM_Model7_nontexas_readiness_20261024/cafo_proximity/receipt.json'
    proximity = json.loads(proximity_path.read_text())
    options = ''.join(f'<option>{html.escape(c)}</option>' for c in sorted({country(s) for s in scenes}))
    report = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cached plume training inputs</title><style>body{background:#0d191f;color:#e1edf1;font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.2}h2{margin-top:40px}a{color:#74dad0}p,li{color:#bdd0d8}article{background:#14262e;border:1px solid #35515e;border-radius:10px;padding:20px;margin:18px 0}img{max-width:100%;image-rendering:pixelated}select{padding:10px;background:#1b333f;color:white;border:1px solid #5a7b8a;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.badge{color:#f3c885;font-weight:700}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #34505b}.sticky{position:sticky;top:0;background:#0d191ff5;padding:12px 0}</style>
    <p class="badge">REAL INPUTS · PREPARATION ONLY · NO NEW MODEL RESULT</p><h1>More plume sites, with observed earlier references.</h1>'''
    report += f'<p>{len(scenes)} new training observations at {len(sites)} sites. International official-training data adds geographic variety; it does not demonstrate performance in another US state. Existing US development and evaluation assignments are retained. These global source sites were used by earlier project models and are not project-untouched data.</p>'
    report += '<h2>Combined data ready for the next comparison</h2><table><tr><th>Split</th><th>Plume observations</th><th>No-plume observations</th></tr>'
    for split in ('train','dev','test'):
        report += f'<tr><td>{split.upper()}</td><td>{old_counts[(split,"PLUME")]}</td><td>{old_counts[(split,"NO_PLUME")]}</td></tr>'
    report += '</table><p>Reference choice uses source metadata and fixed cloud/snow checks. If the nearest reference fails, the next eligible earlier observation is checked. All new arrays were reconstructed independently from their original TIFFs for this review. Shared-grid checks do not prove identical illumination or landmark alignment. Each RGB band uses a pooled two-date display range at least 1,000 DN wide, identical for both dates. These display-balanced images are not calibrated reflectance. Brightness and haze differences remain in some pairs; cloud-screened does not mean perfect reference.</p>'
    report += '<h2>CAFO resources</h2><p>Facility locations can guide follow-up and spatial association checks. They are not methane observations. The cached Iowa inventory has '+str(proximity['unambiguous_facilities'])+' unambiguous facility IDs. None of the 1,498 current MARS/S2CM dataset-site entries lies within 5 km of those points under the provisional coordinate assumption. The 17 cached CAFOSat patches contain 15 distinct RGB arrays; eight have unresolved state-coordinate conflicts. No CAFO-derived methane labels were created.</p>'
    report += '<p><a href="/bundle-assets/GTM_Model7_dataset_repair_summary_v3/index.html">Previous model predictions and failures</a> · <a href="review_receipt.json">Preparation and visual-review evidence</a></p><h2>Inspect one positive and one clear observation per new site</h2><p>Cyan is the released annotation. No prediction is shown here because these inputs have not been used in a new fit.</p><div class="sticky"><label>Country <select id="country"><option value="all">All countries</option>'+options+'</select></label> <label>Class <select id="kind"><option value="all">Both classes</option><option value="PLUME">Plume</option><option value="NO_PLUME">No plume</option></select></label> <span id="count"></span></div>'
    report += ''.join(cards)+'''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!((country.value==='all'||a.dataset.country===country.value)&&(kind.value==='all'||a.dataset.kind===kind.value));if(!a.hidden)n++;});document.getElementById('count').textContent=n+' observations';}const country=document.getElementById('country'),kind=document.getElementById('kind');country.onchange=kind.onchange=filter;filter();</script></html>'''
    (args.output/'index.html').write_text(report, encoding='utf-8')
    if sha(args.manifest)!=manifest_hash or sha(args.combined)!=combined_hash:
        raise ValueError('Input manifest changed')
    receipt = dict(passed=True, manifest_sha256=manifest_hash, combined_sha256=combined_hash,
                   renderer_sha256=sha(Path(__file__)), source_arrays_rebuilt=len(scenes),
                   source_rebuild_function_sha256=sha(ROOT/'tools/GTM_Model7_review_repaired.py'),
                   new_observations=len(scenes), positive_sites=len(sites), positive_observations_by_country=dict(countries),
                   displayed_observations=[s['id'] for s in selected], positive_contact_pages=contacts,
                   image_only_alignment_checks=diagnostics, test_images_opened=0,
                   predictions_generated=0, downloads=0,
                   display_mapping='Pooled two-date RGB per-band 2–98 percentile, minimum width 1000 DN per band; same mapping both dates',
                   assets={p.name:sha(p) for p in args.output.glob('*.png')})
    write(args.output/'review_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('assets','image_only_alignment_checks','displayed_observations')}, indent=2))


if __name__=='__main__':
    main()
