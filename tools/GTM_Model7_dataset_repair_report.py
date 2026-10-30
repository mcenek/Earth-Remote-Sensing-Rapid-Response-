"""Build an evidence-linked overview of the completed real-data repair runs."""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/GTM_Model7_dataset_repair_20261024'
NAMES = {'stable_residual': 'Residual CNN', 'siamese': 'Temporal U-Net'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(x):
    return 'n/a' if x is None else f'{100*x:.3f}%'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    evidence, dev_rows, data_rows, links = {}, [], [], []
    for title, folder in [('MARS, matched clear references', 'mars_clear_data'),
                          ('MethaneS2CM, repaired references', 's2cm_reference_repair_data')]:
        path = BASE / folder / 'manifest.json'
        m = read(path)
        assert m['synthetic'] is False
        evidence[str(path.relative_to(ROOT))] = digest(path)
        for split in ('train', 'dev', 'test'):
            scenes = [s for s in m['scenes'] if s['split'] == split]
            kinds = collections.Counter(s['scene_kind'] for s in scenes)
            data_rows.append(f'<tr><td>{title}</td><td>{split.upper()}</td><td>{kinds["PLUME"]:,}</td><td>{kinds["NO_PLUME"]:,}</td><td>{len({s["physical_location_id"] for s in scenes}):,}</td></tr>')
    variants = [('MARS raw DN', 'mars_calibration_raw_retry1'),
                ('MARS log DN', 'mars_calibration_log_retry1'),
                ('S2CM weighted loss', 's2cm_calibration_retry1'),
                ('S2CM balanced loss', 's2cm_calibration_balanced_v2')]
    for title, folder in variants:
        f = read(BASE / folder / 'freeze.json')
        v = read(BASE / folder / 'verification.json')
        assert v['passed'] and v['test_array_access'] == 'none'
        evidence[str((BASE/folder/'freeze.json').relative_to(ROOT))] = digest(BASE/folder/'freeze.json')
        for model, choice in f['models'].items():
            d = choice['development']
            r = d['pooled']
            dev_rows.append(f'<tr><td>{title}</td><td>{NAMES[model]}</td><td>{choice["steps"]}</td><td>{choice["threshold"]:.3f}</td><td>{percent(r["iou"])}</td><td>{percent(r["precision"])}</td><td>{percent(r["recall"])}</td><td>{percent(d["negative_pixel_fpr"])}</td><td>{d["activated_negative_scenes"]}/{d["negative_scenes"]}</td><td>{"Pass, experimental" if d["qualified"] else "FAIL"}</td></tr>')
    reviews = [('MARS development, 86 scenes', 'mars_log_calibrated_dev_review'),
               ('MARS geographic evaluation, 82 scenes', 'mars_log_calibrated_test_review_v2'),
               ('S2CM weighted development, 1,982 scenes', 's2cm_calibrated_dev_review'),
               ('S2CM balanced development, 1,982 scenes', 's2cm_balanced_calibrated_dev_review')]
    report_bases = {}
    for title, folder in reviews:
        path = BASE / folder / 'verification.json'
        v = read(path)
        assert v['passed'] and v['counts_recomputed']
        evidence[str(path.relative_to(ROOT))] = digest(path)
        pub = v['publication']['stable_residual']
        base = f'/bundle-assets/{pub["experimentId"]}/_versions/{pub["version"]}/'
        assert (ROOT/'outputs/GTM_viewer_bundles'/pub['experimentId']/'_versions'/pub['version']/'index.html').is_file()
        report_bases[folder] = base
        models = ''.join(f'<a href="/?catalog=all&experiment={v["publication"][m]["experimentId"]}#compare">{NAMES[m]} in map viewer</a>' for m in NAMES)
        links.append(f'<article><h3>{title}</h3><a href="{base}index.html">Open all-case comparison</a>{models}</article>')
    final = read(BASE / 'mars_log_calibrated_test_review_v2/comparison.json')
    final_rows = []
    for model in NAMES:
        r, n = final['pooled'][model], final['negatives'][model]
        final_rows.append(f'<tr><td>{NAMES[model]}</td><td>{percent(r["iou"])}</td><td>{percent(r["precision"])}</td><td>{percent(r["recall"])}</td><td>{percent(n["pixel_fpr"])}</td><td>{n["activated"]}/{n["scenes"]}</td><td>FAIL</td></tr>')
    sample_rows, sample_assets = [], []
    for title, folder in [('Partial recovery on a development plume', 'mars_log_calibrated_dev_review'),
                          ('Missed plume at the reserved Mississippi location', 'mars_log_calibrated_test_review_v2')]:
        tiles = []
        for file, label in [('current','Real target'), ('background','Real reference'), ('annotation','Released annotation'), ('overlay','Residual prediction'), ('errors','Residual errors')]:
            source = BASE/folder/'stable_residual/scene_0001'/f'{file}.png'
            assert source.is_file()
            filename = f'{folder}_{file}.png'
            sample_assets.append((source, filename))
            evidence[str(source.relative_to(ROOT))] = digest(source)
            image_url = 'data:image/png;base64,' + base64.b64encode(source.read_bytes()).decode('ascii')
            tiles.append(f'<figure><img src="{image_url}" alt="{label}"><figcaption>{label}</figcaption></figure>')
        tiles = ''.join(tiles)
        sample_rows.append(f'<h3>{title}</h3><div class="samples">{tiles}</div>')
    body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Real-data repair and model comparison</title><style>
body{margin:0;background:#0e1b22;color:#dce9ed;font:16px/1.6 system-ui}main{max-width:1400px;margin:auto;padding:48px 28px}h1{font-size:42px;line-height:1.15;max-width:900px}h2{margin-top:42px}a{color:#70d5d0}article{padding:20px;background:#172b35;border:1px solid #344e5c;border-radius:10px}article a{display:block;margin:10px 0}.links{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.warning{padding:22px;border-left:4px solid #e5a171;background:#28303a}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:12px;border-bottom:1px solid #344e5c;white-space:nowrap}th{color:#95b3c3}.samples{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}figure{margin:0}img{width:100%;image-rendering:pixelated}figcaption{font-size:13px;color:#9bb8c8}code,pre{background:#172b35;padding:6px;border-radius:4px;overflow:auto;font-size:12px}.small{font-size:13px;color:#9bb8c8}@media(max-width:750px){.links{grid-template-columns:1fr}.samples{grid-template-columns:repeat(2,1fr)}h1{font-size:32px}}</style><main><p class="small">ERSRR / OCTOBER 24, 2026 / CACHED REAL IMAGERY ONLY</p><h1>The dataset is repaired. Geographic transfer still fails.</h1><div class="warning">Neither architecture is ready for statewide methane detection. One MARS development result improved, but both models missed the reserved-state plumes. The broader collection did not pass the development checks. No model was promoted.</div>
<h2>What changed</h2><ul><li>Built two separate real-data cohorts. Kept Sentinel L1C and L2A products separate.</li><li>Matched MARS to earlier reviewed no-plume dates at the same site. Checked both cloud masks and grids. Removed two snow-like negative scenes using a frozen image-only rule.</li><li>Changed 112 unusually bright S2CM references to the available alternate reference. Exact dates and cloud masks are still unavailable for that collection.</li><li>Retrained both architectures. Compared raw and log-scaled MARS inputs, plus weighted and balanced pixel losses on S2CM. Rotations and mirrors used training scenes only.</li><li>Saved actual scores, masks, errors and training curves. Rebuilt every displayed case from its original source and checked fresh inference replay.</li></ul>
<h2>Data actually used</h2><div class="table"><table><thead><tr><th>Collection</th><th>Partition</th><th>Plume observations</th><th>No-plume observations</th><th>Unique sites</th></tr></thead><tbody>__DATA__</tbody></table></div><p class="small">Observations are not independent sites. MARS contains 565 scenes across 13 states, but 96 of its 97 training plume observations are in Texas and one is in Oklahoma. Its six development positives are in Texas and New Mexico. S2CM contains 10,195 observations at 1,292 sites across 22 states; the training split covers 20 states. Its final split is mostly California and remains unscored here because development failed.</p>
<h2>Development comparisons</h2><p>Each row uses the fixed development-only checkpoint and cutoff search. All plume and no-plume pixels count toward pooled overlap and precision. These are selection results, not independent validation.</p><p class="small">The two MARS variants share the same draw schedule; input scaling changes. The S2CM balanced variant changes the references, pixel loss and training duration together. It is a recovery experiment, so its change cannot be attributed to the loss alone. Both S2CM variants fail the fixed precision and false-alarm checks.</p><div class="table"><table><thead><tr><th>Input / loss</th><th>Model</th><th>Updates</th><th>Cutoff</th><th>IoU</th><th>Precision</th><th>Recall</th><th>Clear pixel false alarms</th><th>Clear scenes activated</th><th>DEV gate</th></tr></thead><tbody>__DEV__</tbody></table></div>
<h2>Frozen MARS geographic evaluation</h2><p>Two plume observations, in Kansas and Mississippi, plus 80 no-plume observations. These locations were excluded from the new fits, but were used in earlier project work. This is regression evidence, not a fresh national benchmark. No settings were changed after viewing these results.</p><div class="table"><table><thead><tr><th>Model</th><th>IoU</th><th>Precision</th><th>Recall</th><th>Clear pixel false alarms</th><th>Clear scenes activated</th><th>Result</th></tr></thead><tbody>__FINAL__</tbody></table></div>
<h2>Inspect the actual predictions</h2><div class="links">__LINKS__</div>__SAMPLES__<p class="small">Orange is prediction. Cyan is annotation. Error colors: teal correct overlap, pink extra prediction, purple missed plume. Gray is unscored. A blank predicted mask can be a real miss, not a display error.</p>
<h2>What the checks establish</h2><p>Both models can learn small real training examples. Their completed fits and inference replays pass the engineering checks. That does not establish geographic transfer. The MARS fitting positives remain concentrated in one region. The larger collection shows substantial seasonal variation and does not have exact reference dates or cloud masks. No single cause has been proved, and no obvious preprocessing mismatch was found.</p><p>These runs predict binary plume masks. They do not demonstrate quantitative EMIT reconstruction, finer-resolution upscaling, emission rates, CAFO detection accuracy or source localization. The 20 m MARS predictions are repeated for comparison against native 10 m labels.</p><p>The other cached EMIT products contain positive plume support with unknown exterior labels. That exterior was not turned into clear training data. Controlled-release and CAFO facility labels also remain separate because they do not provide equivalent dense plume boundaries.</p>
<h2>Next evidence needed</h2><ol><li>Obtain or restore acquisition dates, cloud masks and plume-label provenance for the larger cached collection.</li><li>Build a small reviewed positive and negative development set across different landscapes. Add more known plume sites outside the current training region.</li><li>Keep a genuinely new geographic evaluation sealed. Select settings only on development locations.</li><li>Require visible plume recovery and low false alarms before a limited state pilot. Do not launch a nationwide scan from these checkpoints.</li></ol><p>No new imagery was downloaded for these runs. The dataset repairs, failed runs and reserved predictions are preserved.</p><h2>Reproduce the artifacts</h2><pre>__COMMANDS__</pre><p class="small">Run from the repository root. Use a new output directory for every inference, calibration or review export. The evidence.json file beside this page records the input artifact hashes.</p></main></html>'''
    commands = '\n'.join([
        '.venv/Scripts/python.exe tools/GTM_Model7_repaired_mars_log_v3_e2e.py --branch mars_repaired --manifest outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json --run outputs/GTM_Model7_dataset_repair_20261024/mars_clear_training_log_v3',
        '.venv/Scripts/python.exe tools/GTM_Model7_repaired_balanced_v2_e2e.py --branch methanes2cm --manifest outputs/GTM_Model7_dataset_repair_20261024/s2cm_reference_repair_data/manifest.json --run outputs/GTM_Model7_dataset_repair_20261024/s2cm_training_balanced_v2',
        '.venv/Scripts/python.exe tools/GTM_Model7_repaired_inference.py infer --freeze outputs/GTM_Model7_dataset_repair_20261024/mars_calibration_log_retry1/freeze.json --split test --output outputs/GTM_Model7_dataset_repair_20261024/mars_log_calibrated_test --verify',
        '.venv/Scripts/python.exe tools/GTM_Model7_dataset_repair_report.py --output outputs/GTM_viewer_bundles/GTM_Model7_dataset_repair_summary_rerun'])
    for key, value in {'DATA': ''.join(data_rows), 'DEV': ''.join(dev_rows), 'FINAL': ''.join(final_rows), 'LINKS': ''.join(links), 'SAMPLES': ''.join(sample_rows), 'COMMANDS': html.escape(commands)}.items():
        body = body.replace('__'+key+'__', value)
    out.mkdir(parents=True)
    (out/'assets').mkdir()
    for source, filename in sample_assets:
        shutil.copyfile(source, out/'assets'/filename)
        assert digest(source) == digest(out/'assets'/filename)
    (out/'index.html').write_text(body, encoding='utf-8')
    (out/'evidence.json').write_text(json.dumps(dict(source_sha256=digest(Path(__file__)), artifacts=evidence, national_readiness=False, zero_new_downloads=True), indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(out), evidence_artifacts=len(evidence), national_readiness=False)))


if __name__ == '__main__':
    main()
