"""Describe size-stratified exposed DEV and matched TRAIN counts after replay.

The v10 protocol specifies these descriptive bins before the fit. They never
select checkpoints, thresholds, scene exclusions or training examples.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, counts):
    total = {k: 0 for k in ('tp', 'fp', 'fn', 'tn')}
    recalls = []
    for sid in rows:
        c = counts[sid]
        if any(not isinstance(c[k], int) or c[k] < 0 for k in total):
            raise ValueError('Malformed saved native counts')
        for k in total:
            total[k] += c[k]
        recalls.append(c['tp'] / max(c['tp'] + c['fn'], 1))
    return dict(scenes=len(rows), native=total, marked_native_pixels=total['tp']+total['fp'],
                f1=2*total['tp']/max(2*total['tp']+total['fp']+total['fn'],1),
                mean_recall=float(np.mean(recalls)) if rows else None,
                pooled_recall=total['tp']/max(total['tp']+total['fn'], 1),
                zero_tp=sum(counts[sid]['tp'] == 0 for sid in rows),
                recovered_ten_percent=sum(r >= .1 for r in recalls),
                recovered_half=sum(r >= .5 for r in recalls))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--measurement', type=Path, required=True)
    p.add_argument('--old-fit', type=Path, required=True)
    p.add_argument('--new-fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    measurement = json.loads(a.measurement.read_text())
    if any(s['split'] not in ('train', 'dev') for s in manifest['scenes']):
        raise ValueError('Only TRAIN/DEV data are allowed')
    methods = {k: measurement[k]['scene_native_confusion'] for k in
               ('reference', 'sampling_reference', 'selected', 'matched_v8_checkpoint_cutoff')}
    train_counts = {}
    identities = {str(a.manifest): sha(a.manifest), str(a.measurement): sha(a.measurement)}
    for key, run in (('v8_update2400_cutoff50', a.old_fit), ('v10_update2400_cutoff50', a.new_fit)):
        replay = json.loads((run/'e2e_receipt.json').read_text())
        receipt = json.loads((run/'receipt.json').read_text())
        if not replay['passed'] or replay['manifest_sha256'] != sha(a.manifest):
            raise ValueError('Completed matching replay is required')
        score = receipt['models']['siamese']['milestones']['2400']['train']
        train_counts[key] = {sid: row['native_confusion'] for sid, row in score['scenes'].items()}
        for name in ('receipt.json', 'e2e_receipt.json'):
            identities[str(run/name)] = sha(run/name)
    output = {}
    for split, models in (('train', train_counts), ('dev', methods)):
        rows = [s for s in manifest['scenes'] if s['split'] == split and s['scene_kind'] == 'PLUME']
        areas = {}
        for s in rows:
            file = a.manifest.parent/s['array_file']
            if sha(file) != s['array_sha256']:
                raise ValueError('Prepared data changed')
            with np.load(file, allow_pickle=False) as z:
                areas[s['id']] = int(((z['target'] == 1) & z['valid'].astype(bool)).sum())
        if len(rows) != (268 if split == 'train' else 46):
            raise ValueError('Positive split membership changed')
        output[split] = []
        for low, high in ((0, 100), (100, 250), (250, 1000), (1000, 10001)):
            members = [s['id'] for s in rows if low <= areas[s['id']] < high]
            output[split].append(dict(positive20_range=[low, high], ids=members,
                                     models={k: summarize(members, c) for k, c in models.items()}))
    dev = [s for s in manifest['scenes'] if s['split']=='dev']
    countries = {}
    for country in sorted({s['source_row']['country'] for s in dev}):
        members = [s['id'] for s in dev if s['source_row']['country']==country]
        countries[country] = {k:summarize(members,c) for k,c in methods.items()}
    a.output.write_text(json.dumps(dict(role='Descriptive TRAIN and exposed DEV only; not independent validation',
        passed=True, source_sha256=sha(Path(__file__)), sources=identities, breakdown=output,country_dev=countries,
        test_array_access='none'), indent=2))
    print(json.dumps({split: [{k: v for k, v in row.items() if k != 'ids'} for row in bins]
                      for split, bins in output.items()}))


if __name__ == '__main__':
    main()
