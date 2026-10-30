"""Describe actual native mask changes without selecting scenes or expanding masks."""
import argparse
import collections
import json

import numpy as np

import GTM_Model7_native_transfer_v13 as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=study.Path,default=study.OUT/'review_recipe.json')
    parser.add_argument('--review',type=study.Path,default=study.OUT/'review')
    parser.add_argument('--output',type=study.Path,default=study.OUT/'mask_changes.json')
    args=parser.parse_args()
    out = study.OUT
    recipe_path = args.recipe
    recipe = json.loads(recipe_path.read_text())
    review = args.review
    verified = json.loads((review / 'verification.json').read_text())
    if not verified['passed'] or verified['recipe_sha256'] != study.sha(recipe_path):
        raise ValueError('Verified full-set render required')
    scenes = [r for r in study.manifest() if r['split'] == 'dev']
    counts = {k: collections.Counter() for k in ('candidate', 'ratio')}
    countries = {}
    rows = []
    for row in scenes:
        sid = row['id']
        cache = recipe['native_data'][sid]
        if study.sha(study.Path(cache['path'])) != cache['sha256']:
            raise ValueError('Native inputs changed')
        with np.load(cache['path'], allow_pickle=False) as z:
            truth, valid = z['y'].astype(bool), z['valid'].astype(bool)
        masks, metrics = {}, {}
        for key, method in recipe['methods'].items():
            path = study.Path(method['folder']) / (sid + '.npz')
            if study.sha(path) != method['records'][sid]['probability_sha256']:
                raise ValueError('Saved predictions changed')
            with np.load(path, allow_pickle=False) as z:
                masks[key] = (z['probability_native'] >= method['cutoff']) & valid
                if not np.array_equal(z['valid_native'].astype(bool), valid):
                    raise ValueError('Compared support differs')
                metrics[key] = study.native_confusion(z['probability_native'], truth, valid, method['cutoff'])
            if metrics[key] != method['records'][sid]['native_confusion']:
                raise ValueError('Compared native counts differ')
        country = row['source_row']['country']
        country_counts = countries.setdefault(country, {k: collections.Counter() for k in masks})
        for key in masks:
            country_counts[key].update(metrics[key])
        for key in counts:
            new, old = masks[key], masks['baseline']
            counts[key].update(tp_gained=int((new & ~old & truth).sum()),
                               tp_lost=int((~new & old & truth).sum()),
                               fp_added=int((new & ~old & ~truth).sum()),
                               fp_removed=int((~new & old & ~truth).sum()),
                               newly_marked=int((new & ~old).sum()),
                               removed_marked=int((~new & old).sum()))
        rows.append(dict(id=sid, country=country, kind=row['scene_kind'],
                         native_truth_pixels=int((truth & valid).sum()), models=metrics))
    for group in countries.values():
        for metric in group.values():
            metric['f1'] = 2 * metric['tp'] / max(2 * metric['tp'] + metric['fp'] + metric['fn'], 1)
    study.save_json(args.output, dict(passed=True, recipe_sha256=study.sha(recipe_path),
                    renderer_verification_sha256=study.sha(review / 'verification.json'),
                    role='Descriptive exposed DEV; no subset promotion or independent US accuracy',
                    changes_from_v11=counts, countries=countries, scenes=rows,
                    script_sha256=study.sha(study.Path(__file__))))
    print(json.dumps({'changes_from_v11': counts, 'countries': countries}))


if __name__ == '__main__':
    main()
