"""Measure the original r4 crop sampler on its frozen training partition only."""
import json
from pathlib import Path
import numpy as np

from train_GTM_Model6_mapper_mars_pilot import ROOT, SEED, HOLDOUT_FOLD, inventory


def main():
    items, _ = inventory(128)
    train = [i for i in items if i['row']['dataset'] == 'MARS' and i['row']['partition'] != HOLDOUT_FOLD]
    rng = np.random.default_rng(SEED)
    positive_draws = lost_positive_draws = positive_tiles = positive_pixels = 0
    draws = 800
    for _ in range(draws):
        item = train[int(rng.integers(len(train)))]
        candidates = np.argwhere((item['target'] > 0) & item['valid'])
        positive_draw = bool(len(candidates) and rng.random() < 0.7)
        if positive_draw:
            cy, cx = candidates[int(rng.integers(len(candidates)))]
            positive_draws += 1
        else:
            cy, cx = rng.integers(64, 137, size=2)
        cy, cx = int(np.clip(cy, 64, 136)), int(np.clip(cx, 64, 136))
        target = item['target'][cy-8:cy+8, cx-8:cx+8]
        valid = item['valid'][cy-8:cy+8, cx-8:cx+8]
        n = int(((target > 0) & valid).sum())
        positive_tiles += int(n > 0)
        positive_pixels += n
        lost_positive_draws += int(positive_draw and n == 0)
        rng.integers(4)  # Same augmentation RNG consumption as r4.
        rng.integers(2)
    report = {
        'scope': 'original r4 sampler reconstruction on training masks only; before augmentation',
        'seed': SEED, 'training_parents': len(train), 'draws': draws,
        'requested_positive_draws': positive_draws, 'requested_positive_draws_returning_empty_tile': lost_positive_draws,
        'positive_tiles': positive_tiles, 'positive_tile_fraction': positive_tiles / draws,
        'positive_pixel_fraction': positive_pixels / (draws * 256),
        'training_ids': [i['row']['id'] for i in train],
    }
    path = ROOT / 'reports/experiments/GTM_Model6_mapper_r4_sampling_audit.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'training_ids'}, indent=2))


if __name__ == '__main__':
    main()
