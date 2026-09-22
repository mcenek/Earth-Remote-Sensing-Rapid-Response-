"""Compare frozen dense masks with spatially constant crop baselines; no tuning."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MethaneS2CM/l2a_location_split_32x32/v5_1_location_test_predictions.npz'

def main():
    with np.load(CACHE, allow_pickle=False) as z:
        prob = z['ersrr_v5_1_probability']
        truth = z['truth'].astype(bool)
        valid = z['observable'].astype(bool)
        decisions = z['ersrr_v5_1_scene_decision'].astype(bool)
    totals = {k: np.zeros(4, dtype=np.int64) for k in [
        'cached_dense', 'constant_crop_mean', 'whole_crop_if_scene_positive', 'every_pixel_positive']}
    changed = 0
    for start in range(0, len(prob), 512):
        end = start + 512
        v, t, a = valid[start:end], truth[start:end], prob[start:end].astype(np.float32)
        dense = a >= 0.4
        mean = (a * v).sum(axis=(1, 2)) / np.maximum(v.sum(axis=(1, 2)), 1)
        flat = np.broadcast_to((mean >= 0.4)[:, None, None], a.shape)
        scene = np.broadcast_to(decisions[start:end, None, None], a.shape)
        changed += int(((dense != flat) & v).sum())
        for key, pred in [('cached_dense', dense), ('constant_crop_mean', flat),
                          ('whole_crop_if_scene_positive', scene), ('every_pixel_positive', np.ones_like(v))]:
            totals[key] += np.array([(pred & t & v).sum(), (pred & ~t & v).sum(),
                                     (~pred & t & v).sum(), (~pred & ~t & v).sum()])
    out = {'diagnostic_only': True, 'threshold': 0.4, 'observable_pixels': int(valid.sum()),
           'dense_vs_flat_changed_pixels': changed, 'dense_vs_flat_agreement': float(1 - changed / valid.sum()),
           'rules': {}}
    for key, (tp, fp, fn, tn) in totals.items():
        out['rules'][key] = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                            'iou': float(tp / max(tp + fp + fn, 1)),
                            'pixel_precision': float(tp / max(tp + fp, 1)),
                            'pixel_recall': float(tp / max(tp + fn, 1))}
    (ROOT / 'reports/research/GTM_Model51_constant_crop_diagnostic.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
