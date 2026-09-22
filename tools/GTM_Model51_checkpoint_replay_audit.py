"""Bounded, post-hoc CPU replay of eight existing v5.1 viewer samples."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / 'Earth-Remote-Sensing-Rapid-Response-'
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'EarthRemoteSensingRapidResponse'))
from train_methanes2cm_v5 import PackedMethaneS2CMDataset
from methanes2cm_v5_model import MethaneS2CMV5Model

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def main():
    torch.set_num_threads(4)
    data = ASSETS / 'EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MethaneS2CM/l2a_location_split_32x32'
    sample_ids = [86360, 83106, 82448, 97322, 98104, 87344, 83149, 83072]
    with np.load(data / 'v5_1_location_test_predictions.npz', allow_pickle=False) as z:
        all_ids = z['sample_id']
        index = {int(s): i for i, s in enumerate(all_ids)}
        selected = np.array([index[s] for s in sample_ids])
        cached = z['ersrr_v5_1_probability'][selected].astype(np.float32)
        truth = z['truth'][selected].astype(bool)
        valid = z['observable'][selected].astype(bool)
        labels = z['label'][selected]
        groups = z['group_id'][selected]
        decisions = z['ersrr_v5_1_scene_decision'][selected]
    records = [dict(id=s, label=int(labels[i]), group_id=str(groups[i]), exact_location_id='not_used_for_replay') for i, s in enumerate(sample_ids)]
    dataset = PackedMethaneS2CMDataset(data / 'v5_location_test_packed.h5', records, augment=False, seed=0)
    batch = [dataset[i] for i in range(len(records))]
    inputs = torch.stack([b['inputs'] for b in batch])
    observable = torch.stack([b['observable'] for b in batch])
    assert np.array_equal(np.stack([b['mask'][0].numpy() > .5 for b in batch]), truth)
    assert np.array_equal(observable[:, 0].numpy() > .5, valid)
    ensemble = json.loads((ROOT / 'reports/experiments/methanes2cm_v5_1_ensemble_validation.json').read_text())
    probabilities = []
    identities = []
    with torch.inference_mode():
        for seed in ensemble['seeds']:
            path = ASSETS / seed['checkpoint']['path']
            sha = digest(path)
            assert sha == seed['checkpoint']['sha256'], 'checkpoint hash mismatch'
            payload = torch.load(path, map_location='cpu', weights_only=True)
            meta = payload['model_metadata']
            model = MethaneS2CMV5Model(**{k: float(meta[k]) for k in ['scene_topk_fraction', 'scene_max_weight', 'context_scene_weight']})
            model.load_state_dict(payload['state_dict'], strict=True)
            model.eval()
            probability = model(inputs, observable)['segmentation_logits'].sigmoid()[:, 0].numpy()
            probabilities.append(probability)
            identities.append(dict(seed=seed['seed'], path=str(path), sha256=sha, mean_spatial_std=float(probability.std(axis=(1, 2)).mean())))
    replay = np.mean(probabilities, axis=0)
    results = []
    for i, sid in enumerate(sample_ids):
        p, c, v = replay[i], cached[i], valid[i]
        results.append(dict(sample_id=sid, scene_outcome=('TP' if decisions[i] else 'FN') if labels[i] else ('FP' if decisions[i] else 'TN'), max_abs_error=float(np.abs(p-c)[v].max()), mean_abs_error=float(np.abs(p-c)[v].mean()), threshold_agreement=float(((p>=.4)==(c>=.4))[v].mean()), replay_spatial_std=float(p[v].std()), replay_positive_fraction=float((p[v]>=.4).mean()), reference_positive_fraction=float(truth[i][v].mean())))
    out = ROOT / 'reports/research'
    report = dict(diagnostic_only=True, no_training=True, checkpoint_hashes_match=True, packed_truth_and_validity_match=True, precision_note='CPU float32 replay compared with historical CUDA autocast inference cached as float16; numerical identity is not expected.', checkpoints=identities, samples=results)
    (out / 'GTM_Model51_checkpoint_replay_audit.json').write_text(json.dumps(report, indent=2))
    cell, pad, top = 192, 12, 44
    image = Image.new('RGB', (4*(cell+pad)+pad, len(sample_ids)*(cell+35)+top), 'white')
    draw = ImageDraw.Draw(image)
    for j, title in enumerate(['Sentinel RGB', 'Reference mask', 'Replayed probability 0..1', 'Replay mask at 0.40']):
        draw.text((pad+j*(cell+pad), 8), title, fill='black')
    for i, sample in enumerate(batch):
        rgb = inputs[i, [4,3,2]].numpy().transpose(1,2,0)
        lo, hi = np.percentile(rgb, [2,98], axis=(0,1))
        rgb = (np.clip((rgb-lo)/np.maximum(hi-lo,1e-6),0,1)*255).astype(np.uint8)
        prob = replay[i]
        rgbmask = np.stack([truth[i]*25, truth[i]*210, truth[i]*230],axis=-1).astype(np.uint8)
        heat = np.stack([prob*255, prob*150, prob*30],axis=-1).clip(0,255).astype(np.uint8)
        mask = np.stack([(prob>=.4)*255,(prob>=.4)*150,(prob>=.4)*30],axis=-1).astype(np.uint8)
        for j, arr in enumerate([rgb,rgbmask,heat,mask]):
            arr[~valid[i]] = [140,140,140]
            image.paste(Image.fromarray(arr).resize((cell,cell),Image.Resampling.NEAREST), (pad+j*(cell+pad),top+i*(cell+35)))
        draw.text((pad,top+i*(cell+35)+cell+5),f"Sample {sample_ids[i]} | scene {results[i]['scene_outcome']} | CPU replay",fill='black')
    image.save(out / 'GTM_Model51_checkpoint_replay_audit.png')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
