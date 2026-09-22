"""Can the mapper fit four actual training patches? No held-fold evaluation."""
from __future__ import annotations

import json
import shutil
import time
import numpy as np
import torch
from rasterio.transform import Affine

from train_GTM_Model6_mapper_mars_pilot import (
    ROOT, MODEL6, SEED, HOLDOUT_FOLD, MultiScaleMapper, _metrics, _save_bundle,
    digest, inventory, multiscale_mapper_loss,
)
from GTM_Model6_mapper_data import transformed_parent, eligible_centers, extract_patch, sample_batch


def main():
    run_id = 'GTM_Model6_mapper_training_sanity_20260922'
    run = ROOT / 'outputs/GTM_Model6' / run_id
    if run.exists():
        raise FileExistsError(run)
    run.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    items, _ = inventory(128)
    train = [i for i in items if i['row']['dataset'] == 'MARS' and i['row']['partition'] != HOLDOUT_FOLD]
    selected, all_contexts, targets, supports = [], [], [], []
    for positive in (True, False):
        used = 0
        for item in train:
            if bool((item['target'] > 0).any()) != positive:
                continue
            view = transformed_parent(item, 0, False)
            centers = eligible_centers(view[1], view[2], view[3], positive=positive)
            if not len(centers):
                continue
            # Choose a spatially central boundary patch from training truth only.
            for center in sorted(centers.tolist(), key=lambda c: (c[0]-100)**2 + (c[1]-100)**2):
                contexts, y, v = extract_patch(view, center)
                fraction = float(y[v].mean())
                if not positive or .15 <= fraction <= .70:
                    break
            else:
                continue
            cy, cx = center
            row = dict(item['row'])
            row['affine'] = list(Affine(*row['affine']) * Affine.translation(cx-8, cy-8))[:6]
            row['shape'] = [16, 16]
            row['crop_top_left'] = [cy-8, cx-8]
            selected.append({'row': row, 'target': y[0].numpy(), 'valid': v[0].numpy(),
                             'rgb': item['rgb'][cy-8:cy+8, cx-8:cx+8]})
            all_contexts.append(contexts)
            targets.append(y)
            supports.append(v)
            used += 1
            if used == 2:
                break
        if used != 2:
            raise ValueError('insufficient observed training patches for sanity check')
    telemetry = {}
    rng = np.random.default_rng(SEED)
    for _ in range(32):
        sample_batch(train, rng, 2, telemetry=telemetry)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    contexts = {k:torch.stack([c[k] for c in all_contexts]).to(device) for k in all_contexts[0]}
    target, support = torch.stack(targets).to(device), torch.stack(supports).to(device)
    model = MultiScaleMapper(input_channels=16, base_channels=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    started, log = time.monotonic(), []
    for step in range(300):
        if time.monotonic() - started > 180:
            raise TimeoutError('three-minute training cap reached')
        output = model(contexts)
        loss = multiscale_mapper_loss(output, target, support, torch.zeros_like(target),
                                     torch.zeros_like(support), enhancement_weight=0, fusion_weight=0)['loss']
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == 299:
            log.append({'step':step+1, 'loss':float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        output = model(contexts)
        probabilities = output['mask_logits'].sigmoid().cpu().numpy()[:, 0]
        branch_iou = {}
        truth = target.bool()
        for name, out in output['branches'].items():
            pred = out['mask_logits'] >= 0
            branch_iou[name] = float((pred & truth & support).sum() / ((pred | truth) & support).sum().clamp_min(1))
    checkpoint = run / 'GTM_Model6_mapper_checkpoint.pt'
    torch.save({'state_dict':model.cpu().state_dict(), 'seed':SEED, 'training_only':True,
                'training_ids':[i['row']['id'] for i in selected], 'base_channels':8, 'input_channels':16}, checkpoint)
    metrics = []
    for index, (p, item) in enumerate(zip(probabilities, selected)):
        metrics.append(_metrics(p, item))
        np.savez_compressed(run / f'prediction_{index:04d}.npz', probability=p, target=item['target'], valid=item['valid'])
    tp, fp, fn = (sum(m[k] for m in metrics) for k in ('tp', 'fp', 'fn'))
    iou = tp / max(tp+fp+fn, 1)
    neg_activation = float(np.mean([m['predicted_fraction'] for m in metrics[2:]]))
    receipt = {
        'scope':'real training patches, deliberately memorized; no held-out accuracy',
        'seed':SEED, 'device':device, 'steps':300, 'learning_rate':3e-4,
        'branch_training':'independent losses, no fusion gradient', 'enhancement_trained':False,
        'training_only_iou':iou, 'training_negative_activation':neg_activation,
        'per_branch_training_iou':branch_iou,
        'gate_passed':iou >= .90 and neg_activation <= .01,
        'training_seconds':time.monotonic()-started, 'sampling_check':telemetry,
        'training_log':log, 'observations':[i['row'] for i in selected], 'metrics':metrics,
        'checkpoint_sha256':digest(checkpoint),
    }
    sources = [__file__, str(ROOT/'tools/train_GTM_Model6_mapper_mars_pilot.py'),
               str(MODEL6/'GTM_Model6_mapper.py'), str(MODEL6/'GTM_Model6_mapper_data.py')]
    source_dir = run / 'source'
    source_dir.mkdir()
    receipt['source_hashes'] = {}
    for filename in sources:
        copied = source_dir / __import__('pathlib').Path(filename).name
        shutil.copyfile(filename, copied)
        receipt['source_hashes'][copied.name] = digest(copied)
    report = '\n'.join([
        '# Real training-patch sanity check', '',
        '**TRAINING FIT ONLY. These exact four patches were optimized repeatedly.**', '',
        f"Training IoU: {iou:.3%}; negative activation: {neg_activation:.3%}.",
        f"Per-branch IoU: {branch_iou}.",
        f"Seed {SEED}; 300 steps; {device}; independent branch BCE+Dice; 3e-4 learning rate.", '',
        'This answers only whether the model and optimizer can learn observed training inputs. It does not measure generalization, EMIT methane intensity, upscaling, CAFO recovery or source localization.', '',
    ])
    bundle = _save_bundle(run, run_id, selected, list(probabilities), [i['valid'] for i in selected], .5,
                         report_text=report, stage='TRAINING FIT ONLY', training_only=True)
    receipt['viewer_bundle'] = str(bundle)
    (run/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    (run/'report.md').write_text(report)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('observations','training_log','metrics')}, indent=2))


if __name__ == '__main__':
    main()
