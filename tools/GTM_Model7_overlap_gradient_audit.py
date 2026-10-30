"""Read-only real TRAIN gradient check under the written October 25 protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import GTM_Model7_repaired_inference as factory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    meta=json.loads((a.run/'metadata.json').read_text())
    receipt=json.loads((a.run/'receipt.json').read_text())
    replay=json.loads((a.run/'e2e_receipt.json').read_text())
    factory.check_implementation(meta);t=factory.t;t.set_branch('mars_repaired')
    if meta['arm']!='site_balanced' or not replay['passed'] or receipt['metadata_sha256']!=t.sha(a.run/'metadata.json') or replay['trainer_sha256']!=meta['trainer_sha256']:
        raise ValueError('Completed frozen v8 replay required')
    manifest=Path(meta['manifest']);schedule=a.run/'draw_schedule.json'
    if t.sha(manifest)!=meta['manifest_sha256'] or t.sha(schedule)!=meta['draw_schedule_file_sha256']:
        raise ValueError('Frozen data/schedule drift')
    rows=json.loads(manifest.read_text())['scenes']
    by_id={s['id']:s for s in rows if s['split']=='train'}
    draws=json.loads(schedule.read_text())[:128]
    if len(draws)!=128 or any(d[0] not in by_id for d in draws):
        raise ValueError('Non-TRAIN diagnostic sample')
    checkpoint=a.run/'siamese/step_2400.pt'
    if t.sha(checkpoint)!=receipt['models']['siamese']['milestones']['2400']['checkpoint_sha256']:
        raise ValueError('Frozen weights drift')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['normalization']!=meta['normalization']:
        raise ValueError('Normalizer drift')
    torch.set_num_threads(1)
    model=t.new_model('siamese');model.load_state_dict(saved['model']);model.train()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    parameters=tuple(model.parameters());cache={};results=[]
    for i in range(16):
        draw=draws[i*8:(i+1)*8]
        if sum(bool(d[4]) for d in draw)!=4:
            raise ValueError('Class balance differs')
        x,y,valid,positive=t.make_batch(manifest,by_id,draw,meta['normalization'],cache)
        _,bce,dice=t.loss_parts(model(x),y,valid,positive)
        gb=torch.autograd.grad(bce,parameters,retain_graph=True)
        gd=torch.autograd.grad(dice,parameters)
        b=torch.cat([v.flatten() for v in gb]);d=torch.cat([v.flatten() for v in gd])
        bn=float(b.norm());dn=float(d.norm());cos=float(torch.dot(b,d)/(b.norm()*d.norm()).clamp_min(1e-20))
        result=dict(batch=i,bce=float(bce),dice=float(dice),bce_gradient_l2=bn,dice_gradient_l2=dn,
                    actual_weighted_dice_to_bce_gradient_ratio=.1*dn/max(bn,1e-20),gradient_cosine=cos)
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError('Nonfinite diagnostic')
        results.append(result)
    if any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()):
        raise ValueError('Read-only diagnostic changed weights')
    protocol=Path(__file__).resolve().parents[1]/'docs/GTM_Model7_overlap_gradient_audit_20261025.md'
    files=[Path(__file__),protocol,a.run/'metadata.json',a.run/'receipt.json',a.run/'e2e_receipt.json',checkpoint,manifest,schedule]
    result=dict(passed=True,sources={str(f.resolve()):t.sha(f) for f in files},batches=results,
        median_actual_weighted_dice_to_bce_gradient_ratio=float(np.median([r['actual_weighted_dice_to_bce_gradient_ratio'] for r in results])),
        median_gradient_cosine=float(np.median([r['gradient_cosine'] for r in results])),weights_unchanged=True,
        scene_ids=[d[0] for d in draws],role='TRAIN-only mechanism diagnostic; no new fit or transfer claim',test_array_access='none')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','batches','scene_ids']}))


if __name__=='__main__':
    main()
