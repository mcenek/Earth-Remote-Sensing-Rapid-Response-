"""Select the single per-scene Dice fit on exposed DEV, never final scenes."""
import argparse
import copy
import json
from pathlib import Path

import GTM_Model7_repaired_inference as factory
import GTM_Model7_calibrate_broader_v7 as c


def rank(row):
    return (row['mean_positive_site_recall'],row['mean_positive_scene_recall'],
            row['pooled']['iou'],-row['clear_pixel_fpr'],-row['step'],row['cutoff'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline-freeze',type=Path,required=True)
    p.add_argument('--baseline-inference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sampling-freeze',type=Path,required=True)
    p.add_argument('--sampling-inference',type=Path,required=True)
    a=p.parse_args()
    meta=json.loads((a.run/'metadata.json').read_text())
    factory.check_implementation(meta)
    t=factory.t;t.set_branch('mars_repaired')
    if meta.get('arm')!='scene_dice':
        raise ValueError('Only the single declared v9 overlap-loss fit is allowed')
    replay=json.loads((a.run/'e2e_receipt.json').read_text())
    receipt=json.loads((a.run/'receipt.json').read_text())
    if (not replay['passed'] or replay['trainer_sha256']!=meta['trainer_sha256'] or
        replay['manifest_sha256']!=meta['manifest_sha256'] or
        receipt['metadata_sha256']!=t.sha(a.run/'metadata.json') or
        any(row['max_probability_error']>1e-6 or row['dev_scenes']!=198 for row in replay['rows'].values())):
        raise ValueError('Matching completed fresh CPU replay required')
    manifest_path=Path(meta['manifest']);manifest=json.loads(manifest_path.read_text())
    if t.sha(manifest_path)!=meta['manifest_sha256'] or any(s['split'] not in ('train','dev') for s in manifest['scenes']):
        raise ValueError('Pinned TRAIN/DEV manifest required')
    scenes=[s for s in manifest['scenes'] if s['split']=='dev']
    if len(scenes)!=198:
        raise ValueError('DEV membership drift')
    samples={s['id']:t.load_scene(manifest_path,s) for s in scenes}
    baseline=json.loads(a.baseline_freeze.read_text())
    baseline_receipt=json.loads((a.baseline_inference/'receipt.json').read_text())
    baseline_replay=json.loads((a.baseline_inference/'replay.json').read_text())
    if (baseline_receipt['freeze_sha256']!=t.sha(a.baseline_freeze) or
        baseline_receipt['manifest_sha256']!=meta['manifest_sha256'] or
        not baseline_replay['passed'] or baseline_replay['receipt_sha256']!=t.sha(a.baseline_inference/'receipt.json')):
        raise ValueError('Matching replayed v7 DEV predictions required')
    basechoice=baseline['models']['siamese']
    if basechoice['steps']!=1200 or basechoice['threshold']!=.7:
        raise ValueError('Comparison must retain frozen selected v7')
    baseprob=c.load_probabilities(a.baseline_inference/'siamese',scenes,samples,baseline_receipt['models']['siamese']['scenes'])
    reference=c.measure(baseprob,samples,scenes,.7,1200)
    sampling=json.loads(a.sampling_freeze.read_text())
    sampling_receipt=json.loads((a.sampling_inference/'receipt.json').read_text())
    sampling_replay=json.loads((a.sampling_inference/'replay.json').read_text())
    sc=sampling['models']['siamese']
    if (sc['steps']!=2400 or sc['threshold']!=.75 or
        sampling_receipt['freeze_sha256']!=t.sha(a.sampling_freeze) or
        sampling_receipt['manifest_sha256']!=meta['manifest_sha256'] or
        not sampling_replay['passed'] or sampling_replay['receipt_sha256']!=t.sha(a.sampling_inference/'receipt.json')):
        raise ValueError('Pinned replayed v8 comparator required')
    sampling_prob=c.load_probabilities(a.sampling_inference/'siamese',scenes,samples,sampling_receipt['models']['siamese']['scenes'])
    sampling_measure=c.measure(sampling_prob,samples,scenes,.75,2400)
    candidates=[]
    for step in (1200,2400,3600,4800):
        milestone=receipt['models']['siamese']['milestones'][str(step)]
        if t.sha(a.run/'siamese'/f'step_{step}.pt')!=milestone['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint drift')
        probabilities=c.load_probabilities(a.run/'siamese'/f'dev_step_{step}',scenes,samples,milestone['dev']['scenes'])
        candidates.extend(c.measure(probabilities,samples,scenes,cutoff,step) for cutoff in c.CUTOFFS)
    eligible=[row for row in candidates if row['qualified']]
    chosen=max(eligible,key=rank) if eligible else next(row for row in candidates if row['step']==1200 and row['cutoff']==.7)
    improved=(chosen['qualified'] and chosen['mean_positive_site_recall']>reference['mean_positive_site_recall'] and
              chosen['pooled']['iou']>=reference['pooled']['iou'] and chosen['clear_pixel_fpr']<=reference['clear_pixel_fpr'])
    a.output.mkdir(parents=True,exist_ok=False)
    step=chosen['step'];checkpoint=(a.run/'siamese'/f'step_{step}.pt').resolve()
    frozen=dict(schema='GTM_Model7_repaired_freeze_v1',selection_kind='scene_dice_exposed_DEV_v9',
        run=str(a.run.resolve()),metadata=copy.deepcopy(meta),training_receipt_sha256=t.sha(a.run/'receipt.json'),
        training_replay_sha256=t.sha(a.run/'e2e_receipt.json'),threshold=chosen['cutoff'],
        selection='Same22cutoffs/4milestones; exposed broader DEV only; strict replacement rule retained',
        test_predictions_used_for_selection=False,improved_development_gate=bool(improved),
        models={'siamese':dict(checkpoint=str(checkpoint),checkpoint_sha256=t.sha(checkpoint),steps=step,
            threshold=chosen['cutoff'],qualified_development_gate=chosen['qualified'],development=chosen)})
    t.write_json(a.output/'freeze.json',frozen)
    t.write_json(a.output/'measurement.json',dict(reference=reference,sampling_reference=sampling_measure,selected=chosen,matched_v8_checkpoint_cutoff=next(row for row in candidates if row['step']==2400 and row['cutoff']==.75),candidates=candidates,
        eligible_rows=len(eligible),improved_development_gate=bool(improved),
        fallback='No eligible candidate: predeclared diagnostic update1200/cutoff.70 only' if not eligible else None))
    t.write_json(a.output/'receipt.json',dict(passed=True,source_sha256=t.sha(Path(__file__)),
        freeze_sha256=t.sha(a.output/'freeze.json'),candidate_rows=len(candidates),
        baseline_freeze_sha256=t.sha(a.baseline_freeze),
        baseline_inference_receipt_sha256=t.sha(a.baseline_inference/'receipt.json'),
        sampling_freeze_sha256=t.sha(a.sampling_freeze),
        sampling_inference_receipt_sha256=t.sha(a.sampling_inference/'receipt.json'),
        evaluation_role='Exposed DEV, no final-site evaluation or independent transfer claim',test_array_access='none'))
    print(json.dumps(dict(step=step,cutoff=chosen['cutoff'],pooled=chosen['pooled'],
        mean_positive_site_recall=chosen['mean_positive_site_recall'],clear_pixel_fpr=chosen['clear_pixel_fpr'],
        recovered_positive_scenes=chosen['recovered_positive_scenes'],eligible_rows=len(eligible),
        improved_development_gate=bool(improved))),flush=True)


if __name__=='__main__':
    main()
