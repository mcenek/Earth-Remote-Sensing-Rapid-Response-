"""Apply the prospectively fixed useful-F1 rule to the one v11 fit."""
import argparse
import copy
import json
from pathlib import Path

import GTM_Model7_repaired_inference as factory
import GTM_Model7_calibrate_broader_v7 as c


def rank(row):
    return (row['pooled']['f1'],row['mean_positive_site_recall'],row['mean_positive_scene_recall'],
            -row['clear_pixel_fpr'],-row['step'],row['cutoff'])


def augment(row, scenes):
    """Derived native counts only, with limits fixed in the written protocol."""
    p = row['pooled']
    p['f1'] = 2*p['tp']/max(2*p['tp']+p['fp']+p['fn'],1)
    row['marked_native_pixels'] = p['tp']+p['fp']
    positives = [row['scene_native_confusion'][s['id']] for s in scenes if s['scene_kind']=='PLUME']
    row['whole_plume_misses'] = sum(c['tp']==0 for c in positives)
    row['half_recovered_positive_scenes'] = sum(c['tp']/max(c['tp']+c['fn'],1)>=.5 for c in positives)
    row['clear_false_positive_pixels'] = sum(row['scene_native_confusion'][s['id']]['fp'] for s in scenes if s['scene_kind']=='NO_PLUME')
    guard = {'historical_eligibility':row['qualified'],
             'clear_pixels_at_most_5140':row['clear_false_positive_pixels']<=5140,
             'clear_activated_at_most_20':row['activated_clear_scenes']<=20,
             'site_recall_at_least_v7':row['mean_positive_site_recall']>=.297463372680822}
    target = {'f1_at_least_70':p['f1']>=.7,'precision_at_least_70':p['precision']>=.7,
              'recall_at_least_65':p['recall']>=.65,
              'site_recall_at_least_50':row['mean_positive_site_recall']>=.5,
              'half_recovered_at_least_23':row['half_recovered_positive_scenes']>=23,
              'clear_pixels_at_most_5140':guard['clear_pixels_at_most_5140'],
              'clear_activated_at_most_20':guard['clear_activated_at_most_20']}
    row.update(selection_gates=guard,selection_eligible=all(guard.values()),
               target_gates=target,useful_target_met=all(target.values()),
               failed_selection_gates=[k for k,v in guard.items() if not v],
               failed_target_gates=[k for k,v in target.items() if not v])
    return row


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
    if meta.get('arm')!='hard_negative_replay':
        raise ValueError('Only the single declared v11 hard-negative fit is allowed')
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
    reference=augment(c.measure(baseprob,samples,scenes,.7,1200),scenes)
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
    sampling_measure=augment(c.measure(sampling_prob,samples,scenes,.75,2400),scenes)
    candidates=[]
    for step in (1200,2400,3600,4800):
        milestone=receipt['models']['siamese']['milestones'][str(step)]
        if t.sha(a.run/'siamese'/f'step_{step}.pt')!=milestone['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint drift')
        probabilities=c.load_probabilities(a.run/'siamese'/f'dev_step_{step}',scenes,samples,milestone['dev']['scenes'])
        candidates.extend(augment(c.measure(probabilities,samples,scenes,cutoff,step),scenes) for cutoff in c.CUTOFFS)
    eligible=[row for row in candidates if row['selection_eligible']]
    chosen=max(eligible,key=rank) if eligible else next(row for row in candidates if row['step']==1200 and row['cutoff']==.7)
    improved=(chosen['selection_eligible'] and chosen['mean_positive_site_recall']>reference['mean_positive_site_recall'] and
              chosen['pooled']['iou']>=reference['pooled']['iou'] and chosen['clear_pixel_fpr']<=reference['clear_pixel_fpr'])
    a.output.mkdir(parents=True,exist_ok=False)
    step=chosen['step'];checkpoint=(a.run/'siamese'/f'step_{step}.pt').resolve()
    frozen=dict(schema='GTM_Model7_repaired_freeze_v1',selection_kind='hard_negative_F1_exposed_DEV_v11',
        run=str(a.run.resolve()),metadata=copy.deepcopy(meta),training_receipt_sha256=t.sha(a.run/'receipt.json'),
        training_replay_sha256=t.sha(a.run/'e2e_receipt.json'),threshold=chosen['cutoff'],
        selection='Prospective F1 ranking with v7 clear-pixel/activation guards; 22 cutoffs/4 milestones; exposed DEV only',
        test_predictions_used_for_selection=False,improved_development_gate=bool(improved),
        useful_target_met=chosen['useful_target_met'],
        models={'siamese':dict(checkpoint=str(checkpoint),checkpoint_sha256=t.sha(checkpoint),steps=step,
            threshold=chosen['cutoff'],qualified_development_gate=chosen['selection_eligible'],development=chosen)})
    t.write_json(a.output/'freeze.json',frozen)
    t.write_json(a.output/'measurement.json',dict(reference=reference,sampling_reference=sampling_measure,selected=chosen,matched_v8_checkpoint_cutoff=next(row for row in candidates if row['step']==2400 and row['cutoff']==.75),candidates=candidates,
        eligible_rows=len(eligible),improved_development_gate=bool(improved),useful_target_met=chosen['useful_target_met'],
        unconstrained_best=max(candidates,key=rank),
        fallback='No eligible candidate: predeclared diagnostic update1200/cutoff.70 only' if not eligible else None))
    t.write_json(a.output/'receipt.json',dict(passed=True,source_sha256=t.sha(Path(__file__)),
        freeze_sha256=t.sha(a.output/'freeze.json'),measurement_sha256=t.sha(a.output/'measurement.json'),candidate_rows=len(candidates),
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
