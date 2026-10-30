"""Select on SELECT only; freeze all arms before locked assessment access.

Failure modes, ranking and useful targets were fixed in the v12 protocol before
this implementation. This command never opens assessment or regression arrays.
"""
import argparse
import json
from pathlib import Path

import GTM_Model7_calibrate_broader_v7 as c
import GTM_Model7_repaired_inference as factory

ARMS = ('control', 'fraction', 'ratio')


def rank(row):
    return (row['pooled']['f1'], row['mean_positive_site_recall'],
            row['mean_positive_scene_recall'], -row['clear_pixel_fpr'],
            -row['step'], row['cutoff'])


def augment(row, scenes):
    p = row['pooled']
    p['f1'] = 2*p['tp']/max(2*p['tp']+p['fp']+p['fn'], 1)
    positives = [row['scene_native_confusion'][s['id']] for s in scenes if s['scene_kind']=='PLUME']
    row['whole_plume_misses'] = sum(x['tp']==0 for x in positives)
    row['half_recovered_positive_scenes'] = sum(x['tp']/max(x['tp']+x['fn'],1)>=.5 for x in positives)
    row['marked_native_pixels'] = p['tp']+p['fp']
    row['clear_activation_rate'] = row['activated_clear_scenes']/max(row['clear_scenes'],1)
    gates = dict(precision_at_least_70=p['precision']>=.7,
                 recall_at_least_30=p['recall']>=.3,
                 clear_pixel_fpr_at_most_001=row['clear_pixel_fpr']<=.001,
                 clear_activation_at_most_15=row['clear_activation_rate']<=.15,
                 half_positive_observations_recover_10=row['recovered_positive_scenes']>=row['positive_scenes']/2)
    targets = dict(f1_at_least_70=p['f1']>=.7, precision_at_least_70=p['precision']>=.7,
                   recall_at_least_65=p['recall']>=.65,
                   clear_pixel_fpr_at_most_001=row['clear_pixel_fpr']<=.001,
                   clear_activation_at_most_15=row['clear_activation_rate']<=.15)
    row.update(selection_gates=gates, selection_eligible=all(gates.values()),
               failed_selection_gates=[k for k,v in gates.items() if not v],
               target_gates=targets, useful_target_met=all(targets.values()),
               failed_target_gates=[k for k,v in targets.items() if not v])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); out = root/'selection'
    if out.exists():
        raise FileExistsError('Selection is immutable: '+str(out))
    manifest_path = root/'data/combined_manifest.json'
    split_receipt_path = root/'data/split_receipt.json'
    split_receipt = json.loads(split_receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    scenes = [s for s in manifest['scenes'] if s['split']=='dev']
    if len(scenes)!=144 or sum(s['scene_kind']=='PLUME' for s in scenes)!=39:
        raise ValueError('Selection membership changed')
    choices, measurements, sources = {}, {}, {}
    samples = None
    for arm in ARMS:
        run = root/('fit_'+arm)
        meta = json.loads((run/'metadata.json').read_text())
        factory.check_implementation(meta); t = factory.t
        if t.sha(manifest_path)!=split_receipt['manifest_sha256'] or meta['manifest_sha256']!=t.sha(manifest_path):
            raise ValueError('Dataset/source drift')
        receipt = json.loads((run/'receipt.json').read_text())
        replay = json.loads((run/'e2e_receipt.json').read_text())
        if not replay['passed'] or receipt['metadata_sha256']!=t.sha(run/'metadata.json'):
            raise ValueError('Matching completed fresh CPU replay required')
        if (meta['arm']!=arm or replay['arm']!=arm or
            replay['trainer_sha256']!=meta['trainer_sha256'] or
            replay['manifest_sha256']!=meta['manifest_sha256'] or
            replay['run_metadata_sha256']!=t.sha(run/'metadata.json') or
            set(replay['milestones'])!={'1200','2400','3600','4800'}):
            raise ValueError('Fresh replay identity/milestones differ')
        for replayed in replay['milestones'].values():
            for split,n in (('train',670),('dev',144)):
                if (replayed[split]['scene_count']!=n or
                    replayed[split]['max_abs_probability20_error']>1e-6 or
                    not replayed[split]['support_and_native_counts_exact']):
                    raise ValueError('Complete exact-support CPU replay required')
        if samples is None:
            samples = {s['id']:t.load_scene(manifest_path,s) for s in scenes}
        rows=[]
        for step in (1200,2400,3600,4800):
            saved=receipt['models']['siamese']['milestones'][str(step)]
            checkpoint=run/'siamese'/f'step_{step}.pt'
            if t.sha(checkpoint)!=saved['checkpoint_sha256']:
                raise ValueError('Weights changed')
            probabilities=c.load_probabilities(run/'siamese'/f'dev_step_{step}',scenes,samples,saved['dev']['scenes'])
            rows.extend(augment(c.measure(probabilities,samples,scenes,cutoff,step),scenes) for cutoff in c.CUTOFFS)
        eligible=[x for x in rows if x['selection_eligible']]
        chosen=max(eligible,key=rank) if eligible else next(x for x in rows if x['step']==1200 and x['cutoff']==.7)
        checkpoint=run/'siamese'/f"step_{chosen['step']}.pt"
        choice=dict(checkpoint=str(checkpoint),checkpoint_sha256=t.sha(checkpoint),steps=chosen['step'],
                    threshold=chosen['cutoff'],qualified_development_gate=chosen['selection_eligible'],development=chosen)
        frozen=dict(schema='GTM_Model7_repaired_freeze_v1',selection_kind='v12_component_heldout_SELECT',
                    run=str(run),metadata=meta,training_receipt_sha256=t.sha(run/'receipt.json'),
                    training_replay_sha256=t.sha(run/'e2e_receipt.json'),threshold=chosen['cutoff'],models={'siamese':choice},
                    selection='SELECT only, 22 cutoffs x 4 checkpoints, fixed F1 ranking and precision/clear guards',
                    test_predictions_used_for_selection=False,locked_assessment_array_access='none',
                    historical_exposure='All sites have project history; this is withheld from these fresh fits only.')
        choices[arm]=frozen
        measurements[arm]=dict(selected=chosen,eligible_rows=len(eligible),candidates=rows,
                                unconstrained_best=max(rows,key=rank),
                                fallback=None if eligible else 'Failed diagnostic: predeclared update1200/cutoff0.70')
        sources[arm]=dict(metadata_sha256=t.sha(run/'metadata.json'),receipt_sha256=t.sha(run/'receipt.json'),
                          replay_sha256=t.sha(run/'e2e_receipt.json'))
    eligible_arms=[arm for arm in ARMS if choices[arm]['models']['siamese']['qualified_development_gate']]
    winner=max(eligible_arms,key=lambda arm:rank(measurements[arm]['selected'])) if eligible_arms else None
    out.mkdir()
    for arm,frozen in choices.items():
        (out/arm).mkdir(); t.write_json(out/arm/'freeze.json',frozen)
    t.write_json(out/'measurements.json',measurements)
    # This final receipt is the gate that permits assessment/regression inference.
    t.write_json(out/'all_arms_frozen.json',dict(passed=True,selected_arm=winner,
        retain_historical_v11=winner is None,manifest_sha256=t.sha(manifest_path),
        split_receipt_sha256=t.sha(split_receipt_path),protocol_sha256=t.sha(t.PROTOCOL),
        selector_sha256=t.sha(Path(__file__)),measurements_sha256=t.sha(out/'measurements.json'),
        freezes={arm:dict(path=str(out/arm/'freeze.json'),sha256=t.sha(out/arm/'freeze.json')) for arm in ARMS},
        sources=sources,assessment_array_access_before_freeze='none',regression_array_access_before_freeze='none'))
    print(json.dumps(dict(winner=winner,arms={arm:dict(step=measurements[arm]['selected']['step'],
        cutoff=measurements[arm]['selected']['cutoff'],eligible_rows=measurements[arm]['eligible_rows'],
        pooled=measurements[arm]['selected']['pooled']) for arm in ARMS})),flush=True)


if __name__=='__main__':
    main()
