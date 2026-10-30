"""Bind replayed frozen predictions to three-model visual comparison recipes."""
import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure

ARMS={'baseline':'control','candidate':'fraction','ratio':'ratio'}
NAMES={'baseline':'Fresh control','candidate':'Native-label loss','ratio':'Native-label loss + ratio'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def f1(c):
    c=c.get('confusion',c)
    return 2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();selection=root/'selection';gate_path=selection/'all_arms_frozen.json'
    gate=json.loads(gate_path.read_text());measurements_path=selection/'measurements.json'
    if sha(measurements_path)!=gate['measurements_sha256']:
        raise ValueError('Frozen selection measurements changed')
    measurements=json.loads(measurements_path.read_text())
    if not gate['passed']:raise ValueError('Frozen all-arm choices required')
    out=root/'review_recipes';out.mkdir(exist_ok=False)
    figure=Figure(figsize=(13,3.8),facecolor='#13242d',layout='constrained')
    axes=figure.subplots(1,3)
    for axis,(key,arm) in zip(axes,ARMS.items()):
        receipt=json.loads((root/('fit_'+arm)/'receipt.json').read_text())
        steps=[1200,2400,3600,4800];history=receipt['models']['siamese']['milestones']
        axis.set_facecolor('#13242d')
        axis.plot(steps,[f1(history[str(s)]['train']['native_pooled']) for s in steps],':o',color='#81d1cb',label='FIT')
        axis.plot(steps,[f1(history[str(s)]['dev']['native_pooled']) for s in steps],'-o',color='#e8b978',label='SELECT')
        axis.axvline(measurements[arm]['selected']['step'],color='#a8bdc7',alpha=.4,linewidth=1)
        axis.set(title=NAMES[key],xlabel='Updates',ylabel='Native pixel F1 at fixed 0.50',ylim=(0,1))
        axis.title.set_color('#e4eef1');axis.xaxis.label.set_color('#b7cbd3');axis.yaxis.label.set_color('#b7cbd3')
        axis.tick_params(colors='#b7cbd3');axis.grid(alpha=.15);axis.legend(facecolor='#13242d',labelcolor='#e4eef1')
        for spine in axis.spines.values():spine.set_color('#49616b')
    learning=out/'learning.png';figure.savefig(learning,dpi=130,facecolor=figure.get_facecolor())
    roles={'dev':'Selection validation | 144 observations, 27 sites; used to choose weights and cutoff',
           'assessment':'Locked assessment | 144 observations, 26 sites; opened after all choices were frozen',
           'regression':'Exposed regression | original 198 DEV observations, 55 sites; no choice made here'}
    assessment=json.loads((root/'evaluation_assessment/receipt.json').read_text())
    winner=gate['selected_arm']
    target=bool(winner and measurements[winner]['selected']['useful_target_met'] and assessment['measurements'][winner]['useful_target_met'])
    for split,role in roles.items():
        evaluation=root/('evaluation_'+split);receipt_path=evaluation/'receipt.json';replay_path=evaluation/'replay.json'
        receipt=json.loads(receipt_path.read_text());replay=json.loads(replay_path.read_text())
        if not replay['passed'] or replay['receipt_sha256']!=sha(receipt_path) or receipt['all_arms_frozen_sha256']!=sha(gate_path):
            raise ValueError('Matching frozen evaluation and fresh replay required')
        methods={}
        for key,arm in ARMS.items():
            frozen_path=Path(gate['freezes'][arm]['path']);frozen=json.loads(frozen_path.read_text());choice=frozen['models']['siamese']
            if sha(frozen_path)!=gate['freezes'][arm]['sha256']:raise ValueError('Frozen choice drift')
            name=NAMES[key]+(' (failed diagnostic)' if not choice['qualified_development_gate'] else '')
            methods[key]=dict(name=name,folder=str(evaluation/arm),cutoff=choice['threshold'],
                checkpoint_sha256=choice['checkpoint_sha256'],records=receipt['models'][arm]['scenes'])
        decision=('SELECT chose '+winner+'.' if winner else 'No arm passed the selection guards. Each failed arm is shown at the predeclared update 1200 / cutoff 0.70. Historical v11 remains the research candidate.')
        scope=role+'. '+decision+' All sites have historical project exposure. These scenes were withheld from the new fits; this does not establish independent US-wide accuracy.'
        recipe=dict(title='Model 7 / '+{'dev':'selection validation','assessment':'locked assessment','regression':'old-site regression'}[split],
            manifest=str(root/'data/combined_manifest.json'),manifest_sha256=gate['manifest_sha256'],evaluation_split=split,
            evaluation_role=role,methods=methods,id_prefix='GTM_Model7_validation_v12_'+split,
            architecture_family='Matched fresh temporal U-Nets',status='Research comparison; useful target '+('passed' if target else 'not met'),
            scope=scope,context='Real FIT and SELECT histories at the same 0.50 display cutoff. The tables and maps use each frozen cutoff. Assessment and regression scores did not choose weights or cutoffs.',
            chart_title='Real learning and transfer',chart_detail='Shared initialization, identical site-balanced draws, four fixed checkpoints per model.',
            learning_image=dict(path=str(learning),sha256=sha(learning)),
            acceptance='Useful target: plume F1 at least 70%, precision at least 70%, recall at least 65%, clear-pixel FPR at most 0.1%, and clear-scene activation at most 15%. No mask dilation, filling, or selective removal.',
            sources=[dict(path=str(path),sha256=sha(path),require_passed=True) for path in (gate_path,receipt_path,replay_path)])
        (out/(split+'.json')).write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),winner=gate['selected_arm'])))


if __name__=='__main__':main()
