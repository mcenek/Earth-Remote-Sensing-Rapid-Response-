"""Package frozen residual-model development choices for independent rendering."""
import argparse
import copy
import json
from pathlib import Path

from matplotlib.figure import Figure

import GTM_Model7_native_residual_v14 as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics',action='store_true',help='Separate, explicitly unconstrained diagnostic views; never promotes a model')
    args=parser.parse_args()
    out=study.OUT
    destination=out/('review_recipe_diagnostics.json' if args.diagnostics else 'review_recipe.json')
    if destination.exists(): raise FileExistsError(destination)
    freeze_path=out/'selection/all_arms_frozen.json'
    frozen=json.loads(freeze_path.read_text())
    summary=json.loads((out/'evaluation/summary.json').read_text())
    if (not frozen['passed'] or frozen['script_sha256']!=study.sha(Path(study.__file__)) or
        frozen['protocol_sha256']!=study.sha(study.PROTOCOL) or
        summary['selection_freeze_sha256']!=study.sha(freeze_path)):
        raise ValueError('Study identity changed')
    selection=json.loads((out/'selection/measurements.json').read_text())
    historical=json.loads((study.TEACHER/'review_recipe.json').read_text())
    baseline=copy.deepcopy(historical['methods']['candidate'])
    baseline.update(name='Retained v11',prediction_grid='20m_repeat',
                    status='RETAINED CONTROL | exposed development candidate; new US validation pending',
                    resolution='20 m model output repeated to 10 m annotation grid')
    methods={'baseline':baseline}
    labels=dict(residual_original_loss='Residual | original loss',residual_balanced_loss='Residual | balanced pixels')
    histories={}
    figure=Figure(figsize=(12,3.6),facecolor='#13242d')
    axes=figure.subplots(1,2)
    items=study.index()
    displayed={}
    for key,arm in zip(('candidate','ratio'),study.ARMS):
        choice=frozen['choices'][arm]; result=summary['models'][arm]
        if (choice['step'],choice['cutoff'])!=(result['step'],result['cutoff']):
            raise ValueError('Choice changed')
        meta=study.fit_meta(arm)
        measured=summary['measurements'][arm]
        folder=out/'evaluation'/arm
        records=result['scenes']
        if args.diagnostics:
            measured=max(selection['arms'][arm]['candidates'],key=study.v13.rank)
            choice=dict(step=measured['step'],cutoff=measured['cutoff'],eligible=measured['selection_eligible'],
                fallback_teacher_identity=False,
                checkpoint_sha256=meta['milestones'][str(measured['step'])]['checkpoint_sha256'])
            folder=out/f'fit_{arm}'/f"dev_step_{measured['step']}"
            records=copy.deepcopy(meta['milestones'][str(measured['step'])]['dev']['scenes'])
            for sid,record in records.items():record['native_confusion']=measured['scene_native_confusion'][sid]
        displayed[arm]=(choice,measured)
        fallback=choice['fallback_teacher_identity']
        methods[key]=dict(name=('DIAGNOSTIC | ' if args.diagnostics else '')+labels[arm]+(' | teacher retained' if fallback else ''),
            status=('UNCONSTRAINED DIAGNOSTIC | rejected by error guards' if args.diagnostics and not choice['eligible'] else
                    'REJECTED CORRECTION | unchanged teacher displayed' if fallback else
                    'QUALIFIED DEVELOPMENT REPLACEMENT | prospective US validation pending'),
            folder=str(folder),cutoff=choice['cutoff'],
            checkpoint_sha256=choice['checkpoint_sha256'],records=records,
            prediction_grid='20m_repeat' if fallback else 'native10m',
            resolution=baseline['resolution'] if fallback else '10 m output grid; SWIR measurements remain 20 m')
        path=out/f'fit_{arm}/history.json'
        history=json.loads(path.read_text())
        measured=[]
        for step in study.STEPS:
            measured.append(study.v13.measure(study.saved_predictions(arm,step,items,meta),items,.65,step))
        histories[arm]={'history_sha256':study.sha(path),'fixed_cutoff_dev':measured}
        axes[0].plot([r['step'] for r in history],[r['loss'] for r in history],label=labels[arm])
        axes[1].plot(study.STEPS,[r['pooled']['f1'] for r in measured],marker='o',label=labels[arm])
    for axis in axes:
        axis.set_facecolor('#13242d');axis.tick_params(colors='#d8e5ed')
        axis.xaxis.label.set_color('#d8e5ed');axis.title.set_color('#d8e5ed')
        axis.set_xlabel('Training updates');axis.grid(alpha=.12)
        axis.legend(facecolor='#21333d',labelcolor='#e4eef1',fontsize=8)
    axes[0].set_title('Recorded training loss; objectives differ')
    axes[1].set_title('Exposed DEV F1 at fixed teacher cutoff 0.65')
    axes[1].axhline(summary['teacher_measurement']['pooled']['f1'],color='#72d6c7',linestyle='--',alpha=.6)
    axes[1].set_ylim(0,1);figure.tight_layout()
    learning=out/('learning_diagnostics' if args.diagnostics else 'learning');learning.mkdir(exist_ok=False)
    figure.savefig(learning/'learning.png',facecolor=figure.get_facecolor(),dpi=140)
    study.save(learning/'measurements.json',histories)
    sources=list(historical['sources'])
    for relative,passed in [('data/receipt.json',False),('data/index.json',False),('preflight/receipt.json',True),
        ('selection/all_arms_frozen.json',True),('selection/measurements.json',False),('evaluation/summary.json',False),
        ]:
        path=out/relative;sources.append(dict(path=str(path),sha256=study.sha(path),require_passed=passed))
    sources.append(dict(path=str(learning/'measurements.json'),sha256=study.sha(learning/'measurements.json'),require_passed=False))
    for arm in study.ARMS:
        for relative,passed in [(f'fit_{arm}/metadata.json',False),(f'replay_{arm}/receipt.json',True)]:
            path=out/relative;sources.append(dict(path=str(path),sha256=study.sha(path),require_passed=passed))
    statements=[]
    for arm,(c,m) in displayed.items():
        p=m['pooled']
        statements.append(f"{labels[arm]}: F1 {p['f1']:.2%}, precision {p['precision']:.2%}, recall {p['recall']:.2%}; "
            f"{m['activated_clear_scenes']}/152 clear observations activated. "
            +('A replacement passed the predeclared error guards.' if c['eligible'] else
              'This unconstrained diagnostic failed the error guards and is NOT selected.' if args.diagnostics else
              'No qualifying replacement; the displayed prediction is the unchanged teacher.'))
    norm=json.loads((study.TEACHER/'fit/metadata.json').read_text())['normalization']
    recipe=dict(manifest=str(study.v13.SOURCE),manifest_sha256=study.v13.SOURCE_SHA,
        evaluation_split='dev',evaluation_role='historically_exposed_development_selection',
        id_prefix='GTM_Model7_residual_v14_20261026'+('_diagnostic' if args.diagnostics else ''),
        title='Residual corrections | unconstrained diagnostics' if args.diagnostics else 'Native residual correction comparison',
        comparison_heading='Retained v11 and two bounded residual studies. Actual saved predictions.',
        architecture_family='early_fusion_unet',status='DEVELOPMENT ONLY | prospective US validation pending',
        scope=('Separate unconstrained diagnostic views. These masks do not replace the frozen selected model. ' if args.diagnostics else '')+
            'All 198 exposed DEV observations: 46 plume and 152 provider no-plume scenes. '
            'No independent national performance claim. Labels are binary plume annotations, not emission rates. '
            '10 m output spacing does not create new 10 m SWIR measurements.',
        acceptance=' '.join(statements)+' No dilation, smoothing, filling or scene exclusions.',
        context='A fixed v11 prediction plus a small, zero-initialized, bounded native-grid correction. '
            'Both arms use identical inputs, initialization, balanced scene draws and 4,800 updates. '
            'Only pixel-class weighting differs. The teacher saw FIT labels; DEV selects cutoffs and checkpoints. '
            'Canonical probabilities were generated and exactly replayed on CPU. '
            'Loss values across the two objectives are not directly comparable.',
        chart_title='Frozen teacher with native correction',
        chart_detail='958 FIT; 198 exposed DEV. No new test images. Fixed physical and clear-error guards.',
        learning_image=dict(path=str(learning/'learning.png'),sha256=study.sha(learning/'learning.png')),
        sources=sources,methods=methods,feature_contract='residual26',frozen_normalization=norm,
        native_data={r['id']:dict(path=str(out/'data'/r['cache']),sha256=r['sha256']) for r in items if r['split']=='dev'})
    study.save(destination,recipe)
    print(json.dumps(dict(recipe=str(destination),acceptance=recipe['acceptance'])))


if __name__=='__main__':main()
