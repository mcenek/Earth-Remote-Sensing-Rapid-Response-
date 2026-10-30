"""Build a source-bound native-model comparison after both choices are frozen."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

import GTM_Model7_native_transfer_v13 as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe-name',default='review_recipe.json')
    args=parser.parse_args()
    if Path(args.recipe_name).name!=args.recipe_name: raise ValueError('Recipe name must be a filename')
    out = study.OUT
    destination = out / args.recipe_name
    if destination.exists():
        raise FileExistsError('Review recipe already exists')
    frozen_path = out / 'selection/rejection_summary.json'
    frozen = json.loads(frozen_path.read_text())
    summary_path = out / 'evaluation/summary.json'
    summary = json.loads(summary_path.read_text())
    if (summary['rejection_summary_sha256'] != study.sha(frozen_path)
            or frozen['hashes']['trainer'] != study.sha(Path(study.__file__))
            or frozen['promotion_allowed'] or not frozen['diagnostic_only']
            or summary['numerical_replay_status']!='FAILED'):
        raise ValueError('Frozen study identity changed')
    historical_path = study.ROOT / 'outputs/GTM_Model7_hard_negative_v11_20261025/review_recipe.json'
    historical = json.loads(historical_path.read_text())
    baseline = copy.deepcopy(historical['methods']['candidate'])
    baseline.update(name='Historical v11', prediction_grid='20m_repeat',
                    status='RETAINED CONTROL | exposed development candidate; new US validation pending',
                    resolution='20 m model grid, repeated to 10 m annotation for comparison')
    methods = {'baseline': baseline}
    labels = dict(native_zero_wind='Rejected native U-Net', native_wind='Rejected native U-Net + wind')
    plot = Figure(figsize=(12, 3.6), facecolor='#13242d')
    axes = plot.subplots(1, 2)
    histories = {}
    for key, arm in zip(('candidate', 'ratio'), study.ARMS):
        choice = frozen['choices'][arm]
        meta = study.fit_metadata(arm)
        result = summary['models'][arm]
        if (result['step'], result['cutoff']) != (choice['step'], choice['cutoff']):
            raise ValueError('Selected inference changed')
        methods[key] = dict(name=labels[arm], folder=str(out / 'evaluation' / arm),
                            cutoff=choice['cutoff'], checkpoint_sha256=choice['checkpoint_sha256'],
                            records=result['scenes'], prediction_grid='native10m',
                            resolution='10 m output grid; SWIR information remains 20 m')
        history_path = out / f'fit_{arm}/history.json'
        history = json.loads(history_path.read_text())
        histories[arm] = {'history_sha256': study.sha(history_path), 'fixed_cutoff_dev': []}
        axes[0].plot([r['step'] for r in history], [r['loss'] for r in history], label=labels[arm], alpha=.8)
        for step in study.MILESTONES:
            predictions = study.saved_predictions(arm, step, study.index(), meta)
            measured = study.measure(predictions, study.index(), .5, step)
            histories[arm]['fixed_cutoff_dev'].append(measured)
        axes[1].plot(study.MILESTONES, [r['pooled']['f1'] for r in histories[arm]['fixed_cutoff_dev']],
                     marker='o', label=labels[arm])
    for axis in axes:
        axis.set_facecolor('#13242d')
        axis.tick_params(colors='#d8e5ed')
        axis.xaxis.label.set_color('#d8e5ed')
        axis.title.set_color('#d8e5ed')
        axis.set_xlabel('Training updates')
        axis.grid(alpha=.12)
        axis.legend(facecolor='#21333d', labelcolor='#e4eef1', fontsize=8)
    axes[0].set_title('Recorded balanced-batch training loss')
    axes[1].set_title('Exposed DEV plume F1 at fixed cutoff 0.50')
    axes[1].set_ylim(0, 1)
    plot.tight_layout()
    learning = out / ('learning' if args.recipe_name=='review_recipe.json' else Path(args.recipe_name).stem+'_learning')
    learning.mkdir(exist_ok=False)
    plot.savefig(learning / 'learning.png', facecolor=plot.get_facecolor(), dpi=140)
    study.save_json(learning / 'measurements.json', histories)
    sources = list(historical['sources'])
    for relative, passed in [('preflight/receipt.json', False), ('data/index.json', False),
                             ('selection/rejection_summary.json', False),
                             ('evaluation/summary.json', False)]:
        path = out / relative
        sources.append(dict(path=str(path), sha256=study.sha(path), require_passed=passed))
    sources.append(dict(path=str(learning/'measurements.json'),sha256=study.sha(learning/'measurements.json'),require_passed=False))
    for arm in study.ARMS:
        for relative, passed in [(f'fit_{arm}/metadata.json', False), (f'replay_diagnostic_{arm}/receipt.json', False)]:
            path = out / relative
            sources.append(dict(path=str(path), sha256=study.sha(path), require_passed=passed))
    measures = summary['measurements']
    statements = []
    for arm in study.ARMS:
        m = measures[arm]
        p = m['pooled']
        statements.append(f"{labels[arm]}: F1 {p['f1']:.2%}, precision {p['precision']:.2%}, recall {p['recall']:.2%}; "
                          f"{m['activated_clear_scenes']}/152 clear scenes activated; "
                          f"selection eligible {m['selection_eligible']}; useful target met {m['useful_target_met']}.")
    native = {item['id']: dict(path=str(out / 'data' / item['cache']), sha256=item['sha256'])
              for item in study.index() if item['split'] == 'dev'}
    recipe = dict(manifest=str(study.SOURCE), manifest_sha256=study.SOURCE_SHA,
                  evaluation_split='dev', evaluation_role='historically_exposed_development_selection',
                  id_prefix='GTM_Model7_native_v13_20261026', title='Rejected native-grid model comparison',
                  architecture_family='early_fusion_unet', status='REJECTED | scientific guards and numerical replay failed',
                  scope='All 198 historically exposed DEV observations: 46 plume and 152 provider no-plume scenes. '
                        'No independent or nationwide accuracy claim. Binary MARS-S2L annotations, not emission rates. '
                        'Native-grid predictions do not create new 10 m SWIR measurements.',
                  acceptance='Both native models are rejected. Historical v11 is retained. CPU/GPU numerical replay also failed its bound. '
                             'These are the predeclared failed fallback maps at update 1200 and cutoff 0.70. '
                             + ' '.join(statements) + ' No dilation, filling, smoothing or scene exclusion.',
                  context='Both fresh 3,356,001-parameter U-Nets use the same native inputs, initial weights, '
                          'balanced 4-plume/4-clear draw schedule and 4,800 updates. Only wind inputs differ. '
                          'Historical v11 has a smaller architecture and different preprocessing and sampler; '
                          'this is a package comparison. Graphs show recorded training-batch loss and DEV F1 '
                          'at a fixed 0.50 cutoff. Tables use frozen failed fallback cutoffs. No candidate was eligible. DEV was used for selection.',
                  chart_title='Native-grid models with and without wind',
                  chart_detail='958 FIT observations; 198 exposed DEV observations. Fresh weights. No new test imagery.',
                  learning_image=dict(path=str(learning / 'learning.png'), sha256=study.sha(learning / 'learning.png')),
                  sources=sources, methods=methods, native_data=native)
    study.save_json(destination, recipe)
    print(json.dumps({'recipe': str(destination), 'sha256': study.sha(destination), 'acceptance': recipe['acceptance']}))


if __name__ == '__main__':
    main()
