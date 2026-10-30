"""Publish broader DEV predictions without relabeling old curves as new DEV.

All source and native-grid checks remain in the existing prediction renderer.
Failure modes were recorded in the broader-dev model protocol before code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matplotlib.figure import Figure

import GTM_Model7_review_repaired as review


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--unet-label', required=True)
    parser.add_argument('--omit-old-history', action='store_true')
    known, remaining = parser.parse_known_args()
    review.NAMES['siamese'] = known.unet_label
    review.HTML = review.HTML.replace('<b>Temporal U-Net</b>', '<b>'+review.html.escape(known.unet_label)+'</b>')
    original_save = review.v.save
    def development_save(path, value):
        if Path(path).name == 'experiment.json':
            value['runGroup'] = 'GTM_Model7_broader_dev_20261024'
            value['architectureFamily'] = 'GTM_Model7_broader_development'
            for scene in value['scenes']:
                scene['split'] = 'Broader DEV | separate from fitting; previously evaluated project sites'
                scene['nativeMetrics']['claim'] = 'Exposed development data used for selection, not an independent test'
                scene['notes'] = scene['notes'].replace(
                    ' Unseen by these weights; historical project exposure is audited and is not ruled out.',
                    ' These locations were evaluated in previous project experiments and are now development data.')
        original_save(path, value)
    review.v.save = development_save
    def broader_history(frozen, output):
        source = Path(frozen['run'])/'receipt.json'
        if review.v.sha(source) != frozen['training_receipt_sha256']:
            raise ValueError('Fitting history changed')
        receipt = json.loads(source.read_text(encoding='utf-8'))
        figure = Figure(figsize=(12, 3.6), facecolor='#13242d')
        axes = figure.subplots(1, 2)
        for axis, metric, title in zip(axes, ('iou', 'false_positive_area_rate'),
                                      ('Plume overlap on positive scenes', 'False activation on no-plume scenes')):
            axis.set_facecolor('#13242d')
            # Only draw models actually fitted in this run. The unchanged
            # residual control has predictions, but no new training curve.
            for model, color in (('stable_residual', '#70d5d0'), ('siamese', '#f4b66d')):
                if model not in receipt['models']:
                    continue
                rows = receipt['models'][model]['milestones']
                steps = sorted(map(int, rows))
                for split, style in (('train', '--'), ('dev', '-')):
                    section = 'positive' if metric == 'iou' else 'negative'
                    values = [100*(rows[str(s)][split][section][metric] or 0) for s in steps]
                    axis.plot(steps, values, style, color=color, linewidth=2,
                              label=review.NAMES[model]+' '+split.upper())
            axis.set_title(title, color='#e4eef1', fontsize=11)
            axis.set_xlabel('Optimizer updates', color='#b7cbd3')
            axis.set_ylabel('Percent', color='#b7cbd3')
            axis.tick_params(colors='#b7cbd3')
            axis.grid(alpha=.15, color='#b7cbd3')
            axis.set_ylim(bottom=0)
        axes[0].legend(fontsize=7, labelcolor='#e4eef1', facecolor='#13242d', edgecolor='#415b67')
        figure.tight_layout()
        path = output/'learning.png'
        figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
        return review.v.image_uri(path)
    review.learning_chart = broader_history
    if known.omit_old_history:
        def frozen_history(frozen, output):
            figure = Figure(figsize=(12, 1.25), facecolor='#13242d')
            axis = figure.subplots()
            axis.set_axis_off()
            axis.text(.02, .7, 'Frozen weights, evaluated on broader development sites',
                      color='#e4eef1', fontsize=13, transform=axis.transAxes)
            axis.text(.02, .3, 'No new fit for this control. Original six-plume DEV curves are omitted here.',
                      color='#b7cbd3', fontsize=11, transform=axis.transAxes)
            path = output/'learning.png'
            figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
            return review.v.image_uri(path)
        review.learning_chart = frozen_history
    sys.argv = [sys.argv[0], *remaining]
    review.main()
    # Retain the adapter identity alongside the renderer's independent checks.
    output = Path(remaining[remaining.index('--output')+1])
    path = output/'verification.json'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    receipt['renderer_adapter_sha256'] = review.v.sha(Path(__file__))
    receipt['unet_label'] = known.unet_label
    receipt['original_dev_history_omitted'] = known.omit_old_history
    review.v.save(path, receipt)


if __name__ == '__main__':
    main()
