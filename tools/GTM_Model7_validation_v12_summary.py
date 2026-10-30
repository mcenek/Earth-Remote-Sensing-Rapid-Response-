"""Describe frozen validation outcomes without making further model choices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();gate_path=root/'selection/all_arms_frozen.json'
    gate=json.loads(gate_path.read_text());measurement_path=root/'selection/measurements.json'
    if not gate['passed'] or sha(measurement_path)!=gate['measurements_sha256']:
        raise ValueError('Frozen all-arm selection required')
    choices=json.loads(measurement_path.read_text());split=json.loads((root/'data/frozen_split.json').read_text())
    sources={str(gate_path):sha(gate_path),str(measurement_path):sha(measurement_path)}
    result=dict(selected_arm=gate['selected_arm'],historical_exposure=split['historical_exposure'],splits={})
    manifest=json.loads((root/'data/combined_manifest.json').read_text())
    by_id={s['id']:s for s in manifest['scenes']}
    for role in ('dev','assessment','regression'):
        receipt_path=root/('evaluation_'+role)/'receipt.json';replay_path=receipt_path.parent/'replay.json'
        receipt=json.loads(receipt_path.read_text());replay=json.loads(replay_path.read_text())
        if not replay['passed'] or replay['receipt_sha256']!=sha(receipt_path):raise ValueError('Evaluation replay required')
        sources[str(receipt_path)]=sha(receipt_path);sources[str(replay_path)]=sha(replay_path)
        result['splits'][role]=receipt['measurements']
        for arm,row in receipt['measurements'].items():
            counts=row['scene_native_confusion']
            positives=[sid for sid in counts if by_id[sid]['scene_kind']=='PLUME']
            row['whole_miss_ids']=[sid for sid in positives if counts[sid]['tp']==0]
            row['smallest_five']=[dict(id=sid,native_plume_pixels=counts[sid]['tp']+counts[sid]['fn'],
                recall=counts[sid]['tp']/max(counts[sid]['tp']+counts[sid]['fn'],1))
                for sid in sorted(positives,key=lambda sid:(counts[sid]['tp']+counts[sid]['fn'],sid))[:5]]
            row['largest_clear_false_alarms']=[dict(id=sid,fp=counts[sid]['fp']) for sid in sorted(
                (sid for sid in counts if by_id[sid]['scene_kind']=='NO_PLUME'),key=lambda sid:(-counts[sid]['fp'],sid))[:10]]
            if role=='assessment':
                # Descriptive uncertainty from whole linked components, never
                # used to change a model or cutoff after seeing assessment.
                groups=split['components']['assess'];keys=('tp','fp','fn','tn')
                matrix=np.array([[sum(counts[sid][key] for sid in group['observation_ids']) for key in keys] for group in groups])
                rng=np.random.default_rng(20261026);draws=rng.integers(len(groups),size=(2000,len(groups)))
                pooled=matrix[draws].sum(axis=1);tp,fp,fn,_=pooled.T
                values=2*tp/np.maximum(2*tp+fp+fn,1)
                row['descriptive_component_bootstrap_f1_95_interval']=np.quantile(values,[.025,.975]).tolist()
                row['bootstrap_components']=len(groups)
    winner=gate['selected_arm']
    result['useful_target_met']=bool(winner and result['splits']['dev'][winner]['useful_target_met'] and
                                    result['splits']['assessment'][winner]['useful_target_met'])
    figure=Figure(figsize=(9,6),facecolor='#13242d',layout='constrained');axis=figure.subplots();axis.set_facecolor('#13242d')
    for arm,color in (('control','#8da8b5'),('fraction','#70d5d0'),('ratio','#e8b978')):
        rows=choices[arm]['candidates']
        axis.scatter([r['pooled']['recall'] for r in rows],[r['pooled']['precision'] for r in rows],s=12,alpha=.3,color=color,label=arm+' (88 fixed choices)')
        chosen=choices[arm]['selected'];axis.scatter([chosen['pooled']['recall']],[chosen['pooled']['precision']],s=160,marker='*',color=color,edgecolor='white')
    recall=np.linspace(.351,1,400);precision=.7*recall/(2*recall-.7)
    axis.plot(recall,precision,'--',color='#e6879e',label='F1 = 70%')
    axis.set(xlim=(0,1),ylim=(0,1),xlabel='Plume pixel recall on SELECT',ylabel='Plume pixel precision on SELECT',
        title='No extra-area shortcut: precision and recall together')
    axis.title.set_color('#e4eef1');axis.xaxis.label.set_color('#b7cbd3');axis.yaxis.label.set_color('#b7cbd3');axis.tick_params(colors='#b7cbd3')
    axis.grid(alpha=.15);axis.legend(facecolor='#13242d',labelcolor='#e4eef1',fontsize=9)
    axis.text(.02,.02,'Stars: frozen choices. Failed arms use the predeclared diagnostic cutoff.\nAll clear observations contribute false pixels. Assessment never selects a point.',transform=axis.transAxes,color='#b7cbd3',fontsize=9)
    for spine in axis.spines.values():spine.set_color('#49616b')
    figure.savefig(root/'selection_precision_recall.png',dpi=130,facecolor=figure.get_facecolor())
    result.update(sources=sources,source_sha256=sha(Path(__file__)))
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    compact={role:{arm:dict(f1=row['pooled']['f1'],precision=row['pooled']['precision'],recall=row['pooled']['recall'],
        marked_pixels=row['marked_native_pixels'],clear_fpr=row['clear_pixel_fpr'],clear_activated=row['activated_clear_scenes'],
        whole_misses=row['whole_plume_misses']) for arm,row in values.items()} for role,values in result['splits'].items()}
    print(json.dumps(dict(winner=winner,useful_target_met=result['useful_target_met'],splits=compact)),flush=True)


if __name__=='__main__':main()
