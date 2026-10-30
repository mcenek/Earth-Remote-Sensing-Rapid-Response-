"""Plot the completed, replayed v8/v11 real-data histories at fixed cutoff .50.

Fail on incomplete milestones, altered metadata, failed replay or count mismatch.
This plot describes fitting and exposed development, not independent testing.
"""
import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-fit',type=Path,required=True)
    p.add_argument('--new-fit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows=[]; sources={}
    for label,run in [('v8',a.old_fit),('v11',a.new_fit)]:
        meta=json.loads((run/'metadata.json').read_text())
        receipt=json.loads((run/'receipt.json').read_text())
        replay=json.loads((run/'e2e_receipt.json').read_text())
        if (receipt['metadata_sha256']!=sha(run/'metadata.json') or not replay['passed'] or
            meta['manifest_sha256']!=replay['manifest_sha256'] or
            meta['trainer_sha256']!=replay['trainer_sha256'] or meta['config']['threshold']!=.5):
            raise ValueError('Matching fixed-cutoff completed replay required')
        milestones=receipt['models']['siamese']['milestones']
        if set(milestones)!={'1200','2400','3600','4800'}:
            raise ValueError('Incomplete learning history')
        for step in [1200,2400,3600,4800]:
            for split,count in [('train',958),('dev',198)]:
                record=milestones[str(step)][split]
                c=record['native_pooled']['confusion']
                if len(record['scenes'])!=count or c!={k:sum(s['native_confusion'][k] for s in record['scenes'].values()) for k in ['tp','fp','fn','tn']}:
                    raise ValueError('History membership or native totals differ')
                rows.append(dict(model=label,split=split,step=step,cutoff=.5,**c,
                    f1=2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1),
                    recall=c['tp']/max(c['tp']+c['fn'],1)))
        for name in ['metadata.json','receipt.json','e2e_receipt.json']:
            sources[str((run/name).resolve())]=sha(run/name)
    a.output.mkdir(parents=True)
    fig=Figure(figsize=(12,3.2),facecolor='#13242d',layout='constrained')
    axes=fig.subplots(1,2)
    for ax,key,title in zip(axes,['f1','recall'],['Plume F1 at fixed cutoff 0.50','Plume recall at fixed cutoff 0.50']):
        ax.set_facecolor('#13242d')
        for model,color in [('v8','#8ca6c4'),('v11','#44dfcc')]:
            for split,style in [('train','--'),('dev','-')]:
                data=[r for r in rows if r['model']==model and r['split']==split]
                ax.plot([r['step'] for r in data],[100*r[key] for r in data],style+'o',color=color,label=model+' '+split.upper())
        if key=='f1':
            ax.axhline(70,color='#e9b976',linewidth=1,label='70% target')
        ax.set(title=title,ylim=(0,100),xlabel='Training updates',ylabel='Percent')
        ax.tick_params(colors='#c8d8df')
        ax.xaxis.label.set_color('#c8d8df');ax.yaxis.label.set_color('#c8d8df');ax.title.set_color('#e4eef1')
        ax.grid(alpha=.12); ax.legend(fontsize=8,facecolor='#20333d',labelcolor='#e4eef1')
    fig.savefig(a.output/'learning.png',dpi=150,facecolor=fig.get_facecolor())
    result=dict(passed=True,source_sha256=sha(Path(__file__)),sources=sources,rows=rows,
        image_sha256=sha(a.output/'learning.png'),role='TRAIN and exposed DEV at .50; no independent transfer claim',test_array_access='none')
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'rows':len(rows),'image':str(a.output/'learning.png')}))


if __name__=='__main__':
    main()
