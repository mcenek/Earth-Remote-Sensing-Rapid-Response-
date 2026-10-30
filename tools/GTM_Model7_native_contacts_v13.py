"""Make repeatable visual-review pages from verified three-model outputs.

Every plume is retained. Additional pages contain the largest clear false alarms;
the untouched interactive report retains all clear observations as well.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):return 'n/a' if value is None else f'{value:.1%}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-label',default='Native model errors')
    p.add_argument('--ratio-label',default='Native + wind errors')
    a=p.parse_args();review=a.review.resolve();out=a.output.resolve()
    verification=json.loads((review/'verification.json').read_text())
    comparison=json.loads((review/'comparison.json').read_text())
    if not verification['passed'] or not verification['counts_recomputed']:
        raise ValueError('Verified rendered predictions required')
    methods=list(verification['publication'])
    if set(methods)!={'baseline','candidate','ratio'}:
        raise ValueError('All three saved model outputs required')
    bundles={k:json.loads((review/k/'experiment.json').read_text()) for k in methods}
    scenes=bundles['baseline']['scenes'];ids=[s['id'] for s in scenes]
    if any([s['id'] for s in bundles[k]['scenes']]!=ids for k in methods):
        raise ValueError('Model scene order differs')
    metrics={row['id']:row['models'] for row in comparison['scenes']}
    positives=[i for i,s in enumerate(scenes) if s['label']=='PLUME']
    clear=[i for i,s in enumerate(scenes) if s['label']=='NO_PLUME']
    clear.sort(key=lambda i:max(metrics[scenes[i]['id']][k]['fp'] for k in methods),reverse=True)
    us_positives=[i for i in positives if scenes[i]['name'].startswith('Plume | United States')]
    groups=[('all_plumes',positives),('largest_clear_false_alarms',clear[:10]),('us_plumes',us_positives)]
    out.mkdir(parents=True,exist_ok=False)
    try:
        regular=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
        bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',14)
    except OSError:
        regular=bold=ImageFont.load_default()
    pages=[]
    for group,indices in groups:
        for offset in range(0,len(indices),5):
            chosen=indices[offset:offset+5]
            image=Image.new('RGB',(1260,110+len(chosen)*260),'#0d191f');draw=ImageDraw.Draw(image)
            draw.text((14,10),verification['evaluation_role']+' | '+group,fill='#e4eef1',font=bold)
            draw.text((14,34),'Teal: overlap | Pink: extra prediction | Purple: missed label | Gray: not scored',fill='#a8c7d1',font=regular)
            headings=['Current Sentinel','Reference Sentinel','Native plume label','v11 errors',a.candidate_label,a.ratio_label]
            for col,label in enumerate(headings):draw.text((14+col*208,67),label,fill='#e4eef1',font=bold)
            for row,index in enumerate(chosen):
                s=scenes[index];y=98+row*260
                common=review/'baseline'/f'scene_{index:04d}'
                paths=[common/'current.png',common/'background.png',common/'annotation.png']+[review/k/f'scene_{index:04d}'/'errors.png' for k in methods]
                for col,path in enumerate(paths):
                    with Image.open(path) as tile:image.paste(tile.convert('RGB').resize((200,200),Image.Resampling.NEAREST),(14+col*208,y))
                draw.text((14,y+204),s['name'],fill='#e4eef1',font=regular)
                draw.text((14,y+224),s['id'],fill='#a8c7d1',font=regular)
                for col,k in enumerate(methods,start=3):
                    m=metrics[s['id']][k];f1=2*m['tp']/max(2*m['tp']+m['fp']+m['fn'],1)
                    plume_f1=f1 if m['tp']+m['fn'] else None
                    draw.text((14+col*208,y+204),f"F1 {percent(plume_f1)}  P {percent(m['precision'])}  R {percent(m['recall'])}",fill='#e4eef1',font=regular)
                    draw.text((14+col*208,y+224),f"TP {m['tp']}  FP {m['fp']}  FN {m['fn']}",fill='#a8c7d1',font=regular)
            path=out/f'{group}_{offset//5+1:02d}.png';image.save(path)
            pages.append(dict(path=str(path),sha256=sha(path),group=group,scene_ids=[scenes[i]['id'] for i in chosen]))
    (out/'receipt.json').write_text(json.dumps(dict(passed=True,source_sha256=sha(Path(__file__)),
        verification_sha256=sha(review/'verification.json'),all_positive_observations=len(positives),
        clear_observations_in_interactive_viewer=len(clear),pages=pages),indent=2))
    print(json.dumps(dict(pages=len(pages),positive_observations=len(positives),output=str(out))))


if __name__=='__main__':main()
