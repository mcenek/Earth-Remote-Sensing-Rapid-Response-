"""Contact sheets of actual, independently verified pair-review exports."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    verify=json.loads((a.review/'verification.json').read_text())
    if not verify['passed'] or not verify['counts_recomputed']:
        raise ValueError('Verified real-data review required')
    methods={k:json.loads((a.review/k/'experiment.json').read_text()) for k in ('baseline','candidate')}
    if [s['id'] for s in methods['baseline']['scenes']] != [s['id'] for s in methods['candidate']['scenes']]:
        raise ValueError('Compared scene order differs')
    rows=methods['baseline']['scenes']
    a.output.mkdir(parents=True,exist_ok=False)
    assets={}
    for start in range(0,len(rows),8):
        page=Image.new('RGB',(1050,8*245+65),'#101e25');draw=ImageDraw.Draw(page)
        draw.text((10,8),'EXPOSED BROADER DEVELOPMENT. No independent validation claim.',fill='white')
        for column,title in enumerate(('Current','Reference','Annotation','Baseline errors','Candidate errors')):
            draw.text((10+210*column,33),title,fill='white')
        for row,s in enumerate(rows[start:start+8]):
            number=start+row
            base=a.review/'baseline'/f'scene_{number:04d}'
            candidate=a.review/'candidate'/f'scene_{number:04d}'
            paths=[base/'current.png',base/'background.png',base/'annotation.png',base/'errors.png',candidate/'errors.png']
            y=60+245*row
            draw.text((10,y),s['name'],fill='white')
            for column,path in enumerate(paths):
                assets[str(path)]=sha(path)
                with Image.open(path) as im:
                    page.paste(im.convert('RGB').resize((200,200),Image.Resampling.NEAREST),(10+210*column,y+20))
            for column,k in ((3,'baseline'),(4,'candidate')):
                m=methods[k]['scenes'][number]['nativeMetrics']
                draw.text((10+210*column,y+222),f'TP {m["tp"]} FP {m["fp"]}',fill='white')
        page.save(a.output/f'contact_{start//8:03d}.png')
    (a.output/'receipt.json').write_text(json.dumps(dict(passed=True,source_sha256=sha(Path(__file__)),
        review_verification_sha256=sha(a.review/'verification.json'),scenes=len(rows),assets=assets),indent=2))
    print(json.dumps(dict(scenes=len(rows),pages=(len(rows)+7)//8)))


if __name__=='__main__':
    main()
