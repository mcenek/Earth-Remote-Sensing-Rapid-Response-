"""Build the meeting PDF from verified saved evidence, never model inference.

Before writing, require the evidence manifest and all selected panel files.
Render all pages after creation; do not infer layout quality from text extraction.
"""
from pathlib import Path
import json
import shutil
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'GTM_Viewer/friday_20261002'
OUTPUT=ROOT/'output/pdf/GTM_Model6_Friday_brief_20261002.pdf'
W,H=960,640
INK='#20392c';MUTED='#516359';GREEN='#315e4b';PAPER='#f6f5ef'


def paragraph(c,text,x,y,width,size=12,color=INK,bold=False):
    style=ParagraphStyle('body',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,
                         leading=size*1.36,textColor=HexColor(color),spaceAfter=0)
    p=Paragraph(text,style);_,height=p.wrap(width,1000);p.drawOn(c,x,H-y-height)
    return y+height


def image(c,path,x,y,width,height):
    obj=ImageReader(str(path));iw,ih=obj.getSize();ratio=min(width/iw,height/ih)
    dw,dh=iw*ratio,ih*ratio;c.drawImage(obj,x+(width-dw)/2,H-y-(height+dh)/2,width=dw,height=dh,mask='auto')


def page(c,number,title,subtitle):
    c.setFillColor(HexColor(PAPER));c.rect(0,0,W,H,fill=1,stroke=0)
    paragraph(c,'GTM / METHANE RESEARCH     |     JOSHUA KRASNOGOROV     |     OCTOBER 2, 2026',32,19,890,9,GREEN)
    paragraph(c,title,32,45,896,26,bold=True)
    paragraph(c,subtitle,32,85,896,12,MUTED)
    c.setStrokeColor(HexColor('#c7d1c8'));c.line(32,29,W-32,29)
    paragraph(c,'Prepared September 30. Saved evidence with limits; no new model trained.',32,H-22,780,8,MUTED)
    paragraph(c,f'{number} / 4',884,H-22,44,8,MUTED)


def panels(c,case,y=146,height=276):
    for i,p in enumerate(case['panels']):
        x=32+i*307;width=282
        paragraph(c,p['title'],x,y-25,width,12,bold=True)
        c.setFillColor(HexColor('#e8ede7'));c.rect(x,H-y-height,width,height,fill=1,stroke=0)
        if p['image']:
            image(c,DEMO/p['image'],x,y,width,height)
        else:
            paragraph(c,'No prediction for this scene',x+20,y+height*.34,width-40,16,bold=True)
            paragraph(c,'Data checks must pass before a compatible model is trained or evaluated here.',x+20,y+height*.52,width-40,11,MUTED)


def main():
    source=(DEMO/'assets/data.js').read_text(encoding='utf-8')
    data=json.loads(source.split('=',1)[1].strip().rstrip(';'))
    cases=data['cases']
    for case in cases:
        for p in case['panels']:
            if p['image'] and not (DEMO/p['image']).is_file():raise FileNotFoundError(p['image'])
    new=next(x for x in cases if x['kind']=='data')
    partial=next(x for x in cases if x['id'].endswith('8207778b'))
    miss=next(x for x in cases if x['id'].endswith('d303fa1d'))
    false=next(x for x in cases if x['id'].endswith('a8675c89'))
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(OUTPUT),pagesize=(W,H));c.setTitle('Methane research visual review for Martin');c.setAuthor('Joshua Krasnogorov')
    page(c,1,'A new pair we can inspect','Volusia landfill, Florida | June 1, 2025 | Sentinel observed 12.7 minutes after EMIT')
    panels(c,new)
    paragraph(c,'What I have now',32,444,896,16,bold=True)
    paragraph(c,'I can place the provider\'s EMIT plume annotation over actual Sentinel imagery and inspect the retrieval. The full annotation fits inside the crop. The five target bands are downloaded; the two missing SWIR bands were completed in a bounded 2.75 MiB pass.',32,472,896,12)
    paragraph(c,'This is a data example, not a model result. The reference is natively 60 m. A finer display does not create finer measured truth. Label-zero semantics, retrieval units and radiometric conversion still need verification; a temporal reference and paired negative are also missing.',32,530,896,11,MUTED)
    paragraph(c,'Source: UNEP-IMEO / MARS-Hyperspectral-EMIT-v2025 (CC-BY-NC-SA-4.0); Sentinel-2 via Earth Search. Source links and receipts accompany the offline review.',32,585,896,8,MUTED)
    c.showPage()
    page(c,2,'A localized result does not establish a working detector','Saved R5 mean prediction | November 1, 2023 | Fixed threshold 0.50 | Reused development data')
    panels(c,partial)
    paragraph(c,'This example overlaps part of the reviewed plume',32,444,896,16,bold=True)
    paragraph(c,'On the supported central area, this scene has 38.8% intersection-over-union (IoU). But the full 65-scene development run has only 2.15% pooled IoU and 2.25% precision. The run failed its quality gate. These images are diagnostic examples, not an estimate of general performance.',32,472,896,12)
    paragraph(c,'Cyan: reviewed reference boundary. Orange/red: saved positive prediction. Dimmed area: no model output or no valid comparison. Only the central 88 x 88 pixels of the 200 x 200 source are supported. The reference here is a MARS reviewed mask, not quantitative EMIT enhancement.',32,548,896,10,MUTED)
    c.showPage()
    page(c,3,'The misses and false positives matter','Both examples use the same saved model and cutoff. The interactive demo includes all three positives and two largest false-positive controls.')
    for row,case in enumerate((miss,false)):
        y=145+row*213
        paragraph(c,'Missed plume' if row==0 else 'Largest false-positive control',32,y-19,896,12,bold=True)
        for col,p in enumerate(case['panels']):
            x=32+col*307;image(c,DEMO/p['image'],x,y,282,171)
            paragraph(c,p['title'],x,y+174,282,9,MUTED)
    paragraph(c,'The missed scene recovers 0 of 461 labeled plume pixels. The no-plume control activates 869 of 7,744 evaluated pixels. These failures are why visual checks and reviewed controls are required before another training run or a US-wide scan.',32,566,896,11)
    c.showPage()
    page(c,4,'Martin\'s architecture and the decisions we need','The US screening goal stays the same. The next milestone needs trustworthy supervision and a clear acceptance test.')
    image(c,DEMO/'assets/GTM_Model6_architecture.png',30,112,615,393)
    y=124
    for title,body in [
        ('Agree on the first success criterion','Start with reliable plume regions and controlled false alarms; retain finer methane reconstruction as a separately validated goal.'),
        ('Obtain the label definitions','Can Martin connect us to the provider or exact processing documentation for mask zero, retrieval units, quality flags and reviewed negatives?'),
        ('Validate source locations later','Request independent source and wind information for the Gaussian-like origin fit. A landfill or CAFO coordinate alone is not source truth.')]:
        y=paragraph(c,title,674,y,250,12,bold=True)+7
        y=paragraph(c,body,674,y,250,11,MUTED)+21
    paragraph(c,'My next experiment',32,515,896,15,bold=True)
    paragraph(c,'After data checks pass: split by site, create eight rotation/mirror training views, train four scales independently, then compare a fixed average with a combiner trained on predictions from excluded sites. Review every positive, every miss, and the largest false positives before expanding.',32,542,896,11)
    c.showPage();c.save()
    shutil.copyfile(OUTPUT,DEMO/OUTPUT.name)
    print(json.dumps({'pdf':str(OUTPUT),'demo_copy':str(DEMO/OUTPUT.name),'pages':4}))


if __name__=='__main__':main()
