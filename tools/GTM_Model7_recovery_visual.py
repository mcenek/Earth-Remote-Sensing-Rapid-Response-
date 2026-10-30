"""Render stable-ID TRAIN-only diagnostics. Protocol: recovery_visual_acceptance.md."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import GTM_Model7_review_temporal as view

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/GTM_Model7_recovery_20261023'
DATA = ROOT / 'outputs/GTM_Model7_temporal_20261023/data'

def probe_sheet():
    folder=RUN/'probe_visual'
    receipt=json.loads((folder/'receipt.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    sheet=Image.new('RGB',(1100,1060),(13,25,31))
    draw=ImageDraw.Draw(sheet)
    draw.text((12,10),'TRAIN MEMORIZATION CHECK | same four fixed real crops, 300 steps | not unseen-scene evidence',font=font,fill='#e5eff2')
    arrays={}
    for name in ('residual','siamese'):
        file=folder/(name+'_probe.npz')
        assert view.sha(file)==receipt['models'][name]['array_sha256']
        with np.load(file) as z:
            arrays[name]={k:z[k] for k in z.files}
    for i in range(4):
        d=arrays['siamese']
        # Shared per-crop RGB display stretch only; prediction cutoff stays 0.5.
        current=np.moveaxis(d['image12'][i,:6],0,-1)
        valid=d['valid'][i].astype(bool)
        rgb=np.stack([np.clip((current[...,b]-np.percentile(current[...,b][valid],2))/max(np.ptp(np.percentile(current[...,b][valid],[2,98])),1e-8),0,1) for b in (2,1,0)],-1)
        rgb=(rgb*255).astype(np.uint8)
        truth=d['target'][i]
        annotation=rgb.copy()
        annotation[(truth==1)&valid]=(46,224,219)
        images=[rgb,annotation,view.overlay(rgb,arrays['residual']['probability_final'][i],truth,valid)[0],view.overlay(rgb,d['probability_initial'][i],truth,valid)[0],view.overlay(rgb,d['probability_final'][i],truth,valid)[0]]
        y=60+i*250
        draw.text((12,y),('PLUME' if i in (0,2) else 'NO PLUME')+' | '+receipt['draws'][i][0],font=font,fill='#bdd0d7')
        for j,(a,title) in enumerate(zip(images,['Current crop','Reviewed label','Residual after 300','U-Net before','U-Net after 300'])):
            sheet.paste(Image.fromarray(a).convert('RGB').resize((196,196),Image.Resampling.NEAREST),(12+j*220,y+26))
            draw.text((12+j*220,y+225),title,font=font,fill='#bdd0d7')
    sheet.save(folder/'memorization.png')

def main():
    manifest = json.loads((DATA/'manifest.json').read_text())
    positive = sorted((s for s in manifest['scenes'] if s['split']=='train' and s['source_row']['label_state']=='PLUME'), key=lambda s:s['id'])[:6]
    negative = sorted((s for s in manifest['scenes'] if s['split']=='train' and s['source_row']['label_state']=='NO_PLUME'), key=lambda s:s['id'])[:2]
    out = RUN/'train_visual'
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    rows = positive+negative
    sheet = Image.new('RGB',(1300,70+len(rows)*250),(13,25,31))
    draw = ImageDraw.Draw(sheet)
    draw.text((12,10),'TRAINING SCENES ONLY | stable IDs, not best cases | orange prediction / cyan label boundary',font=font,fill='#e5eff2')
    columns = ['Current Sentinel','Reviewed annotation','Original U-Net, w10 / 360','Revised U-Net, w2 / 1800','Revised error map']
    for i,s in enumerate(rows):
        with np.load(DATA/s['array_file']) as z:
            d={k:z[k] for k in z.files}
        view.original_arrays(s,d)
        rgb,_,_=view.paired_rgb(d['target_image'],d['reference_image'],d['valid'])
        label,valid=d['label_native'],d['valid_native']
        annotation=rgb.copy()
        positive_mask=(label==1)&valid
        annotation[positive_mask]=(annotation[positive_mask]*.35+np.array((46,224,219))*.65).astype(np.uint8)
        ps=[]
        for folder in [RUN/'train_diagnostics/siamese_w10/train_predictions_step_360',RUN/'stability_followup/siamese_w2/train_predictions_step_1800']:
            with np.load(folder/(s['id']+'.npz')) as z:
                assert np.array_equal(z['valid20'],d['valid'])
                ps.append(np.repeat(np.repeat(z['probability20'],2,0),2,1))
        images=[rgb,annotation,view.overlay(rgb,ps[0],label,valid)[0],view.overlay(rgb,ps[1],label,valid)[0],view.overlay(rgb,ps[1],label,valid)[1]]
        y=70+i*250
        draw.text((12,y),s['source_row']['label_state']+' | '+s['id'],font=font,fill='#bdd0d7')
        for j,(a,title) in enumerate(zip(images,columns)):
            sheet.paste(Image.fromarray(a).convert('RGB').resize((200,200),Image.Resampling.NEAREST),(12+j*260,y+25))
            draw.text((12+j*260,y+226),title,font=font,fill='#bdd0d7')
    sheet.save(out/'train_contact_sheet.png')
    sheet.crop((0,0,1300,820)).save(out/'train_preview.png')
    view.save(out/'receipt.json',dict(training_only=True,selection='first six positive and first two negative TRAIN IDs',scene_ids=[s['id'] for s in rows],manifest_sha256=view.sha(DATA/'manifest.json')))

if __name__=='__main__':
    probe_sheet()
    main()
