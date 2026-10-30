"""Publish a frozen before/after pair on identical confirmation scenes.

Failure cases and end-to-end acceptance preceded implementation in
docs/GTM_Model7_recovery_visual_acceptance.md.
"""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import GTM_Model7_review_temporal as v

ROOT=v.ROOT
RUN=ROOT/'outputs/GTM_Model7_recovery_20261023'
KEYS=('residual','siamese')  # DOM slots only, actual architecture comes from receipt.

def main():
    parser=argparse.ArgumentParser()
    for name in ('data','inference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--before',required=True)
    parser.add_argument('--after',required=True)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    receipt=json.loads((args.inference/'receipt.json').read_text())
    replay=json.loads((args.inference/'replay.json').read_text())
    if not replay['passed'] or v.sha(args.inference/'receipt.json')!=replay['receipt_sha256']:
        raise ValueError('Independent checkpoint replay required')
    manifest=json.loads((args.data/'manifest.json').read_text())
    if v.sha(args.data/'manifest.json')!=receipt['manifest_sha256']:
        raise ValueError('Confirmation manifest changed')
    tests=sorted((s for s in manifest['scenes'] if s['split']=='test'),key=lambda s:(s['scene_kind']!='PLUME',s['id']))
    selected=dict(zip(KEYS,(args.before,args.after)))
    entries={m:receipt['models'][selected[m]] for m in KEYS}
    names={m:entries[m]['name'] for m in KEYS}
    ids={m:'GTM_Model7_recovery_20261023_'+selected[m] for m in KEYS}
    immutable={str(p):v.sha(p) for p in args.inference.rglob('*') if p.is_file()}
    for m in KEYS:
        e=entries[m]
        if v.sha(e['checkpoint'])!=e['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
    args.output.mkdir(parents=True,exist_ok=False)
    for m in KEYS:
        (args.output/m).mkdir()
    totals={m:dict(tp=0,fp=0,fn=0,tn=0) for m in KEYS}
    negatives={m:dict(scenes=0,activated=0,fp=0,tn=0) for m in KEYS}
    controls={m:dict(tp=0,fp=0,fn=0,tn=0) for m in ('all_positive','all_negative')}
    cases=[]; records=[]; scene_infos={m:[] for m in KEYS}; rows=[]; sheet_rows=[]
    for i,s in enumerate(tests):
        file=args.data/s['array_file']
        if v.sha(file)!=s['array_sha256']:
            raise ValueError('Confirmation array changed')
        with np.load(file,allow_pickle=False) as z:
            d={k:z[k] for k in z.files}
        for source in s['sources']:
            if v.sha(source['path'])!=source['sha256']:
                raise ValueError('Source TIFF changed')
        grid=v.original_arrays(s,d)
        rgb,background,stretch=v.paired_rgb(d['target_image'],d['reference_image'],d['valid'])
        label,valid=d['label_native'],d['valid_native'].astype(bool)
        annotation=rgb.copy()
        positive=(label==1)&valid
        annotation[positive]=(annotation[positive]*.35+np.array((46,224,219))*.65).astype(np.uint8)
        kind=s['scene_kind']; title=('Plume' if kind=='PLUME' else 'No plume')+' | '+s['satellite_time'][:10]+' | '+s['id'].replace('MARS_','')[:8]
        figures={}; metrics={}; arrays={}
        for control in controls:
            c=v.counts(np.ones_like(label) if control=='all_positive' else np.zeros_like(label),label,valid)
            for k in c: controls[control][k]+=c[k]
        record=dict(id=s['id'],kind=kind,models={})
        for m in KEYS:
            pred=args.inference/selected[m]/(s['id']+'.npz')
            accepted=entries[m]['scenes'][s['id']]
            if v.sha(pred)!=accepted['prediction_sha256']:
                raise ValueError('Inference array changed')
            with np.load(pred,allow_pickle=False) as z:
                p=z['probability_native']; coarse=z['probability20']
                if not np.array_equal(z['valid20'],d['valid']) or not np.array_equal(p,np.repeat(np.repeat(coarse,2,0),2,1)):
                    raise ValueError('Support or model grid changed')
            if not np.isfinite(p[valid]).all() or np.any((p[valid]<0)|(p[valid]>1)):
                raise ValueError('Invalid model scores')
            c=v.counts(p,label,valid)
            if c!=accepted['counts']: raise ValueError('Native confusion differs')
            measured=v.ratios(c); metrics[m]=measured; arrays[m]=p
            for k in c: totals[m][k]+=c[k]
            if kind=='NO_PLUME':
                n=negatives[m]; n['scenes']+=1; n['activated']+=int(c['fp']>0); n['fp']+=c['fp']; n['tn']+=c['tn']
            folder=args.output/m/f'scene_{i:03d}'; folder.mkdir()
            painted,error,heat=v.overlay(rgb,p,label,valid)
            for name,a in dict(current=rgb,background=background,annotation=annotation,overlay=painted,errors=error,score=heat).items():
                Image.fromarray(a).save(folder/(name+'.png'))
            figures[m]={name:v.image_uri(folder/(name+'.png')) for name in ('overlay','errors','score')}
            bounds=v.map_assets(folder,rgb,p,label,valid,grid)
            ref=s['reference_scene_id'].split('_')[2][:8]
            ref=ref[:4]+'-'+ref[4:6]+'-'+ref[6:8]
            outcome='False positive' if kind=='NO_PLUME' and c['fp'] else 'Correct negative' if kind=='NO_PLUME' else 'Missed plume' if c['tp']==0 else 'Partial overlap'
            scene_infos[m].append(dict(id=s['id'],name=title,task='plume_segmentation',bounds=bounds,date=s['satellite_time'][:10],referenceDate=ref,
                rgb=folder.name+'/satellite.png',predictionImage=folder.name+'/prediction.png',truthImage=folder.name+'/truth.png',grid=folder.name+'/grid.json',
                predictionAvailable=True,decisionThreshold=.5,dataKind='measured',dataset='MARS',label=kind,split='Frozen official-test geographic confirmation',
                notes='All configuration choices were frozen before these rasters were opened. More than 100 km from every original cohort scene; no shared target/reference acquisition. Three US sites only.',
                source=s['target_scene_id'],predictionHash=v.sha(pred),checkpointHash=entries[m]['checkpoint_sha256'],outcome=outcome,
                truthLabel='MARS reviewed annotation' if kind=='PLUME' else 'Reviewed no plume',truthNote='Delivered 10 m mask; actual 20 m output repeats without smoothing.',
                rgbNote='Real Sentinel-2 L1C target; display-only contrast stretch',resolution='20 m model output, scored on delivered 10 m annotation',
                qualityFlags=['held_out_from_this_model','three_confirmation_sites_only','reference_cloud_mask_unavailable'],
                nativeMetrics={**measured,'source_valid_pixels':int(valid.sum()),'support_fraction':float(valid.mean()),'claim':'Frozen geographic confirmation'}))
            record['models'][selected[m]]={**measured,'prediction_sha256':v.sha(pred)}
        records.append(record)
        common=args.output/'residual'/f'scene_{i:03d}'
        cases.append(dict(id=s['id'],title=title,kind=kind,current=v.image_uri(common/'current.png'),background=v.image_uri(common/'background.png'),annotation=v.image_uri(common/'annotation.png'),models=figures,metrics=metrics,group=s['group_id'],source=s['target_scene_id'],reference=s['reference_scene_id'],target_time=s['satellite_time'],reference_time=s['reference_time'],later_reference=s['target_minus_reference_hours']<0,rgb_display_ranges=stretch))
        rows.append('<tr><td>'+html.escape(title)+'</td>'+''.join(f'<td>{v.percent(metrics[m]["iou"])}</td><td>{metrics[m]["fp"]:,}</td>' for m in KEYS)+'</tr>')
        sheet_rows.append((title,[rgb,annotation]+[v.overlay(rgb,arrays[m],label,valid)[0] for m in KEYS]))
    pooled={m:v.ratios(totals[m]) for m in KEYS}
    for m in KEYS:
        n=negatives[m]; n['pixel_fpr']=n['fp']/(n['fp']+n['tn']) if n['fp']+n['tn'] else None
        if pooled[m]['tp']!=entries[m]['native']['confusion']['tp']:
            raise ValueError('Aggregate count mismatch')
    summary=dict(manifest_sha256=v.sha(args.data/'manifest.json'),pooled=pooled,negatives=negatives,controls={k:v.ratios(c) for k,c in controls.items()},scenes=records,test_scenes=len(tests),positive_test_scenes=sum(s['scene_kind']=='PLUME' for s in tests),quality_exclusions=manifest['quality_exclusion_rows'],zero_new_imagery_downloads=True)
    v.save(args.output/'comparison.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    sheet=Image.new('RGB',(1040,len(sheet_rows)*254),(13,25,31)); draw=ImageDraw.Draw(sheet)
    for i,(title,images) in enumerate(sheet_rows):
        y=i*254; draw.text((12,y+5),title,font=font,fill='#e5eff2')
        for j,(a,name) in enumerate(zip(images,['Current Sentinel','Reviewed annotation','Original checkpoint','Revised checkpoint'])):
            sheet.paste(Image.fromarray(a).convert('RGB').resize((200,200),Image.Resampling.NEAREST),(12+j*260,y+28)); draw.text((12+j*260,y+230),name,font=font,fill='#bdd0d7')
    sheet.save(args.output/'all_scenes.png'); sheet.crop((0,0,1040,min(6,len(sheet_rows))*254)).save(args.output/'comparison_preview.png')
    options=''.join(f'<option value="{i}">{html.escape(c["title"])}</option>' for i,c in enumerate(cases))
    stats=''.join(f'<tr><td>{html.escape(names[m])}</td><td>{v.percent(pooled[m]["iou"])}</td><td>{v.percent(pooled[m]["precision"])}</td><td>{v.percent(pooled[m]["recall"])}</td><td>{negatives[m]["activated"]}/{negatives[m]["scenes"]}</td><td>{v.percent(negatives[m]["pixel_fpr"])}</td></tr>' for m in KEYS)
    body=v.HTML
    start=body.index('<div class="scope">'); end=body.index('<div class="controls">',start)
    body=body[:start]+'<div class="scope"><strong>Frozen confirmation, not a tuning set.</strong> These are three US sites more than 100 km from every original training, development and test scene. We fixed all revisions before opening these rasters. The complete selection contains three plume observations and 24 reviewed no-plume observations. Quality exclusions: '+html.escape(json.dumps(manifest['quality_exclusions']))+'. This is a small retrospective test. It cannot establish national performance.</div>\n'+body[end:]
    start=body.index('<details><summary>Training and evidence limits</summary>'); end=body.index('</details>',start)+len('</details>')
    body=body[:start]+'''<details><summary>Training and evidence limits</summary><p>Input band order, native label alignment and source hashes were independently checked. Training uses 32 plume and 120 reviewed no-plume observations, but the positive training scenes cover only eight physical sites in one broad geographic component. Development has negatives only. This limits model selection and generalization.</p><p>The original residual network failed a fixed four-crop memorization check. The original U-Net passed that check, but its broader predictions followed field boundaries. Recovery trials reduced the positive loss weight, increased updates, tested more same-facility negative crops, and separately tested a normalized residual network. All trial choices used training and development only. The exact frozen checkpoint and configuration are included in the inference receipt.</p><p>Some references were acquired after their target observation. A separately verified reference cloud mask is unavailable. These are MARS reviewed Sentinel plume labels, not NASA EMIT measurements, methane emission rates or fine-resolution methane reconstruction. A no-plume observation does not prove that a site never emits. Three confirmation sites cannot represent the US. Kansas is geographically unseen to these fitted models, but prior project exposure is not excluded. No new imagery was downloaded.</p></details>'''+body[end:]
    body=body.replace('Two models.<br>The same unseen scenes.','Recovery check.<br>Did the predictions improve?').replace('Real temporal models | options 1 and 2','Methane model recovery | real confirmation scenes')
    pstart=body.index('<p>Option 1 learns'); pend=body.index('</p>',pstart)+4
    body=body[:pstart]+'<p>Compare the original checkpoint and a frozen revision on exactly the same scenes. Orange is the saved model prediction at 0.50. Cyan is the reviewed annotation. Use the error display to see false alarms and missed plume pixels.</p>'+body[pend:]
    body=body.replace('Open option 1 on map','Open original on map').replace('Open option 2 on map','Open revision on map').replace('1 / Residual CNN',html.escape(names['residual'])).replace('2 / Temporal U-Net',html.escape(names['siamese']))
    body=body.replace('Temporal residual CNN saved prediction',html.escape(names['residual'])+' saved prediction').replace('Temporal U-Net saved prediction',html.escape(names['siamese'])+' saved prediction')
    body=body.replace('No new imagery was downloaded.</p></details>','No new imagery was downloaded.</p><p>Runtime audit: the stabilized residual was fitted with PyTorch 2.11; original models and U-Net revisions used PyTorch 2.4. Replaying the frozen stabilized checkpoint in its training runtime reproduced every confirmation score exactly, so the displayed results are unchanged. The before/after comparison combines training and model changes; it does not isolate architecture alone.</p></details>')
    body=body.replace('Residual IoU','Original IoU').replace('Residual FP cells','Original FP cells').replace('U-Net IoU','Revision IoU').replace('U-Net FP cells','Revision FP cells')
    body=body.replace('Fitting progress','Can the pipeline learn a known example?').replace('__TRAINING__. Both curves show training loss only. A falling curve does not establish success on unseen scenes.','The four fixed real training crops below compare the original residual CNN and U-Net after 300 updates. This is memorization evidence only. It is not a held-out result. The U-Net fits the label while the residual CNN fails.').replace('Training loss by fixed epoch for both temporal models','Four fixed training crops before and after fitting')
    body=body.replace("'/?catalog=all&experiment=GTM_Model7_temporal_20261023_'+m+'&scene='","'/?catalog=all&experiment='+"+json.dumps(ids)+"[m]+'&scene='")
    for marker,value in dict(OPTIONS=options,STATS=stats,ROWS=''.join(rows),CASES=json.dumps(cases).replace('</','<\\/'),SUMMARY=json.dumps(summary).replace('</','<\\/'),CURVES=v.image_uri(RUN/'probe_visual/memorization.png')).items():
        body=body.replace('__'+marker+'__',value)
    for m in KEYS:
        (args.output/m/'index.html').write_text(body,encoding='utf-8')
        experiment=dict(id=ids[m],name=names[m],lifecycle='experimental',priority=False,dataKind='measured',architectureFamily='GTM_Model7_recovery',runGroup='GTM_Model7_recovery_20261023',status='FROZEN REAL GEOGRAPHIC CONFIRMATION',description=f'IoU {v.percent(pooled[m]["iou"])}; {negatives[m]["activated"]}/{negatives[m]["scenes"]} no-plume observations activate.',notes='Configuration frozen before opening confirmation rasters. Three US sites only. All failures remain visible.',decisionThreshold=.5,nativeAggregate={**pooled[m],'scenes':len(tests)},metrics={'Confirmation IoU':v.percent(pooled[m]['iou']),'No-plume false alarms':f'{negatives[m]["activated"]}/{negatives[m]["scenes"]}','Threshold':'0.50 fixed'},scenes=scene_infos[m],reportUrl='index.html')
        v.save(args.output/m/'experiment.json',experiment); v.load_bundle(args.output/m/'experiment.json','bundle-assets/')
    (args.output/'index.html').write_text(body,encoding='utf-8')
    if immutable!={str(p):v.sha(p) for p in args.inference.rglob('*') if p.is_file()}:
        raise ValueError('Rendering changed inference evidence')
    verification=dict(passed=True,original_confirmation_rasters_rebuilt=len(tests),native_counts_independently_recomputed=True,prediction_repeat_verified=True,model_artifacts_unchanged=True,manifest_sha256=v.sha(args.data/'manifest.json'),renderer_sha256=v.sha(__file__))
    if args.publish:
        verification['publication']={m:v.publish(args.output/m,ROOT/'outputs/GTM_viewer_bundles') for m in KEYS}
    v.save(args.output/'verification.json',verification)
    print(json.dumps(dict(pooled=pooled,negatives=negatives,verification=verification),indent=2))

if __name__=='__main__':main()
