"""Infer a frozen checkpoint list and reproduce it in a separate process.

See docs/GTM_Model7_recovery_visual_acceptance.md. This tool never fits a model.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import GTM_Model7_train_temporal as old


def load_model(entry):
    if entry['architecture'] == 'stable_residual':
        spec=importlib.util.spec_from_file_location('stable_residual',entry['architecture_source'])
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model=module.new_model()
    else:
        model=old.new_model(entry['architecture'])
    state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    model.eval()
    return model,state['normalization']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--model',help='Infer only this frozen entry, preserving its training runtime')
    args=parser.parse_args()
    freeze=json.loads(args.freeze.read_text())
    if freeze['threshold'] != .5 or not freeze['configuration_frozen_before_confirmation']:
        raise ValueError('Missing confirmation freeze')
    for entry in freeze['models']:
        if old.sha(Path(entry['checkpoint'])) != entry['checkpoint_sha256'] or old.sha(Path(entry['architecture_source'])) != entry['architecture_source_sha256']:
            raise ValueError('Checkpoint or architecture changed after freeze')
    manifest_path=args.data/'manifest.json'
    manifest,scenes=old.load_manifest(manifest_path,inspect_test_labels=True)
    tests=[s for s in scenes if s['split']=='test']
    out=args.output
    if not args.verify:
        out.mkdir(parents=True,exist_ok=False)
    original=json.loads((out/'receipt.json').read_text()) if args.verify else None
    reports={}
    for entry in freeze['models']:
        if args.model and entry['id']!=args.model:
            continue
        model,norm=load_model(entry)
        folder=out/entry['id']
        if not args.verify:
            folder.mkdir()
        totals={k:0 for k in ('tp','fp','fn','tn')}
        rows={}
        for scene in tests:
            data=old.load_scene(manifest_path,scene,labels=True)
            p=old.infer(model,old.prepare(data,norm),data['valid'])
            native=np.repeat(np.repeat(p,2,0),2,1)
            counts=old.confusion(native,data['label_native'],data['valid_native'].astype(bool))
            file=folder/(scene['id']+'.npz')
            if args.verify:
                with np.load(file,allow_pickle=False) as z:
                    if not np.array_equal(p,z['probability20']) or not np.array_equal(native,z['probability_native']) or not np.array_equal(data['valid'],z['valid20']):
                        raise ValueError('Checkpoint replay mismatch: '+scene['id'])
                accepted=original['models'][entry['id']]['scenes'][scene['id']]
                if counts != accepted['counts'] or old.sha(file)!=accepted['prediction_sha256']:
                    raise ValueError('Prediction hash or counts changed')
            else:
                np.savez_compressed(file,probability20=p,probability_native=native,valid20=data['valid'])
            rows[scene['id']]={'counts':counts,'prediction_sha256':old.sha(file)}
            for key in counts:
                totals[key]+=counts[key]
        reports[entry['id']]={**entry,'scenes':rows,'native':old.score(totals),'normalization':norm}
    receipt={'schema':'GTM_Model7_frozen_confirmation_inference_v1','freeze_sha256':old.sha(args.freeze),'manifest_sha256':old.sha(manifest_path),'inference_source_sha256':old.sha(Path(__file__)),'torch_version':torch.__version__,'models':reports,'scene_count':len(tests),'training_performed':False}
    if args.verify:
        old.write_json(out/'replay.json',{'passed':True,'model_count':len(reports),'scenes_per_model':len(tests),'exact_saved_prediction_replay':True,'native_counts_reproduced':True,'receipt_sha256':old.sha(out/'receipt.json')})
    else:
        old.write_json(out/'receipt.json',receipt)
    print(json.dumps({'verified':args.verify,'models':{k:v['native'] for k,v in reports.items()}},indent=2))

if __name__=='__main__':
    main()
