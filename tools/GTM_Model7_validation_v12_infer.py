"""Evaluate all frozen arms once, or independently replay saved CPU predictions.

Checks and failure controls are specified in the v12 protocol. The all-arm freeze
receipt must exist and match before any locked scene array can be opened.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import GTM_Model7_repaired_inference as factory
import GTM_Model7_calibrate_broader_v7 as c
from GTM_Model7_validation_v12_select import ARMS, augment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',choices=('dev','assessment','regression'),required=True)
    p.add_argument('--verify',action='store_true')
    a=p.parse_args(); root=a.root.resolve(); gate_path=root/'selection/all_arms_frozen.json'
    gate=json.loads(gate_path.read_text()); sha=factory.t.sha
    if not gate['passed'] or set(gate['freezes'])!=set(ARMS):
        raise ValueError('All arms must be frozen before locked access')
    for arm in ARMS:
        if sha(Path(gate['freezes'][arm]['path']))!=gate['freezes'][arm]['sha256']:
            raise ValueError('Frozen choice changed')
    manifest_path=root/'data/combined_manifest.json'
    if sha(manifest_path)!=gate['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    manifest=json.loads(manifest_path.read_text()); scenes=[s for s in manifest['scenes'] if s['split']==a.split]
    expected={'dev':(144,39),'assessment':(144,41),'regression':(198,46)}[a.split]
    if (len(scenes),sum(s['scene_kind']=='PLUME' for s in scenes))!=expected:
        raise ValueError('Evaluation membership changed')
    out=root/('evaluation_'+a.split)
    if not a.verify:
        out.mkdir(exist_ok=False)
        receipt=dict(passed=True,split=a.split,all_arms_frozen_sha256=sha(gate_path),
                     manifest_sha256=sha(manifest_path),models={},measurements={})
    else:
        receipt=json.loads((out/'receipt.json').read_text())
        if receipt['all_arms_frozen_sha256']!=sha(gate_path) or receipt['manifest_sha256']!=sha(manifest_path):
            raise ValueError('Evaluation receipt identity changed')
    replay={}
    for arm in ARMS:
        frozen=json.loads(Path(gate['freezes'][arm]['path']).read_text());meta=frozen['metadata']
        factory.check_implementation(meta);t=factory.t
        if sha(t.PROTOCOL)!=gate['protocol_sha256']:
            raise ValueError('Frozen protocol changed')
        choice=frozen['models']['siamese']; t.THRESHOLD=choice['threshold']
        checkpoint=Path(choice['checkpoint'])
        if sha(checkpoint)!=choice['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['normalization']!=meta['normalization'] or state['steps']!=choice['steps']:
            raise ValueError('Checkpoint normalization/step changed')
        model=t.new_model('siamese');model.load_state_dict(state['model']);model.eval()
        samples={s['id']:t.load_scene(manifest_path,s) for s in scenes}
        folder=out/arm
        if not a.verify:
            measured=t.evaluate(model,manifest_path,scenes,meta['normalization'],a.split,folder)
            receipt['models'][arm]=dict(**measured,threshold=choice['threshold'],
                checkpoint_sha256=choice['checkpoint_sha256'],freeze_sha256=gate['freezes'][arm]['sha256'])
            probabilities=c.load_probabilities(folder,scenes,samples,measured['scenes'])
            receipt['measurements'][arm]=augment(c.measure(probabilities,samples,scenes,choice['threshold'],choice['steps']),scenes)
        else:
            maximum=0.0
            for s in scenes:
                sid=s['id'];data=samples[sid];file=folder/(sid+'.npz');accepted=receipt['models'][arm]['scenes'][sid]
                if sha(file)!=accepted['probability_sha256']:
                    raise ValueError('Saved prediction changed')
                probability=t.predict(model,t.prepare(data,meta['normalization']))
                with np.load(file,allow_pickle=False) as z:
                    error=float(np.max(np.abs(probability-z['probability20'])));maximum=max(maximum,error)
                    native=np.repeat(np.repeat(probability,2,0),2,1)
                    if error>1e-6 or not np.array_equal(z['valid_native'],data['valid_native'].astype(bool)):
                        raise ValueError('CPU score/support replay differs')
                    nc=c.counts(native,data['label_native'],data['valid_native'].astype(bool),choice['threshold'])
                    if nc!=accepted['native_confusion']:
                        raise ValueError('CPU native counts differ')
            replay[arm]=dict(scenes=len(scenes),maximum_probability_error=maximum)
        print(json.dumps(dict(arm=arm,split=a.split,replay=a.verify,scenes=len(scenes))),flush=True)
    if a.verify:
        t.write_json(out/'replay.json',dict(passed=True,fresh_process=True,models=replay,
            receipt_sha256=sha(out/'receipt.json'),all_arms_frozen_sha256=sha(gate_path),source_sha256=sha(Path(__file__))))
    else:
        t.write_json(out/'receipt.json',receipt)


if __name__=='__main__':
    main()
