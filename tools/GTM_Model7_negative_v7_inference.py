"""Frozen baseline/selected U-Nets on reserved no-plume sites only.

Acceptance and failure modes: negative-only-test protocol, written first.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import GTM_Model7_repaired_inference as factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    def sha(value):
        return factory.t.sha(Path(value))
    gate = json.loads(a.gate.read_text(encoding='utf-8'))
    for item in ('selection', 'reservation', 'baseline', 'selected'):
        record = gate[item]
        if sha(record['path']) != record['sha256']:
            raise ValueError('Final gate drift: '+item)
    selection = json.loads(Path(gate['selection']['path']).read_text(encoding='utf-8'))
    if not selection['eligible'] or not selection['promoted_for_final_evaluation']:
        raise ValueError('Eligible improved DEV selection required')
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    scenes = manifest['scenes']
    if not scenes or any(s['scene_kind'] != 'NO_PLUME' or s['split'] != 'test' for s in scenes):
        raise ValueError('Only frozen no-plume TEST observations are allowed')
    # The preparation receipt pins the reservation and independently replays
    # source TIFFs. The root gate predates any final source decoding.
    prepared = json.loads((a.manifest.parent/'e2e_receipt.json').read_text(encoding='utf-8'))
    if not prepared['passed'] or prepared['reservation_sha256'] != gate['reservation']['sha256']:
        raise ValueError('Verified negative-only preparation required')
    if prepared['manifest_sha256'] != sha(a.manifest):
        raise ValueError('Prepared manifest changed')
    identity = dict(gate_sha256=sha(a.gate), manifest_sha256=sha(a.manifest),
                    reservation_sha256=gate['reservation']['sha256'])
    if a.verify:
        receipt = json.loads((a.output/'receipt.json').read_text(encoding='utf-8'))
        if any(receipt[k] != v for k, v in identity.items()):
            raise ValueError('Saved final inference identity changed')
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        receipt = dict(schema='GTM_Model7_negative_only_inference_v7', **identity,
                       role='fresh_no_plume_sites_only_no_recall_claim',
                       inference_sha256=sha(Path(__file__)), models={})
    maximum = 0.
    for key in ('baseline', 'selected'):
        frozen = json.loads(Path(gate[key]['path']).read_text(encoding='utf-8'))
        meta, choice = frozen['metadata'], frozen['models']['siamese']
        factory.check_implementation(meta)
        t = factory.t
        t.set_branch(meta['config']['branch'])
        if sha(meta['manifest']) != meta['manifest_sha256']:
            raise ValueError('Fit manifest changed')
        fit = json.loads(Path(meta['manifest']).read_text(encoding='utf-8'))['scenes']
        fit = [s for s in fit if s['split'] in ('train', 'dev')]
        for field in ('id', 'physical_location_id'):
            if {s[field] for s in fit} & {s[field] for s in scenes}:
                raise ValueError('Final site/observation overlap')
        if sha(choice['checkpoint']) != choice['checkpoint_sha256']:
            raise ValueError('Final weights changed')
        checkpoint = torch.load(choice['checkpoint'], map_location='cpu', weights_only=False)
        if checkpoint['normalization'] != meta['normalization'] or checkpoint['steps'] != choice['steps']:
            raise ValueError('Checkpoint preprocessing mismatch')
        model = t.new_model('siamese')
        model.load_state_dict(checkpoint['model'])
        cutoff = choice.get('threshold', frozen['threshold'])
        t.THRESHOLD = cutoff
        counts = dict(tp=0, fp=0, fn=0, tn=0)
        records = {}
        folder = a.output/key
        if not a.verify:
            folder.mkdir()
        for scene in scenes:
            if t.sha(a.manifest.parent/scene['array_file']) != scene['array_sha256']:
                raise ValueError('Final array drift')
            d = t.load_scene(a.manifest, scene)
            if d['label_native'].any() or d['target'].any():
                raise ValueError('Unexpected positive label')
            coarse = t.predict(model, t.prepare(d, meta['normalization']))
            native = np.repeat(np.repeat(coarse, t.native_factor(), 0), t.native_factor(), 1)
            c = t.confusion(native, d['label_native'], d['valid_native'])
            path = folder/(scene['id']+'.npz')
            if a.verify:
                expected = receipt['models'][key]['scenes'][scene['id']]
                if sha(path) != expected['probability_sha256'] or c != expected['native_confusion']:
                    raise ValueError('Final saved counts changed')
                with np.load(path, allow_pickle=False) as z:
                    maximum = max(maximum, float(np.abs(z['probability20']-coarse).max()))
                    if not np.array_equal(z['probability_native'], native) or not np.array_equal(z['valid_native'], d['valid_native']):
                        raise ValueError('Final fresh-process replay differs')
            else:
                np.savez_compressed(path, probability20=coarse, probability_native=native,
                                    valid20=d['valid'], valid_native=d['valid_native'])
            records[scene['id']] = dict(native_confusion=c, probability_sha256=sha(path))
            for k, value in c.items():
                counts[k] += value
        if not a.verify:
            receipt['models'][key] = dict(checkpoint_sha256=choice['checkpoint_sha256'],
                cutoff=cutoff, scenes=records, native_confusion=counts)
        print(json.dumps(dict(model=key, scenes=len(scenes), replay=a.verify)), flush=True)
    if a.verify:
        factory.t.write_json(a.output/'replay.json', dict(passed=True, **identity,
            receipt_sha256=sha(a.output/'receipt.json'), maximum_probability_error=maximum,
            fresh_process=True))
    else:
        factory.t.write_json(a.output/'receipt.json', receipt)


if __name__ == '__main__':
    main()
