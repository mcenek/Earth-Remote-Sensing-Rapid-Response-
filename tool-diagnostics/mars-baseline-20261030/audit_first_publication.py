"""Independently check saved review scores, live assets and checkpoint replay."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from evaluate_released_marss2l import ReleasedUNet


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / 'outputs/GTM_Model8_mars_baseline_20261030'
    review = run/'review/first_pass/holdout'
    summary = json.loads((review/'summary.json').read_text())
    records = json.loads((review/'scene_records.json').read_text())['scenes']
    frozen = set(json.loads((ROOT/'tool-diagnostics/mars-baseline-20261030/final_candidate_ids.json').read_text()))
    assert {row['id'] for row in records} == frozen and len(records) == 60
    receipt = {'scenes':len(records), 'methods':{}, 'live_assets':0, 'replay':[]}
    for method in ('baseline','released_reference'):
        native = [0,0,0,0]
        negatives = {'scenes':0, 'activated':0}
        all_y, all_p = [], []
        for row in records:
            path = review/method/row['id']/'prediction.npz'
            assert digest(path) == row['models'][method]['prediction_sha256']
            with np.load(path, allow_pickle=False) as saved:
                mask, score, label = saved['valid'].astype(bool), saved['probability'], saved['target']
                assert score.shape == (200,200) and np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
                y, p = label[mask].astype(bool), score[mask] >= .5
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[False,True]).ravel()
            expected = row['models'][method]
            assert [int(tn),int(fp),int(fn),int(tp)] == [expected[k] for k in ('tn','fp','fn','tp')]
            native = [a+int(b) for a,b in zip(native,(tn,fp,fn,tp))]
            all_y.append(y)
            all_p.append(p)
            if row['source_row']['isplume'].lower() != 'true':
                negatives['scenes'] += 1
                negatives['activated'] += int(fp > 0)
        f1 = float(f1_score(np.concatenate(all_y), np.concatenate(all_p), zero_division=0))
        assert abs(f1-summary['pooled'][method]['f1']) < 1e-12
        assert negatives['activated'] == summary['negative'][method]['activated']
        receipt['methods'][method] = {'f1':f1, 'tn_fp_fn_tp':native, 'negative_scenes':negatives}
    for role in ('validation','holdout'):
        publication = json.loads((run/'review/first_pass'/role/'publication.json').read_text())
        for published in publication.values():
            manifest = Path(published['manifest'])
            data = json.loads(manifest.read_text())
            references = {data['reportUrl']}
            for scene in data['scenes']:
                for key in ('rgb','reference90Rgb','truthImage','predictionImage','disagreementImage','grid'):
                    references.add(scene[key])
            for relative in references:
                path = manifest.parent/relative
                with urlopen('http://127.0.0.1:8766/bundle-assets/'+data['id']+'/'+relative, timeout=20) as response:
                    assert response.status == 200
                    assert hashlib.sha256(response.read()).hexdigest() == digest(path)
                receipt['live_assets'] += 1
    torch.set_num_threads(2)
    checkpoint = run/'fit/epoch_000.pt'
    assert digest(checkpoint) == summary['checkpoint_sha256']
    model = ReleasedUNet(16).eval()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    # Replay both US plumes and all four US negative observations in a fresh process.
    us = [row for row in records if row['source_row']['country'] == 'United States of America']
    assert len(us) == 6
    for row in us:
        with np.load(review/'baseline'/row['id']/'prediction.npz', allow_pickle=False) as saved:
            with torch.no_grad():
                actual = torch.sigmoid(model(torch.from_numpy(saved['input'][None])))[0].numpy()
            assert np.array_equal(actual, saved['probability'])
        receipt['replay'].append(row['id'])
    receipt['state'] = 'passed'
    (review/'independent_root_audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
