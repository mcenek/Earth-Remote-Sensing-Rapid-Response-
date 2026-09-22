"""Recompute historical MARS headline AP and mask counts, without inference/tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / 'Earth-Remote-Sensing-Rapid-Response-'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    report = json.loads((ROOT/'reports/experiments/mars_spatial_prithvi_ensemble_paper_posttest.json').read_text())
    protocol = json.loads((ROOT/'configs/mars_spatial_prithvi_ensemble_paper_protocol.json').read_text())
    diag_path = ASSETS/protocol['inputs']['diagnostic']['path']
    assert digest(diag_path)==protocol['inputs']['diagnostic']['sha256']
    score_record = report['label_free_score_cache']
    score_path = ASSETS/score_record['path']
    assert digest(score_path)==score_record['sha256']
    with np.load(diag_path,allow_pickle=False) as z:
        keys=['aligned_sample_ids','labels','baseline_scores','candidate_scores','baseline_pixels','candidate_pixels','test_only']
        d={k:z[k] for k in keys}
    with np.load(score_path,allow_pickle=False) as z:
        ids=z['sample_ids'].astype(str)
        scores=z['calibrated_scores']
    lookup={sid:i for i,sid in enumerate(d['aligned_sample_ids'].astype(str))}
    indices=np.array([lookup[sid] for sid in ids])
    assert len(set(indices))==len(indices)
    candidate=d['candidate_scores'].astype(np.float64).copy()
    candidate[indices]=scores
    dense=d['candidate_pixels'].astype(np.int64).copy()
    rejected=d['candidate_scores']<report['architecture']['mask_gate_cutoff']
    dense[rejected,2]+=dense[rejected,0]
    dense[rejected,0:2]=0
    result={'diagnostic_only':True,'source_hashes_match':True,'no_training_or_retuning':True,'views':{}}
    for name,sel in [('full',np.ones(len(candidate),dtype=bool)),('test_only_sites',d['test_only'].astype(bool))]:
        base_counts=d['baseline_pixels'][sel].sum(axis=0)
        cand_counts=dense[sel].sum(axis=0)
        values=dict(rows=int(sel.sum()),baseline_ap=float(average_precision_score(d['labels'][sel],d['baseline_scores'][sel])),candidate_ap=float(average_precision_score(d['labels'][sel],candidate[sel])),baseline_iou=float(base_counts[0]/base_counts.sum()),candidate_iou=float(cand_counts[0]/cand_counts.sum()),baseline_tp_fp_fn=base_counts.astype(int).tolist(),candidate_tp_fp_fn=cand_counts.astype(int).tolist())
        original=report['views'][name]['metrics']
        expected=dict(baseline_ap=original['baseline']['average_precision'],candidate_ap=original['candidate']['average_precision'],baseline_iou=original['baseline']['pixels']['intersection_over_union'],candidate_iou=original['candidate']['pixels']['intersection_over_union'])
        values['max_abs_report_difference']=max(abs(values[k]-v) for k,v in expected.items())
        assert values['max_abs_report_difference']<1e-12
        result['views'][name]=values
    (ROOT/'reports/research/GTM_Model0_mars_headline_recomputation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
