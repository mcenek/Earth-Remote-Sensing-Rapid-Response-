"""Bind fixed, exactly replayed released predictions to independent review."""
import copy
import json
from pathlib import Path

import GTM_Model7_released_comparison_v15 as study


def main():
    out=study.OUT
    destination=out/'review_recipe.json'
    if destination.exists(): raise FileExistsError(destination)
    preflight=study.preflight_receipt()
    historical=json.loads((study.v13.ROOT/'outputs/GTM_Model7_hard_negative_v11_20261025/review_recipe.json').read_text())
    baseline=copy.deepcopy(historical['methods']['candidate'])
    baseline.update(name='Retained v11',prediction_grid='20m_repeat',
        status='EXPOSED DEVELOPMENT CONTROL | no new national validation',
        resolution='20 m model output repeated to 10 m annotation grid')
    methods={'baseline':baseline}
    sources=list(historical['sources'])
    for rel,passed in [('data/receipt.json',False),('data/index.json',False),('preflight/receipt.json',True),
                       ('preflight/teacher_score_verification.json',True)]:
        path=out/rel
        sources.append(dict(path=str(path),sha256=study.sha(path),require_passed=passed))
    statements=[]
    for key,arm,label in [('candidate','mars-s2l','Released MARS-S2L'),('ratio','ch4net','Released CH4Net')]:
        summary_path=out/'inference'/arm/'summary.json'
        replay_path=out/'replay'/arm/'receipt.json'
        result=json.loads(summary_path.read_text()); replay=json.loads(replay_path.read_text())
        if (not replay['passed'] or not replay['exact_probability_label_support_binary'] or
            replay['inference_summary_sha256']!=study.sha(summary_path) or
            result['preflight_sha256']!=study.sha(out/'preflight/receipt.json') or
            result['script_sha256']!=study.sha(Path(study.__file__)) or
            result['protocol_sha256']!=study.sha(study.PROTOCOL) or
            result['scene_count']!=198 or result['exact_valid_half_scores']!=0 or result['cutoff']!=.5):
            raise ValueError('Released inference/replay identity changed')
        for path,passed in [(summary_path,False),(replay_path,True)]:
            sources.append(dict(path=str(path),sha256=study.sha(path),require_passed=passed))
        methods[key]=dict(name=label,status='EXTERNALLY TRAINED REFERENCE | training locations overlap these scenes',
            folder=str(out/'inference'/arm),cutoff=.5,checkpoint_sha256=result['checkpoint_sha256'],
            records=result['scenes'],prediction_grid='native10m',
            resolution='10 m output grid; native SWIR information is 20 m')
        p=result['native_pooled']
        statements.append(f"{label}: F1 {p['f1']:.2%}, precision {p['precision']:.2%}, recall {p['recall']:.2%}; "
            f"{result['clear_activated_scenes']}/152 clear observations marked.")
    recipe=dict(manifest=str(study.v13.SOURCE),manifest_sha256=study.v13.SOURCE_SHA,
        evaluation_split='dev',evaluation_role='historically_exposed_development_external_reference',
        id_prefix='GTM_Model7_released_v15_20261026',title='Local detector and released methane models',
        comparison_heading='Actual saved predictions from three fixed checkpoints. All 198 observations remain visible.',
        architecture_family='early_fusion_unet',status='EXTERNAL REFERENCE | no independent national validation',
        scope='46 real plume and 152 provider no-plume observations from the already exposed development set. '
            'Both releases list 113 of these observations at their training locations and 11 at validation locations. '
            'This is not a fresh test or a training-data-controlled architecture contest. Binary labels do not measure emission rate.',
        acceptance=' '.join(statements)+' Cutoffs were fixed before inference: v11 0.65; both releases 0.5. '
            'No dilation, smoothing, component removal or scene exclusions. No model is promoted from this comparison.',
        context='MARS-S2L uses the current and separately selected earlier Sentinel images, a spectral ratio, wind and current cloud mask. '
            'CH4Net uses only the current six spectral bands. Both release configurations list 38,366 training samples. '
            'The earlier reference selection differs from the original release benchmark. '
            'Both models were run locally on CPU and exactly replayed in fresh processes. No retraining or download occurred.',
        chart_title='Fixed pretrained inference comparison',
        chart_detail='198 exposed observations. Published cutoff 0.5. Source reconstruction and exact CPU replay required.',
        feature_contract='released16',sources=sources,methods=methods,
        native_data={r['id']:dict(path=str(out/'data'/r['cache']),sha256=r['sha256']) for r in study.index()})
    study.save(destination,recipe)
    print(json.dumps({'recipe':str(destination),'acceptance':recipe['acceptance']}))


if __name__=='__main__':main()
