"""Read-only Model6 preparation status. No training or acquisition dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('GTM_Model6_config.json')


def validate_config(config: dict) -> list[str]:
    errors = []
    expected = {
        'schema_version': 1,
        'model_id': 'GTM_Model6',
        'lifecycle': 'pilot_complete_not_promoted',
        'objective': 'sentinel_plume_extent_and_candidate_source_detection',
        'geography': 'United States',
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f'{key} must be {value!r} in this preparation entry point')
    augmentation = config.get('augmentation', {})
    if augmentation.get('rotation_degrees') != [0, 15, 30, 45]:
        errors.append('Expected the four proposed rotations: 0, 15, 30, 45')
    if augmentation.get('horizontal_mirror') != [False, True]:
        errors.append('Expected both unmirrored and horizontally mirrored views')
    if augmentation.get('views_per_parent') != 8 or augmentation.get('order') != 'mirror_then_rotate':
        errors.append('Expected eight total views, mirror then rotate')
    branches = config.get('branches', {})
    if branches.get('context_windows_pixels') != [16, 32, 64, 128] or branches.get('independent_weights') is not True:
        errors.append('Expected four independently trained context-window branches')
    evaluation = config.get('evaluation', {})
    for key in ['parent_groups_split_before_augmentation', 'visual_review_required', 'new_confirmation_cohort_required']:
        if evaluation.get(key) is not True:
            errors.append(f'evaluation.{key} must be true')
    if evaluation.get('primary_validation_augmented') is not False:
        errors.append('Primary validation must retain original observations')
    if evaluation.get('first_task') != 'sentinel_mask_detection_pilot':
        errors.append('First experiment must be the Sentinel mask detection pilot')
    if evaluation.get('outer_folds') != 5 or evaluation.get('outer_split') != 'group_held_out_by_25km_and_acquisition_before_augmentation':
        errors.append('The pilot must use five grouped outer folds before augmentation')
    if evaluation.get('inner_threshold_calibration') is not False or evaluation.get('fixed_cutoff') != 0.5:
        errors.append('The first pilot must use a fixed 0.5 cutoff without inner threshold calibration')
    if config.get('merger', {}).get('training_predictions') != 'training_partition_out_of_fold':
        errors.append('Merger training must use training-partition out-of-fold predictions')
    if config.get('initial_checkpoint') is not None:
        errors.append('The clean-slate default does not load a historical checkpoint')
    for key in ['automatic_training', 'downloads_enabled']:
        if config.get('execution', {}).get(key) is not False:
            errors.append(f'execution.{key} must remain false until a separate implementation exists')
    if config.get('branches', {}).get('implemented') is not True:
        errors.append('plume branches must reflect the implemented pilot')
    for key in ['merger', 'source']:
        if config.get(key, {}).get('implemented') is not False:
            errors.append(f'{key}.implemented must remain false until that component is implemented')
    if config.get('output', {}).get('fine_resolution_validated') is not False:
        errors.append('No fine-resolution methane validation result exists yet')
    return errors


def status(config: dict, root: Path = ROOT) -> dict:
    errors = validate_config(config)
    catalog_path = root / config['paths']['data_sources']
    catalog = json.loads(catalog_path.read_text(encoding='utf-8-sig'))
    locations = []
    for source in catalog['sources']:
        locations.append({
            'id': source['id'],
            'approved_for_model6': source['approved_for_model6'],
            'candidate_locations': [
                {'path': path, 'exists': (root / path).is_dir()}
                for path in source['candidate_paths']
            ],
            'limitations': source['limitations'],
        })
    diagnostic_path = root / 'outputs/GTM_Model6/GTM_Model6_legacy_reconstruction_01/GTM_Model6_results.json'
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8')) if diagnostic_path.is_file() else None
    native_path = root / 'outputs/GTM_Model6/GTM_Model6_native_multiscale_01/GTM_Model6_native_results.json'
    native = json.loads(native_path.read_text(encoding='utf-8')) if native_path.is_file() else None
    plume_path = root / 'outputs/GTM_Model6/GTM_Model6_plume_01/GTM_Model6_plume_results.json'
    plume = json.loads(plume_path.read_text(encoding='utf-8')) if plume_path.is_file() else None
    mapper_path = root / 'outputs/GTM_Model6/GTM_Model6_mapper_mars_pilot_20260922_r5/GTM_Model6_mapper_mars_pilot_results.json'
    mapper = json.loads(mapper_path.read_text(encoding='utf-8')) if mapper_path.is_file() else None
    repair_path = root / 'outputs/GTM_Model6/GTM_Model6_georgia_crop_repair_20260922/report.json'
    repair = json.loads(repair_path.read_text(encoding='utf-8')) if repair_path.is_file() else None
    pair_qa_path = repair_path.parent / 'qa_v2/GTM_Model6_georgia_repaired_pair_qa.json'
    pair_qa = json.loads(pair_qa_path.read_text(encoding='utf-8')) if pair_qa_path.is_file() else None
    return {
        'model_id': config['model_id'],
        'scope': 'Sentinel-only plume-mask pilot supervised by EMIT positives; future fusion/source work listed separately',
        'lifecycle': config['lifecycle'],
        'configuration_valid': not errors,
        'configuration_errors': errors,
        'scientific_readiness': 'not_established',
        'approved_manifest': config.get('approved_manifest'),
        'training_implemented': (root / 'research/GTM_Model6/GTM_Model6_plumes.py').is_file(),
        'trained_result_available': (plume is not None and plume.get('complete') is True) or diagnostic is not None or native is not None or mapper is not None,
        'mapper_development_diagnostic': {
            'trained_result_available': mapper is not None,
            'results': str(mapper_path) if mapper else None,
            'status': mapper.get('status') if mapper else 'not_run',
            'scope': 'reused 65-case MARS fold; first16 Sentinel features; mask-only supervision',
            'native_aggregate': mapper.get('aggregate') if mapper else None,
            'promotion': 'rejected: low precision and surface-pattern false positives',
            'quantitative_emit_trained': False,
            'nationally_validated': False,
            'viewer': 'http://127.0.0.1:8766/?experiment=GTM_Model6_mapper_mars_pilot_20260922_r5_mean#compare' if mapper else None,
        },
        'native_pair_crop_repair': {
            'report': str(repair_path) if repair else None,
            'status': repair.get('status') if repair else 'not_run',
            'scope': 'one peak-centered near-time Georgia Sentinel target crop; input-data repair only',
            'training_cohort_approved': False,
            'quantitative_prediction_available': False,
            'local_qa_status': pair_qa.get('status') if pair_qa else 'not_run',
            'local_qa_report': str(pair_qa_path) if pair_qa else None,
            'remaining_requirements': ['resolve Sentinel already-applied BOA versus literal offset metadata', 'quantitative EMIT QA and plume-outline semantics', 'appropriate Sentinel temporal reference', 'independent positive/negative observations and grouped splits'],
        },
        'plume_pilot': {
            'trained_result_available': plume is not None and plume.get('complete') is True,
            'run_complete': plume.get('complete') if plume else False,
            'nationally_validated': False,
            'promotion': 'rejected: excess background activation and missed EMIT support',
            'followup_results': str(root / 'outputs/GTM_Model6/GTM_Model6_plume_evidence_02/GTM_Model6_plume_results.json'),
            'results': str(plume_path) if plume else None,
            'fixed_cutoff': 0.5,
            'metrics_scope': 'MARS reviewed masks; EMIT positive-footprint proxy only',
            'fine_resolution_validated': False,
        },
        'full_design_implemented': False,
        'engineering_diagnostic': {
            'trained_result_available': diagnostic is not None,
            'status': diagnostic.get('status') if diagnostic else 'not_run',
            'results': str(diagnostic_path) if diagnostic else None,
            'summary': diagnostic.get('summary_macro_observation_mae') if diagnostic else None,
            'fine_resolution_validated': False,
            'viewer': 'http://127.0.0.1:8766/?experiment=GTM_Model6_emit_only_f0#compare' if diagnostic else None,
        },
        'native_multiscale_diagnostic': {
            'trained_result_available': native is not None,
            'status': native.get('status') if native else 'not_run',
            'results': str(native_path) if native else None,
            'input': 'EMIT only',
            'windows': [16, 32, 64, 128],
            'combination': 'uniform mean, not learned merger',
            'held_acquisitions': 3 if native else 0,
            'fine_resolution_validated': False,
            'viewer': 'http://127.0.0.1:8766/?experiment=GTM_Model6_native_uniform_mean#compare' if native else None,
        },
        'next_pass': 'Acquire independently located, closely timed US positive pairs and reviewed negatives; compare the capstone baseline on common compatible scenes before further architecture sweeps',
        'source_locations_only_not_data_validation': locations,
        'blockers': config['blockers'],
        'design': str(root / config['design']),
        'run_root': str(root / config['paths']['run_root']),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'validate'], nargs='?', default='status')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        errors = validate_config(config)
        if args.command == 'validate':
            result = {'configuration_valid': not errors, 'errors': errors, 'scope': 'Sentinel mask pilot configuration; use status for completed run outputs', 'full_design_data_validated': False, 'full_design_training_implemented': False}
        elif errors:
            result = {'configuration_valid': False, 'errors': errors}
        else:
            result = status(config)
        print(json.dumps(result, indent=2))
        return 1 if errors else 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({'configuration_valid': False, 'error': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
