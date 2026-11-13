"""Prospective score evaluation; no files, model inference or test-set tuning.

Bundles: {split: validation|test, samples: [{sample_id, group_id, label: 0|1,
valid: bool, cnn: finite score|null, temporal_attention: finite score|null}]}.
Scores may be logits; larger means more positive. Both models must share QA.
Group IDs must already reflect independently adjudicated geographic clusters.
Empirical validation FPR is not a population false-alert guarantee.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from sklearn.metrics import average_precision_score

MODELS = ('cnn', 'temporal_attention')


#### SCORE CONTRACT AND VALIDATION-ONLY OPERATING POINT ####

def _bundle(bundle: dict, split: str) -> list[dict]:
    if not isinstance(bundle, dict) or bundle.get('split') != split:
        raise ValueError(f'Expected explicit {split} split')
    rows = bundle.get('samples')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty samples required')
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each sample must be an object')
        for key in ('sample_id', 'group_id'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'{key} must be a nonempty string')
        if row['sample_id'] in ids:
            raise ValueError('Duplicate sample_id')
        ids.add(row['sample_id'])
        if type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise ValueError('Labels must be binary integers')
        if type(row.get('valid')) is not bool:
            raise ValueError('valid must be an explicit boolean')
        for name in MODELS:
            score = row.get(name)
            if not row['valid'] and score is None:
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError('Valid scenes require finite scores for both models')
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _counts(labels: np.ndarray, decisions: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & decisions))
    fp = int(np.sum((labels == 0) & decisions))
    fn = int(np.sum((labels == 1) & ~decisions))
    tn = int(np.sum((labels == 0) & ~decisions))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=_ratio(tp, tp+fn),
                false_positive_rate=_ratio(fp, fp+tn), precision=_ratio(tp, tp+fp))


def _decisions(scores: np.ndarray, rule: dict) -> np.ndarray:
    if rule.get('kind') == 'reject_all':
        return np.zeros(scores.shape, dtype=bool)
    threshold = rule.get('threshold')
    if (rule.get('kind') != 'threshold' or rule.get('comparator') != '>='
            or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)):
        raise ValueError('Invalid frozen operating rule')
    return scores >= threshold


def calibrate(bundle: dict, *, max_fpr: float) -> dict:
    """Maximize validation recall within empirical FPR; break ties conservatively.

    Ties prefer fewer false positives, then a higher threshold. A zero-recall
    solution becomes explicit reject_all, avoiding Infinity in JSON artifacts.
    """
    rows = _bundle(bundle, 'validation')
    if isinstance(max_fpr, bool) or not isinstance(max_fpr, (int, float)) or not 0 <= max_fpr <= 1:
        raise ValueError('max_fpr must be finite in [0, 1]')
    valid = [r for r in rows if r['valid']]
    labels = np.array([r['label'] for r in valid], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Calibration requires observed positives and negatives')
    rules = {}
    negatives = int((labels == 0).sum())
    for name in MODELS:
        scores = np.array([r[name] for r in valid], dtype=float)
        best_key = (0, 0, math.inf)
        rule = {'kind': 'reject_all'}
        # Sort once, evaluate entire tied score groups with cumulative counts.
        order = np.argsort(-scores, kind='stable')
        ordered = scores[order]
        ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
        tp = np.cumsum(labels[order])[ends]
        fp = (ends + 1) - tp
        for index in np.flatnonzero(fp / negatives <= max_fpr):
            threshold = float(ordered[ends[index]])
            key = (int(tp[index]), -int(fp[index]), threshold)
            if key > best_key:
                best_key = key
                rule = {'kind': 'threshold', 'threshold': threshold, 'comparator': '>='}
        rules[name] = {'rule': rule, 'validation': _counts(labels, _decisions(scores, rule))}
    return {
        'schema_version': 1, 'fit_split': 'validation', 'max_empirical_fpr': max_fpr,
        'validation_sha256': hashlib.sha256(json.dumps(bundle, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'validation_ids': sorted(r['sample_id'] for r in rows),
        'validation_groups': sorted({r['group_id'] for r in rows}),
        'valid_scenes': len(valid), 'abstained_scenes': len(rows)-len(valid),
        'negative_scenes': negatives, 'fpr_step_per_negative_scene': 1 / negatives,
        'population_risk_guarantee': False, 'models': rules,
    }


#### FIXED-RULE METRICS AND PAIRED GEOGRAPHIC RESAMPLING ####

def evaluate_pair(bundle: dict, calibration: dict, *, replicates: int = 1000,
                  seed: int = 20260907, confidence: float = .95) -> dict:
    rows = _bundle(bundle, 'test')
    if (not isinstance(calibration, dict) or calibration.get('schema_version') != 1
            or calibration.get('fit_split') != 'validation'
            or not isinstance(calibration.get('models'), dict)
            or set(calibration['models']) != set(MODELS)):
        raise ValueError('Require a validation calibration for both models')
    for key in ('validation_ids', 'validation_groups'):
        values = calibration.get(key)
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)):
            raise ValueError('Calibration must retain validation identities and groups')
    for model in MODELS:
        entry = calibration['models'][model]
        if not isinstance(entry, dict) or not isinstance(entry.get('rule'), dict):
            raise ValueError('Each calibrated model requires a frozen operating rule')
    if ({r['sample_id'] for r in rows} & set(calibration['validation_ids'])
            or {r['group_id'] for r in rows} & set(calibration['validation_groups'])):
        raise ValueError('Evaluation overlaps validation sample IDs or geographic groups')
    if type(replicates) is not int or not 1 <= replicates <= 100000:
        raise ValueError('replicates must be an integer in [1, 100000]')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    valid = [r for r in rows if r['valid']]
    if not valid:
        raise ValueError('No observed evaluation scenes')
    labels = np.array([r['label'] for r in valid], dtype=int)
    group_names, groups = np.unique([r['group_id'] for r in valid], return_inverse=True)
    scores = {m: np.array([r[m] for r in valid], dtype=float) for m in MODELS}
    predictions = {m: _decisions(scores[m], calibration['models'][m]['rule']) for m in MODELS}
    both_classes = set(labels.tolist()) == {0, 1}
    metrics = {
        m: {'average_precision': float(average_precision_score(labels, scores[m])) if both_classes else None,
            'fixed_operating_point': _counts(labels, predictions[m])}
        for m in MODELS
    }
    bootstrap = {'unit': 'geographic group', 'groups': len(group_names), 'seed': seed,
                 'replicates': replicates, 'confidence': confidence, 'intervals': None,
                 'degenerate_replicates': 0,
                 'scope': 'conditional on fitted models, frozen rules and observed groups; excludes training/calibration uncertainty'}
    if not both_classes or len(group_names) < 2:
        bootstrap['unavailable_reason'] = 'Need both classes and at least two geographic groups'
    else:
        rng = np.random.default_rng(seed)
        deltas = {key: [] for key in ('average_precision', 'recall', 'false_positive_rate')}
        for _ in range(replicates):
            multiplicities = rng.multinomial(len(group_names), np.full(len(group_names), 1/len(group_names)))
            weight = multiplicities[groups]
            positives = int(weight[labels == 1].sum())
            negatives = int(weight[labels == 0].sum())
            if not positives or not negatives:
                bootstrap['degenerate_replicates'] += 1
                continue
            per_model = {}
            for name in MODELS:
                per_model[name] = (
                    float(average_precision_score(labels, scores[name], sample_weight=weight)),
                    float(weight[(labels == 1) & predictions[name]].sum() / positives),
                    float(weight[(labels == 0) & predictions[name]].sum() / negatives))
            for index, key in enumerate(deltas):
                deltas[key].append(per_model['temporal_attention'][index] - per_model['cnn'][index])
        if bootstrap['degenerate_replicates']:
            bootstrap['unavailable_reason'] = 'Class-degenerate draws encountered; no conditional-on-survival interval reported'
        else:
            tail = (1-confidence)/2
            bootstrap['intervals'] = {
                key: {'lower': float(np.quantile(values, tail)),
                      'upper': float(np.quantile(values, 1-tail))}
                for key, values in deltas.items()
            }
    return {'scope': 'fixed-rule paired evaluation; no threshold fitting',
            'valid_scenes': len(valid), 'abstained_scenes': len(rows)-len(valid),
            'positive_scenes_abstained': sum(r['label'] == 1 and not r['valid'] for r in rows),
            'negative_scenes_abstained': sum(r['label'] == 0 and not r['valid'] for r in rows),
            'models': metrics, 'paired_delta_bootstrap': bootstrap}


#### DENSE METRICS FOR ONE ANNOTATED SCENE ####

def dense_metrics(probability, truth, sensor_valid, *, annotation_available: bool,
                  threshold: float = .5) -> dict:
    """Pixel QA never implies annotation. Undefined ratios are null, not perfect."""
    p, y, valid = np.asarray(probability), np.asarray(truth), np.asarray(sensor_valid)
    if p.ndim != 2 or p.shape != y.shape or p.shape != valid.shape or valid.dtype != np.bool_:
        raise ValueError('Dense inputs must have equal HxW shape and boolean QA')
    if type(annotation_available) is not bool:
        raise ValueError('annotation_available must be explicit boolean')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('Dense threshold must be in [0, 1]')
    if not annotation_available or not valid.any():
        return {'eligible': False, 'reason': 'missing_annotation' if not annotation_available else 'no_valid_pixels'}
    if not np.isfinite(p[valid]).all() or not ((p[valid] >= 0) & (p[valid] <= 1)).all():
        raise ValueError('Supported probabilities must be finite in [0, 1]')
    if not np.isin(y[valid], (0, 1)).all():
        raise ValueError('Supported truth must be binary')
    counts = _counts(y[valid], p[valid] >= threshold)
    return {'eligible': True, 'valid_pixels': int(valid.sum()), **counts,
            'iou': _ratio(counts['tp'], counts['tp'] + counts['fp'] + counts['fn'])}
