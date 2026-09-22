"""Registered, support-aware native context sampling for the mapper.

Transform each complete parent once, then cut all contexts and the label from
that same field. Positive draws must contain an observed plume pixel, and the
largest context must fit inside the original (rotated) source footprint.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from GTM_Model6_mapper import CONTEXT_WINDOWS, PREDICTION_WINDOW

TRANSFORMS = tuple((angle, mirror) for mirror in (False, True) for angle in (0, 15, 30, 45))


def transformed_parent(item, angle, mirror):
    if (angle, mirror) not in TRANSFORMS:
        raise ValueError("use one of the eight declared Martin transforms")
    x = torch.from_numpy(item['x'][:16]).float()[None]
    y = torch.from_numpy(item['target']).float()[None, None]
    v = torch.from_numpy(item['valid']).float()[None, None]
    if x.shape[-2:] != y.shape[-2:] or y.shape != v.shape:
        raise ValueError("parent input/label/support grids must match")
    footprint = torch.ones_like(v)
    if mirror:
        x, y, v = (a.flip(-1) for a in (x, y, v))
    radians = math.radians(angle)
    theta = x.new_tensor([[[math.cos(radians), -math.sin(radians), 0],
                           [math.sin(radians), math.cos(radians), 0]]])
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    rotated_x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    rotated_y = F.grid_sample(y, grid, mode='nearest', align_corners=False)
    # Bilinear support requires all contributing input pixels to be valid.
    rotated_v = F.grid_sample(v, grid, align_corners=False) >= 0.999
    geometry = F.grid_sample(footprint, grid, align_corners=False) >= 0.999
    return rotated_x[0], rotated_y[0], rotated_v[0], geometry[0, 0].numpy()


def _box_sum(values, size):
    integral = np.pad(np.asarray(values, dtype=np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]


def eligible_centers(target, support, footprint, *, positive):
    largest = max(CONTEXT_WINDOWS)
    h, w = footprint.shape
    if min(h, w) < largest:
        return np.empty((0, 2), dtype=int)
    margin = (largest - PREDICTION_WINDOW) // 2
    geometry = _box_sum(footprint, largest) == largest * largest
    truth = np.asarray(target).squeeze() > 0
    valid = np.asarray(support).squeeze().astype(bool)
    positive_count = _box_sum(truth & valid, PREDICTION_WINDOW)[margin:margin+h-largest+1, margin:margin+w-largest+1]
    valid_count = _box_sum(valid, PREDICTION_WINDOW)[margin:margin+h-largest+1, margin:margin+w-largest+1]
    eligible = geometry & (valid_count >= 0.9 * PREDICTION_WINDOW**2)
    eligible &= positive_count > 0 if positive else positive_count == 0
    return np.argwhere(eligible) + largest // 2


def extract_patch(view, center):
    x, y, support, footprint = view
    row, col = map(int, center)
    contexts = {}
    for size in CONTEXT_WINDOWS:
        top, left = row - size // 2, col - size // 2
        patch = x[:, top:top+size, left:left+size]
        if patch.shape[-2:] != (size, size) or not footprint[top:top+size, left:left+size].all():
            raise ValueError("context crosses unobserved source footprint")
        contexts[f'S{size}'] = patch
    half = PREDICTION_WINDOW // 2
    return contexts, y[:, row-half:row+half, col-half:col+half], support[:, row-half:row+half, col-half:col+half]


def sample_batch(items, rng, batch_size, *, telemetry=None):
    if batch_size < 2:
        raise ValueError("balanced sampling requires batch size >= 2")
    positives = [i for i in items if ((i['target'] > 0) & i['valid']).any()]
    negatives = [i for i in items if not (i['target'] > 0).any() and i['row'].get('fully_labeled', True)]
    if not positives or not negatives:
        raise ValueError("need both positive parents and reviewed negative parents")
    slots = np.array([True] * ((batch_size + 1) // 2) + [False] * (batch_size // 2))
    rng.shuffle(slots)
    all_contexts = {f'S{size}': [] for size in CONTEXT_WINDOWS}
    targets, supports = [], []
    for positive in slots:
        pool = positives if positive else negatives
        for attempt in range(64):
            item = pool[int(rng.integers(len(pool)))]
            angle, mirror = TRANSFORMS[int(rng.integers(len(TRANSFORMS)))]
            view = transformed_parent(item, angle, mirror)
            centers = eligible_centers(view[1], view[2], view[3], positive=positive)
            if len(centers):
                break
        else:
            raise ValueError("no observed context can satisfy the requested sampling class")
        center = centers[int(rng.integers(len(centers)))]
        contexts, target, valid = extract_patch(view, center)
        if positive and not (target.bool() & valid).any():
            raise AssertionError("positive sampler returned an empty target")
        for name, value in contexts.items():
            all_contexts[name].append(value)
        targets.append(target)
        supports.append(valid)
        if telemetry is not None:
            telemetry['draws'] = telemetry.get('draws', 0) + 1
            telemetry['positive_draws'] = telemetry.get('positive_draws', 0) + int(positive)
            telemetry['rejected_views'] = telemetry.get('rejected_views', 0) + attempt
            telemetry['positive_pixels'] = telemetry.get('positive_pixels', 0) + int((target.bool() & valid).sum())
            telemetry['supported_pixels'] = telemetry.get('supported_pixels', 0) + int(valid.sum())
    return {name: torch.stack(values) for name, values in all_contexts.items()}, torch.stack(targets), torch.stack(supports)
