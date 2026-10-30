"""Check published prediction bundles over HTTP against verified local exports.

Repeat with --review for each rendered comparison and --base-url for its server.
Failure cases were specified before implementation in the diversity protocol:
stale bytes, missing assets, mismatched native counts, and a static mask that uses
a different threshold from the interactive grid must stop publication review.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

from GTM_Model0_publish_viewer import source_references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--negative-only', action='store_true', help='Require the explicitly reserved no-plume-only review role')
    args = parser.parse_args()
    verification = json.loads((args.review / 'verification.json').read_text())
    comparison = json.loads((args.review / 'comparison.json').read_text())
    if not verification['passed'] or not verification['counts_recomputed']:
        raise ValueError('Independent source/native-count review is required')
    if args.negative_only and verification.get('evaluation_role') != 'reserved_no_plume_only':
        raise ValueError('Negative-only exemption requires an explicit reserved-clear review')
    checks, models = [], {}
    for model, publication in verification['publication'].items():
        manifest_path = Path(publication['manifest'])
        bundle = json.loads(manifest_path.read_text())
        if bundle['id'] != publication['experimentId']:
            raise ValueError('Publication identity changed')
        prefix = args.base_url.rstrip('/') + '/bundle-assets/' + bundle['id'] + '/'
        served = {}
        for relative in dict.fromkeys(row[2] for row in source_references(bundle)):
            expected = (manifest_path.parent / relative).read_bytes()
            with urlopen(prefix + relative, timeout=30) as response:
                actual = response.read()
                if response.status != 200 or actual != expected:
                    raise ValueError('HTTP asset differs: ' + relative)
            served[relative] = actual
            checks.append(dict(model=model, asset=relative, bytes=len(actual),
                               sha256=hashlib.sha256(actual).hexdigest()))
        records = {row['id']: row for row in comparison['scenes']}
        kinds = {}
        for scene in bundle['scenes']:
            if scene['nativeMetrics'] != {**scene['nativeMetrics'], **records[scene['id']]['models'][model]}:
                raise ValueError('Viewer native metrics differ from independent counts')
            if not scene['predictionAvailable']:
                raise ValueError('Scene has no saved prediction')
            cutoff = scene['decisionThreshold']
            if cutoff != bundle['decisionThreshold']:
                raise ValueError('Scene cutoff changed')
            grid = json.loads(served[scene['grid']])
            scores = np.array([np.nan if value is None else value for value in grid['probability']])
            scores = scores.reshape(grid['height'], grid['width'])
            if scene.get('predictionImage'):
                image = np.asarray(Image.open(io.BytesIO(served[scene['predictionImage']])))
                if not np.array_equal(image[..., 3] > 0, np.isfinite(scores) & (scores >= cutoff)):
                    raise ValueError('Static prediction differs from interactive grid at frozen cutoff')
            kinds[scene['label']] = kinds.get(scene['label'], 0) + 1
        if args.negative_only:
            if set(kinds) != {'NO_PLUME'}:
                raise ValueError('Reserved-clear review must contain only no-plume scenes')
        elif not all(kinds.get(kind) for kind in ('PLUME', 'NO_PLUME')):
            raise ValueError('Both real classes must be present')
        for key in ('tp', 'fp', 'fn', 'tn'):
            total = sum(scene['nativeMetrics'][key] for scene in bundle['scenes'])
            if total != comparison['pooled'][model][key] or total != bundle['nativeAggregate'][key]:
                raise ValueError('Viewer aggregate counts differ')
        models[model] = dict(experiment_id=bundle['id'], scenes=len(bundle['scenes']),
                             kinds=kinds, cutoff=bundle['decisionThreshold'],
                             report_url=prefix + bundle['reportUrl'])
    result = dict(passed=True, review=str(args.review), models=models,
                  served_assets=len(checks), checks=checks,
                  renderer_verification_sha256=hashlib.sha256((args.review / 'verification.json').read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
