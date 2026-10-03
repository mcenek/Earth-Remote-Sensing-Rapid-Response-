"""Exercise live viewer publication through its public CLI and HTTP API."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / 'tools/GTM_Model0_local_viewer.py'
PUBLISHER = ROOT / 'tools/GTM_Model0_publish_viewer.py'
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9p3gAAAABJRU5ErkJggg==')


def request(url, etag=None):
    headers = {'If-None-Match': etag} if etag else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=8) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as error:
        return error.code, dict(error.headers), error.read()


def prepare(folder, scene_count, label='first'):
    folder.mkdir(parents=True, exist_ok=True)
    scenes = []
    for number in range(scene_count):
        name = f'scene{number}.png'
        (folder / name).write_bytes(PNG)
        scenes.append({'id': f'scene{number}', 'name': f'Saved scene {number}',
                       'bounds': [[30.0, -100.0], [30.1, -99.9]], 'rgb': name})
    manifest = {'id': 'live_e2e', 'name': 'Live E2E evidence', 'status': label,
                'run': {'state': 'running', 'progress': {
                    'completed': scene_count, 'total': 3, 'unit': 'scenes'}},
                'scenes': scenes}
    (folder / 'experiment.json').write_text(json.dumps(manifest), encoding='utf-8')


def publish(folder, bundles):
    result = subprocess.run([sys.executable, str(PUBLISHER), '--bundle-dir', str(folder),
                             '--bundles-dir', str(bundles)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError('Publish failed: ' + result.stderr)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=ROOT / 'outputs/GTM_viewer_logs/live_viewer_e2e.json')
    args = parser.parse_args()
    checks = []

    def check(name, condition, detail=None):
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(name + (': ' + str(detail) if detail else ''))

    with tempfile.TemporaryDirectory(prefix='gtm-viewer-e2e-') as temporary:
        temp = Path(temporary)
        bundles = temp / 'bundles'
        bundles.mkdir()
        prepared = temp / 'prepared'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        process = subprocess.Popen([sys.executable, '-u', str(VIEWER), '--host', '127.0.0.1',
                                    '--port', str(port), '--bundles-dir', str(bundles),
                                    '--poll-seconds', '2', '--name', 'E2E backend'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        base = f'http://127.0.0.1:{port}'
        try:
            for _ in range(60):
                try:
                    health_status, _, health_body = request(base + '/api/health')
                    if health_status == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                raise AssertionError('Viewer did not start')
            check('legacy health app', json.loads(health_body)['app'] == 'GTM local experiment viewer')
            status, headers, body = request(base + '/api/experiments')
            registry = json.loads(body)
            check('fresh portable scenes', status == 200 and any(
                e['id'] == 'GTM_Model6_mapper_r5_portable' and e['scenes'] for e in registry['experiments']))
            presentation = registry.get('presentation', {})
            check('prediction-first default', registry.get('activeExperimentId') == 'GTM_Model6_mapper_r5_portable'
                  and presentation.get('defaultView') == 'compare')
            cases = presentation.get('experiments', [])
            check('two portable presentation collections', [case['id'] for case in cases] == [
                'GTM_Model6_mapper_r5_portable', 'GTM_Model6_emit_evidence_portable'])
            inspected = 0
            for case in cases:
                saved = next(e for e in registry['experiments'] if e['id'] == case['id'])
                opening = next(s for s in saved['scenes'] if s['id'] == case['sceneId'])
                for item in saved['scenes']:
                    for key in ('rgb', 'truthImage', 'predictionImage', 'grid'):
                        asset_status, _, payload = request(base + '/' + item[key])
                        check('presentation asset ' + item['id'] + ' ' + key, asset_status == 200 and bool(payload))
                    numeric = json.loads(payload)
                    check('saved probabilities ' + item['id'], any(
                        value is not None for value in numeric.get('probability', [])))
                    if item['id'] == opening['id']:
                        cutoff = item.get('decisionThreshold', saved.get('decisionThreshold', .5))
                        check('opening prediction has visible pixels ' + saved['id'], any(
                            value is not None and value >= cutoff and (not numeric.get('valid') or numeric['valid'][index])
                            for index, value in enumerate(numeric['probability'])))
                    inspected += 1
            check('nine presentation scenes', inspected == 9)
            check('live metadata and no-store', registry['live']['serverName'] == 'E2E backend'
                  and headers.get('Cache-Control') == 'no-store')
            etag = headers.get('ETag')
            check('conditional catalog', request(base + '/api/experiments', etag)[0] == 304)
            prepare(prepared, 1)
            first = publish(prepared, bundles)
            time.sleep(2.1)
            status, headers, body = request(base + '/api/experiments', etag)
            registry = json.loads(body)
            experiment = next(e for e in registry['experiments'] if e['id'] == 'live_e2e')
            first_revision = experiment['live']['revision']
            old_asset = experiment['scenes'][0]['rgb']
            check('first scene arrives', status == 200 and len(experiment['scenes']) == 1)
            check('run state passed through', experiment['run']['progress']['completed'] == 1)
            check('published asset serves', request(base + '/' + old_asset)[2] == PNG)
            prepare(prepared, 2, 'expanded')
            second = publish(prepared, bundles)
            time.sleep(2.1)
            _, headers, body = request(base + '/api/experiments', headers['ETag'])
            registry = json.loads(body)
            experiment = next(e for e in registry['experiments'] if e['id'] == 'live_e2e')
            check('expanded revision arrives', len(experiment['scenes']) == 2
                  and experiment['live']['revision'] != first_revision)
            check('old version remains readable', first['version'] != second['version']
                  and request(base + '/' + old_asset)[2] == PNG)
            manifest = bundles / 'live_e2e/experiment.json'
            saved = manifest.read_bytes()
            manifest.write_text('{bad json', encoding='utf-8')
            time.sleep(2.1)
            _, _, body = request(base + '/api/experiments')
            registry = json.loads(body)
            check('malformed update retains last good', any(
                e['id'] == 'live_e2e' and len(e['scenes']) == 2 for e in registry['experiments'])
                  and registry['local']['warnings'])
            manifest.write_text(json.dumps({'id': 'live_e2e', 'name': 'Missing',
                                            'scenes': [{'id': 'x', 'name': 'X', 'bounds': [[30, -100], [31, -99]], 'rgb': 'gone.png'}]}), encoding='utf-8')
            time.sleep(2.1)
            _, _, body = request(base + '/api/experiments')
            registry = json.loads(body)
            check('missing asset retains last good', any(
                e['id'] == 'live_e2e' and len(e['scenes']) == 2 for e in registry['experiments'])
                  and any('Missing asset' in warning for warning in registry['local']['warnings']))
            manifest.write_bytes(saved)
            current_asset = bundles / 'live_e2e' / '_versions' / second['version'] / 'scene0.png'
            current_asset.write_bytes(PNG + b'changed')
            time.sleep(2.1)
            _, _, body = request(base + '/api/experiments')
            registry = json.loads(body)
            changed = next(e for e in registry['experiments'] if e['id'] == 'live_e2e')
            check('same-name asset changes revision', changed['live']['revision'] != experiment['live']['revision'])
            check('asset response no-store', request(base + '/' + changed['scenes'][0]['rgb'])[1].get(
                'Cache-Control') == 'no-store')
            check('traversal blocked', request(base + '/bundle-assets/../GTM_Viewer/index.html')[0] == 404)
            outside = temp / 'outside.txt'
            outside.write_text('outside', encoding='utf-8')
            link = bundles / 'live_e2e' / 'escape.txt'
            try:
                link.symlink_to(outside)
                check('symlink blocked', request(base + '/bundle-assets/live_e2e/escape.txt')[0] == 404)
            except (OSError, NotImplementedError):
                checks.append({'name': 'symlink blocked', 'passed': None,
                               'detail': 'Symlink creation unavailable on this host'})
            invalid = bundles / 'invalid'
            invalid.mkdir()
            (invalid / 'experiment.json').write_text('{"id":"invalid","name":"Bad","scenes":[{"id":"x"}]}', encoding='utf-8')
            time.sleep(2.1)
            _, _, body = request(base + '/api/experiments')
            registry = json.loads(body)
            check('invalid new scene isolated', not any(e['id'] == 'invalid' for e in registry['experiments'])
                  and any(e['id'] == 'live_e2e' for e in registry['experiments']))
            unsafe = temp / 'unsafe_prepared'
            unsafe.mkdir()
            (unsafe / 'experiment.json').write_text(json.dumps({
                'id': 'unsafe', 'name': 'Unsafe', 'scenes': [
                    {'id': 'x', 'name': 'X', 'rgb': '../outside.txt'}]}), encoding='utf-8')
            refused = subprocess.run([sys.executable, str(PUBLISHER), '--bundle-dir', str(unsafe),
                                      '--bundles-dir', str(bundles)], capture_output=True, text=True)
            check('unsafe manifest path rejected', refused.returncode != 0
                  and not (bundles / 'unsafe/experiment.json').exists())
            help_result = subprocess.run([sys.executable, str(VIEWER), '--help'], capture_output=True, text=True)
            check('remote host CLI documented', help_result.returncode == 0 and '--host' in help_result.stdout)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        artifact = {'result': 'passed', 'checks': checks, 'backend': 'temporary stdlib HTTP child',
                    'publisher': str(PUBLISHER), 'viewer': str(VIEWER)}
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
        print(str(args.artifact.resolve()))


if __name__ == '__main__':
    main()
