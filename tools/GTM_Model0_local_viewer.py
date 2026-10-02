"""Local GTM experiment viewer. Requests serve saved artifacts; they never run inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import socket
import threading
import time
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'GTM_Viewer'
BUNDLES = ROOT / 'outputs/GTM_viewer_bundles'
PORTABLE = UI / 'portable_bundles'
DOCS = ROOT / 'docs'
ASSET_KEYS = ('rgb', 'emit', 'truthImage', 'predictionImage', 'probabilityImage',
              'disagreementImage', 'observedImage', 'coarseImage', 'residualImage',
              'reference90Rgb', 'reference365Rgb')
MAX_MANIFEST = 50 * 1024 * 1024
ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def inside(root: Path, relative: str) -> Path:
    """Resolve a URL or manifest path without allowing an absolute or symlink escape."""
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Unsafe asset path')
    parts = PurePosixPath(relative).parts
    if relative.startswith('/') or any(part in ('.', '..') for part in parts) or ':' in parts[0]:
        raise ValueError('Unsafe asset path')
    base = root.resolve()
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base):
        raise ValueError('Path outside viewer directory')
    return target


def validate_run(value):
    if not isinstance(value, dict) or value.get('state') not in ('queued', 'running', 'completed', 'failed'):
        raise ValueError('Invalid run state')
    if set(value) - {'state', 'progress', 'startedAt', 'updatedAt', 'message'}:
        raise ValueError('Unknown run field')
    progress = value.get('progress')
    if progress is not None:
        if not isinstance(progress, dict) or set(progress) != {'completed', 'total', 'unit'}:
            raise ValueError('Invalid run progress')
        completed, total = progress['completed'], progress['total']
        if (type(completed) is not int or type(total) is not int or completed < 0
                or total < completed or not isinstance(progress['unit'], str)
                or not progress['unit'].strip()):
            raise ValueError('Invalid run progress')
    for key in ('startedAt', 'updatedAt'):
        if key in value:
            stamp = value[key]
            if not isinstance(stamp, str):
                raise ValueError('Invalid run timestamp')
            try:
                parsed = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
                if parsed.tzinfo is None:
                    raise ValueError()
            except ValueError as error:
                raise ValueError('Invalid run timestamp') from error
    if 'message' in value and (not isinstance(value['message'], str) or len(value['message']) > 2000):
        raise ValueError('Invalid run message')
    return value


def load_bundle(manifest: Path, route_prefix: str):
    """Return a fully checked experiment and revision, or raise without exposing a partial one."""
    before = manifest.stat()
    if before.st_size > MAX_MANIFEST:
        raise ValueError('Manifest exceeds 50 MiB')
    raw = manifest.read_bytes()
    after = manifest.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Manifest changed during read')
    data = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Manifest must be a JSON object')
    experiment = data.get('experiment', data)
    if not isinstance(experiment, dict):
        raise ValueError('Invalid experiment')
    experiment = copy.deepcopy(experiment)
    identifier = experiment.get('id')
    scenes = experiment.get('scenes')
    if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier) or not isinstance(scenes, list):
        raise ValueError('Need valid experiment id and scenes')
    if len(scenes) > 1000:
        raise ValueError('Too many scenes')
    if not isinstance(experiment.get('name'), str) or not experiment['name'].strip():
        raise ValueError('Need experiment name')
    if 'run' in experiment:
        validate_run(experiment['run'])
    digest = hashlib.sha256(raw)
    referenced = {}

    def asset(value):
        if not isinstance(value, str):
            raise ValueError('Invalid asset reference')
        if re.match(r'^data:image/(png|jpeg|webp);base64,', value):
            return value
        path = inside(manifest.parent, value)
        if not path.is_file():
            raise ValueError('Missing asset: ' + value)
        stat = path.stat()
        referenced[value] = (stat.st_size, stat.st_mtime_ns)
        encoded = '/'.join(quote(part, safe='') for part in PurePosixPath(value).parts)
        return route_prefix + quote(manifest.parent.name, safe='') + '/' + encoded

    seen_scenes = set()
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get('id'), str) or not scene['id'].strip():
            raise ValueError('Invalid scene id')
        if not isinstance(scene.get('name'), str) or not scene['name'].strip():
            raise ValueError('Invalid scene name')
        if scene['id'] in seen_scenes:
            raise ValueError('Duplicate scene id')
        seen_scenes.add(scene['id'])
        if scene.get('coordinateSystem') == 'pixel':
            size = scene.get('imageSize', [32, 32])
            if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n < 1 for n in size):
                raise ValueError('Invalid image dimensions')
        else:
            bounds = scene.get('bounds')
            if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(not isinstance(pair, list) or len(pair) != 2 or
                    any(type(n) not in (int, float) for n in pair) or
                    not -85 <= pair[0] <= 85 or not -180 <= pair[1] <= 180
                    for pair in bounds) or
                bounds[0][0] >= bounds[1][0] or bounds[0][1] >= bounds[1][1]):
                raise ValueError('Invalid scene bounds')
        for key in ASSET_KEYS:
            if scene.get(key) is not None:
                scene[key] = asset(scene[key])
        if isinstance(scene.get('grid'), str):
            scene['grid'] = asset(scene['grid'])
        elif scene.get('grid') is not None and not isinstance(scene['grid'], dict):
            raise ValueError('Invalid grid')
    if experiment.get('reportUrl') is not None:
        experiment['reportUrl'] = asset(experiment['reportUrl'])
    for relative, signature in sorted(referenced.items()):
        digest.update(relative.encode('utf-8'))
        digest.update(str(signature).encode('ascii'))
    revision = digest.hexdigest()[:20]
    experiment['live'] = {'revision': revision, 'updatedAt': datetime.fromtimestamp(
        after.st_mtime, timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}
    return experiment, revision


class ViewerState:
    def __init__(self, bundles_dir=BUNDLES, poll_seconds=5, server_name=None):
        self.bundles_dir = Path(bundles_dir).resolve()
        self.poll_seconds = max(2, poll_seconds)
        self.server_name = server_name or 'GTM viewer'
        self.lock = threading.RLock()
        self.next_check = 0.0
        self.cached = {}
        self.snapshot = None

    def registry(self):
        with self.lock:
            if self.snapshot is None or time.monotonic() >= self.next_check:
                self._refresh()
            return copy.deepcopy(self.snapshot)

    def _refresh(self):
        base = json.loads((UI / 'GTM_Model0_registry.json').read_text(encoding='utf-8'))
        indexed = {entry['id']: entry for entry in base['experiments']}
        warnings = []
        seen = set()
        manifests = sorted(PORTABLE.glob('*/experiment.json')) + sorted(self.bundles_dir.glob('*/experiment.json'))
        for manifest in manifests:
            key = str(manifest.resolve())
            try:
                portable = manifest.parent.parent.resolve() == PORTABLE.resolve()
                root = PORTABLE if portable else self.bundles_dir
                if manifest.is_symlink() or not manifest.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Manifest symlink or path escape')
                prefix = 'portable-bundles/' if portable else 'bundle-assets/'
                experiment, revision = load_bundle(manifest, prefix)
                identifier = experiment['id']
                if identifier in seen:
                    raise ValueError('Duplicate bundle experiment id: ' + identifier)
                seen.add(identifier)
                self.cached[key] = (experiment, revision)
                indexed[identifier] = experiment
            except (ValueError, TypeError, KeyError, OSError, UnicodeError) as error:
                warnings.append(manifest.parent.name + ': ' + str(error))
                if key in self.cached and self.cached[key][0]['id'] not in seen:
                    indexed[self.cached[key][0]['id']] = copy.deepcopy(self.cached[key][0])
                    seen.add(self.cached[key][0]['id'])
                    warnings.append(manifest.parent.name + ': showing last valid publication')
        current = {str(path.resolve()) for path in manifests}
        for key, (experiment, _) in self.cached.items():
            if key not in current and Path(key).parent.parent == self.bundles_dir and experiment['id'] not in seen:
                indexed[experiment['id']] = copy.deepcopy(experiment)
                warnings.append(Path(key).parent.name + ': manifest unavailable; showing last valid version')
        for identifier, experiment in indexed.items():
            if identifier in {'GTM_Model0', 'GTM_Model1', 'GTM_Model2', 'GTM_Model3', 'GTM_Model4', 'GTM_Model5'}:
                experiment['lifecycle'] = 'archived'
                experiment['priority'] = False
                if not experiment.get('status', '').startswith('ARCHIVED'):
                    experiment['status'] = 'ARCHIVED · ' + experiment.get('status', 'Historical evidence')
        preferred = base.get('activeExperimentId', 'GTM_Model6')
        base['activeExperimentId'] = preferred if preferred in indexed else (
            'GTM_Model6_mapper_r5_portable' if 'GTM_Model6_mapper_r5_portable' in indexed else 'GTM_Model6')
        base['experiments'] = list(indexed.values())
        base['local'] = {'bundleDirectory': str(self.bundles_dir), 'warnings': warnings,
                         'mode': 'Local read-only viewer'}
        # Hash the visible payload before adding checkedAt, which changes every scan.
        revision = hashlib.sha256(json.dumps(base, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:20]
        base['live'] = {'revision': revision, 'checkedAt': utc_now(),
                        'pollSeconds': self.poll_seconds, 'serverName': self.server_name}
        self.snapshot = base
        self.next_check = time.monotonic() + self.poll_seconds


class ViewerHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = unquote(urlsplit(self.path).path)
        if route == '/api/experiments':
            try:
                registry = self.server.viewer.registry()
                etag = '"' + registry['live']['revision'] + '"'
                if self.headers.get('If-None-Match') == etag:
                    self.send_response(304)
                    self.send_header('ETag', etag)
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    return
                return self.send_bytes(json.dumps(registry).encode(), 'application/json; charset=utf-8', etag)
            except Exception as error:
                return self.send_error(500, str(error))
        if route == '/api/health':
            return self.send_bytes(json.dumps({'app': 'GTM local experiment viewer', 'version': 1,
                                               'local_only': self.server.server_address[0] in ('127.0.0.1', 'localhost', '::1')}).encode(), 'application/json')
        try:
            if route.startswith('/bundle-assets/'):
                path = inside(self.server.viewer.bundles_dir, route[len('/bundle-assets/'):])
            elif route.startswith('/portable-bundles/'):
                path = inside(PORTABLE, route[len('/portable-bundles/'):])
            elif route.startswith('/research-docs/'):
                path = inside(DOCS, route[len('/research-docs/'):])
            else:
                path = inside(UI, route.lstrip('/') or 'index.html')
            if not path.is_file():
                return self.send_error(404)
            content_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            return self.send_bytes(path.read_bytes(), content_type)
        except (ValueError, OSError):
            return self.send_error(404)

    def send_bytes(self, content, content_type, etag=None):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if etag:
            self.send_header('ETag', etag)
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith('GET /api/health'):
            return
        super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--bundles-dir', type=Path, default=BUNDLES)
    parser.add_argument('--poll-seconds', type=float, default=5)
    parser.add_argument('--name', default=None, help='Display name shown by the viewer')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    args.bundles_dir.mkdir(parents=True, exist_ok=True)
    server = ViewerHTTPServer((args.host, args.port), Handler)
    server.viewer = ViewerState(args.bundles_dir, args.poll_seconds, args.name)
    url = f'http://{args.host}:{server.server_address[1]}/'
    print('GTM local experiment viewer: ' + url, flush=True)
    print('Experiment folder: ' + str(args.bundles_dir.resolve()), flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
