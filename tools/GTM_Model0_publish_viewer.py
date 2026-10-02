"""Publish a prepared GTM viewer bundle through an atomic manifest switch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from GTM_Model0_local_viewer import ASSET_KEYS, BUNDLES, ID_PATTERN, inside, load_bundle


def source_references(experiment):
    references = []
    for scene in experiment['scenes']:
        for key in (*ASSET_KEYS, 'grid'):
            value = scene.get(key)
            if isinstance(value, str) and not value.startswith('data:image/'):
                references.append((scene, key, value))
    value = experiment.get('reportUrl')
    if isinstance(value, str) and not value.startswith('data:image/'):
        references.append((experiment, 'reportUrl', value))
    return references


def copy_verified(source, destination):
    before = source.stat()
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, destination.open('xb') as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source asset changed during copy: ' + str(source))
    if destination.stat().st_size != before.st_size:
        raise ValueError('Incomplete asset copy: ' + str(source))
    shutil.copystat(source, destination)
    return digest.hexdigest()


def publish(bundle_dir: Path, bundles_dir: Path):
    bundle_dir = bundle_dir.resolve()
    manifest = bundle_dir / 'experiment.json'
    raw = manifest.read_bytes()
    load_bundle(manifest, 'bundle-assets/')
    if manifest.read_bytes() != raw:
        raise ValueError('Prepared manifest changed during validation')
    data = json.loads(raw.decode('utf-8-sig'))
    experiment = data.get('experiment', data)
    identifier = experiment['id']
    if not ID_PATTERN.fullmatch(identifier):
        raise ValueError('Unsafe experiment id')
    destination = bundles_dir.resolve() / identifier
    if destination.is_symlink():
        raise ValueError('Destination folder is a symlink')
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.resolve().is_relative_to(bundles_dir.resolve()):
        raise ValueError('Destination outside publication folder')
    version = uuid.uuid4().hex
    versions = destination / '_versions'
    if versions.is_symlink():
        raise ValueError('Destination version folder is a symlink')
    stage = versions / version
    stage.mkdir(parents=True, exist_ok=False)
    references = source_references(experiment)
    copied = {}
    try:
        for holder, key, relative in references:
            source = inside(bundle_dir, relative)
            target = stage.joinpath(*Path(relative).parts)
            if relative not in copied:
                copied[relative] = copy_verified(source, target)
            holder[key] = '_versions/' + version + '/' + relative
        staged_manifest = destination / ('experiment.' + version + '.tmp')
        with staged_manifest.open('x', encoding='utf-8') as writer:
            json.dump(data, writer, ensure_ascii=False, indent=2)
            writer.write('\n')
            writer.flush()
            os.fsync(writer.fileno())
        load_bundle(staged_manifest, 'bundle-assets/')
        if manifest.read_bytes() != raw:
            raise ValueError('Prepared manifest changed during publication')
        # The staged file is the sole mutable publication pointer.
        os.replace(staged_manifest, destination / 'experiment.json')
    except Exception:
        # Validate the exact resolved target before recursive cleanup on every platform.
        if (stage.exists() and not stage.is_symlink() and
                stage.resolve().parent == versions.resolve() and
                versions.resolve().parent == destination.resolve()):
            shutil.rmtree(stage)
        temporary = destination / ('experiment.' + version + '.tmp')
        temporary.unlink(missing_ok=True)
        raise
    return {'experimentId': identifier, 'version': version,
            'manifest': str(destination / 'experiment.json'), 'assets': len(copied)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True, help='Prepared folder containing experiment.json')
    parser.add_argument('--bundles-dir', type=Path, default=BUNDLES)
    args = parser.parse_args()
    print(json.dumps(publish(args.bundle_dir, args.bundles_dir), indent=2))


if __name__ == '__main__':
    main()
