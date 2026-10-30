"""Stage a small, pinned Hugging Face pilot against the existing shared budget.

No archive extraction, labeling, data consolidation, or automatic retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

from bounded_http_range import Budget

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/GTM_Model0_hf_pilot_20260925'
LEDGER = ROOT / 'outputs/cafo_pilot_2026-09-11/range_cache'
REPOS = ('oishee3003/CAFOSat', 'UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025')
ARCHIVES = {'CA_filtered.tar.gz', 'ID_filtered.tar.gz', 'LA_filtered.tar.gz', 'WA_filtered.tar.gz'}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def request_bytes(url, budget, cap, exact=None, data=None):
    budget.reserve(cap)
    headers = {'User-Agent': 'ERSRR-bounded-research/1.0', 'Accept-Encoding': 'identity'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    chunks, received = [], 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers, data=data), timeout=45) as response:
        if response.status != 200:
            raise ValueError(f'Unexpected HTTP status {response.status}')
        length = response.headers.get('Content-Length')
        if length and (int(length) > cap or (exact is not None and int(length) != exact)):
            raise ValueError('Response length disagrees with the pinned bound')
        while received < cap:
            chunk = response.read(min(65536, cap - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            budget.received(len(chunk))
        if exact is None and received == cap:
            raise ValueError('Metadata reached its size cap; refuse potentially truncated catalog')
        if exact is not None and received != exact:
            raise ValueError(f'Truncated transfer: {received} of {exact} bytes')
        more_pages = bool(response.headers.get('Link'))
    return b''.join(chunks), more_pages


def catalog(repo, budget):
    folder = STAGE / repo.replace('/', '__')
    manifest_path = folder / 'catalog.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding='utf-8'))
    refs, _ = request_bytes(f'https://huggingface.co/api/datasets/{repo}/refs', budget, 32768)
    ref_info = json.loads(refs)
    revision = next(item['targetCommit'] for item in ref_info['branches'] if item['name'] == 'main')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Invalid pinned revision')
    tree, paginated = request_bytes(
        f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=false&expand=false&limit=100',
        budget, 524288)
    files = json.loads(tree)
    if not isinstance(files, list):
        raise ValueError('File catalog is not a list')
    result = {'repo': repo, 'revision': revision, 'root_listing_paginated': paginated,
              'files': files, 'scope': 'root listing only; no dataset row analysis'}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'refs.json').write_bytes(refs)
    (folder / 'root_tree.json').write_bytes(tree)
    save_json(manifest_path, result)
    return result


def check_content(body, item):
    if len(body) != item['size']:
        raise ValueError('Size mismatch')
    sha256 = hashlib.sha256(body).hexdigest()
    if item.get('lfs'):
        if sha256 != item['lfs']['oid']:
            raise ValueError('LFS SHA-256 mismatch')
    else:
        blob = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
        if blob != item['oid']:
            raise ValueError('Git blob hash mismatch')
    return sha256


def list_metadata_folder(repo, relative, budget):
    if (repo, relative) != (REPOS[1], 'csvs'):
        raise ValueError('Only the published EMIT csvs folder is in this pilot')
    root = catalog(repo, budget)
    if not any(f['path'] == relative and f['type'] == 'directory' for f in root['files']):
        raise ValueError('Folder is absent from the pinned root listing')
    out = STAGE / repo.replace('/', '__') / 'catalog_csvs.json'
    if out.exists():
        return json.loads(out.read_text(encoding='utf-8'))
    payload, paginated = request_bytes(
        f'https://huggingface.co/api/datasets/{repo}/tree/{root["revision"]}/csvs?recursive=false&expand=false&limit=100',
        budget, 524288)
    result = {'repo': repo, 'revision': root['revision'], 'folder': relative,
              'paginated': paginated, 'files': json.loads(payload)}
    save_json(out, result)
    return result


def fetch_file(repo, relative, budget):
    if repo not in REPOS:
        raise ValueError('Repository is outside this pilot')
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative:
        raise ValueError('Unsafe relative path')
    data = catalog(repo, budget)
    entries = data['files']
    if relative.startswith('csvs/'):
        entries = list_metadata_folder(repo, 'csvs', budget)['files']
        if any(term in rel.name.lower() for term in ('test', 'val')):
            raise ValueError('Held-out metadata is outside this training-only pilot')
    item = next((f for f in entries if f['path'] == relative and f['type'] == 'file'), None)
    if not item:
        raise ValueError('File is absent from the pinned root catalog')
    size = int(item['size'])
    allowed_archive = repo == REPOS[0] and relative in ARCHIVES
    allowed_metadata = rel.suffix.lower() in {'.md', '.csv', '.json', '.txt'} and size <= 2 * 1024 * 1024
    allowed_metadata |= repo == REPOS[1] and relative == 'csvs/train_t_v2025A.csv' and size <= 5 * 1024 * 1024
    if not (allowed_archive or allowed_metadata) or not 0 < size <= 20 * 1024 * 1024:
        raise ValueError('File exceeds the pilot allowlist or size limit')
    out = STAGE / repo.replace('/', '__') / 'originals' / relative
    out.resolve().relative_to(STAGE.resolve())
    url = f'https://huggingface.co/datasets/{repo}/resolve/{data["revision"]}/{urllib.parse.quote(relative)}'
    reused = out.exists()
    if reused:
        sha256 = check_content(out.read_bytes(), item)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_suffix(out.suffix + '.partial')
        with partial.open('xb') as handle:
            body, _ = request_bytes(url, budget, size, exact=size)
            sha256 = check_content(body, item)
            handle.write(body)
        if out.exists():
            raise FileExistsError('Output appeared while downloading; refusing overwrite')
        partial.rename(out)
    result = {'repo': repo, 'revision': data['revision'], 'path': relative, 'url': url,
              'local_path': str(out), 'bytes': size, 'sha256': sha256,
              'upstream_hash_verified': True, 'reused': reused, 'extracted': False}
    if not reused:
        save_json(out.with_suffix(out.suffix + '.receipt.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', action='store_true')
    parser.add_argument('--list', action='append', default=[], help='repo:csvs for the EMIT metadata catalog')
    parser.add_argument('--fetch', action='append', default=[], help='repo:path from pinned root catalog')
    args = parser.parse_args()
    if not args.catalog and not args.fetch and not args.list:
        parser.error('Choose --catalog, --list or --fetch')
    budget = Budget(LEDGER, 100 * 1024 * 1024)
    timestamp = datetime.now(timezone.utc)
    report = {'created_at': timestamp.isoformat(), 'budget_before': budget.snapshot(), 'catalogs': [], 'files': []}
    try:
        if args.catalog:
            for repo in REPOS:
                data = catalog(repo, budget)
                report['catalogs'].append(data)
        for value in args.list:
            repo, relative = value.split(':', 1)
            report['catalogs'].append(list_metadata_folder(repo, relative, budget))
        for value in args.fetch:
            repo, relative = value.split(':', 1)
            result = fetch_file(repo, relative, budget)
            report['files'].append(result)
            print(json.dumps({'staged': result['path'], 'bytes': result['bytes'], 'verified': True}), flush=True)
        report['status'] = 'complete'
    except Exception as error:
        report['status'] = 'stopped'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['budget_after'] = budget.snapshot()
        save_json(STAGE / 'runs' / (timestamp.strftime('%Y%m%dT%H%M%S%fZ') + '.json'), report)
        save_json(STAGE / 'latest_stage_receipt.json', report)
        print(json.dumps({'status': report['status'], 'budget': report['budget_after']}), flush=True)


if __name__ == '__main__':
    main()
