"""Package the declared Friday review assets and verify the extracted ZIP."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'GTM_Viewer/friday_20261002'
OUT = ROOT / 'output/friday_20261002'
MANIFEST = 'SHA256SUMS.json'
REQUIRED = (
    'index.html', 'demo.css', 'demo.js', 'README.txt', 'verification.json',
    'assets/data.js', 'assets/GTM_Model6_architecture.svg',
    'assets/GTM_Model6_architecture.png', 'GTM_Model6_Friday_brief_20261002.pdf',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_asset(review_dir, rel):
    if not isinstance(rel, str) or not rel or '\\' in rel or ':' in rel:
        raise ValueError(f'Unsafe artifact path: {rel!r}')
    parts = PurePosixPath(rel).parts
    if (not parts or PurePosixPath(rel).is_absolute() or
            any(part in ('.', '..') for part in parts) or
            PurePosixPath(rel).as_posix() != rel or
            any(part.casefold() == MANIFEST.casefold() for part in parts)):
        raise ValueError(f'Unsafe artifact path: {rel!r}')
    path = review_dir
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Linked artifact path: {rel!r}')
    if not path.is_file() or not path.resolve().is_relative_to(review_dir.resolve()):
        raise ValueError(f'Missing or unsafe artifact: {rel!r}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=DEMO)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    review_dir = args.review_dir
    output_dir = args.output_dir
    if review_dir.is_symlink() or not review_dir.is_dir():
        raise ValueError(f'Missing or linked review directory: {review_dir}')

    data_path = safe_asset(review_dir, 'assets/data.js')
    data = json.loads(data_path.read_text(encoding='utf-8').split('=', 1)[1].strip().rstrip(';'))
    if data['architectureImage'] != 'assets/GTM_Model6_architecture.svg' or data['pdf'] != 'GTM_Model6_Friday_brief_20261002.pdf':
        raise ValueError('Unexpected architecture or PDF asset')
    panels = [panel['image'] for case in data['cases'] for panel in case['panels'] if panel['image']]
    if any(not isinstance(rel, str) or not rel.startswith('assets/case_') or not rel.endswith('.png') for rel in panels):
        raise ValueError('Unexpected case panel asset')
    names = [*REQUIRED, *panels]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError('Duplicate artifact name')
    files = {rel: safe_asset(review_dir, rel) for rel in names}
    hashes = {rel: digest(files[rel]) for rel in sorted(files)}
    manifest = review_dir / MANIFEST
    if manifest.is_symlink():
        raise ValueError('Linked hash manifest')
    manifest.write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8', newline='\n')
    manifest_hash = digest(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'GTM_Model6_Friday_review_20261002.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for rel, path in files.items():
            archive.write(path, rel)
        archive.write(manifest, MANIFEST)

    extracted = Path(tempfile.mkdtemp(prefix='verified_extract_', dir=output_dir))
    expected = set(files) | {MANIFEST}
    with zipfile.ZipFile(target) as archive:
        entries = archive.namelist()
        if len(entries) != len(expected) or set(entries) != expected:
            raise ValueError('Unexpected ZIP entries')
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        archive.extractall(extracted)
    actual = {p.relative_to(extracted).as_posix() for p in extracted.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Unexpected extracted files')
    if digest(extracted / MANIFEST) != manifest_hash or json.loads((extracted / MANIFEST).read_text(encoding='utf-8')) != hashes:
        raise ValueError('Extracted hash manifest mismatch')
    for rel, expected_hash in hashes.items():
        if digest(extracted / rel) != expected_hash:
            raise ValueError(f'Extracted artifact mismatch: {rel}')
    receipt = {'zip': str(target), 'bytes': target.stat().st_size, 'sha256': digest(target),
               'files_verified': len(hashes), 'extracted_for_browser_check': str(extracted),
               'cases': len(data['cases']), 'no_training_or_satellite_transfer': True}
    (output_dir / 'package_verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
