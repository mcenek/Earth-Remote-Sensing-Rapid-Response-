"""One-member, 1 MiB transport-capped packaging probe; never bulk acquisition.

This diagnostic is for a specifically identified training reference, not a
change to the failed 480m cohort. It cannot verify the whole archive checksum.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'e654cd777f85d701a4aff2894f513dcdf6981531'
ARCHIVE = 'dataset_part_139.tar.gz'
MEMBER = 'data/original/geo/train/s2/254920/s2_pre.tif'
MAXIMUM_NETWORK_BYTES = 1024 * 1024
MAXIMUM_EXPANDED_BYTES = 16 * 1024 * 1024


def extract_first_expected(payload: bytes, expected: str) -> bytes:
    expanded = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(
        payload, MAXIMUM_EXPANDED_BYTES
    )
    with tarfile.open(fileobj=io.BytesIO(expanded), mode='r:') as archive:
        member = archive.next()
        if member is None or not member.isfile() or member.name != expected:
            raise ValueError('First regular member is not the exact selected training reference')
        if not 0 < member.size <= MAXIMUM_EXPANDED_BYTES:
            raise ValueError('Member exceeds decompression budget')
        if member.offset_data + member.size > len(expanded):
            raise ValueError('Complete member is not present within capped prefix; do not extend automatically')
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError('Selected member cannot be read')
        result = handle.read(member.size)
        if len(result) != member.size:
            raise ValueError('Truncated selected member')
        return result


def main() -> None:
    import tifffile

    report_path = ROOT / 'reports/acquisition/methaneunion_one_member_probe.json'
    if report_path.exists():
        raise FileExistsError('Probe already recorded; no automatic repeat download')
    url = f'https://huggingface.co/datasets/yuyao42/MethaneUnion/resolve/{REVISION}/{ARCHIVE}'
    request = Request(url, headers={'Range': f'bytes=0-{MAXIMUM_NETWORK_BYTES-1}',
                                    'Accept-Encoding': 'identity'})
    with urlopen(request, timeout=30) as response:
        # Do not accept an ignored range and accidentally consume a whole shard.
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes 0-'):
            raise ValueError('Server did not honor the prefix request')
        payload = response.read(MAXIMUM_NETWORK_BYTES)
    report = {'revision': REVISION, 'archive': ARCHIVE, 'member': MEMBER,
              'network_bytes': len(payload), 'network_cap': MAXIMUM_NETWORK_BYTES,
              'whole_archive_hash_verified': False, 'training_eligible': False,
              'scope': 'new original-scale mapping feasibility only; no model or held evaluation'}
    try:
        member = extract_first_expected(payload, MEMBER)
        with tifffile.TiffFile(io.BytesIO(member)) as tif:
            page = tif.pages[0]
            report['tiff'] = {'pages': len(tif.pages), 'shape': list(page.shape),
                              'dtype': str(page.dtype), 'samples_per_pixel': page.samplesperpixel,
                              'geotiff_metadata': tif.geotiff_metadata,
                              'description': page.description[:2000]}
        destination = ROOT / '.research/methaneunion_original_probe/254920_s2_pre.tif'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError('Refusing to overwrite an existing member')
        destination.write_bytes(member)
        report.update(status='member_recovered', member_bytes=len(member),
                      member_sha256=hashlib.sha256(member).hexdigest(),
                      saved_path=str(destination))
    except (ValueError, tarfile.TarError, EOFError) as exc:
        report.update(status='bounded_probe_failed', reason=str(exc))
    report_path.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
