"""Bounded metadata-only Sentinel-2 discovery at up to five active CAFO locations.

No raster downloads, pagination, labels or training. L2A is a prospective
facility-imagery source, not a substitute for the frozen MARS L1C contract.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = 'https://earth-search.aws.element84.com/v1/search'
MAX_RESPONSE = 1048576


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cloud-max', type=float)
    parser.add_argument('--facility-ids', nargs='+', help='At most five explicit inventory IDs')
    args = parser.parse_args()
    if args.cloud_max is not None and not 0 <= args.cloud_max <= 100:
        raise ValueError('cloud-max must be in [0,100]')
    if args.output.exists():
        raise FileExistsError('Preserve existing catalog evidence')
    with args.csv.open(encoding='utf-8-sig', newline='') as source:
        rows = sorted((r for r in csv.DictReader(source) if r['opStatus'].strip().lower() == 'active'),
                      key=lambda r: r['progid'])
    if args.facility_ids:
        if len(args.facility_ids)>5 or len(set(args.facility_ids))!=len(args.facility_ids):
            raise ValueError('Need one to five unique facility IDs')
        selected = {r['progid']: r for r in rows if r['progid'] in args.facility_ids}
        if len(selected)!=len(args.facility_ids):
            raise ValueError('Requested IDs must all be active inventory records')
        rows = [selected[key] for key in args.facility_ids]
    else:
        rows = rows[:3]
    report = {'scope': 'metadata feasibility only, not selected training images',
              'source_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              'endpoint': ENDPOINT, 'collection': 'sentinel-2-l2a',
              'year': 2024, 'year_status': 'planning assumption matching inventory filename',
              'max_queries': len(rows), 'max_bytes_per_response': MAX_RESPONSE,
              'raster_bytes_downloaded': 0, 'queries': []}
    report['scene_cloud_max'] = args.cloud_max
    for row in rows:
        lat, lon = float(row['Latitude']), float(row['Longitude'])
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Invalid source coordinates')
        params = {'collections': 'sentinel-2-l2a', 'limit': 2,
                  'bbox': f'{lon-.0001},{lat-.0001},{lon+.0001},{lat+.0001}',
                  'datetime': '2024-01-01T00:00:00Z/2024-12-31T23:59:59Z'}
        if args.cloud_max is not None:
            params['query'] = json.dumps({'eo:cloud_cover': {'lt': args.cloud_max}})
        entry = {'facility_id': row['progid'], 'latitude': lat, 'longitude': lon,
                 'url': ENDPOINT + '?' + urlencode(params)}
        try:
            with urlopen(Request(entry['url'], headers={'User-Agent': 'ERSRR-catalog-feasibility/1'}), timeout=30) as response:
                raw = response.read(MAX_RESPONSE+1)
                if len(raw) > MAX_RESPONSE:
                    raise ValueError('Catalog response exceeded byte cap')
            data = json.loads(raw)
            features = data.get('features', [])
            if not isinstance(features, list) or len(features) > 2:
                raise ValueError('Catalog did not honor requested limit')
            entry.update(status='ok', bytes_received=len(raw),
                         more_items_available=any(x.get('rel') == 'next' for x in data.get('links', [])),
                         items=[{'id': f['id'], 'datetime': f['properties'].get('datetime'),
                                 'cloud_cover': f['properties'].get('eo:cloud_cover'),
                                 'collection': f.get('collection'),
                                 'assets': {name: {'href': a.get('href'), 'bands': a.get('eo:bands'),
                                                   'raster_bands': a.get('raster:bands')}
                                            for name, a in f.get('assets', {}).items()
                                            if name in ('blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'scl')}}
                                for f in features])
        except Exception as error:
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
        report['queries'].append(entry)
        if entry['status']=='failed':
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'planned_queries': len(rows), 'queries': len(report['queries']), 'successful': sum(q['status']=='ok' for q in report['queries']),
                      'raster_bytes_downloaded': 0}))


if __name__ == '__main__':
    main()
