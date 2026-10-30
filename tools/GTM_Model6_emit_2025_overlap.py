"""Training-only US EMIT intake against existing identity/coordinate inventories.

No network, rasters, model outcomes, or protected labels are used. CSV source
rows are immediately projected to explicitly allowed identity fields.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT.parent / 'Earth-Remote-Sensing-Rapid-Response-'
STAGE = ROOT / 'outputs/GTM_Model0_hf_pilot_20260925'
OUT = STAGE / 'emit_followup'
NEW = STAGE / 'UNEP-IMEO__MARS-Hyperspectral-EMIT-v2025/originals/csvs/train_t_v2025A.csv'
SHA = 'cbbee1489168630a4966c17117e65b831e580718339a0ba1c09e697f9f5a62ea'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def project(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield {k: row.get(k, '') for k in fields}


def coordinate(row, lat='lat', lon='lon'):
    if not row.get(lat) or not row.get(lon): return None
    a, b = float(row[lat]), float(row[lon])
    if not math.isfinite(a+b) or not -90 <= a <= 90 or not -180 <= b <= 180:
        raise ValueError('Invalid geographic coordinate')
    return a, b


def distance(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    delta_p, delta_l = p2-p1, math.radians(b[1]-a[1])
    z = math.sin(delta_p/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(delta_l/2)**2
    return 12742.0176 * math.asin(math.sqrt(min(1, max(0, z))))


def main():
    if (OUT / 'overlap_audit.json').exists(): raise FileExistsError('Preserve the existing audit')
    if digest(NEW) != SHA: raise ValueError('Training manifest changed')
    fields = ('id','tile','tile_date','location_name','location_lat','location_lon','folder_name',
              'country','crs','percentage_clear','isplume')
    new = list(project(NEW, fields))
    if len({r['id'] for r in new}) != len(new): raise ValueError('Duplicate training sample IDs')
    new_by_name = defaultdict(set)
    for r in new:
        point = coordinate(r,'location_lat','location_lon')
        if point: new_by_name[r['location_name']].add(point)
    sources = [{'path':str(NEW),'sha256':SHA,'fields_used':fields}]
    points, mars_by_name = set(), defaultdict(set)
    existing_ids, protected_ids, old_names, protected_names = set(),set(),set(),set()
    mars_paths = [HIST/'.research/source_audit_20260715/mars_s2l_current_validated_images_all.csv',
                  ROOT/'EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L/validated_images_all.csv']
    for path in mars_paths:
        allowed = ('id_loc_image','location_name','lat','lon','split_name')
        count = 0
        for r in project(path,allowed):
            point = coordinate(r)
            if point: points.add(point); mars_by_name[r['location_name']].add(point)
            count += 1
        sources.append({'path':str(path),'sha256':digest(path),'rows':count,'fields_used':allowed,
                        'policy':'all locations excluded at <=25 km, irrespective of split'})
    base = ROOT/'EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MethaneS2CM/l2a_location_split_32x32'
    for split in ('train','test'):
        path = base/f'{split}.csv'
        allowed = ('id','latitude','longitude')
        count = 0
        for r in project(path,allowed):
            point = coordinate(r,'latitude','longitude')
            if point: points.add(point)
            count += 1
        sources.append({'path':str(path),'sha256':digest(path),'rows':count,'fields_used':allowed})
    # Historical HSI holdouts are identity-only. Coordinates are resolved only
    # from names found in geographic metadata, never from their raster payloads.
    hsi = HIST/'.research/mars_hyperspectral_transfer'
    unresolved = set()
    for split, file in [('train','train_t_v4a.csv'),('validation','val_t_v4a.csv'),('test','test_t_v4a.csv')]:
        path = hsi/'EMIT'/file
        allowed = ('id','location_name','location_lat','location_lon','country')
        count = 0
        for r in project(path,allowed):
            existing_ids.add(r['id']); old_names.add(r['location_name'])
            if split != 'train':
                protected_ids.add(r['id']); protected_names.add(r['location_name'])
                coord = coordinate(r,'location_lat','location_lon')
                resolved = {coord} if coord else new_by_name[r['location_name']] | mars_by_name[r['location_name']]
                points.update(resolved)
                if not resolved and r['country'] in ('United States','United States of America','USA'):
                    unresolved.add(r['location_name'])
            count += 1
        sources.append({'path':str(path),'sha256':digest(path),'rows':count,'fields_used':allowed,'split':split})
    # Existing Model6 observations are development-used, regardless of their old fold.
    path = ROOT/'outputs/GTM_Model6/GTM_Model6_plume_01/GTM_Model6_plume_manifest.json'
    for source in json.loads(path.read_text())['observations']:
        r = {k:source.get(k) for k in ('id','lat','lon')}
        existing_ids.add(r['id'])
        point = coordinate(r)
        if point: points.add(point)
    sources.append({'path':str(path),'sha256':digest(path),'fields_used':['id','lat','lon']})
    union_test = ROOT/'.research/external/MethaneUnion-release/datasets/geo_split/480m_GSD/test.csv'
    if not union_test.exists(): raise FileNotFoundError('Known MethaneUnion protected identity source missing')
    for r in project(union_test,('id','latitude','longitude')):
        point = coordinate(r,'latitude','longitude')
        if point: points.add(point)
    sources.append({'path':str(union_test),'sha256':digest(union_test),'fields_used':['id','latitude','longitude']})
    cache, evaluated = {}, []
    for row in new:
        if row['country'] != 'United States of America': continue
        point = coordinate(row,'location_lat','location_lon')
        if point not in cache: cache[point] = min(distance(point,p) for p in points) if point else None
        reasons = []
        if not point: reasons.append('missing_coordinate')
        if row['id'] in existing_ids: reasons.append('already_known_sample_id')
        if row['id'] in protected_ids or row['location_name'] in protected_names:
            reasons.append('historical_EMIT_holdout_identity')
        if cache[point] is not None and cache[point] <= 25: reasons.append('within_25km_known_or_protected_location')
        if float(row['percentage_clear']) < 90: reasons.append('under_90pct_clear')
        evaluated.append({**row,'nearest_inventory_km':cache[point], 'exclusions':reasons,
                          'previous_HSI_location_name':row['location_name'] in old_names})
    eligible = [r for r in evaluated if not r['exclusions']]
    report = {'scope':'US training-only identity/coordinate audit; no protected outcomes',
              'revision':'4e6c50999eb430fa33abc9488b71cd1a0dcd7991','sources':sources,
              'distance_km':25,'minimum_clear_pct':90,'known_coordinate_points':len(points),
              'us_rows':len(evaluated),'eligible_rows':len(eligible),
              'eligible_location_names':len({r['location_name'] for r in eligible}),
              'eligible_declared_isplume':dict(Counter(r['isplume'] for r in eligible)),
              'exclusion_counts_nonexclusive':dict(Counter(reason for r in evaluated for reason in r['exclusions'])),
              'unresolved_historical_US_holdout_names':sorted(unresolved),
              'caveats':['Eligibility is for development inspection, not independent evaluation.',
                         'New release validation/test coordinates were not acquired; training provenance is the provider split.',
                         'No label masks or cross-sensor geometry have been verified.'],
              'observations':evaluated}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'overlap_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (OUT/'eligible_training_candidates.json').write_text(json.dumps(eligible,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('observations','sources')},indent=2))


if __name__ == '__main__': main()
