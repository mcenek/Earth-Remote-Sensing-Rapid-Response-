"""Build a metadata-only facility follow-up queue, never methane labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'EarthRemoteSensingRapidResponse/Polygon_CSV_Files/iowa_cafos_2024_arcgis_api.csv'
MANIFESTS = {
    'MARS': ROOT / 'outputs/GTM_Model7_dataset_repair_20261024/mars_clear_data/manifest.json',
    'S2CM': ROOT / 'outputs/GTM_Model7_dataset_repair_20261024/s2cm_reference_repair_data/manifest.json',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    q = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742 * math.asin(min(1, math.sqrt(q)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): sha(p) for p in [INVENTORY, *MANIFESTS.values()]}
    with INVENTORY.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_id = defaultdict(list)
    invalid = []
    for index, row in enumerate(rows, 2):
        try:
            point = tuple(float(row[k]) for k in ('Latitude', 'Longitude'))
            if not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180):
                raise ValueError('Out of range')
        except (ValueError, TypeError):
            invalid.append(index)
            continue
        by_id[row['OBJECTID']].append((index, row, point))
    facilities, conflicts = [], []
    for identifier, entries in sorted(by_id.items()):
        points = sorted(set(x[2] for x in entries))
        record = dict(objectid=identifier, source_csv_rows=[x[0] for x in entries],
                      statuses=sorted(set(x[1]['opStatus'].strip() for x in entries)))
        if len(points) != 1:
            conflicts.append(dict(**record, coordinates=points))
            continue
        facilities.append(dict(**record, latitude=points[0][0], longitude=points[0][1],
                               source_accuracy_values=sorted(set(x[1]['Accuracy'] for x in entries))))
    coords = np.radians([[f['latitude'], f['longitude']] for f in facilities])
    nearest, queue = [], []
    for dataset, manifest_path in MANIFESTS.items():
        scenes = json.loads(manifest_path.read_text())['scenes']
        sites = defaultdict(list)
        for scene in scenes:
            sites[scene['physical_location_id']].append(scene)
        for site, items in sorted(sites.items()):
            points = {(x['latitude'], x['longitude']) for x in items}
            if len(points) != 1 or len({x['split'] for x in items}) != 1:
                raise ValueError('Ambiguous methane site point or split')
            lat, lon = next(iter(points))
            y, x = np.radians((lat, lon))
            q = np.sin((coords[:, 0]-y)/2)**2 + np.cos(y)*np.cos(coords[:, 0])*np.sin((coords[:, 1]-x)/2)**2
            distances = 12742*np.arcsin(np.sqrt(np.clip(q, 0, 1)))
            index = int(np.argmin(distances))
            facility, distance = facilities[index], float(distances[index])
            independent = scalar_km((lat, lon), (facility['latitude'], facility['longitude']))
            if abs(distance-independent) > 1e-7:
                raise ValueError('Distance replay disagreement')
            record = dict(dataset=dataset, physical_location_id=site,
                          latitude=lat, longitude=lon, split=items[0]['split'],
                          scene_ids=[x['id'] for x in items], observation_count=len(items),
                          closest_inventory_facility=facility, distance_km=distance,
                          within_2km=distance <= 2, within_5km=distance <= 5,
                          interpretation='Facility proximity only. No methane label or detection accuracy.')
            nearest.append(record)
            if distance <= 5:
                queue.append(record)
    for p, expected in hashes.items():
        if sha(Path(p)) != expected:
            raise ValueError('Input changed during metadata join')
    queue.sort(key=lambda r: (r['distance_km'], r['dataset'], r['physical_location_id']))
    receipt = dict(schema='GTM_Model7_cafo_proximity_queue_v1', source_sha256=hashes,
                   exporter_sha256=sha(Path(__file__)), source_rows=len(rows),
                   unique_objectids=len(by_id), unambiguous_facilities=len(facilities),
                   conflicting_coordinate_objectids=conflicts, invalid_csv_rows=invalid,
                   methane_sites_checked=dict(Counter(r['dataset'] for r in nearest)),
                   queue_sites_within_5km=dict(Counter(r['dataset'] for r in queue)),
                   queue_sites_within_2km=dict(Counter(r['dataset'] for r in queue if r['within_2km'])),
                   distance_replay_checks=len(nearest), methane_labels_created=0,
                   rasters_opened=0, downloads=0,
                   coordinate_assumption='Inventory decimal lat/lon treated as WGS84-like for provisional candidates; source datum undocumented.',
                   caution='Proximity and Active status do not establish methane on any date. No inventory match does not establish no plume.',
                   passed=True)
    for name, data in [('receipt.json', receipt), ('candidate_queue.json', queue),
                       ('nearest_facilities.json', nearest), ('facility_points.json', facilities)]:
        (args.output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256', 'conflicting_coordinate_objectids')}, indent=2))


if __name__ == '__main__':
    main()
