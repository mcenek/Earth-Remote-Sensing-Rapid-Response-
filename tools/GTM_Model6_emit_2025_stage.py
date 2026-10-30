"""Bounded metadata queries and exact-file staging for the approved EMIT pilot."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import urllib.parse

from GTM_Model0_hf_bounded_stage import Budget, LEDGER, request_bytes, save_json, check_content
from GTM_Model6_emit_2025_overlap import OUT, STAGE, distance

REPO = 'UNEP-IMEO/MARS-Hyperspectral-EMIT-v2025'
REVISION = '4e6c50999eb430fa33abc9488b71cd1a0dcd7991'
ROLES = ('scl','blue','green','red','nir','swir16','swir22')


def read(name): return json.loads((OUT/name).read_text())


def dt(value): return datetime.fromisoformat(value.replace('Z','+00:00'))


def solar_rank(row):
    time = dt(row['tile_date']).astimezone(timezone.utc)
    solar = (time.hour + time.minute/60 + time.second/3600 + float(row['location_lon'])/15) % 24
    delta = abs(solar-10.5)
    return min(delta,24-delta), -float(row['percentage_clear']), row['id']


def plan():
    if (OUT/'query_plan.json').exists(): return read('query_plan.json')
    rows = read('eligible_training_candidates.json')
    ordered = {flag: sorted([r for r in rows if r['isplume']==flag],key=solar_rank) for flag in ('True','False')}
    selected=[]
    for _ in range(6):
        for flag in ('True','False'):
            for row in ordered[flag]:
                point = float(row['location_lat']),float(row['location_lon'])
                if all(distance(point,(float(r['location_lat']),float(r['location_lon'])))>25 for r in selected):
                    selected.append(row)
                    break
    result = {'selection':'up to 12 training rows; alternating declared source positives/controls, ranked by solar-time distance to 10:30 then clear fraction and ID; >25 km between selected coordinates',
              'max_offset_hours':1,'minimum_dense_transfer_minutes':None,'dense_transfer_max_minutes':15,
              'max_scene_cloud_pct':30,'metadata_cap_per_query':131072,'rows':selected,
              'overlap_audit_sha256':hashlib.sha256((OUT/'overlap_audit.json').read_bytes()).hexdigest()}
    save_json(OUT/'query_plan.json',result)
    return result


def query(budget):
    config=plan(); results=[]
    for index,row in enumerate(config['rows']):
        out=OUT/'catalog_queries'/f'{index:02d}.json'
        if out.exists():
            results.append(json.loads(out.read_text())); continue
        time=dt(row['tile_date']); lat=float(row['location_lat']); lon=float(row['location_lon'])
        payload={'collections':['sentinel-2-l2a'],'intersects':{'type':'Point','coordinates':[lon,lat]},
                 'datetime':f'{(time-timedelta(hours=1)).isoformat()}/{(time+timedelta(hours=1)).isoformat()}',
                 'limit':4}
        entry={'row':row,'request':payload,'items':[],'status':'started'}
        try:
            body,_=request_bytes('https://earth-search.aws.element84.com/v1/search',budget,131072,
                                 data=json.dumps(payload).encode())
            data=json.loads(body)
            entry['response_sha256']=hashlib.sha256(body).hexdigest()
            entry['response']=data
            for item in data.get('features',[]):
                props=item['properties']; assets=item['assets']; when=dt(props['datetime'])
                offset=(when-time).total_seconds()/60
                if abs(offset)>60 or props.get('eo:cloud_cover',101)>30 or not set(ROLES)<=set(assets): continue
                if item.get('collection')!='sentinel-2-l2a': continue
                entry['items'].append({'id':item['id'],'datetime':props['datetime'],'offset_minutes':offset,
                                       'cloud_cover':props.get('eo:cloud_cover'),'item':item})
            entry['items'].sort(key=lambda i:(abs(i['offset_minutes']),i['cloud_cover'],i['id']))
            entry['status']='matched' if entry['items'] else 'no_qualifying_match'
        except Exception as error:
            entry.update(status='error',error=f'{type(error).__name__}: {error}')
        save_json(out,entry);results.append(entry)
        print(json.dumps({'index':index,'location':row['location_name'],'status':entry['status'],
                          'matches':len(entry['items'])}),flush=True)
        if entry['status']=='error': break
    result={'queries':results,'budget':budget.snapshot(),'training_approved':False}
    save_json(OUT/'sentinel_query_summary.json',result)
    return { 'queries':len(results),'matched_queries':sum(bool(r['items']) for r in results),
             'budget':budget.snapshot() }


def native_catalog(budget, sample_id):
    candidates=read('eligible_training_candidates.json')
    row=next((r for r in candidates if r['id']==sample_id),None)
    if row is None: raise ValueError('Sample is outside audited training candidate set')
    folder=row['folder_name']; year=row['tile_date'][:4]
    if '/' in folder or '\\' in folder or '..' in folder or year not in ('2023','2025'):
        raise ValueError('Unsupported training folder')
    output=OUT/'native_catalogs'/f'{sample_id}.json'
    if output.exists(): return json.loads(output.read_text())
    path=f'EMIT_{year}/{folder}'
    url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{urllib.parse.quote(path)}?recursive=false&expand=false&limit=100'
    body,paginated=request_bytes(url,budget,65536)
    result={'sample':row,'folder':path,'revision':REVISION,'files':json.loads(body),'paginated':paginated}
    if paginated: raise ValueError('Native folder listing unexpectedly paginated')
    save_json(output,result)
    return result


def acquire_native(budget, sample_id):
    data=native_catalog(budget,sample_id)
    allowed={'info.json','plumemask.tif','rgb.tif','mag1c.tif','mf.tif','wmf.tif'}
    files=[f for f in data['files'] if f['type']=='file' and PurePosixPath(f['path']).name in allowed]
    if not {'plumemask.tif','rgb.tif'} <= {PurePosixPath(f['path']).name for f in files}:
        raise ValueError('Required native mask or RGB missing')
    if sum(f['size'] for f in files)>4*1024*1024 or any(f['size']>2*1024*1024 for f in files):
        raise ValueError('Native sample exceeds 4 MiB acquisition ceiling')
    records=[]
    for item in files:
        name=PurePosixPath(item['path']).name
        out=OUT/'native'/sample_id/name
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{urllib.parse.quote(item["path"])}'
        if not out.exists():
            body,_=request_bytes(url,budget,item['size'],exact=item['size'])
            sha=check_content(body,item)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('xb') as f:f.write(body)
        else: sha=check_content(out.read_bytes(),item)
        records.append({'path':str(out),'repository_path':item['path'],'bytes':item['size'],'sha256':sha})
    receipt={'sample':data['sample'],'revision':REVISION,'files':records,'budget':budget.snapshot(),
             'metadata_source':'pinned training CSV; v2025 sample folders have no info.json',
             'scope':'provider training data for intake review only; no prediction or training'}
    save_json(OUT/'native'/sample_id/'receipt.json',receipt)
    return {'sample':sample_id,'files':len(records),'bytes':sum(r['bytes'] for r in records),'budget':budget.snapshot()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','query','catalog','acquire'])
    parser.add_argument('--sample')
    args=parser.parse_args()
    budget=Budget(LEDGER,100*1024*1024)
    if args.mode=='plan': result=plan()
    elif args.mode=='query':result=query(budget)
    elif args.mode=='catalog':result=native_catalog(budget,args.sample)
    else: result=acquire_native(budget,args.sample)
    print(json.dumps(result,indent=2),flush=True)
