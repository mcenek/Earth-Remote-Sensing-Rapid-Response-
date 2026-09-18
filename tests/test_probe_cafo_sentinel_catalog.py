import json

from tools import probe_cafo_sentinel_catalog as catalog


def test_selected_catalog_stops_on_first_failure(tmp_path, monkeypatch):
    source = tmp_path/'sites.csv'
    source.write_text('progid,opStatus,Latitude,Longitude\na,Active,41,-92\nb,Active,42,-93\n')
    output = tmp_path/'result.json'
    monkeypatch.setattr('sys.argv', ['probe','--csv',str(source),'--output',str(output),
                                     '--facility-ids','b','a','--cloud-max','20'])
    calls=[]
    def fail(request, timeout):
        calls.append(request.full_url)
        raise OSError('network unavailable')
    monkeypatch.setattr(catalog, 'urlopen', fail)
    catalog.main()
    report=json.loads(output.read_text())
    assert len(calls)==1
    assert report['max_queries']==2
    assert len(report['queries'])==1
    assert report['queries'][0]['facility_id']=='b'
    assert report['queries'][0]['status']=='failed'
