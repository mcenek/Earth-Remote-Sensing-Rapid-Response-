"""Check the actual served repair overview and all linked reports/images."""
import argparse
import base64
import hashlib
import json
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'a':
            self.links.append(('page', a['href']))
        if tag == 'img':
            self.links.append(('image', a['src']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='http://127.0.0.1:8766')
    p.add_argument('--report', default='GTM_Model7_dataset_repair_summary')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    url = args.base + '/bundle-assets/' + args.report + '/index.html'
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read()
        assert response.status == 200
    parser = Links()
    parser.feed(body.decode())
    rows = []
    for kind, path in parser.links:
        if kind == 'image' and path.startswith('data:image/png;base64,'):
            data = base64.b64decode(path.split(',', 1)[1], validate=True)
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
            rows.append(dict(kind=kind, embedded=True, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            continue
        target = urllib.parse.urljoin(url, path)
        assert urllib.parse.urlsplit(target).netloc == urllib.parse.urlsplit(args.base).netloc
        with urllib.request.urlopen(target, timeout=30) as response:
            data = response.read()
            assert response.status == 200
        if kind == 'image':
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
        else:
            assert b'<!doctype html' in data[:80].lower()
        rows.append(dict(kind=kind, url=target, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    assert sum(r['kind'] == 'image' for r in rows) == 10
    assert b'__' not in body
    receipt = dict(passed=True, summary_sha256=hashlib.sha256(body).hexdigest(),
                   http_requests=sum(not r.get('embedded', False) for r in rows)+1, resources=rows,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(passed=True, http_requests=receipt['http_requests'], images=10)))


if __name__ == '__main__':
    main()
