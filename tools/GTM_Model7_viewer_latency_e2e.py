"""Measure the running viewer's unchanged catalog across refresh intervals."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
rows = []
for index in range(3):
    if index:
        time.sleep(6)
    start = time.perf_counter()
    with urllib.request.urlopen('http://127.0.0.1:8766/api/experiments', timeout=180) as response:
        payload = response.read()
    elapsed = time.perf_counter()-start
    data = json.loads(payload)
    sizes = {e['id']: len(e['scenes']) for e in data['experiments']}
    assert len(sizes) == 77
    assert sum(n == 1982 for n in sizes.values()) == 4
    rows.append(dict(seconds=elapsed, bytes=len(payload), runs=len(sizes), scene_counts=sizes,
                     revision=data['live']['revision']))
    print(json.dumps(dict(request=index, seconds=elapsed, runs=len(sizes))), flush=True)
args.output.write_text(json.dumps(dict(passed=True, requests=rows,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
