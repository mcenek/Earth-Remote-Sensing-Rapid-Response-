"""Read-only experiment-report inventory across the three research checkouts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['ersrr-ordinal-win', 'Earth-Remote-Sensing-Rapid-Response-', 'ersrr-publish-2026-09-04']

def main():
    inventory = {}
    errors = []
    for name in NAMES:
        directory = ROOT.parent / name / 'reports/experiments'
        entries = {}
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in {'.json', '.md'}:
                continue
            raw = p.read_bytes()
            rel = p.relative_to(directory).as_posix()
            normalized = raw.decode('utf-8-sig').replace('\r\n', '\n').strip()
            entry = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw)}
            if p.suffix == '.json':
                try:
                    value = json.loads(raw.decode('utf-8-sig'))
                    normalized = json.dumps(value,sort_keys=True,separators=(',',':'))
                    if isinstance(value, dict):
                        entry['scope'] = value.get('scope', '')
                        entry['decision'] = value.get('decision', '')
                        entry['top_level_keys'] = list(value)
                except Exception as e:
                    errors.append({'path':str(p), 'error':str(e)})
            entry['normalized_sha256'] = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            entries[rel] = entry
        inventory[name] = entries
    primary = inventory[NAMES[0]]
    summary = {'counts':{name:len(entries) for name,entries in inventory.items()}, 'parse_errors':errors,
               'comparison':{name:{'only_in_other':sorted(set(entries)-set(primary)), 'only_in_primary':sorted(set(primary)-set(entries)), 'different_content':sorted(k for k in entries.keys() & primary.keys() if entries[k]['sha256'] != primary[k]['sha256'])} for name,entries in inventory.items() if name != NAMES[0]}}
    output = ROOT / 'reports/research'
    for name, comparison in summary['comparison'].items():
        comparison['different_normalized_content'] = [k for k in comparison['different_content'] if inventory[name][k]['normalized_sha256'] != primary[k]['normalized_sha256']]
    (output/'GTM_Model0_experiment_inventory.json').write_text(json.dumps({'summary':summary,'inventory':inventory},indent=2),encoding='utf-8')
    with (output/'GTM_Model0_experiment_inventory.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['checkout','relative_report','bytes','sha256','scope','decision'])
        for name, entries in inventory.items():
            for path,entry in entries.items():
                writer.writerow([name,path,entry['bytes'],entry['sha256'],entry.get('scope',''),entry.get('decision','')])
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
