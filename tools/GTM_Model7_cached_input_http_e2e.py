"""Verify every published input-review PNG against the on-disk frozen receipt."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.folder/'review_receipt.json').read_text())
    checks = []
    for name in ['index.html', 'review_receipt.json', *receipt['assets']]:
        expected = (args.folder/name).read_bytes()
        with urlopen(args.url.rstrip('/')+'/'+name, timeout=20) as response:
            assert response.status == 200
            actual = response.read()
        assert actual == expected, name
        if name.endswith('.png'):
            assert actual.startswith(b'\x89PNG\r\n\x1a\n'), name
            assert hashlib.sha256(actual).hexdigest() == receipt['assets'][name]
        checks.append(dict(asset=name, bytes=len(actual), sha256=hashlib.sha256(actual).hexdigest()))
    result = dict(passed=True, url=args.url, checks=checks,
                  predictions_generated=0, downloads_of_new_imagery=0)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, verified_served_assets=len(checks))))


if __name__ == '__main__':
    main()
