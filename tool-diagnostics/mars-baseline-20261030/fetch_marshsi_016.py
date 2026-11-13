#!/usr/bin/env python3
"""Fetch the official 0.1.6 pure Python wheel for a source-only comparison."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
URL = "https://pypi.org/pypi/marshsi/0.1.6/json"

with urlopen(URL, timeout=30) as response:
    metadata = json.load(response)
wheel = next(item for item in metadata["urls"] if item["filename"] == "marshsi-0.1.6-py3-none-any.whl")
with urlopen(wheel["url"], timeout=60) as response:
    content = response.read()
digest = hashlib.sha256(content).hexdigest()
if digest != wheel["digests"]["sha256"] or len(content) != wheel["size"]:
    raise ValueError("PyPI wheel digest or length mismatch")
path = HERE / "marshsi-0.1.6-py3-none-any.whl"
path.write_bytes(content)
print(json.dumps({"wheel": str(path), "sha256": digest, "bytes": len(content)}))
