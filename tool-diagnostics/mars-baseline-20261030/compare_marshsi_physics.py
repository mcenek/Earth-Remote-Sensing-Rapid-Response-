#!/usr/bin/env python3
"""Compare 0.1.4 active physics definitions to the 0.1.6 wheel as source only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
WHEEL = HERE / "marshsi-0.1.6-py3-none-any.whl"
NAMES = (
    "marshsi/lut.py", "marshsi/quantification.py",
    "marshsi/output_Tch4_LUT_AMF_VZA_0_v2.nc",
)


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def definitions(content: bytes) -> dict:
    tree = ast.parse(content.decode("utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for name in names:
                if name.isupper():
                    result[name] = ast.dump(node.value, include_attributes=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay", type=Path)
    args = parser.parse_args()
    overlay = args.overlay.resolve()
    result = {"wheel_sha256": sha(WHEEL.read_bytes()), "files": {}}
    with zipfile.ZipFile(WHEEL) as archive:
        for name in NAMES:
            old = (overlay / name).read_bytes()
            new = archive.read(name)
            item = {"installed_014_sha256": sha(old), "official_016_sha256": sha(new),
                    "bytes_identical": old == new}
            if name.endswith(".py"):
                old_defs = definitions(old)
                new_defs = definitions(new)
                item["ast_changed_definitions"] = sorted(
                    key for key in old_defs.keys() | new_defs.keys() if old_defs.get(key) != new_defs.get(key)
                )
            result["files"][name] = item
    path = HERE / "marshsi_physics_comparison.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
