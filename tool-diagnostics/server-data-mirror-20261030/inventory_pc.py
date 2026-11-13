"""Read-only path/size inventory of the existing provider catalog on this PC."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    counts = Counter()
    sizes = Counter()
    missing = []
    mismatches = []
    started = time.monotonic()
    with args.catalog.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            name = row["path"]
            target = root.joinpath(*name.split("/"))
            group = "/".join(name.split("/")[:2])
            counts["expected"] += 1
            sizes["expected"] += row["size"]
            try:
                actual = target.stat().st_size
            except FileNotFoundError:
                counts["missing"] += 1
                sizes["missing"] += row["size"]
                if len(missing) < 20:
                    missing.append(name)
            else:
                counts["present"] += 1
                sizes["present"] += actual
                counts[group] += 1
                sizes[group] += actual
                if actual != row["size"]:
                    counts["size_mismatch"] += 1
                    if len(mismatches) < 20:
                        mismatches.append({"path": name, "expected": row["size"], "actual": actual})
            if counts["expected"] % 25000 == 0:
                print(json.dumps({"checked": counts["expected"], "present": counts["present"],
                                  "missing": counts["missing"]}), flush=True)
    report = {"root": str(root), "catalog": str(args.catalog.resolve()),
              "counts": dict(counts), "bytes": dict(sizes),
              "missing_examples": missing, "size_mismatch_examples": mismatches,
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "verification": "Presence and size only; no full file-content hashing in this inventory."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
