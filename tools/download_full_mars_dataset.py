#!/usr/bin/env python3
"""Download the pinned MARS-S2L release to a directory on the ML server.

Install: python -m pip install 'huggingface_hub>=1.29.0'
Estimate: python tools/download_full_mars_dataset.py /data/MARS-S2L
Download: python tools/download_full_mars_dataset.py /data/MARS-S2L --download

The estimate queries the Hub and checks the local cache; it transfers no imagery.
The dataset and released models are licensed CC BY-NC-SA 4.0 by their authors.
"""
import argparse
import shutil
from pathlib import Path

REPO = "UNEP-IMEO/MARS-S2L"
REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("destination", type=Path, help="Directory for the dataset")
    parser.add_argument("--download", action="store_true", help="Transfer files after estimating size")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent transfers (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(1, "Install huggingface_hub first: python -m pip install 'huggingface_hub>=1.29.0'\n")

    destination = args.destination.expanduser().resolve()
    options = dict(repo_id=REPO, repo_type="dataset", revision=REVISION,
                   local_dir=destination, max_workers=args.workers)
    files = snapshot_download(**options, dry_run=True)
    total = sum(item.file_size for item in files)
    remaining = sum(item.file_size for item in files if item.will_download)
    print(f"Release: {REPO} at {REVISION}", flush=True)
    print(f"Destination: {destination}", flush=True)
    print(f"Files: {len(files):,}; total: {total / 1e9:.2f} GB; estimated remaining: {remaining / 1e9:.2f} GB", flush=True)
    if not args.download:
        print("Estimate only. Add --download to start the transfer.")
        return
    parent = destination
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < remaining + 5 * 1024**3:
        parser.exit(1, "Not enough free space for the estimated download plus a 5 GiB reserve.\n")
    result = snapshot_download(**options)
    print(f"Download complete: {result}")


if __name__ == "__main__":
    main()
