"""Fetch only the public Cambridge example and the two SDK registries."""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re


RUN = Path(__file__).resolve().parent / "local-run"
HOST = "data.source.coop"
PREFIX = "/tessera/tessera"
GRID = "grid_0.15_52.05"
FILES = {
    "npy/v1.1-cam/manifest.parquet": "registries/manifest.parquet",
    "landmasks/v1.1/landmasks.parquet": "registries/landmasks.parquet",
    f"landmasks/v1.1/{GRID}.tiff": f"tile/global_0.1_degree_tiff_all/{GRID}.tiff",
    f"npy/v1.1-cam/2024/{GRID}/{GRID}_scales.npy": f"tile/global_0.1_degree_representation/2024/{GRID}/{GRID}_scales.npy",
    f"npy/v1.1-cam/2024/{GRID}/{GRID}.npy": f"tile/global_0.1_degree_representation/2024/{GRID}/{GRID}.npy",
}
HEADER_ALLOWANCE = 65_536


def fetch(cap):
    if cap <= HEADER_ALLOWANCE * len(FILES):
        raise ValueError("Byte cap must exceed per-request header allowance")
    RUN.mkdir(parents=True, exist_ok=True)
    rows = []
    charged = 0
    try:
        for remote, local in FILES.items():
            target = RUN / local
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            if target.exists() or partial.exists():
                raise FileExistsError(target)
            if charged + HEADER_ALLOWANCE >= cap:
                raise RuntimeError("Smoke-test budget exhausted")
            connection = http.client.HTTPSConnection(HOST, timeout=25)
            entry = {"url": f"https://{HOST}{PREFIX}/{remote}", "path": local, "bytes": 0}
            rows.append(entry)
            charged += HEADER_ALLOWANCE
            try:
                connection.request("GET", f"{PREFIX}/{remote}", headers={"Accept-Encoding": "identity"})
                response = connection.getresponse()
                entry["status"] = response.status
                if response.status != 200 or response.getheader("Content-Encoding", "identity") != "identity":
                    raise RuntimeError(f"Unexpected HTTP response {response.status} for {remote}")
                length = int(response.getheader("Content-Length", "0"))
                if not 0 < length <= cap - charged:
                    raise RuntimeError(f"Unsafe Content-Length {length} for {remote}")
                etag = response.getheader("ETag", "").strip('"')
                sha = hashlib.sha256()
                md5 = hashlib.md5(usedforsecurity=False)
                with partial.open("xb") as output:
                    while entry["bytes"] < length:
                        chunk = response.read(min(262_144, length - entry["bytes"]))
                        if not chunk:
                            raise OSError(f"Truncated transfer for {remote}")
                        entry["bytes"] += len(chunk)
                        charged += len(chunk)
                        output.write(chunk)
                        sha.update(chunk)
                        md5.update(chunk)
                if re.fullmatch(r"[0-9a-f]{32}", etag) and md5.hexdigest() != etag:
                    raise OSError(f"ETag checksum mismatch for {remote}")
                partial.rename(target)
                with target.open("rb") as saved:
                    assert hashlib.file_digest(saved, "sha256").hexdigest() == sha.hexdigest()
                entry.update(sha256=sha.hexdigest(), etag=etag)
                print(f"Fetched {local}: {length:,} bytes", flush=True)
            finally:
                connection.close()
    finally:
        receipt = RUN / "download.json"
        payload = (json.dumps({"objects": rows, "charged_bytes": charged, "cap_bytes": cap}, indent=2) + "\n").encode()
        with receipt.open("xb") as output:
            output.write(payload)
        assert receipt.read_bytes() == payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-bytes", type=int, required=True,
                        help="Maximum response body bytes plus per-request header allowance")
    fetch(parser.parse_args().cap_bytes)
