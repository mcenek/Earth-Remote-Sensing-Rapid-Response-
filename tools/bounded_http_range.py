"""A small, conservative, disk-backed HTTP range reader.

The reader is intentionally single-process.  It never follows a response which
does not prove that it is a bounded range response.
"""
from __future__ import annotations

import hashlib
import json
import re
import os
import urllib.parse
import urllib.request
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cache_dir: Path, limit_bytes: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limit_bytes = int(limit_bytes)
        self._path = self.cache_dir / "budget.json"
        try:
            state = json.loads(self._path.read_text())
        except FileNotFoundError:
            state = {}
        if self.limit_bytes <= 0:
            raise ValueError('Budget must be positive')
        if state:
            self.limit_bytes = min(self.limit_bytes, int(state['limit_bytes']))
        self.charged_bytes = int(state.get("charged_bytes", 0))
        self.received_bytes = int(state.get("received_bytes", 0))
        if self.charged_bytes < 0 or not 0 <= self.received_bytes <= self.charged_bytes:
            raise ValueError('Corrupt budget state; refuse to reset')

    def _save(self) -> None:
        temporary = self._path.with_suffix('.tmp')
        temporary.write_text(json.dumps({"limit_bytes": self.limit_bytes,
                                          "charged_bytes": self.charged_bytes,
                                          "received_bytes": self.received_bytes},
                                         sort_keys=True))
        os.replace(temporary, self._path)

    def reserve(self, amount: int) -> None:
        amount = int(amount)
        if amount < 0 or self.charged_bytes + amount > self.limit_bytes:
            raise BudgetExceeded("HTTP range budget exhausted")
        self.charged_bytes += amount
        self._save()                 # charge survives every network failure

    def received(self, amount: int) -> None:
        self.received_bytes += int(amount)
        self._save()

    def snapshot(self) -> dict:
        return {"limit_bytes": self.limit_bytes,
                "charged_bytes": self.charged_bytes,
                "received_bytes": self.received_bytes,
                "remaining_bytes": self.limit_bytes - self.charged_bytes}


_CR = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.I)
_HOST = "sentinel-cogs.s3.us-west-2.amazonaws.com"


class HTTPRangeReader:
    def __init__(self, url: str, budget: Budget, block_size: int = 262144):
        p = urllib.parse.urlsplit(url)
        if p.scheme != "https" or p.hostname != _HOST or not p.path:
            raise ValueError("only HTTPS sentinel-cogs S3 paths are allowed")
        self.url, self.budget = url, budget
        self.name = 'bounded-remote.tif'
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self._key = hashlib.sha256(url.encode()).hexdigest()
        self._dir = budget.cache_dir / self._key
        self._meta_path = self._dir / "meta.json"
        self._meta = self._load_meta()
        self._pos = 0
        self._size = self._meta.get("total")
        self._etag = self._meta.get("etag")

    def _load_meta(self):
        try:
            m = json.loads(self._meta_path.read_text())
            if m.get("url") == self.url and int(m.get("block_size")) == self.block_size:
                return m
        except FileNotFoundError:
            pass
        return {"url": self.url, "block_size": self.block_size, "blocks": {}}

    def _save_meta(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(self._meta, sort_keys=True))

    def _ensure_size(self):
        if self._size is not None:
            return
        self._fetch(0)

    def _cached(self, index):
        item = self._meta.get("blocks", {}).get(str(index))
        if not item:
            return None
        path = self._dir / (str(index) + ".bin")
        try:
            data = path.read_bytes()
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError
            return data
        except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
            raise ValueError('Corrupt cached block; refusing automatic redownload') from error

    def _fetch(self, index):
        cached = self._cached(index)
        if cached is not None:
            return cached
        start = index * self.block_size
        end = start + self.block_size - 1
        self.budget.reserve(self.block_size)
        headers = {"Range": f"bytes={start}-{end}"}
        if self._etag:
            headers["If-Match"] = self._etag
        response = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=20)
        try:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status != 206:
                raise IOError("server did not return HTTP 206")
            match = _CR.fullmatch(response.headers.get("Content-Range", "").strip())
            if not match:
                raise IOError("missing or invalid Content-Range")
            got_start, got_end, total = map(lambda x: int(x) if x != "*" else None, match.groups())
            if got_start != start or total is None or got_end < got_start or got_end != min(end, total-1):
                raise IOError("Content-Range does not match requested range")
            if self._size is not None and total != self._size:
                raise IOError("remote object size changed")
            etag = response.headers.get("ETag")
            if not etag:
                raise IOError('ETag required for consistent range reads')
            if self._etag and etag != self._etag:
                raise IOError("remote object ETag changed")
            body = response.read(got_end - got_start + 1)
            self.budget.received(len(body))
            if len(body) != got_end - got_start + 1:
                raise IOError("short HTTP range response")
        finally:
            response.close()
        self._size = total
        self._etag = self._etag or etag
        self._meta.update(total=total, etag=self._etag)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / (str(index) + ".bin")).write_bytes(body)
        self._meta.setdefault("blocks", {})[str(index)] = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        self._save_meta()
        return body

    def tell(self):
        return self._pos

    def __len__(self):
        self._ensure_size()
        return self._size

    def seek(self, offset, whence=0):
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self._pos + offset
        elif whence == 2:
            self._ensure_size()
            new = self._size + offset
        else:
            raise ValueError("invalid whence")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return new

    def readable(self): return True
    def seekable(self): return True
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError("unbounded read is disabled")
        if size == 0:
            return b""
        self._ensure_size()
        if self._pos >= self._size:
            return b""
        want = min(size, self._size - self._pos)
        out = bytearray()
        while want:
            index, intra = divmod(self._pos, self.block_size)
            data = self._fetch(index)
            take = min(want, len(data) - intra)
            if take <= 0:
                break
            out += data[intra:intra + take]
            self._pos += take
            want -= take
        return bytes(out)
