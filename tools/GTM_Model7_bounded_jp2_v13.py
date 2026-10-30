"""Frozen Sentinel L1C JP2 reader backed only by charged HTTP 206 ranges.

Use with ``rasterio.open(url, driver='JP2OpenJPEG', opener=FrozenJP2Opener(...))``.
The opener refuses sidecars, other URLs, writes and unbounded file-like reads.
It never delegates to GDAL's network filesystem. This is a transport adapter,
not a verified transfer-size estimate for real Sentinel JP2 codestreams.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request

from bounded_http_range import Budget

_HOST = "sentinel-s2-l1c.s3.amazonaws.com"
_PATH = re.compile(r"^/tiles/[0-9]{1,2}/[A-Z]/[A-Z]{2}/20[0-9]{2}/(?:[1-9]|1[0-2])/(?:[1-9]|[12][0-9]|3[01])/[0-9]+/B(?:0[2-4]|08|11|12)\.jp2$")
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)$", re.I)


def validate_frozen_url(url: str, allowed_urls) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (url not in allowed_urls or parsed.scheme != "https" or parsed.netloc != _HOST
            or not _PATH.fullmatch(parsed.path) or parsed.query or parsed.fragment):
        raise ValueError("URL is not an exact frozen Sentinel L1C JP2 asset")


def load_frozen_urls(preflight_path: Path, expected_sha256: str) -> frozenset[str]:
    """Load only the saved three-event L1C current/reference asset list."""
    body = Path(preflight_path).read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("Frozen JP2 asset preflight hash mismatch")
    document = json.loads(body)
    events = document.get("events", [])
    if document.get("status") != "offline_preflight_only_no_transfer" or len(events) != 3:
        raise ValueError("Unexpected JP2 asset preflight")
    urls = []
    for event in events:
        for role in ("same_domain_current", "same_domain_reference"):
            scene = event[role]
            if scene["collection"] != "sentinel-2-l1c":
                raise ValueError("Frozen scene is not L1C")
            assets = scene["spectral_assets"]
            if set(assets) != {"blue", "green", "red", "nir", "swir16", "swir22"}:
                raise ValueError("Frozen six-band set changed")
            urls.extend(info["normalized_https_url"] for info in assets.values())
    if len(urls) != 36 or len(set(urls)) != 36:
        raise ValueError("Frozen URL count or uniqueness changed")
    allowed = frozenset(urls)
    for url in allowed:
        validate_frozen_url(url, allowed)
    return allowed


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _default_transport(request):
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=20)


class JP2RangeReader:
    def __init__(self, url: str, budget: Budget, *, allowed_urls, block_size=262144,
                 transport=None):
        validate_frozen_url(url, allowed_urls)
        if not 256 <= int(block_size) <= 1048576:
            raise ValueError("Block size outside bounded 256 B to 1 MiB range")
        self.url = url
        self.name = url
        self.budget = budget
        self.block_size = int(block_size)
        self.transport = transport or _default_transport
        self._dir = Path(budget.cache_dir) / "jp2-" / hashlib.sha256(url.encode()).hexdigest()
        self._meta_path = self._dir / "meta.json"
        self._meta = self._load_meta()
        self._size = self._meta.get("total")
        self._etag = self._meta.get("etag")
        self._pos = 0
        self.closed = False

    def _load_meta(self):
        if not self._meta_path.exists():
            return {"url": self.url, "block_size": self.block_size, "blocks": {}}
        value = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if (value.get("url") != self.url or value.get("block_size") != self.block_size
                or not isinstance(value.get("blocks"), dict)):
            raise ValueError("Changed JP2 cache identity")
        if value.get("total") is not None and (not isinstance(value["total"], int) or value["total"] <= 0):
            raise ValueError("Invalid cached JP2 size")
        return value

    def _save_meta(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        temporary = self._meta_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._meta, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._meta_path)

    def _cached(self, index):
        entry = self._meta["blocks"].get(str(index))
        if entry is None:
            return None
        path = self._dir / f"{index}.bin"
        try:
            content = path.read_bytes()
            if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Corrupt cached JP2 range; refusing redownload") from error
        return content

    def _fetch(self, index):
        cached = self._cached(index)
        if cached is not None:
            return cached
        start = index * self.block_size
        if self._size is not None and start >= self._size:
            return b""
        end = start + self.block_size - 1
        if self._size is not None:
            end = min(end, self._size - 1)
        self.budget.reserve(self.block_size + 1)
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self._etag:
            headers["If-Match"] = self._etag
        request = urllib.request.Request(self.url, headers=headers)
        response = self.transport(request)
        try:
            if response.geturl() != self.url:
                raise OSError("Redirect or URL substitution refused")
            if getattr(response, "status", None) != 206:
                raise OSError("HTTP 206 required")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise OSError("Encoded range response refused")
            match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", "").strip())
            if not match:
                raise OSError("Valid finite Content-Range required")
            got_start, got_end, total = map(int, match.groups())
            if got_start != start or total <= 0 or got_end != min(end, total - 1) or got_end < got_start:
                raise OSError("Content-Range differs from requested block")
            if self._size is not None and total != self._size:
                raise OSError("Remote object size changed")
            etag = response.headers.get("ETag")
            if not etag or (self._etag and etag != self._etag):
                raise OSError("ETag missing or changed")
            expected = got_end - got_start + 1
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != expected:
                raise OSError("Content-Length differs from range")
            body = response.read(expected)
            self.budget.received(len(body))
            if len(body) != expected:
                raise OSError("Short HTTP range body")
        finally:
            response.close()
        self._size = total
        self._etag = etag
        self._meta.update(total=total, etag=etag)
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{index}.bin").write_bytes(body)
        self._meta["blocks"][str(index)] = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        self._save_meta()
        return body

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Closed JP2 reader")
        if size is None or size < 0:
            raise ValueError("Unbounded JP2 read refused")
        if size == 0:
            return b""
        if self._size is None:
            self._fetch(0)
        if self._pos >= self._size:
            return b""
        remaining = min(size, self._size - self._pos)
        chunks = bytearray()
        while remaining:
            index, intra = divmod(self._pos, self.block_size)
            block = self._fetch(index)
            part = block[intra:intra+remaining]
            if not part:
                raise OSError("Missing range bytes")
            chunks.extend(part)
            self._pos += len(part)
            remaining -= len(part)
        return bytes(chunks)

    def seek(self, offset, whence=0):
        if self.closed:
            raise ValueError("Closed JP2 reader")
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self._pos + offset
        elif whence == 2:
            if self._size is None:
                self._fetch(0)
            target = self._size + offset
        else:
            raise ValueError("Invalid seek mode")
        if target < 0:
            raise ValueError("Negative seek")
        self._pos = target
        return target

    def tell(self):
        return self._pos

    def close(self):
        self.closed = True

    def readable(self):
        return True

    def seekable(self):
        return True

    def __len__(self):
        if self._size is None:
            self._fetch(0)
        return self._size

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FrozenJP2Opener:
    def __init__(self, allowed_urls, budget: Budget, *, block_size=262144, transport=None):
        self.allowed_urls = frozenset(allowed_urls)
        if not self.allowed_urls:
            raise ValueError("Empty frozen URL set")
        for url in self.allowed_urls:
            validate_frozen_url(url, self.allowed_urls)
        self.budget = budget
        self.block_size = block_size
        self.transport = transport

    def __call__(self, fp, mode="rb"):
        if mode not in ("r", "rb"):
            raise ValueError("Read-only JP2 opener")
        if fp not in self.allowed_urls:
            raise FileNotFoundError(fp)
        return JP2RangeReader(fp, self.budget, allowed_urls=self.allowed_urls,
                              block_size=self.block_size, transport=self.transport)
