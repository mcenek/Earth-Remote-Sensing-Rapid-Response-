#!/usr/bin/env python3
"""Inventory and optionally fetch only frozen Model8 MARS assets.

The default is a read-only local hash audit. Downloads require an explicit
cumulative response-body byte cap. A durable ledger charges retries and
partial responses, including conservative reservations after abrupt crashes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"
URL_PREFIX = f"/datasets/UNEP-IMEO/MARS-S2L/resolve/{REVISION}/"
CHUNK = 1024 * 1024
MARGIN = 5 * 1024**3
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "handoff/catalogs/GTM_Model8_mars_required_20261030.jsonl.gz"
DEFAULT_CATALOG_SHA256 = "b20d27dd58f2fe765e12b72f35d87803a75b3199e0db7bea284441a5432fe77d"
PROFILE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def profile_paths(root: Path, profile: str | None) -> dict[str, Path]:
    if profile is not None and not PROFILE_PATTERN.fullmatch(profile):
        raise ValueError("Profile must be a lowercase alphanumeric, underscore, or hyphen slug of 1..64 characters")
    prefix = ".gtm_model8_fetch" if profile is None else f".gtm_model8_fetch_profile-{profile}"
    return {"receipt": root / f"{prefix}_receipt.json",
            "audit": root / f"{prefix}_audit.jsonl",
            "transfer": root / f"{prefix}_transfer.jsonl",
            "progress": root / f"{prefix}_progress.json"}


def acquire_claim(path: Path, wait_seconds: int) -> None:
    """Wait only for the shared cache claim; never delete or adopt its owner."""
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Existing cache operation; inspect its PID before retrying: {path}") from error
            print(json.dumps({"phase": "waiting_for_cache", "claim": str(path),
                              "seconds_remaining": round(remaining)}), flush=True)
            time.sleep(min(30, remaining))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as claim:
            json.dump({"pid": os.getpid(), "started_unix": time.time()}, claim)
    except BaseException:
        # This process created the claim but could not record ownership.
        path.unlink(missing_ok=True)
        raise


def checked_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or value.startswith("/") or "\\" in value or ":" in value or
            any(piece in {"", ".", ".."} for piece in value.split("/"))):
        raise ValueError(f"Unsafe catalog path: {value!r}")
    return path


def validate_item(item: dict) -> dict:
    required = {"path", "size", "source_url", "remote_oid", "remote_oid_type"}
    if not required <= item.keys():
        raise ValueError(f"Catalog object missing fields: {required - item.keys()}")
    path = checked_path(item["path"])
    if type(item["size"]) is not int or item["size"] < 0:
        raise ValueError(f"Invalid size for {path}")
    kind = item["remote_oid_type"]
    if kind not in {"sha256_lfs", "git_blob_sha1", "sha256_content"}:
        raise ValueError(f"Unsupported object hash type: {kind}")
    digest = item["remote_oid"]
    if not isinstance(digest, str) or len(digest) != (40 if kind == "git_blob_sha1" else 64) or any(
            letter not in "0123456789abcdef" for letter in digest):
        raise ValueError(f"Invalid object hash for {path}")
    url = urlsplit(item["source_url"])
    if (url.scheme != "https" or url.netloc != "huggingface.co" or
            url.path != URL_PREFIX + quote(str(path), safe="/") or url.query not in ("", "download=true") or
            url.fragment):
        raise ValueError(f"Unpinned or mismatched source URL for {path}")
    return item


def load_catalog(path: Path) -> tuple[list[dict], str]:
    identity = sha256(path)
    if path.resolve() == DEFAULT_CATALOG.resolve() and identity != DEFAULT_CATALOG_SHA256:
        raise ValueError("Bundled Model8 asset catalog SHA-256 changed")
    opener = gzip.open if path.suffix == ".gz" else open
    objects = {}
    with opener(path, "rt", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = validate_item(json.loads(line))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Catalog line {number}: {error}") from error
            old = objects.setdefault(item["path"], item)
            if old != item:
                raise ValueError(f"Conflicting duplicate catalog path: {item['path']}")
    if not objects:
        raise ValueError("Catalog is empty")
    return list(objects.values()), identity


def digest_file(path: Path, kind: str, size: int) -> str:
    digest = hashlib.sha1() if kind == "git_blob_sha1" else hashlib.sha256()
    if kind == "git_blob_sha1":
        digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(root: Path, item: dict) -> Path:
    target = root.joinpath(*checked_path(item["path"]).parts)
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError(f"Asset path escapes data root: {item['path']}")
    parent = target.parent
    while parent != root:
        if parent.is_symlink():
            raise ValueError(f"Symlinked asset parent: {parent}")
        parent = parent.parent
    return target


def audit_one(root: Path, item: dict) -> dict:
    target = destination(root, item)
    record = {"path": item["path"], "size": item["size"]}
    if target.is_symlink():
        return {**record, "state": "corrupt", "error": "Destination is a symlink"}
    if not target.exists():
        return {**record, "state": "missing"}
    if not target.is_file():
        return {**record, "state": "corrupt", "error": "Destination is not a file"}
    actual_size = target.stat().st_size
    if actual_size != item["size"]:
        return {**record, "state": "corrupt", "error": f"Size {actual_size} differs"}
    actual = digest_file(target, item["remote_oid_type"], item["size"])
    if actual != item["remote_oid"]:
        return {**record, "state": "corrupt", "error": f"Hash mismatch: {actual}"}
    return {**record, "state": "verified"}


class BudgetExceeded(RuntimeError):
    pass


class TransferBudget:
    """Persistent shared budget; reserve before reads, refund unused bytes."""
    def __init__(self, ledger: Path, identity: dict, cap: int, previous: int = 0,
                 progress_path: Path | None = None):
        self.lock = threading.Lock()
        self.ledger = ledger
        self.cap = cap
        self.progress_path = progress_path
        self.last_progress = time.monotonic()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        if ledger.exists():
            with ledger.open(encoding="utf-8") as source:
                header = json.loads(next(source))
                if header.get("identity") != identity:
                    raise ValueError("Transfer ledger belongs to another data root or catalog")
                self.used = header["base_bytes"]
                for line in source:
                    self.used += json.loads(line)["delta"]
            if self.used < previous:
                raise ValueError("Transfer ledger is behind prior receipt; inspect before resuming")
        else:
            self.used = previous
            with ledger.open("x", encoding="utf-8") as output:
                output.write(json.dumps({"identity": identity, "base_bytes": previous}) + "\n")
                output.flush()
                os.fsync(output.fileno())
        self.output = ledger.open("a", encoding="utf-8")

    def close(self) -> None:
        self.output.close()

    def _charge(self, delta: int) -> None:
        self.output.write(json.dumps({"delta": delta}, separators=(",", ":")) + "\n")
        self.output.flush()
        os.fsync(self.output.fileno())
        self.used += delta

    def read(self, stream, requested: int = CHUNK) -> bytes:
        with self.lock:
            allowed = min(requested, self.cap - self.used)
            if allowed <= 0:
                raise BudgetExceeded(f"Cumulative network response budget exhausted at {self.used} bytes")
            self._charge(allowed)
        try:
            block = stream.read(allowed)
        except BaseException:
            # Bytes possibly delivered before a failed read cannot be measured;
            # retain the reservation conservatively across crashes and retries.
            raise
        with self.lock:
            if len(block) < allowed:
                self._charge(len(block) - allowed)
            if self.progress_path is not None and time.monotonic() - self.last_progress >= 10:
                progress = {"phase": "transfer", "network_response_bytes": self.used,
                            "max_network_bytes": self.cap}
                atomic_json(self.progress_path, progress)
                print(json.dumps(progress), flush=True)
                self.last_progress = time.monotonic()
        return block


def download_one(root: Path, item: dict, budget: TransferBudget, attempts: int = 3) -> dict:
    target = destination(root, item)
    if target.exists() or target.is_symlink():
        return {"path": item["path"], "state": "failed", "error": "Destination appeared during download"}
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(1, attempts + 1):
        temporary = target.with_name(target.name + ".part." + uuid.uuid4().hex)
        try:
            request = Request(item["source_url"], headers={"User-Agent": "ERSRR-Model8-pinned-assets/1"})
            with urlopen(request, timeout=90) as response, temporary.open("xb") as output:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Download redirected away from HTTPS")
                while output.tell() < item["size"]:
                    block = budget.read(response, min(CHUNK, item["size"] - output.tell()))
                    if not block:
                        raise ValueError(f"Incomplete response: {output.tell()} of {item['size']}")
                    output.write(block)
            if temporary.stat().st_size != item["size"]:
                raise ValueError(f"Incomplete response: {temporary.stat().st_size} of {item['size']}")
            actual = digest_file(temporary, item["remote_oid_type"], item["size"])
            if actual != item["remote_oid"]:
                raise ValueError(f"Downloaded hash mismatch: {actual}")
            try:
                os.link(temporary, target)  # Atomic creation; never replace a concurrent file.
            except OSError as error:
                return {"path": item["path"], "state": "failed", "size": item["size"],
                        "attempts": attempt, "error": f"Atomic no-overwrite link failed: {error}",
                        "network_bytes_at_finish": budget.used}
            return {"path": item["path"], "state": "verified_download",
                    "size": item["size"], "attempts": attempt,
                    "network_bytes_at_finish": budget.used}
        except HTTPError as error:
            try:
                # Error bodies also consume the common budget.
                while budget.read(error, 64 * 1024):
                    pass
            except BudgetExceeded as limit:
                return {"path": item["path"], "state": "failed", "size": item["size"],
                        "attempts": attempt, "error": str(limit),
                        "network_bytes_at_finish": budget.used}
            except (OSError, ValueError):
                pass
            errors.append(f"HTTP {error.code}")
            if error.code not in (429, 500, 502, 503, 504):
                break
        except BudgetExceeded as error:
            return {"path": item["path"], "state": "failed", "size": item["size"],
                    "attempts": attempt, "error": str(error),
                    "network_bytes_at_finish": budget.used}
        except (OSError, URLError, ValueError) as error:
            errors.append(str(error))
            if isinstance(error, ValueError) and not str(error).startswith("Incomplete response"):
                break
        finally:
            temporary.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 4))
    return {"path": item["path"], "state": "failed", "size": item["size"],
            "attempts": len(errors), "errors": errors,
            "network_bytes_at_finish": budget.used,
            "note": "All response bytes on retries count toward the cumulative budget; partial files were discarded."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--profile", help="Separate receipt and budget namespace for a supplemental catalog")
    parser.add_argument("--wait-for-cache", type=int, default=0,
                        help="Seconds to wait for the shared cache claim; default is fail fast")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-network-bytes", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.max_errors < 1:
        parser.error("workers must be 1..8 and max-errors positive")
    if args.wait_for_cache < 0:
        parser.error("wait-for-cache must be nonnegative")
    if args.profile is not None and not PROFILE_PATTERN.fullmatch(args.profile):
        parser.error("profile must be a lowercase slug of 1..64 letters, digits, underscores, or hyphens")
    if args.download and (args.max_network_bytes is None or args.max_network_bytes < 1):
        parser.error("--download requires a positive cumulative --max-network-bytes")
    if not args.download and args.max_network_bytes is not None:
        parser.error("--max-network-bytes requires --download")
    root = args.data_root.resolve()
    if not root.is_dir():
        parser.error(f"Data root must already exist: {root}")
    claim_path = root / ".gtm_model8_fetch.claim"
    own_claim = False
    try:
        acquire_claim(claim_path, args.wait_for_cache)
        own_claim = True
        items, catalog_hash = load_catalog(args.catalog)
        identity = {"data_root": str(root), "catalog_sha256": catalog_hash}
        if args.profile is not None:
            identity["profile"] = args.profile
        paths = profile_paths(root, args.profile)
        receipt_path = paths["receipt"]
        audit_path = paths["audit"]
        ledger_path = paths["transfer"]
        audit_run_id = f"{time.time_ns()}-{os.getpid()}"
        previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
        if previous and previous.get("identity") != identity:
            raise ValueError("Existing fetch receipt belongs to another data root or catalog")
        receipt = {"identity": identity, "state": "inventory", "objects": len(items),
                   "audit_run_id": audit_run_id,
                   "logical_bytes": sum(item["size"] for item in items),
                   "network_response_bytes": previous.get("network_response_bytes", 0) if previous else 0,
                   "budget_note": "Cumulative response body bytes include retries. Interrupted reads may conservatively charge a reserved chunk.",
                   "partial_note": "Failed attempt temporary files are discarded; retry response bytes remain charged."}
        missing, corrupt = [], []
        atomic_json(receipt_path, receipt)
        last_progress = time.monotonic()
        with audit_path.open("a", encoding="utf-8", newline="\n") as audit:
            for index, item in enumerate(items, 1):
                result = audit_one(root, item)
                audit.write(json.dumps({"audit_run_id": audit_run_id, **result}, separators=(",", ":")) + "\n")
                if result["state"] == "missing":
                    missing.append(item)
                elif result["state"] == "corrupt":
                    corrupt.append(result)
                if index % 100 == 0 or time.monotonic() - last_progress >= 10:
                    audit.flush()
                    receipt.update(checked=index, missing=len(missing), corrupt_count=len(corrupt))
                    atomic_json(receipt_path, receipt)
                    print(json.dumps({"phase": "inventory", "checked": index, "total": len(items),
                                      "missing": len(missing), "corrupt": len(corrupt)}), flush=True)
                    last_progress = time.monotonic()
        receipt.update(verified_existing=len(items) - len(missing) - len(corrupt),
                       missing=len(missing), missing_bytes=sum(item["size"] for item in missing),
                       corrupt=corrupt[:args.max_errors], audit=str(audit_path))
        atomic_json(receipt_path, receipt)
        print(json.dumps({"phase": "inventory_complete", "verified_existing": receipt["verified_existing"],
                          "missing": len(missing), "missing_bytes": receipt["missing_bytes"],
                          "corrupt": len(corrupt)}), flush=True)
        if corrupt:
            receipt.update(state="failed", error="Existing corrupt assets were left untouched")
            atomic_json(receipt_path, receipt)
            return 1
        if not args.download or not missing:
            receipt["state"] = "complete" if not missing else "incomplete_dry_inventory"
            atomic_json(receipt_path, receipt)
            return 0 if not missing else 1
        free = shutil.disk_usage(root).free
        if free < receipt["missing_bytes"] + MARGIN:
            raise RuntimeError(f"Insufficient free disk: {free} < missing {receipt['missing_bytes']} + 5 GiB")
        budget = TransferBudget(ledger_path, identity, args.max_network_bytes,
                                receipt["network_response_bytes"],
                                paths["progress"])
        try:
            if budget.used >= budget.cap:
                receipt.update(state="incomplete_budget", network_response_bytes=budget.used,
                               error="Cumulative response budget already exhausted")
                atomic_json(receipt_path, receipt)
                return 1
            receipt.update(state="downloading", network_response_bytes=budget.used,
                           max_network_bytes=args.max_network_bytes)
            atomic_json(receipt_path, receipt)
            failures, downloaded = [], 0
            last_progress = time.monotonic()
            iterator = iter(missing)
            with audit_path.open("a", encoding="utf-8", newline="\n") as audit, \
                    ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = set()
                def submit_next() -> None:
                    try:
                        pending.add(pool.submit(download_one, root, next(iterator), budget))
                    except StopIteration:
                        pass
                for _ in range(args.workers):
                    submit_next()
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            result = future.result()
                        except BudgetExceeded as error:
                            result = {"state": "failed", "error": str(error)}
                        except Exception as error:
                            result = {"state": "failed", "error": repr(error)}
                        audit.write(json.dumps({"audit_run_id": audit_run_id, **result}, separators=(",", ":")) + "\n")
                        audit.flush()
                        if result["state"] == "verified_download":
                            downloaded += 1
                        else:
                            failures.append(result)
                        processed = downloaded + len(failures)
                        if processed % 100 == 0 or time.monotonic() - last_progress >= 10 or result["state"] == "failed":
                            receipt.update(downloaded=downloaded, failed=failures[:args.max_errors],
                                           network_response_bytes=budget.used,
                                           remaining=len(missing) - downloaded)
                            atomic_json(receipt_path, receipt)
                            print(json.dumps({"phase": "download", "processed": processed,
                                              "downloaded": downloaded, "missing_total": len(missing),
                                              "failures": len(failures),
                                              "network_response_bytes": budget.used}), flush=True)
                            last_progress = time.monotonic()
                        if len(failures) < args.max_errors and budget.used < budget.cap:
                            submit_next()
                    if len(failures) >= args.max_errors:
                        break
            receipt.update(state="complete" if downloaded == len(missing) else "incomplete",
                           downloaded=downloaded, failed=failures[:args.max_errors],
                           remaining=len(missing) - downloaded,
                           network_response_bytes=budget.used)
            atomic_json(receipt_path, receipt)
            return 0 if receipt["state"] == "complete" else 1
        finally:
            budget.close()
    except Exception as error:
        print(json.dumps({"state": "failed", "error": str(error)}, indent=2), file=sys.stderr)
        return 1
    finally:
        if own_claim:
            claim_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
