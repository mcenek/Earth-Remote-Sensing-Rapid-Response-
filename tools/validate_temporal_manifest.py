#!/usr/bin/env python3
"""Validate a prospective, imagery-independent temporal dataset manifest.

Schema (``schema_version == "temporal-manifest/v1"``)::

    {"schema_version": "temporal-manifest/v1", "samples": [
      {"sample_id": "...", "split": "train|validation|test",
       "site_id": "...", "event_id": "...", "latitude": 0.0,
       "longitude": 0.0,
       "target": {"path": "...", "provenance": "...",
                   "annotation_available": true, "mask_path": "..."},
       "reference": {"available": true, "path": "...", "provenance": "..."}}
    ]}

An unavailable reference is represented explicitly as
``{"available": false, "path": null, "provenance": null}``.

Provenance values are expected to identify an observation or product
instance, rather than only a generic provider name; exact repeated values are
treated as possible cross-split reuse. The report is metadata-only:
``imagery_validated`` is always false.

Paths are interpreted relative to the manifest file when ``base_dir`` is
provided. This checks metadata and optional local file presence only; it does
not establish site independence from coordinates or validate imagery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "temporal-manifest/v1"
SPLITS = frozenset(("train", "validation", "test"))


def _error(errors: list[dict[str, str]], code: str, message: str, path: str) -> None:
    errors.append({"code": code, "message": message, "path": path})


def _report(errors: list[dict[str, str]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": not errors,
        "errors": errors,
        "summary": summary,
        "metadata_only": True,
        "imagery_validated": False,
    }


def validate_manifest(
    manifest: Any, *, base_dir: str | Path | None = None, check_files: bool = False
) -> dict[str, Any]:
    """Return a deterministic validation report; never accesses the network."""
    errors: list[dict[str, str]] = []
    if not isinstance(manifest, dict):
        _error(errors, "root_type", "manifest must be an object", "$")
        return _report(errors, {"samples": 0})
    unknown_root = sorted(set(manifest) - {"schema_version", "samples"})
    for key in unknown_root:
        _error(errors, "unknown_field", "field is not part of the v1 schema", key)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _error(errors, "schema_version", f"must equal {SCHEMA_VERSION!r}", "schema_version")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        _error(errors, "samples_type", "samples must be an array", "samples")
        return _report(errors, {"samples": 0})
    if not samples:
        _error(errors, "empty_samples", "samples must contain at least one sample", "samples")

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    seen_ids: set[str] = set()
    groups: dict[str, dict[str, set[str]]] = {"site_id": {}, "event_id": {}}
    coordinates: dict[tuple[float, float], set[str]] = {}
    paths: dict[str, set[str]] = {}
    provenances: dict[str, set[str]] = {}
    counts = {split: 0 for split in sorted(SPLITS)}
    required = {"sample_id", "split", "site_id", "event_id", "latitude", "longitude", "target", "reference"}
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            _error(errors, "sample_type", "sample must be an object", prefix)
            continue
        for key in sorted(set(sample) - required):
            # mask_path is nested under target; every sample key is fixed.
            _error(errors, "unknown_field", "field is not part of the v1 schema", f"{prefix}.{key}")
        missing = sorted(required - sample.keys())
        for key in missing:
            _error(errors, "required", f"missing required field {key!r}", f"{prefix}.{key}")
        sid = sample.get("sample_id")
        split = sample.get("split")
        if not isinstance(sid, str) or not sid.strip():
            _error(errors, "sample_id", "must be a non-empty string", f"{prefix}.sample_id")
        elif sid in seen_ids:
            _error(errors, "duplicate_sample_id", f"duplicate sample_id {sid!r}", f"{prefix}.sample_id")
        else:
            seen_ids.add(sid)
        split_valid = isinstance(split, str) and split in SPLITS
        if not split_valid:
            _error(errors, "split", "must be train, validation, or test", f"{prefix}.split")
        else:
            counts[split] += 1
        site = sample.get("site_id")
        event = sample.get("event_id")
        for name, value in (("site_id", site), ("event_id", event)):
            if not isinstance(value, str) or not value.strip():
                _error(errors, name, "must be a non-empty string", f"{prefix}.{name}")
            elif split_valid:
                previous = groups[name].setdefault(value, set())
                previous.add(split)
                if len(previous) > 1:
                    _error(errors, f"{name}_split_leakage", f"{name} occurs across splits", f"{prefix}.{name}")
        lat, lon = sample.get("latitude"), sample.get("longitude")
        valid_coord = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in (lat, lon))
        if not valid_coord or not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
            _error(errors, "coordinates", "latitude/longitude must be finite and within [-90,90]/[-180,180]", f"{prefix}.latitude")
        elif split_valid:
            point = (float(lat), float(lon))
            splits_at_point = coordinates.setdefault(point, set())
            splits_at_point.add(split)
            if len(splits_at_point) > 1:
                _error(errors, "coordinate_split_leakage", "identical coordinates occur across splits", f"{prefix}.latitude")
        for kind in ("target", "reference"):
            item = sample.get(kind)
            item_prefix = f"{prefix}.{kind}"
            if not isinstance(item, dict):
                _error(errors, "provenance", f"{kind} must be an object", item_prefix)
                continue
            allowed = {"path", "provenance", "available"} if kind == "reference" else {"path", "provenance", "annotation_available", "mask_path"}
            for key in sorted(set(item) - allowed):
                _error(errors, "unknown_field", "field is not part of the v1 schema", f"{item_prefix}.{key}")
            if kind == "reference":
                available = item.get("available")
                if not isinstance(available, bool):
                    _error(errors, "reference_availability", "available must be boolean", f"{item_prefix}.available")
                    available = None
                if available is False:
                    if item.get("path") is not None or item.get("provenance") is not None:
                        _error(errors, "reference_fields", "path and provenance must be null when reference is unavailable", item_prefix)
                    continue
            for key in ("path", "provenance"):
                value = item.get(key)
                if not isinstance(value, str) or not value.strip():
                    _error(errors, "provenance" if key == "provenance" else "path", f"{key} must be a non-empty string", f"{item_prefix}.{key}")
                elif key == "path" and check_files and root is not None and not (root / value).is_file():
                    _error(errors, "missing_file", "path does not exist as a file", f"{item_prefix}.{key}")
                if split_valid and isinstance(value, str) and value.strip():
                    identity = str((root / value).resolve()) if key == "path" else value.strip()
                    registry = paths if key == "path" else provenances
                    prior_splits = registry.setdefault(identity, set())
                    prior_splits.add(split)
                    if len(prior_splits) > 1:
                        label = "resolved asset path" if key == "path" else "observation provenance"
                        _error(errors, "observation_split_leakage", f"same {label} occurs across splits", f"{item_prefix}.{key}")
            if kind == "target":
                available = item.get("annotation_available")
                if not isinstance(available, bool):
                    _error(errors, "annotation_availability", "annotation_available must be boolean", f"{item_prefix}.annotation_available")
                mask = item.get("mask_path")
                if available is True and (not isinstance(mask, str) or not mask.strip()):
                    _error(errors, "missing_annotation", "mask_path is required when annotation_available is true", f"{item_prefix}.mask_path")
                if available is False and mask not in (None, ""):
                    _error(errors, "annotation_mask", "mask_path must be absent or null when annotation_available is false", f"{item_prefix}.mask_path")
                if check_files and root is not None and isinstance(mask, str) and mask.strip() and not (root / mask).is_file():
                    _error(errors, "missing_file", "path does not exist as a file", f"{item_prefix}.mask_path")
    return _report(errors, {"samples": len(samples), "by_split": counts})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args(argv)
    try:
        with args.manifest.open(encoding="utf-8") as source:
            manifest = json.load(source)
        report = validate_manifest(manifest, base_dir=args.manifest.parent, check_files=args.check_files)
    except (OSError, json.JSONDecodeError) as exc:
        report = _report(
            [{"code": "read_error", "message": str(exc), "path": str(args.manifest)}],
            {"samples": 0},
        )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
