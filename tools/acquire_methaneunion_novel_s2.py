#!/usr/bin/env python3
"""Acquire and adjudicate the frozen novel MethaneUnion Sentinel-2 training rows.

Only the 1,632 preregistered rows from the released training manifest are retained.
Archives are downloaded, verified, streamed once, receipted, and deleted one at a
time so the 145 GiB release is never resident on disk.  No released test raster is
opened or extracted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "4")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import numpy as np
import rasterio
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/methaneunion_novel_s2_acquisition_protocol.json"
ASSET_COLUMNS = {
    "S2_t0_path": "target",
    "S2_pre_path": "reference90",
    "S2_pre_pre_path": "reference365",
    "S2_plume_label_path": "mask",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def checked_repo_path(relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        path = Path(os.path.abspath(candidate))
    else:
        path = Path(os.path.abspath(ROOT / candidate))
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path escapes repository root: {relative}")
    return path


def checked_member_path(relative: str, row_id: int, role: str) -> PurePosixPath:
    value = PurePosixPath(relative)
    expected_name = {
        "target": "s2.tif",
        "reference90": "s2_pre.tif",
        "reference365": "s2_pre_pre.tif",
        "mask": "s2_plume.tif",
    }[role]
    expected_parts = ("data", "480m_GSD", "s2", str(row_id), expected_name)
    if value.is_absolute() or ".." in value.parts or value.parts != expected_parts:
        raise ValueError(f"Unsafe or unexpected MethaneUnion asset path: {relative!r}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_file_identity(path: Path, identity: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(identity["bytes"]) or sha256(path) != identity["sha256"]:
        raise ValueError(f"Pinned file identity mismatch: {path}")


def read_frozen_rows(
    manifest_path: Path, candidate_path: Path
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, str]]]:
    candidates = load_json(candidate_path)
    candidate_ids = [int(value) for value in candidates["candidate_ids"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Frozen candidate IDs are not unique")
    candidate_set = set(candidate_ids)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        selected = {
            int(row["id"]): row
            for row in csv.DictReader(handle)
            if int(row["id"]) in candidate_set
        }
    if set(selected) != candidate_set:
        missing = sorted(candidate_set - set(selected))
        raise ValueError(f"Training manifest lacks {len(missing)} frozen candidate IDs")

    rows: list[dict[str, Any]] = []
    expected: dict[str, tuple[int, str]] = {}
    for index, row_id in enumerate(sorted(candidate_ids)):
        source = selected[row_id]
        row = {
            "id": row_id,
            "label": int(source["label"]),
            "latitude": float(source["latitude"]),
            "longitude": float(source["longitude"]),
            "available_sensor": source["available_sensor"],
        }
        if "S2" not in {part.strip() for part in source["available_sensor"].split(",")}:
            raise ValueError(f"Frozen candidate {row_id} lacks Sentinel-2")
        for column, role in ASSET_COLUMNS.items():
            path = checked_member_path(source[column].strip(), row_id, role)
            member = path.as_posix()
            if member in expected:
                raise ValueError(f"Duplicate expected archive member: {member}")
            expected[member] = (index, role)
            row[column] = member
        rows.append(row)
    return rows, expected


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat_left, lat_right = map(math.radians, (left[0], right[0]))
    dlat = math.radians(right[0] - left[0])
    dlon = math.radians(right[1] - left[1])
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat_left) * math.cos(lat_right) * math.sin(dlon / 2) ** 2
    )
    return 12742.0 * math.asin(min(1.0, math.sqrt(value)))


def connected_groups(
    rows: Sequence[Mapping[str, Any]], distance_km: float
) -> tuple[list[str], dict[str, Any]]:
    coordinates = sorted({(float(row["latitude"]), float(row["longitude"])) for row in rows})
    parent = list(range(len(coordinates)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(coordinates)):
        for right in range(left):
            if haversine_km(coordinates[left], coordinates[right]) <= distance_km:
                union(left, right)

    members: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for index, point in enumerate(coordinates):
        members[find(index)].append(point)
    ordered = sorted(
        members.values(), key=lambda values: (min(values)[0], min(values)[1], len(values))
    )
    point_to_group: dict[tuple[float, float], str] = {}
    for index, points in enumerate(ordered):
        group_id = f"mu25-{index:03d}"
        for point in points:
            point_to_group[point] = group_id
    row_groups = [
        point_to_group[(float(row["latitude"]), float(row["longitude"]))] for row in rows
    ]
    group_labels: dict[str, set[int]] = defaultdict(set)
    group_rows = Counter(row_groups)
    for row, group_id in zip(rows, row_groups, strict=True):
        group_labels[group_id].add(int(row["label"]))
    summary = {
        "distance_km_inclusive": distance_km,
        "coordinates": len(coordinates),
        "connected_groups": len(ordered),
        "groups_with_label": {
            str(label): sum(label in values for values in group_labels.values())
            for label in (0, 1)
        },
        "mixed_label_groups": sum(len(values) == 2 for values in group_labels.values()),
        "maximum_coordinates_per_group": max(map(len, ordered)),
        "maximum_rows_per_group": max(group_rows.values()),
    }
    return row_groups, summary


def archive_local_path(scratch: Path, downloaded: str | Path) -> Path:
    path = Path(downloaded).resolve()
    base = scratch.resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"Downloader returned path outside scratch directory: {path}")
    return path


def safe_output_path(output_root: Path, member: str) -> Path:
    value = PurePosixPath(member)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Unsafe archive member: {member!r}")
    path = Path(os.path.abspath(output_root.joinpath(*value.parts)))
    base = Path(os.path.abspath(output_root))
    if path != base and base not in path.parents:
        raise ValueError(f"Archive member escapes output root: {member!r}")
    return path


def atomic_copy_member(source: Any, destination: Path, maximum_bytes: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError(f"Selected archive member exceeds {maximum_bytes} bytes")
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if total <= 0:
            raise ValueError("Selected archive member is empty")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"bytes": total, "sha256": digest.hexdigest()}


def stream_selected_archive(
    archive_path: Path,
    expected: Mapping[str, tuple[int, str]],
    already_seen: set[str],
    output_root: Path,
    maximum_member_bytes: int,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if member.name not in expected:
                continue
            if member.name in already_seen or any(item["member"] == member.name for item in found):
                raise ValueError(f"Duplicate selected member across archives: {member.name}")
            if not member.isfile() or member.size <= 0 or member.size > maximum_member_bytes:
                raise ValueError(f"Invalid selected archive member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not open selected archive member: {member.name}")
            destination = safe_output_path(output_root, member.name)
            with source:
                identity = atomic_copy_member(source, destination, maximum_member_bytes)
            if identity["bytes"] != member.size:
                raise ValueError(f"Truncated selected archive member: {member.name}")
            found.append({"member": member.name, **identity})
    return sorted(found, key=lambda item: item["member"])


def load_receipts(
    receipt_root: Path,
    archives: Sequence[Mapping[str, Any]],
    expected: Mapping[str, tuple[int, str]],
    output_root: Path,
) -> tuple[set[str], set[str]]:
    seen: set[str] = set()
    complete: set[str] = set()
    frozen = {item["name"]: item for item in archives}
    if not receipt_root.exists():
        return seen, complete
    for path in sorted(receipt_root.glob("*.json")):
        receipt = load_json(path)
        name = receipt["archive"]["name"]
        if name not in frozen or receipt["archive"] != frozen[name]:
            raise ValueError(f"Receipt does not match frozen archive identity: {path}")
        for item in receipt["selected_members"]:
            if item["member"] in seen:
                raise ValueError(f"Duplicate selected member in receipts: {item['member']}")
            if item["member"] not in expected:
                raise ValueError(f"Receipt contains a member outside the frozen subset: {path}")
            extracted = safe_output_path(output_root, item["member"])
            verify_file_identity(extracted, item)
            seen.add(item["member"])
        complete.add(name)
    return seen, complete


def acquire_archives(
    protocol: Mapping[str, Any],
    expected: Mapping[str, tuple[int, str]],
    output_root: Path,
    scratch: Path,
    receipt_root: Path,
) -> dict[str, Any]:
    archives = protocol["source"]["archives"]
    seen, complete = load_receipts(receipt_root, archives, expected, output_root)
    unknown = seen - set(expected)
    if unknown:
        raise ValueError(f"Receipts contain {len(unknown)} members outside the frozen subset")
    scratch.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    downloaded_bytes = 0
    for index, archive in enumerate(archives, start=1):
        name = archive["name"]
        if name in complete:
            print(f"[{index:02d}/{len(archives)}] Reusing verified receipt {name}", flush=True)
            continue
        print(f"[{index:02d}/{len(archives)}] Downloading {name}", flush=True)
        downloaded = hf_hub_download(
            repo_id=protocol["source"]["repo_id"],
            filename=name,
            revision=protocol["source"]["revision"],
            repo_type="dataset",
            local_dir=scratch,
        )
        archive_path = archive_local_path(scratch, downloaded)
        verify_file_identity(archive_path, archive)
        downloaded_bytes += archive_path.stat().st_size
        print(f"[{index:02d}/{len(archives)}] Verified; scanning {name}", flush=True)
        found = stream_selected_archive(
            archive_path,
            expected,
            seen,
            output_root,
            int(protocol["extraction"]["maximum_member_bytes"]),
        )
        receipt = {
            "schema_version": 1,
            "archive": dict(archive),
            "selected_members": found,
            "selected_members_count": len(found),
        }
        receipt_path = receipt_root / f"{name}.json"
        atomic_write(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        seen.update(item["member"] for item in found)
        archive_path.unlink()
        print(
            f"[{index:02d}/{len(archives)}] Retained {len(found)} members; "
            f"{len(seen)}/{len(expected)} total",
            flush=True,
        )
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Pinned archives lack {len(missing)} frozen candidate assets")
    return {
        "archives_receipted": len(complete) + (len(archives) - len(complete)),
        "members_retained": len(seen),
        "network_bytes_this_execution": downloaded_bytes,
    }


def same_grid(left: rasterio.DatasetReader, right: rasterio.DatasetReader) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and tuple(left.transform) == tuple(right.transform)
    )


def adjudicate_rows(
    rows: Sequence[Mapping[str, Any]],
    row_groups: Sequence[str],
    output_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raster = protocol["raster_contract"]
    selected_bands = [int(value) + 1 for value in raster["selected_zero_based_band_indices"]]
    expected_shape = tuple(raster["frame_shape"])
    min_valid = float(raster["minimum_valid_fraction_per_frame"])
    results: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row, group_id in zip(rows, row_groups, strict=True):
        paths = {
            role: safe_output_path(output_root, str(row[column]))
            for column, role in ASSET_COLUMNS.items()
        }
        reasons: list[str] = []
        frame_valid: dict[str, float] = {}
        mask_pixels = 0
        try:
            with (
                rasterio.open(paths["target"]) as target,
                rasterio.open(paths["reference90"]) as reference90,
                rasterio.open(paths["reference365"]) as reference365,
                rasterio.open(paths["mask"]) as mask_source,
            ):
                frames = {
                    "target": target,
                    "reference90": reference90,
                    "reference365": reference365,
                }
                for role, source in frames.items():
                    if (source.count, source.height, source.width) != expected_shape:
                        reasons.append(f"{role}_shape")
                        continue
                    values = source.read(selected_bands).astype(np.float64, copy=False)
                    finite_positive = np.all(np.isfinite(values) & (values > 0), axis=0)
                    frame_valid[role] = float(np.mean(finite_positive))
                    if frame_valid[role] < min_valid:
                        reasons.append(f"{role}_radiometric_support")
                    if any(float(np.nanstd(values[index])) <= 0.0 for index in range(values.shape[0])):
                        reasons.append(f"{role}_constant_band")
                if any(not same_grid(target, source) for source in (reference90, reference365, mask_source)):
                    reasons.append("grid_mismatch")
                if mask_source.count != 1 or (
                    mask_source.height,
                    mask_source.width,
                ) != expected_shape[1:]:
                    reasons.append("mask_shape")
                else:
                    mask = mask_source.read(1).astype(np.float64, copy=False)
                    if not np.all(np.isfinite(mask)) or np.any(mask < 0):
                        reasons.append("mask_nonfinite_or_negative")
                    else:
                        mask_pixels = int(np.count_nonzero(mask > 0))
                        if (int(row["label"]) == 1) != (mask_pixels > 0):
                            reasons.append("label_mask_disagreement")
        except (OSError, rasterio.errors.RasterioError, ValueError) as error:
            reasons.append(f"raster_open:{type(error).__name__}")
        reasons = sorted(set(reasons))
        reason_counts.update(reasons)
        results.append(
            {
                "id": int(row["id"]),
                "label": int(row["label"]),
                "group_id": group_id,
                "usable": not reasons,
                "exclusion_reasons": reasons,
                "mask_positive_pixels": mask_pixels,
                "valid_fraction": frame_valid,
            }
        )

    usable = [item for item in results if item["usable"]]
    usable_groups = {item["group_id"] for item in usable}
    groups_with_label = {
        str(label): len({item["group_id"] for item in usable if item["label"] == label})
        for label in (0, 1)
    }
    summary = {
        "rows_total": len(results),
        "rows_usable": len(usable),
        "usable_by_label": dict(sorted(Counter(item["label"] for item in usable).items())),
        "usable_groups": len(usable_groups),
        "usable_groups_with_label": groups_with_label,
        "exclusion_reasons": dict(sorted(reason_counts.items())),
        "negative_provenance": "event-centered near-plume background; not deployment-FPR evidence",
    }
    return results, summary


def evaluate_gates(summary: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "minimum_usable_rows": summary["rows_usable"] >= int(gates["minimum_usable_rows"]),
        "minimum_usable_positive_rows": int(summary["usable_by_label"].get(1, 0))
        >= int(gates["minimum_usable_rows_per_label"]),
        "minimum_usable_negative_rows": int(summary["usable_by_label"].get(0, 0))
        >= int(gates["minimum_usable_rows_per_label"]),
        "minimum_usable_groups": summary["usable_groups"]
        >= int(gates["minimum_usable_25km_groups"]),
        "minimum_positive_groups": int(summary["usable_groups_with_label"]["1"])
        >= int(gates["minimum_usable_25km_groups_per_label"]),
        "minimum_negative_groups": int(summary["usable_groups_with_label"]["0"])
        >= int(gates["minimum_usable_25km_groups_per_label"]),
    }


def tracked_dirty(root: Path) -> list[str]:
    result = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=root, text=True
    )
    return [line for line in result.splitlines() if line.strip()]


def ignored(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative], cwd=root, check=False
    ).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL.relative_to(ROOT)))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen identities, rows, asset paths, and groups without network or raster access.",
    )
    args = parser.parse_args()
    protocol_path = checked_repo_path(args.protocol)
    protocol = load_json(protocol_path)
    for key in ("candidate_inventory", "metadata_audit", "training_manifest"):
        identity = protocol["inputs"][key]
        verify_file_identity(checked_repo_path(identity["path"]), identity)
    rows, expected = read_frozen_rows(
        checked_repo_path(protocol["inputs"]["training_manifest"]["path"]),
        checked_repo_path(protocol["inputs"]["candidate_inventory"]["path"]),
    )
    frozen = protocol["frozen_counts"]
    if len(rows) != int(frozen["rows"]) or len(expected) != int(frozen["assets"]):
        raise ValueError("Resolved candidate counts do not match the frozen protocol")
    actual_labels = {str(key): value for key, value in sorted(Counter(row["label"] for row in rows).items())}
    if actual_labels != frozen["rows_by_label"]:
        raise ValueError("Resolved candidate labels do not match the frozen protocol")
    row_groups, group_summary = connected_groups(rows, float(protocol["geography"]["group_km"]))
    if group_summary != frozen["metadata_group_summary"]:
        raise ValueError("Recomputed geographic grouping does not match the frozen protocol")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "protocol": protocol_path.relative_to(ROOT).as_posix(),
                    "protocol_sha256": sha256(protocol_path),
                    "rows": len(rows),
                    "assets": len(expected),
                    "geography": group_summary,
                    "network_accessed": False,
                    "raster_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if tracked_dirty(ROOT):
        raise RuntimeError("Refusing acquisition from a dirty tracked worktree")

    output_root = checked_repo_path(protocol["outputs"]["raw_root"])
    scratch = checked_repo_path(protocol["outputs"]["scratch_root"])
    receipts = checked_repo_path(protocol["outputs"]["receipt_root"])
    report_path = checked_repo_path(protocol["outputs"]["report"])
    rows_path = checked_repo_path(protocol["outputs"]["row_adjudication"])
    for bulk_path in (output_root, scratch, receipts):
        if not ignored(ROOT, bulk_path):
            raise ValueError(f"Bulk acquisition path is not ignored by Git: {bulk_path}")

    acquisition = acquire_archives(protocol, expected, output_root, scratch, receipts)
    row_results, adjudication = adjudicate_rows(rows, row_groups, output_root, protocol)
    gates = evaluate_gates(adjudication, protocol["gates"])
    atomic_write(rows_path, json.dumps(row_results, indent=2, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(protocol_path),
            "status": protocol["status"],
        },
        "source": {
            "repo_id": protocol["source"]["repo_id"],
            "revision": protocol["source"]["revision"],
            "license": protocol["source"]["license"],
            "released_partition": "train only",
        },
        "geography": group_summary,
        "acquisition": acquisition,
        "adjudication": adjudication,
        "gates": {"results": gates, "passed": all(gates.values())},
        "access_ledger": {
            "released_training_rasters_opened": True,
            "released_test_rasters_opened": False,
            "mars_folds_0_1_2_or_official_opened": False,
            "protected_or_external_outcomes_opened": False,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tool": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "tool_sha256": sha256(Path(__file__).resolve()),
        },
    }
    atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
