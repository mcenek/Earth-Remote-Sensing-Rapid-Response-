"""Strict prepared-data boundary for the three-CNN methane experiment.

No array is resampled or repaired here. A rejected observation needs a reviewed
preparation step before it may enter training or scoring.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np


SCHEMA = "GTM_Model6_three_cnn_manifest_v1"
SCENES_SCHEMA = "GTM_Model6_three_cnn_scenes_v1"
BANDS = ["B2", "B3", "B4", "B11", "B12"]
ROLES = {"branch_fit", "fusion_fit", "validation", "test"}
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_NPZ_BYTES = 256 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_OBSERVATIONS = 10000
ARRAY_NAMES = {"image", "input_valid", "target", "target_valid"}
SCENE_ARRAY_NAMES = {"image", "input_valid"}


class ContractError(ValueError):
    """A prepared observation does not meet the reviewed-data contract."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(where, message):
    raise ContractError(f"{where}: {message}")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("manifest JSON", f"duplicate key {key!r}")
        result[key] = value
    return result


def _nonfinite_json(value):
    _fail("manifest JSON", f"nonfinite constant {value} is not valid JSON")


def _fixture_marker(value):
    return isinstance(value, str) and bool(re.search(r"\b(?:synthetic|fixture)\b", value, re.I))


def _fields(value, required, where):
    if not isinstance(value, dict) or set(value) != set(required):
        missing = sorted(set(required) - set(value)) if isinstance(value, dict) else sorted(required)
        extra = sorted(set(value) - set(required)) if isinstance(value, dict) else []
        _fail(where, f"fields mismatch (missing={missing}, extra={extra})")


def _string(value, where):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        _fail(where, "must be a nonempty string of at most 512 characters")
    return value


def _timestamp(value, where):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", value):
        _fail(where, "must be an explicit ISO 8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(where, "invalid UTC timestamp")


def _reviewed_true(value, where):
    if value is not True:
        _fail(where, "must be true after review")


def _decimal(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(where, "must be a finite number")
    return Decimal(str(value))


def _projected_crs(crs, data_kind, where):
    _string(crs, where)
    if data_kind == "synthetic":
        if crs != "LOCAL_METRE_GRID":
            _fail(where, "synthetic observations require LOCAL_METRE_GRID")
        return
    if crs == "LOCAL_METRE_GRID":
        _fail(where, "measured observations require a projected metre CRS")
    match = re.fullmatch(r"EPSG:(\d{4,5})", crs)
    if not match:
        _fail(where, "measured CRS must be an explicitly supported projected metre EPSG code")
    code = int(match.group(1))
    supported = (32601 <= code <= 32660 or 32701 <= code <= 32760
                 or 26901 <= code <= 26998 or code in {3857, 5070, 6933, 3035, 3413, 3995})
    if not supported:
        _fail(where, "CRS is not a supported projected metre EPSG code")


def _grid(grid, data_kind, where):
    _fields(grid, {"shape", "crs", "affine"}, where)
    shape = grid["shape"]
    if (not isinstance(shape, list) or len(shape) != 2 or
            any(type(item) is not int or not 1 <= item <= MAX_DIMENSION for item in shape)):
        _fail(where, f"shape must contain two positive dimensions <= {MAX_DIMENSION}")
    _projected_crs(grid["crs"], data_kind, f"{where}.crs")
    affine = grid["affine"]
    if not isinstance(affine, list) or len(affine) != 6:
        _fail(where, "affine must have six coefficients")
    a = [_decimal(item, f"{where}.affine[{index}]") for index, item in enumerate(affine)]
    if a[0] <= 0 or a[4] >= 0 or a[1] != 0 or a[3] != 0:
        _fail(where, "grid must be north-up, nonrotated, and have positive metre cell size")
    return shape, a


def _matching_grids(input_grid, target_grid, kind, where):
    ishape, ia = _grid(input_grid, kind, f"{where}.input.grid")
    tshape, ta = _grid(target_grid, kind, f"{where}.target.grid")
    if input_grid["crs"] != target_grid["crs"]:
        _fail(where, "input and target CRS differ")
    if ia[0] != 20 or ia[4] != -20:
        _fail(where, "Sentinel input cells must be exactly 20 metres")
    if ia[2] != ta[2] or ia[5] != ta[5]:
        _fail(where, "input and target grid origins differ")
    if ishape[0] % tshape[0] or ishape[1] % tshape[1]:
        _fail(where, "target must aggregate an integer number of input cells")
    factor_y, factor_x = ishape[0] // tshape[0], ishape[1] // tshape[1]
    if factor_y != factor_x or ta[0] != ia[0] * factor_x or ta[4] != ia[4] * factor_y:
        _fail(where, "target cell size and shape must be exact integer aggregation of input")
    return factor_x


def _resolve_npz(manifest_path, relative, where):
    _string(relative, where)
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".npz":
        _fail(where, "must be a relative NPZ path inside the manifest directory")
    base = manifest_path.parent.resolve()
    array_path = (base / rel).resolve()
    if not array_path.is_relative_to(base) or not array_path.is_file():
        _fail(where, "NPZ file is missing or escapes the manifest directory")
    if array_path.stat().st_size > MAX_NPZ_BYTES:
        _fail(where, "NPZ file exceeds size bound")
    return array_path


def _check_binary(array, where):
    if array.dtype.kind not in "biu" or not np.all((array == 0) | (array == 1)):
        _fail(where, "must contain only integer/bool 0 or 1")


def _input_fingerprint(arrays):
    digest = hashlib.sha256()
    for name in ("image", "input_valid"):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        # Normalize dtype so a byte-order or float-width change cannot hide reuse.
        normalized = array.astype("<f8" if name == "image" else "u1", copy=False)
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def _read_npz(row, required_names):
    where = f"observation {row.get('id', '<unknown>')}"
    if "_array_path" not in row:
        _fail(where, "row has no resolved _array_path; call a manifest loader first")
    path = Path(row["_array_path"])
    if not path.is_file() or path.stat().st_size > MAX_NPZ_BYTES:
        _fail(where, "NPZ file missing or exceeds size bound")
    if sha256(path) != row["input"]["sha256"]:
        _fail(where, "NPZ SHA-256 differs from manifest")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if set(names) != {name + ".npy" for name in required_names} or len(names) != len(required_names):
                _fail(where, f"NPZ must contain exactly {', '.join(sorted(required_names))}")
            if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
                _fail(where, "NPZ uncompressed size exceeds bound")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in required_names}
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
        if isinstance(exc, ContractError):
            raise
        _fail(where, f"cannot read NPZ: {exc}")
    return arrays


def _validate_image_support(arrays, row, where):
    image, iv = arrays["image"], arrays["input_valid"]
    ih, iw = row["input"]["grid"]["shape"]
    if image.shape != (ih, iw, 5) or iv.shape != (ih, iw):
        _fail(where, "image/input_valid shape or five-band channel count differs from grid")
    if image.dtype.kind != "f":
        _fail(where, "image must use floating-point reflectance")
    _check_binary(iv, f"{where}.input_valid")
    if np.isinf(image).any() or not np.isfinite(image[iv.astype(bool)]).all():
        _fail(where, "supported image pixels must be finite; infinity is never permitted")
    if not np.any(iv):
        _fail(where, "input has empty valid support")


def load_arrays(row):
    """Load and validate the four arrays in one already validated paired row."""
    where = f"observation {row.get('id', '<unknown>')}"
    arrays = _read_npz(row, ARRAY_NAMES)
    _validate_image_support(arrays, row, where)
    iv = arrays["input_valid"]
    target, tv = arrays["target"], arrays["target_valid"]
    th, tw = row["target"]["grid"]["shape"]
    if target.shape != (th, tw) or tv.shape != (th, tw):
        _fail(where, "target/target_valid shape differs from native target grid")
    if target.dtype.kind != "f":
        _fail(where, "target must use a floating-point array")
    _check_binary(tv, f"{where}.target_valid")
    if not np.all(np.isnan(target[~tv.astype(bool)])):
        _fail(where, "invalid/unknown target cells must be NaN")
    valid_target = target[tv.astype(bool)]
    if not np.isfinite(valid_target).all() or not np.all((valid_target == 0) | (valid_target == 1)):
        _fail(where, "valid target cells must be finite binary 0/1")
    data_kind = "synthetic" if row["input"]["grid"]["crs"] == "LOCAL_METRE_GRID" else "measured"
    factor = _matching_grids(row["input"]["grid"], row["target"]["grid"], data_kind, where)
    full_support = iv.astype(bool).reshape(th, factor, tw, factor).all(axis=(1, 3))
    if not np.any(tv):
        _fail(where, "target has empty reviewed support")
    if np.any(tv.astype(bool) & ~full_support):
        _fail(where, "target_valid extends beyond fully valid input support")
    return arrays


def load_scene_arrays(row):
    """Load only Sentinel reflectance and its support from an input-only scene."""
    where = f"scene {row.get('id', '<unknown>')}"
    arrays = _read_npz(row, SCENE_ARRAY_NAMES)
    _validate_image_support(arrays, row, where)
    data_kind = "synthetic" if row["input"]["grid"]["crs"] == "LOCAL_METRE_GRID" else "measured"
    _, affine = _grid(row["input"]["grid"], data_kind, f"{where}.input.grid")
    if affine[0] != 20 or affine[4] != -20:
        _fail(where, "Sentinel input cells must be exactly 20 metres")
    return arrays


def _footprint(row):
    grid = row["input"]["grid"]
    h, w = grid["shape"]
    a = grid["affine"]
    return (Decimal(str(a[2])), Decimal(str(a[2])) + Decimal(str(a[0])) * w,
            Decimal(str(a[5])) + Decimal(str(a[4])) * h, Decimal(str(a[5])))


def _intersects(left, right):
    lx0, lx1, ly0, ly1 = left
    rx0, rx1, ry0, ry1 = right
    return lx0 < rx1 and rx0 < lx1 and ly0 < ry1 and ry0 < ly1


def _read_manifest_root(path, expected_schema):
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        _fail("manifest", "file missing or exceeds size bound")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"),
                         object_pairs_hook=_json_object, parse_constant=_nonfinite_json)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("manifest", f"cannot read JSON: {exc}")
    _fields(raw, {"schema", "data_kind", "bands", "observations"}, "manifest")
    if raw["schema"] != expected_schema:
        _fail("manifest.schema", f"expected {expected_schema}")
    if raw["data_kind"] not in ("synthetic", "measured"):
        _fail("manifest.data_kind", "must be synthetic or measured")
    if raw["bands"] != BANDS:
        _fail("manifest.bands", f"must be exactly {BANDS}")
    observations = raw["observations"]
    if not isinstance(observations, list) or not 1 <= len(observations) <= MAX_OBSERVATIONS:
        _fail("manifest.observations", "must be a nonempty bounded list")
    return manifest_path, raw, observations


def _validate_input_metadata(inp, data_kind, where):
    _fields(inp, {"path", "sha256", "sensor", "product_level", "units",
                  "radiometry_verified", "acquired_at", "grid"}, where)
    if inp["sensor"] != "Sentinel-2" or inp["product_level"] not in ("L1C", "L2A") or inp["units"] != "reflectance":
        _fail(where, "input requires Sentinel-2 L1C/L2A reflectance")
    _reviewed_true(inp["radiometry_verified"], f"{where}.radiometry_verified")
    _timestamp(inp["acquired_at"], f"{where}.acquired_at")
    if not isinstance(inp["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", inp["sha256"]):
        _fail(where, "sha256 must be lowercase SHA-256 hex")
    _, affine = _grid(inp["grid"], data_kind, f"{where}.grid")
    if affine[0] != 20 or affine[4] != -20:
        _fail(where, "Sentinel input cells must be exactly 20 metres")


def load_manifest(path):
    """Validate a v1 paired manifest and every linked NPZ, returning resolved rows."""
    manifest_path, raw, observations = _read_manifest_root(path, SCHEMA)
    rows = []
    ids = set()
    role_identities = {}
    by_role = {}
    measured_crs = None
    for index, source_row in enumerate(observations):
        where = f"observations[{index}]"
        _fields(source_row, {"id", "parent_id", "site", "group", "source_ids", "role",
                             "target_kind", "input", "target", "review"}, where)
        row = copy.deepcopy(source_row)
        for key in ("id", "parent_id", "site", "group"):
            _string(row[key], f"{where}.{key}")
        if row["id"] in ids:
            _fail(where, f"duplicate observation ID {row['id']}")
        ids.add(row["id"])
        if not isinstance(row["source_ids"], list) or not row["source_ids"]:
            _fail(where, "source_ids must be a nonempty list of unique IDs")
        for source in row["source_ids"]:
            _string(source, f"{where}.source_ids")
        if len(set(row["source_ids"])) != len(row["source_ids"]):
            _fail(where, "source_ids must be a nonempty list of unique IDs")
        _string(row["role"], f"{where}.role")
        if row["role"] not in ROLES or row["target_kind"] not in ("airborne", "emit"):
            _fail(where, "invalid role or target_kind")
        if row["role"] == "fusion_fit" and row["target_kind"] != "emit":
            _fail(where, "fusion_fit requires EMIT supervision")
        inp = row["input"]
        _validate_input_metadata(inp, raw["data_kind"], f"{where}.input")
        _fields(row["target"], {"kind", "units", "acquired_at", "support_reviewed", "reviewer",
                                "source", "grid"}, f"{where}.target")
        target = row["target"]
        if target["kind"] != "plume_mask" or target["units"] != "binary":
            _fail(where, "target must be a binary plume_mask")
        _timestamp(target["acquired_at"], f"{where}.target.acquired_at")
        _reviewed_true(target["support_reviewed"], f"{where}.target.support_reviewed")
        _string(target["reviewer"], f"{where}.target.reviewer")
        _string(target["source"], f"{where}.target.source")
        _fields(row["review"], {"status", "alignment_verified", "timing_verified",
                                "independence_verified", "reviewer", "notes"}, f"{where}.review")
        review = row["review"]
        if review["status"] != "approved":
            _fail(where, "review.status must be approved")
        for key in ("alignment_verified", "timing_verified", "independence_verified"):
            _reviewed_true(review[key], f"{where}.review.{key}")
        _string(review["reviewer"], f"{where}.review.reviewer")
        _string(review["notes"], f"{where}.review.notes")
        _matching_grids(inp["grid"], target["grid"], raw["data_kind"], where)
        if raw["data_kind"] == "measured":
            crs = inp["grid"]["crs"]
            if measured_crs is None:
                measured_crs = crs
            elif crs != measured_crs:
                _fail(where, "mixed measured CRSs are unsupported; prepare all observations in one common projected metre CRS before checking footprint independence")
            provenance = [row["id"], row["parent_id"], row["site"], row["group"],
                          *row["source_ids"], inp["path"], target["source"],
                          target["reviewer"], review["reviewer"], review["notes"]]
            if any(_fixture_marker(value) for value in provenance):
                _fail(where, "measured observation contains a synthetic fixture identity, source, path, or reviewer")
        array_path = _resolve_npz(manifest_path, inp["path"], f"{where}.input.path")
        row["_array_path"] = str(array_path)
        arrays = load_arrays(row)
        fingerprint = _input_fingerprint(arrays)
        for kind, value in (("parent_id", row["parent_id"]), ("site", row["site"]),
                            ("group", row["group"]), ("array_path", str(array_path)),
                            ("npz_sha256", inp["sha256"]), ("input_content", fingerprint)):
            key = (kind, value)
            if key in role_identities and role_identities[key] != row["role"]:
                _fail(where, f"cross-role {kind} leakage: {value}")
            role_identities[key] = row["role"]
        for source in row["source_ids"]:
            key = ("source_id", source)
            if key in role_identities and role_identities[key] != row["role"]:
                _fail(where, f"cross-role source_id leakage: {source}")
            role_identities[key] = row["role"]
        footprint = _footprint(row)
        for prior, prior_footprint in by_role.get(inp["grid"]["crs"], []):
            if prior["role"] != row["role"] and _intersects(footprint, prior_footprint):
                _fail(where, f"footprint overlaps cross-role observation {prior['id']}")
        by_role.setdefault(inp["grid"]["crs"], []).append((row, footprint))
        rows.append(row)
    return {"schema": raw["schema"], "data_kind": raw["data_kind"], "bands": BANDS.copy(),
            "observations": rows, "manifest_sha256": sha256(manifest_path), "path": str(manifest_path)}


def load_scenes(path):
    """Validate input-only Sentinel scenes for inference without methane targets."""
    manifest_path, raw, observations = _read_manifest_root(path, SCENES_SCHEMA)
    rows = []
    ids = set()
    sources = set()
    paths = set()
    contents = set()
    footprints = []
    measured_crs = None
    for index, source_row in enumerate(observations):
        where = f"observations[{index}]"
        _fields(source_row, {"id", "source_ids", "input", "review"}, where)
        row = copy.deepcopy(source_row)
        _string(row["id"], f"{where}.id")
        if row["id"] in ids:
            _fail(where, f"duplicate scene ID {row['id']}")
        ids.add(row["id"])
        source_ids = row["source_ids"]
        if not isinstance(source_ids, list) or not source_ids:
            _fail(where, "source_ids must be a nonempty list")
        for source in source_ids:
            _string(source, f"{where}.source_ids")
        if len(set(source_ids)) != len(source_ids):
            _fail(where, "duplicate source_ids within scene")
        inp = row["input"]
        _validate_input_metadata(inp, raw["data_kind"], f"{where}.input")
        _fields(row["review"], {"status", "reviewer", "notes"}, f"{where}.review")
        review = row["review"]
        if review["status"] != "approved":
            _fail(where, "review.status must be approved")
        _string(review["reviewer"], f"{where}.review.reviewer")
        _string(review["notes"], f"{where}.review.notes")
        if raw["data_kind"] == "measured":
            crs = inp["grid"]["crs"]
            if measured_crs is None:
                measured_crs = crs
            elif crs != measured_crs:
                _fail(where, "mixed measured CRSs are unsupported; prepare all scenes in one common projected metre CRS before checking footprint independence")
            provenance = [row["id"], *source_ids, inp["path"], review["reviewer"], review["notes"]]
            if any(_fixture_marker(value) for value in provenance):
                _fail(where, "measured scene contains synthetic fixture identity, source, path, or reviewer")
        elif (not _fixture_marker(row["id"]) or
              not all(_fixture_marker(source) for source in source_ids) or
              not _fixture_marker(review["reviewer"])):
            _fail(where, "synthetic scenes require explicit synthetic identities, sources, and reviewer")
        array_path = _resolve_npz(manifest_path, inp["path"], f"{where}.input.path")
        row["_array_path"] = str(array_path)
        arrays = load_scene_arrays(row)
        fingerprint = _input_fingerprint(arrays)
        if str(array_path) in paths or fingerprint in contents:
            _fail(where, "duplicate scene NPZ path or input array content")
        paths.add(str(array_path))
        contents.add(fingerprint)
        for source in source_ids:
            if source in sources:
                _fail(where, f"source identity reused across scenes: {source}")
            sources.add(source)
        footprint = _footprint(row)
        for prior, prior_footprint in footprints:
            if inp["grid"]["crs"] == prior["input"]["grid"]["crs"] and _intersects(footprint, prior_footprint):
                _fail(where, f"footprint overlaps scene {prior['id']}")
        footprints.append((row, footprint))
        rows.append(row)
    return {"schema": raw["schema"], "data_kind": raw["data_kind"], "bands": BANDS.copy(),
            "observations": rows, "manifest_sha256": sha256(manifest_path), "path": str(manifest_path)}
