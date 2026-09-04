from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_methaneunion_novel_s2 import (  # noqa: E402
    adjudicate_rows,
    checked_member_path,
    connected_groups,
    evaluate_gates,
    read_frozen_rows,
    stream_selected_archive,
)


def tiff_bytes(values: np.ndarray, transform=None) -> bytes:
    transform = transform or from_origin(0, 48, 10, 10)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=values.shape[-2],
            width=values.shape[-1],
            count=1 if values.ndim == 2 else values.shape[0],
            dtype=values.dtype,
            crs="EPSG:32611",
            transform=transform,
        ) as target:
            target.write(values if values.ndim == 3 else values[None])
        return memory.read()


def row(row_id: int, label: int, latitude: float, longitude: float) -> dict[str, object]:
    base = f"data/480m_GSD/s2/{row_id}"
    return {
        "id": row_id,
        "label": label,
        "latitude": latitude,
        "longitude": longitude,
        "available_sensor": "S2",
        "S2_t0_path": f"{base}/s2.tif",
        "S2_pre_path": f"{base}/s2_pre.tif",
        "S2_pre_pre_path": f"{base}/s2_pre_pre.tif",
        "S2_plume_label_path": f"{base}/s2_plume.tif",
    }


def protocol() -> dict[str, object]:
    return {
        "raster_contract": {
            "selected_zero_based_band_indices": [1, 2, 3, 7, 10, 11],
            "frame_shape": [12, 48, 48],
            "minimum_valid_fraction_per_frame": 0.70,
        }
    }


def write_assets(root: Path, item: dict[str, object], *, bad_negative_mask=False) -> None:
    frames = np.full((12, 48, 48), 1000, dtype=np.uint16)
    for band in range(12):
        frames[band] += np.arange(48, dtype=np.uint16)[:, None] + band
    mask = np.zeros((48, 48), dtype=np.uint8)
    if int(item["label"]) == 1 or bad_negative_mask:
        mask[20:24, 20:24] = 1
    mapping = {
        "S2_t0_path": frames,
        "S2_pre_path": frames + 10,
        "S2_pre_pre_path": frames + 20,
        "S2_plume_label_path": mask,
    }
    for column, values in mapping.items():
        path = root.joinpath(*Path(str(item[column])).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tiff_bytes(values))


def test_checked_member_path_rejects_traversal_and_wrong_identifier() -> None:
    assert checked_member_path("data/480m_GSD/s2/7/s2.tif", 7, "target").name == "s2.tif"
    for value in ("../7/s2.tif", "data/480m_GSD/s2/8/s2.tif"):
        try:
            checked_member_path(value, 7, "target")
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe path was accepted")


def test_read_frozen_rows_resolves_only_selected_training_ids() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "train.csv"
        fields = list(row(7, 0, 1.0, 2.0))
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row(7, 0, 1.0, 2.0))
            writer.writerow(row(8, 1, 3.0, 4.0))
        candidates = root / "candidates.json"
        candidates.write_text(json.dumps({"candidate_ids": [7]}), encoding="utf-8")
        rows, expected = read_frozen_rows(manifest, candidates)
        assert [item["id"] for item in rows] == [7]
        assert len(expected) == 4
        assert all("/8/" not in member for member in expected)


def test_connected_groups_uses_transitive_25km_components() -> None:
    rows = [
        row(1, 0, 0.0, 0.0),
        row(2, 1, 0.0, 0.20),
        row(3, 0, 0.0, 0.40),
        row(4, 1, 0.0, 1.0),
    ]
    groups, summary = connected_groups(rows, 25.0)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert summary["connected_groups"] == 2
    assert summary["mixed_label_groups"] == 1


def test_stream_archive_extracts_only_frozen_members() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive_path = root / "part.tar.gz"
        expected_name = "data/480m_GSD/s2/7/s2.tif"
        payload = b"selected"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name, content in ((expected_name, payload), ("data/480m_GSD/s2/99/s2.tif", b"sealed")):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        output = root / "raw"
        found = stream_selected_archive(
            archive_path, {expected_name: (0, "target")}, set(), output, 1024
        )
        assert [item["member"] for item in found] == [expected_name]
        assert output.joinpath(*PurePath(expected_name).parts).read_bytes() == payload
        assert not output.joinpath(*PurePath("data/480m_GSD/s2/99/s2.tif").parts).exists()


def PurePath(value: str) -> Path:
    return Path(*value.split("/"))


def test_adjudication_accepts_valid_rows_and_rejects_nonzero_negative_mask() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good = row(7, 1, 1.0, 2.0)
        bad = row(8, 0, 3.0, 4.0)
        write_assets(root, good)
        write_assets(root, bad, bad_negative_mask=True)
        results, summary = adjudicate_rows(
            [good, bad], ["mu25-000", "mu25-001"], root, protocol()
        )
        assert results[0]["usable"] is True
        assert results[1]["usable"] is False
        assert "label_mask_disagreement" in results[1]["exclusion_reasons"]
        assert summary["rows_usable"] == 1


def test_evaluate_gates_is_fail_closed() -> None:
    summary = {
        "rows_usable": 19,
        "usable_by_label": {0: 10, 1: 9},
        "usable_groups": 4,
        "usable_groups_with_label": {"0": 4, "1": 3},
    }
    gates = {
        "minimum_usable_rows": 20,
        "minimum_usable_rows_per_label": 9,
        "minimum_usable_25km_groups": 4,
        "minimum_usable_25km_groups_per_label": 3,
    }
    result = evaluate_gates(summary, gates)
    assert result["minimum_usable_rows"] is False
    assert all(value for key, value in result.items() if key != "minimum_usable_rows")


def test_frozen_archive_inventory_is_complete_and_contiguous() -> None:
    value = json.loads(
        (ROOT / "configs/methaneunion_novel_s2_acquisition_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    archives = value["source"]["archives"]
    assert len(archives) == 144
    assert [item["name"] for item in archives] == [
        f"dataset_part_{index:03d}.tar.gz" for index in range(1, 145)
    ]
    assert sum(int(item["bytes"]) for item in archives) == int(
        value["source"]["archives_total_bytes"]
    )
    assert all(len(item["sha256"]) == 64 for item in archives)
