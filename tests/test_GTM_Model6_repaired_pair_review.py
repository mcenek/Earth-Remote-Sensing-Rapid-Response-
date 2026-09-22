"""Regression checks for scientific misinterpretation and provenance failures."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from GTM_Model6_review_repaired_pair import common_native_support, plm_characterization, radiometry_check, verify_hash


def test_already_corrected_flag_and_offset_conflict_cannot_approve_conversion():
    values = np.array([200, 500, 1500])
    result = radiometry_check({"earthsearch:boa_offset_applied": True},
                             {"scale": .0001, "offset": -.1}, values, np.ones(3, dtype=bool))
    assert result["offset_metadata_conflict"]
    assert not result["effective_reflectance_conversion_approved"]
    assert result["literal_offset_negative_fraction"] == pytest.approx(2 / 3)
    assert values.tolist() == [200, 500, 1500]


def test_continuous_plm_keeps_negative_enhancement_without_binary_target():
    result = plm_characterization(np.array([-20., 5., 1000., -9999.]), np.array([True, True, True, False]))
    assert not result["binary_label_created"]
    assert result["statistics"]["negative_pixels"] == 1
    assert result["statistics"]["min"] == -20
    assert result["statistics"]["positive_value_pixels"] == 2


def test_shifted_native_window_cannot_be_combined():
    base = {"crs": "EPSG:4326", "shape": [2, 2], "transform": [1, 0, 0, 0, -1, 2],
            "window": [0, 0, 2, 2], "window_transform": [1, 0, 0, 0, -1, 2]}
    products = {name: dict(base) for name in ("enh", "sens", "uncert")}
    masks = {name: np.ones((2, 2), dtype=bool) for name in products}
    products["uncert"]["window_transform"] = [1, 0, .5, 0, -1, 2]
    with pytest.raises(ValueError, match="grid or geometry"):
        common_native_support(products, masks, masks)


def test_crop_file_changed_since_acquisition_is_rejected(tmp_path):
    crop = tmp_path / "crop.tif"
    crop.write_bytes(b"modified raster")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_hash(crop, "0" * 64)
