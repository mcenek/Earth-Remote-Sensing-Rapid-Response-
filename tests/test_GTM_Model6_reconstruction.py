from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gtm_model6_reconstruction", ROOT / "research/GTM_Model6/GTM_Model6_reconstruction.py"
)
assert SPEC and SPEC.loader
try:
    reconstruction = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = reconstruction
    SPEC.loader.exec_module(reconstruction)
    IMPORT_ERROR = None
except (ImportError, OSError, PermissionError) as exc:  # pragma: no cover - host dependency
    reconstruction = None
    IMPORT_ERROR = exc


@unittest.skipIf(reconstruction is None, f"Model6 runtime dependencies unavailable: {IMPORT_ERROR}")
class ReconstructionContractTests(unittest.TestCase):
    def test_masked_pool_is_area_mean_and_reports_support(self):
        torch = reconstruction.torch
        value = torch.tensor([[[[1.0, 3.0], [5.0, 99.0]]]])
        valid = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
        pooled, support = reconstruction.pool_valid(value, valid, 2)
        self.assertAlmostEqual(float(pooled[0, 0, 0, 0]), 3.0)
        self.assertAlmostEqual(float(support[0, 0, 0, 0]), 0.75)

    def test_degrade_does_not_halo_invalid_target_values(self):
        torch = reconstruction.torch
        # The invalid 999 value is surrounded by valid zero cells. It must not
        # influence either the masked mean or the interpolated reconstruction.
        target = torch.zeros(1, 1, 4, 4)
        target[:, :, 1, 1] = 999.0
        valid = torch.ones_like(target)
        valid[:, :, 1, 1] = 0.0
        baseline, coverage, nearest = reconstruction.degrade(target, valid)
        self.assertEqual(float(baseline.abs().max()), 0.0)
        self.assertEqual(float(nearest.abs().max()), 0.0)
        self.assertGreaterEqual(float(coverage.min()), 0.0)

    def test_rotation_keeps_padded_nodata_out_of_target_and_support(self):
        torch = reconstruction.torch
        target = torch.ones(1, 1, 9, 9)
        valid = torch.ones_like(target)
        sentinel = torch.zeros(1, 5, 9, 9)
        _, rotated, mask = reconstruction.augment(sentinel, target, valid, 45, False)
        # Rotated corners are padding and must be excluded; a valid interior
        # remains available. This prevents zero padding becoming a target halo.
        self.assertEqual(float(rotated[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(mask[0, 0, 0, 0]), 0.0)
        self.assertGreater(int(mask.sum()), 0)
        self.assertTrue(bool(torch.all(rotated[mask > 0] == 1)))

    def test_model_output_is_one_channel_and_matches_target_grid(self):
        torch = reconstruction.torch
        model = reconstruction.ReconstructionUNet()
        output = model(torch.zeros(2, 7, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))

    def test_degradation_discards_zero_mean_subpixel_detail(self):
        torch = reconstruction.torch
        # Distinct high-frequency detail with zero mean in every 2x2 block is
        # indistinguishable after the declared 2x2 area-mean degradation.
        low = torch.zeros(1, 1, 4, 4)
        detail = torch.tensor([[[[1., -1., 2., -2.], [-1., 1., -2., 2.],
                                [3., -3., 4., -4.], [-3., 3., -4., 4.]]]])
        valid = torch.ones_like(low)
        base_low, cov_low, _ = reconstruction.degrade(low, valid)
        base_detail, cov_detail, _ = reconstruction.degrade(low + detail, valid)
        self.assertTrue(torch.equal(base_low, base_detail))
        self.assertTrue(torch.equal(cov_low, cov_detail))

    def test_invalid_target_magnitude_has_no_effect_on_degradation(self):
        torch = reconstruction.torch
        target = torch.arange(16., dtype=torch.float32).reshape(1, 1, 4, 4)
        valid = torch.ones_like(target)
        valid[:, :, 1, 1] = 0
        altered = target.clone(); altered[:, :, 1, 1] = 1e12
        self.assertTrue(torch.equal(reconstruction.degrade(target, valid)[0],
                                    reconstruction.degrade(altered, valid)[0]))

    def test_zero_head_initialization_is_the_bilinear_baseline(self):
        torch = reconstruction.torch
        model = reconstruction.ReconstructionUNet()
        base = torch.randn(2, 1, 16, 16)
        x = torch.cat([torch.randn(2, 5, 16, 16), base, torch.ones_like(base)], 1)
        with torch.no_grad():
            prediction = model(x)
        self.assertTrue(torch.equal(prediction, base))

    @unittest.skipUnless((ROOT / "EarthRemoteSensingRapidResponse/Dataset").exists(),
                         "legacy dataset unavailable")
    def test_inventory_groups_keep_same_emit_and_nearby_observations_together(self):
        try:
            rows, _ = reconstruction.inventory()
        except (ImportError, OSError, PermissionError) as exc:
            self.skipTest(f"raster backend unavailable: {exc}")
        accepted = [row for row in rows if row.get("eligible_engineering_only")]
        by_emit = {}
        for row in accepted:
            if row.get("emit_time"):
                by_emit.setdefault(row["emit_time"], []).append(row)
        for same_emit in by_emit.values():
            self.assertEqual(len({row["group"] for row in same_emit}), 1)
        for i, left in enumerate(accepted):
            for right in accepted[:i]:
                ax = (left["bounds_wgs84"][0] + left["bounds_wgs84"][2]) / 2
                ay = (left["bounds_wgs84"][1] + left["bounds_wgs84"][3]) / 2
                bx = (right["bounds_wgs84"][0] + right["bounds_wgs84"][2]) / 2
                by = (right["bounds_wgs84"][1] + right["bounds_wgs84"][3]) / 2
                distance = 111.2 * ((ay - by) ** 2 +
                                    ((ax - bx) * __import__("math").cos(
                                        __import__("math").radians((ay + by) / 2))) ** 2) ** 0.5
                if distance < 20:
                    self.assertEqual(left["group"], right["group"])


if __name__ == "__main__":
    unittest.main()
