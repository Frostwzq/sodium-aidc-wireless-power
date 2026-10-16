import unittest

from src.config import Config
from src.l2_field.scaling import all_scenario_reports, compare_prediction_to_aalto_band, scenario_scaling_report


class ScalingTests(unittest.TestCase):
    def test_three_uncertainty_intervals_and_p0c_warning(self) -> None:
        reports = all_scenario_reports()
        self.assertEqual([r["efficiency_interval_assumption"] for r in reports], [(0.85, 0.92), (0.78, 0.90), (0.70, 0.85)])
        p0c = reports[2]
        self.assertEqual(p0c["scale_from_aalto_40w"], 750.0)
        self.assertIn("超出已验证区间", p0c["warning"])
        self.assertIsNone(p0c["prediction_error_pct"])

    def test_efficiency_range_is_constraint_not_calibration(self) -> None:
        self.assertTrue(compare_prediction_to_aalto_band(0.90)["within_band"])
        self.assertFalse(compare_prediction_to_aalto_band(0.95)["within_band"])
        self.assertIsNone(compare_prediction_to_aalto_band(None)["within_band"])
        self.assertEqual(scenario_scaling_report(Config.for_scenario("A_subzone"))["scale_from_aalto_40w"], 75.0)


if __name__ == "__main__":
    unittest.main()
