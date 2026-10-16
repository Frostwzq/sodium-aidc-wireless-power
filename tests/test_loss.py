import unittest

from src.l2_field.loss import (
    F_DELTA_BASE_HZ,
    loss_breakdown,
    loss_fraction_curves,
    optimize_delta_f,
    test_idle_loss_not_zero as run_idle_loss_check,
)
from src.l2_field.thermal import scenario_thermal_assessment


class P5LossTests(unittest.TestCase):
    def test_idle_loss_not_zero(self) -> None:
        report = run_idle_loss_check()
        self.assertGreater(report["total_loss_w"], 0.0)
        self.assertGreater(report["P_sw_idle_coss_w"], 0.0)

    def test_loss_breakdown_has_four_bounded_terms(self) -> None:
        report = loss_breakdown(12_000.0, F_DELTA_BASE_HZ)
        self.assertEqual(set(report["components_w"]), {"P_sw", "P_cond", "P_mag", "P_ctrl"})
        for interval in report["components_w"].values():
            self.assertLessEqual(interval["lower"], interval["nominal"])
            self.assertLessEqual(interval["nominal"], interval["upper"])
        self.assertAlmostEqual(sum(report["component_fraction"].values()), 1.0)

    def test_delta_f_optimum_is_selected_from_declared_scan(self) -> None:
        result = optimize_delta_f(12_000.0, 200.0, 2_000.0, 200.0)
        self.assertEqual(result["scan_hz"]["points"], 10)
        self.assertIn(result["optimal_delta_f_hz"], [row[0] for row in result["total_loss_curve_w"]])

    def test_component_fraction_curve_series_cover_power_and_delta_f(self) -> None:
        curves = loss_fraction_curves(delta_f_step_hz=2_000.0, power_step_w=3_000.0)
        self.assertEqual(set(curves["versus_delta_f"]), {"P0-A", "P0-B", "P0-C"})
        self.assertEqual(set(curves["versus_power"]), {"200", "2000", "10000", "20000"})
        self.assertEqual(len(curves["versus_delta_f"]["P0-C"]["delta_f_hz"]), 10)
        self.assertEqual(len(curves["versus_power"]["2000"]["power_w"]), 11)

    def test_p0c_thermal_check_reports_power_cap(self) -> None:
        result = scenario_thermal_assessment("C_full_rack")
        self.assertGreaterEqual(result["feasible_field_power_upper_w"], 0.0)
        self.assertLessEqual(result["feasible_field_power_upper_w"], result["field_power_w"])
        self.assertEqual(result["conclusion"], "P0-C 不可行")


if __name__ == "__main__":
    unittest.main()
