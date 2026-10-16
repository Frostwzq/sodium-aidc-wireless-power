import unittest

from src.l2_field.magnet import CoreGeometry, N87_3C90, saturation_check, scenario_core_assessment, thermal_check


class MagnetTests(unittest.TestCase):
    def test_temperature_adjusted_saturation(self) -> None:
        result = saturation_check(0.42, N87_3C90, ambient_c=45.0)
        self.assertAlmostEqual(result["b_sat_hot_t"], 0.4365, places=4)
        self.assertFalse(result["saturated"])
        self.assertTrue(result["within_redline"])
        self.assertTrue(saturation_check(0.50, N87_3C90, 45.0)["saturated"])

    def test_thermal_range_and_unresolved_scenario(self) -> None:
        natural = thermal_check(25.0, 0.05, 10.0, ambient_c=45.0)
        forced = thermal_check(25.0, 0.05, 20.0, ambient_c=45.0)
        self.assertEqual(natural["temperature_rise_k"], 50.0)
        self.assertEqual(forced["temperature_rise_k"], 25.0)
        self.assertEqual(natural["cooling_mode"], "natural")
        self.assertEqual(forced["cooling_mode"], "forced_air")
        assessment = scenario_core_assessment(None, None, CoreGeometry(0.0025, 0.04, 0.04, 0.05), 10.0)
        self.assertIsNone(assessment["feasible"])
        self.assertIn("不可行" if assessment["feasible"] is False else "未定", assessment["conclusion"])


if __name__ == "__main__":
    unittest.main()
