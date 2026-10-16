import unittest

from src.l2_field.freqswitch import frequency_switch_report


class P6FrequencySwitchTests(unittest.TestCase):
    def test_frequency_switch_reports_rf_zero_and_filtered_output(self) -> None:
        report = frequency_switch_report()
        output = report["output"]
        self.assertTrue(output["instantaneous_blind_during_switch"])
        self.assertEqual(output["instantaneous_envelope_minimum_v"], 0.0)
        self.assertGreater(output["minimum_rectified_voltage_v"], 0.0)
        self.assertLessEqual(output["max_duration_below_5pct_band_s"],
                             output["voltage_tolerance_window_s"])


if __name__ == "__main__":
    unittest.main()
