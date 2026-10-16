import unittest

from src.l2_field.zvs import compare_deadtime_strategies, test_no_degrade_after_zvs as run_no_degrade_check


class P6ZvsTests(unittest.TestCase):
    def test_shared_and_independent_deadtime_tradeoff(self) -> None:
        report = compare_deadtime_strategies()
        shared, independent = report["strategies"]
        self.assertEqual(shared["timer_channels"], 1)
        self.assertEqual(independent["timer_channels"], 2)
        self.assertLess(independent["estimated_excess_third_quadrant_loss_w"],
                        shared["estimated_excess_third_quadrant_loss_w"])
        self.assertLess(independent["inter_tone_phase_mismatch_deg"],
                        shared["inter_tone_phase_mismatch_deg"])

    def test_no_degrade_after_zvs(self) -> None:
        report = run_no_degrade_check()
        self.assertEqual([row["rho"] for row in report["rho_results"]], [0.05, 0.10])
        self.assertTrue(report["all_rho_pass"])
        self.assertTrue(report["constraint_degenerate"])
        self.assertEqual(report["status"], "PASS_BUT_DEGENERATE_BASELINE")
        self.assertEqual(report["domain"]["samples_per_tx"], 64)


if __name__ == "__main__":
    unittest.main()
