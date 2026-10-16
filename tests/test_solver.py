import unittest

from src.l2_field.array import F1_PHASE_PATTERN_DEG, F2_PHASE_PATTERN_DEG, classify_intersection, phase_patterns
from src.l2_field.solver import (
    LOAD_INDEPENDENCE_TOLERANCE,
    derive_epsilon_threshold,
    search_feasible_sizes,
    test_blind_spot_exists as run_blind_spot_check,
    test_load_independence as run_load_independence_check,
    test_no_blind_spot as run_no_blind_spot_check,
)


class ArrayPatternTests(unittest.TestCase):
    def test_phase_patterns_tile_partial_non_four_sizes(self) -> None:
        f1, f2 = phase_patterns(5, 3)
        self.assertEqual(f1.shape, (3, 5))
        self.assertEqual(f2.shape, (3, 5))
        self.assertTrue((F2_PHASE_PATTERN_DEG == (F1_PHASE_PATTERN_DEG + 90.0) % 360.0).all())

    def test_adjacent_clusters_swap_frequency_maps(self) -> None:
        f1, f2 = phase_patterns(8, 4)
        self.assertEqual(f1[0, 0], F1_PHASE_PATTERN_DEG[0, 0])
        self.assertEqual(f1[0, 4], F2_PHASE_PATTERN_DEG[0, 0])
        self.assertEqual(f2[0, 4], F1_PHASE_PATTERN_DEG[0, 0])
        self.assertEqual(classify_intersection([0, 0, 0, 0], [0, 90, 180, 270]), "diamond")
        self.assertEqual(classify_intersection([0, 0, 180, 180], [90, 90, 270, 270]), "cross")


class SolverTests(unittest.TestCase):
    def test_blind_spot_exists(self) -> None:
        report = run_blind_spot_check()
        self.assertGreater(report["blind_count"], 0)
        self.assertGreater(report["point_count"], 0)
        self.assertEqual(report["grid_per_tx"], 64)

    def test_load_independence(self) -> None:
        report = run_load_independence_check()
        # This assertion deliberately preserves the P4 acceptance criterion.
        # With the supplied LCC assumptions the measured value exceeds 1e-6,
        # so unittest must show the criterion failure rather than mask it.
        worst_relative_change = max(item["i_tx_relative_change"] for item in report["results"])
        self.assertLess(worst_relative_change, LOAD_INDEPENDENCE_TOLERANCE)

    def test_no_blind_spot(self) -> None:
        threshold = derive_epsilon_threshold()["epsilon_th_v"]
        report = run_no_blind_spot_check(epsilon_th_v=threshold)
        self.assertGreaterEqual(report["min_rms_v"], threshold)

    def test_non_four_cluster_boundary_no_blind_spot(self) -> None:
        threshold = derive_epsilon_threshold()["epsilon_th_v"]
        report = run_no_blind_spot_check(5, 4, epsilon_th_v=threshold)
        self.assertIsNotNone(report["boundary_min_rms_v"])
        self.assertGreaterEqual(report["boundary_min_rms_v"], threshold)

    def test_non_multiple_array_size_scan_reports_feasible_set(self) -> None:
        report = search_feasible_sizes(((4, 4), (5, 3), (5, 4)), samples_per_tx=8)
        self.assertEqual(report["feasible_sizes"], [])
        self.assertEqual({(item["nx"], item["ny"]) for item in report["tested_sizes"]},
                         {(4, 4), (5, 3), (5, 4)})


if __name__ == "__main__":
    unittest.main()
