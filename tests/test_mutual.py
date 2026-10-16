import unittest

from src.l2_field.coil import RectangularWinding
from src.l2_field.mutual import (
    coupling_matrix,
    mutual_inductance_coaxial_closed_form_h,
    mutual_inductance_h,
    scenario_b_grid_c_assumptions,
    scenario_b_heatmap_data,
    transmitter_mutual_matrix,
    tx_rx_mutual_vector,
    validate_mutual_matrix,
)


class MutualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tx = RectangularWinding(0.157, 0.157, 17, 0.004617647058823529)
        self.rx = RectangularWinding(0.255, 0.115)

    def test_coaxial_closed_form_and_numerical_agree(self) -> None:
        closed = mutual_inductance_coaxial_closed_form_h(self.tx, self.rx, 0.035)
        automatic = mutual_inductance_h(self.tx, self.rx, (0.0, 0.0, 0.035))
        numerical = mutual_inductance_h(
            self.tx, self.rx, (0.0, 0.0, 0.035), segments_per_side=3, method="numerical"
        )
        self.assertGreater(closed, 0.0)
        self.assertEqual(automatic, closed)
        self.assertLess(abs(numerical - closed) / abs(closed), 0.03)

    def test_tx_matrix_reciprocity_and_pair_coupling(self) -> None:
        positions = [(-0.31, 0.0, 0.0), (0.0, 0.0, 0.0), (0.31, 0.0, 0.0)]
        matrix = transmitter_mutual_matrix(self.tx, positions, 21e-6, segments_per_side=8)
        report = validate_mutual_matrix(matrix)
        self.assertTrue(report["symmetric"])
        self.assertTrue(report["reciprocal"])
        self.assertLess(report["max_abs_pair_k"], 0.3)
        self.assertEqual(matrix[0][1], matrix[1][0])
        self.assertTrue(all(value == 0.0 for row, values in enumerate(coupling_matrix(matrix))
                            for column, value in enumerate(values) if row == column))

    def test_off_axis_reciprocity_when_coils_are_exchanged(self) -> None:
        forward = mutual_inductance_h(
            self.tx, self.rx, (0.21, -0.08, 0.035), segments_per_side=5, method="numerical"
        )
        reverse = mutual_inductance_h(
            self.rx, self.tx, (-0.21, 0.08, -0.035), segments_per_side=5, method="numerical"
        )
        self.assertAlmostEqual(forward, reverse, delta=1e-12)

    def test_rx_vector_and_b_heatmap_data(self) -> None:
        grid = scenario_b_grid_c_assumptions()
        self.assertEqual(len(grid["tx_positions_m"]), 16)
        self.assertEqual(len(grid["rx_positions_m"]), 24)
        vector = tx_rx_mutual_vector(
            self.tx, grid["tx_positions_m"], self.rx, grid["rx_positions_m"][0], segments_per_side=4
        )
        self.assertEqual(len(vector), 16)
        heatmap = scenario_b_heatmap_data(
            self.tx, self.rx, segments_per_side=4
        )
        self.assertEqual(heatmap["M_heatmap_shape"], [16, 16])
        self.assertEqual(heatmap["m_heatmap_shape"], [24, 16])
        self.assertTrue(heatmap["matrix_validation"]["coupling_ok"])
        self.assertIn("[C]", heatmap["provenance"])


if __name__ == "__main__":
    unittest.main()
