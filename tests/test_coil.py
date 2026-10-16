import unittest

from src.l2_field.coil import (
    FluxPipeRx,
    RectangularWinding,
    inverse_reference_spiral,
    mutual_inductance_neumann_h,
    skin_depth_m,
    tolerance_analysis,
)


class CoilTests(unittest.TestCase):
    def test_reference_geometry_and_ac_range(self) -> None:
        tx, estimate = inverse_reference_spiral()
        self.assertEqual(tx.turns, 17)
        self.assertAlmostEqual(estimate.inductance_h * 1e6, 21.85, places=1)
        self.assertAlmostEqual((estimate.inductance_h / 21e-6 - 1) * 100, 4.0, places=0)
        self.assertLess(skin_depth_m() * 1e3, 0.15)
        self.assertLess(estimate.resistance_ac_range_ohm[0], estimate.resistance_ac_range_ohm[1])
        self.assertLess(estimate.q_range[0], estimate.q_range[1])

    def test_neumann_mutual_changes_with_gap_and_tolerance(self) -> None:
        tx, _ = inverse_reference_spiral()
        tx_loop = RectangularWinding(tx.outer_side_m, tx.outer_side_m, tx.turns, tx.radial_pitch_m)
        rx = FluxPipeRx(RectangularWinding(0.255, 0.115, 1, 0.0), core_cross_section_m2=0.0025)
        m_near = mutual_inductance_neumann_h(tx_loop, rx.winding, (0.0, 0.0, 0.035), segments_per_side=6)
        m_far = mutual_inductance_neumann_h(tx_loop, rx.winding, (0.0, 0.0, 0.045), segments_per_side=6)
        self.assertNotEqual(m_near, 0.0)
        self.assertLess(abs(m_far), abs(m_near))
        report = tolerance_analysis(tx_loop, rx, (0.0, 0.0, 0.035), segments_per_side=4)
        self.assertGreater(report["max_abs_change_pct"], 0.0)
        self.assertEqual(len(report["samples"]), 12)


if __name__ == "__main__":
    unittest.main()
