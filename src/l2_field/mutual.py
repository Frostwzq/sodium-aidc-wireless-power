"""Mutual-inductance matrices and B-scenario heatmap data for P3.

All positions are coil-center coordinates in metres.  The computations use
filament rectangular windings in free space; ferrite/flux-pipe corrections
remain unresolved.  Scenario coordinates are explicitly illustrative [C]
assumptions, not a measured rack layout.
"""

from __future__ import annotations

from math import asinh, isclose, sqrt
from typing import Any, Sequence

import numpy as np

from src.l2_field.coil import RectangularWinding

Position3 = tuple[float, float, float]
Angles3 = tuple[float, float, float]
MATRIX_TOLERANCE_H = 1e-6
MAX_PAIR_COUPLING = 0.3


def _validate_position(position: Position3) -> None:
    if len(position) != 3:
        raise ValueError("coil positions must contain x, y, z")


def _loop_segments(winding: RectangularWinding) -> list[tuple[str, float, float, float, float, int]]:
    """Return oriented x/y line segments for each rectangular turn."""
    winding.validate()
    segments: list[tuple[str, float, float, float, float, int]] = []
    for turn in range(winding.turns):
        width = winding.outer_width_m - 2.0 * turn * winding.radial_pitch_m
        height = winding.outer_height_m - 2.0 * turn * winding.radial_pitch_m
        x0, x1 = -width / 2.0, width / 2.0
        y0, y1 = -height / 2.0, height / 2.0
        # CCW traversal viewed from +z.
        segments.extend([
            ("x", x0, x1, y0, 0.0, 1),
            ("y", y0, y1, x1, 0.0, 1),
            ("x", x0, x1, y1, 0.0, -1),
            ("y", y0, y1, x0, 0.0, -1),
        ])
    return segments


def _segment_arrays(
    winding: RectangularWinding,
    position_m: Position3,
    angles_deg: Angles3,
    segments_per_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized midpoint segments (start points, directed dl vectors)."""
    if segments_per_side < 1:
        raise ValueError("segments_per_side must be >= 1")
    _validate_position(position_m)
    winding.validate()
    starts: list[tuple[float, float, float]] = []
    vectors: list[tuple[float, float, float]] = []
    for turn in range(winding.turns):
        width = winding.outer_width_m - 2.0 * turn * winding.radial_pitch_m
        height = winding.outer_height_m - 2.0 * turn * winding.radial_pitch_m
        x0, x1 = -width / 2.0, width / 2.0
        y0, y1 = -height / 2.0, height / 2.0
        edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                 ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        for start, end in edges:
            for part in range(segments_per_side):
                t0, t1 = part / segments_per_side, (part + 1) / segments_per_side
                x_a = start[0] + (end[0] - start[0]) * t0
                y_a = start[1] + (end[1] - start[1]) * t0
                x_b = start[0] + (end[0] - start[0]) * t1
                y_b = start[1] + (end[1] - start[1]) * t1
                starts.append((x_a, y_a, 0.0))
                vectors.append((x_b - x_a, y_b - y_a, 0.0))
    starts_np = np.asarray(starts, dtype=float)
    vectors_np = np.asarray(vectors, dtype=float)
    ax, ay, az = np.radians(np.asarray(angles_deg, dtype=float))
    rotate_x = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rotate_y = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rotate_z = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rotation = rotate_z @ rotate_y @ rotate_x
    starts_np = starts_np @ rotation.T + np.asarray(position_m, dtype=float)
    vectors_np = vectors_np @ rotation.T
    return starts_np, vectors_np


def _mutual_inductance_neumann_vectorized_h(
    tx: RectangularWinding,
    rx: RectangularWinding,
    relative_position_m: Position3,
    rx_angles_deg: Angles3,
    segments_per_side: int,
) -> float:
    tx_start, tx_dl = _segment_arrays(tx, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), segments_per_side)
    rx_start, rx_dl = _segment_arrays(rx, relative_position_m, rx_angles_deg, segments_per_side)
    tx_mid = tx_start + tx_dl * 0.5
    rx_mid = rx_start + rx_dl * 0.5
    displacement = rx_mid[None, :, :] - tx_mid[:, None, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance == 0.0):
        raise ValueError("coils intersect; Neumann mutual integral is singular")
    dot = tx_dl @ rx_dl.T
    return 1e-7 * float(np.sum(dot / distance))


def _parallel_segment_integral(a0: float, a1: float, b0: float, b1: float, separation_m: float) -> float:
    """Closed double line integral for two parallel segments (units: m)."""
    if separation_m <= 0:
        raise ValueError("coaxial closed form requires nonzero separation")

    def primitive(t: float) -> float:
        q = separation_m
        return sqrt(t * t + q * q) - t * asinh(t / q)

    return (
        primitive(a1 - b0) - primitive(a0 - b0)
        - primitive(a1 - b1) + primitive(a0 - b1)
    )


def mutual_inductance_coaxial_closed_form_h(
    tx: RectangularWinding,
    rx: RectangularWinding,
    axial_gap_m: float,
) -> float:
    """Closed-form Neumann sum for coaxial, parallel rectangular windings.

    Perpendicular segment pairs contribute zero; parallel pairs use the
    analytic double integral.  This supports rectangular coils and multiple
    concentric turns without numerical quadrature.
    """
    if axial_gap_m == 0:
        raise ValueError("coaxial loops must have nonzero axial gap")
    tx_segments, rx_segments = _loop_segments(tx), _loop_segments(rx)
    total = 0.0
    for axis_a, a0, a1, fixed_a, z_a, dir_a in tx_segments:
        for axis_b, b0, b1, fixed_b, z_b, dir_b in rx_segments:
            if axis_a != axis_b:
                continue
            transverse = fixed_b - fixed_a
            separation = sqrt(transverse * transverse + (axial_gap_m + z_b - z_a) ** 2)
            integral = _parallel_segment_integral(a0, a1, b0, b1, separation)
            total += dir_a * dir_b * integral
    # Neumann's prefactor mu0/(4*pi) is exactly 1e-7 H/m in SI.
    return -1e-7 * total


def mutual_inductance_h(
    tx: RectangularWinding,
    rx: RectangularWinding,
    relative_position_m: Position3,
    rx_angles_deg: Angles3 = (0.0, 0.0, 0.0),
    segments_per_side: int = 8,
    method: str = "auto",
) -> float:
    """Mutual inductance; coaxial uses closed form, displaced uses integration."""
    _validate_position(relative_position_m)
    if method not in {"auto", "closed_form", "numerical"}:
        raise ValueError("method must be 'auto', 'closed_form', or 'numerical'")
    coaxial = (
        isclose(relative_position_m[0], 0.0, abs_tol=1e-15)
        and isclose(relative_position_m[1], 0.0, abs_tol=1e-15)
        and all(isclose(angle, 0.0, abs_tol=1e-12) for angle in rx_angles_deg)
        and not isclose(relative_position_m[2], 0.0, abs_tol=1e-15)
    )
    if method == "closed_form" and not coaxial:
        raise ValueError("closed-form method is only available for coaxial parallel coils")
    if method == "closed_form" or (method == "auto" and coaxial):
        return mutual_inductance_coaxial_closed_form_h(tx, rx, relative_position_m[2])
    return _mutual_inductance_neumann_vectorized_h(
        tx, rx, relative_position_m, rx_angles_deg, segments_per_side
    )


def tx_rx_mutual_vector(
    tx_winding: RectangularWinding,
    tx_positions_m: Sequence[Position3],
    rx_winding: RectangularWinding,
    rx_position_m: Position3,
    rx_angles_deg: Angles3 = (0.0, 0.0, 0.0),
    segments_per_side: int = 8,
) -> list[float]:
    """Return m_k(x): each transmitter's mutual inductance to an Rx pose."""
    _validate_position(rx_position_m)
    values = []
    for tx_position in tx_positions_m:
        _validate_position(tx_position)
        relative = tuple(rx_position_m[i] - tx_position[i] for i in range(3))
        values.append(mutual_inductance_h(
            tx_winding, rx_winding, relative, rx_angles_deg, segments_per_side
        ))
    return values


def transmitter_mutual_matrix(
    winding: RectangularWinding,
    tx_positions_m: Sequence[Position3],
    self_inductance_h: float | Sequence[float],
    segments_per_side: int = 8,
) -> list[list[float]]:
    """Build the reciprocal Tx matrix; the diagonal contains self L values."""
    if not tx_positions_m:
        raise ValueError("at least one transmitter position is required")
    for position in tx_positions_m:
        _validate_position(position)
    if isinstance(self_inductance_h, (int, float)):
        diagonal = [float(self_inductance_h)] * len(tx_positions_m)
    else:
        diagonal = [float(value) for value in self_inductance_h]
    if len(diagonal) != len(tx_positions_m) or any(value <= 0 for value in diagonal):
        raise ValueError("provide one positive self-inductance per transmitter")

    count = len(tx_positions_m)
    matrix = [[0.0] * count for _ in range(count)]
    for i in range(count):
        matrix[i][i] = diagonal[i]
        for j in range(i + 1, count):
            relative = tuple(tx_positions_m[j][axis] - tx_positions_m[i][axis] for axis in range(3))
            value = mutual_inductance_h(
                winding, winding, relative, segments_per_side=segments_per_side
            )
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def coupling_matrix(mutual_matrix_h: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return pair coupling coefficients; diagonal is zero (self-pairs excluded)."""
    count = len(mutual_matrix_h)
    if count == 0 or any(len(row) != count for row in mutual_matrix_h):
        raise ValueError("mutual matrix must be nonempty and square")
    result = [[0.0] * count for _ in range(count)]
    for i in range(count):
        if mutual_matrix_h[i][i] <= 0:
            raise ValueError("self-inductance diagonal entries must be positive")
        for j in range(i + 1, count):
            if mutual_matrix_h[j][j] <= 0:
                raise ValueError("self-inductance diagonal entries must be positive")
            value = mutual_matrix_h[i][j] / sqrt(mutual_matrix_h[i][i] * mutual_matrix_h[j][j])
            result[i][j] = result[j][i] = value
    return result


def validate_mutual_matrix(
    mutual_matrix_h: Sequence[Sequence[float]], tolerance_h: float = MATRIX_TOLERANCE_H
) -> dict[str, Any]:
    """Check square shape, symmetry/reciprocity, and the pair-coupling limit."""
    count = len(mutual_matrix_h)
    if count == 0 or any(len(row) != count for row in mutual_matrix_h):
        raise ValueError("mutual matrix must be nonempty and square")
    max_asymmetry_h = max(
        abs(mutual_matrix_h[i][j] - mutual_matrix_h[j][i])
        for i in range(count) for j in range(count)
    )
    k_matrix = coupling_matrix(mutual_matrix_h)
    max_pair_k = max(
        (abs(k_matrix[i][j]) for i in range(count) for j in range(count) if i != j),
        default=0.0,
    )
    return {
        "symmetric": max_asymmetry_h <= tolerance_h,
        "reciprocal": max_asymmetry_h <= tolerance_h,
        "max_asymmetry_h": max_asymmetry_h,
        "tolerance_h": tolerance_h,
        "max_abs_pair_k": max_pair_k,
        "coupling_limit": MAX_PAIR_COUPLING,
        "coupling_ok": max_pair_k < MAX_PAIR_COUPLING,
        "k_matrix_diagonal": "zero; k_ij constraint applies only to distinct coils",
    }


def scenario_b_grid_c_assumptions(
    tx_pitch_m: float = 0.31,
    rx_gap_m: float = 0.035,
) -> dict[str, Any]:
    """Illustrative [C] 4x4 Tx and 24-Rx grid for heatmap plumbing.

    Coordinates use local face axes x/y over a 1.2 m x 1.2 m illustrative
    coverage face, centered at x=y=0. Tx coils occupy a centered 4x4 grid;
    24 Rx poses use a 6x4 grid at the Aalto 35 mm gap along z. P0's 0.6 m
    third extent and flux-pipe core are not modeled in this 2-D example.
    """
    if tx_pitch_m <= 0 or rx_gap_m <= 0:
        raise ValueError("grid pitch and Rx gap must be positive")
    tx_axis = tuple((index - 1.5) * tx_pitch_m for index in range(4))
    tx_positions = [(x, y, 0.0) for y in tx_axis for x in tx_axis]
    rx_x = tuple(-0.5 + index * 0.2 for index in range(6))
    rx_y = tuple(-0.45 + index * 0.3 for index in range(4))
    rx_positions = [(x, y, rx_gap_m) for y in rx_y for x in rx_x]
    return {
        "tx_positions_m": tx_positions,
        "rx_positions_m": rx_positions,
        "tx_pitch_m": tx_pitch_m,
        "rx_gap_m": rx_gap_m,
        "provenance": "[C] illustrative regular grid; not a measured AIDC layout",
        "scope": "local 1.2 x 1.2 m face; free-space filament coils; no flux-pipe correction",
    }


def scenario_b_heatmap_data(
    tx_winding: RectangularWinding,
    rx_winding: RectangularWinding,
    self_inductance_h: float = 21e-6,
    tx_pitch_m: float = 0.31,
    rx_gap_m: float = 0.035,
    segments_per_side: int = 8,
) -> dict[str, Any]:
    """Return matrices/vectors suitable for M and 24-position heatmaps."""
    grid = scenario_b_grid_c_assumptions(tx_pitch_m, rx_gap_m)
    tx_positions = grid["tx_positions_m"]
    rx_positions = grid["rx_positions_m"]
    mutual_tx = transmitter_mutual_matrix(
        tx_winding, tx_positions, self_inductance_h, segments_per_side
    )
    vectors = [tx_rx_mutual_vector(
        tx_winding, tx_positions, rx_winding, position,
        segments_per_side=segments_per_side,
    ) for position in rx_positions]
    report = validate_mutual_matrix(mutual_tx)
    return {
        **grid,
        "M_tx_h": mutual_tx,
        "k_tx": coupling_matrix(mutual_tx),
        "m_rx_by_load_h": vectors,
        "M_heatmap_shape": [len(mutual_tx), len(mutual_tx)],
        "m_heatmap_shape": [len(vectors), len(tx_positions)],
        "matrix_validation": report,
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": {
            "windings": "RectangularWinding",
            "tx_positions_m": "N center coordinates (x,y,z)",
            "rx_position_m": "one receiver center coordinate (x,y,z)",
        },
        "outputs": {
            "M_tx_h": "N x N matrix with self inductance on diagonal",
            "m_rx_h": "N-element Tx-Rx mutual vector",
            "heatmap": "M matrix plus 24 x N receiver-position matrix",
        },
        "valid_ranges": {
            "coaxial_gap_m": "nonzero",
            "segments_per_side": ">= 1 for numerical off-axis integration",
            "pair_coupling": "abs(k_ij) < 0.3 for i != j",
        },
        "TODOs": [
            "TODO[实测]: replace illustrative [C] grid with actual AIDC Tx/Rx coordinates.",
            "TODO[实测/FEM]: apply flux-pipe core correction to free-space mutual values.",
        ],
    }
