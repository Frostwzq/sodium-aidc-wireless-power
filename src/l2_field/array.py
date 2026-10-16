"""4x4 multi-frequency Tx phase patterns and arbitrary-size cluster tiling."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

F1_PHASE_PATTERN_DEG = np.array(
    [
        [0.0, 90.0, 0.0, 90.0],
        [270.0, 270.0, 180.0, 180.0],
        [270.0, 270.0, 180.0, 180.0],
        [0.0, 90.0, 0.0, 90.0],
    ],
    dtype=float,
)
F2_PHASE_PATTERN_DEG = np.mod(F1_PHASE_PATTERN_DEG + 90.0, 360.0)


def phase_patterns(
    nx: int,
    ny: int,
    swap_adjacent_clusters: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile f1/f2 maps into an arbitrary nx-by-ny array.

    Every 4x4 cluster alternates between the two supplied maps, so adjacent
    clusters exchange diamond/cross frequency assignments. Partial clusters
    are cropped, allowing non-multiples of four.
    """
    if nx < 1 or ny < 1:
        raise ValueError("array dimensions nx and ny must be positive")
    p1 = np.empty((ny, nx), dtype=float)
    p2 = np.empty((ny, nx), dtype=float)
    for row in range(ny):
        for column in range(nx):
            tile_row, tile_column = row // 4, column // 4
            swap = swap_adjacent_clusters and (tile_row + tile_column) % 2 == 1
            local_row, local_column = row % 4, column % 4
            if swap:
                p1[row, column] = F2_PHASE_PATTERN_DEG[local_row, local_column]
                p2[row, column] = F1_PHASE_PATTERN_DEG[local_row, local_column]
            else:
                p1[row, column] = F1_PHASE_PATTERN_DEG[local_row, local_column]
                p2[row, column] = F2_PHASE_PATTERN_DEG[local_row, local_column]
    return p1, p2


def classify_intersection(
    phases_f1_deg: Sequence[float], phases_f2_deg: Sequence[float]
) -> str:
    """Classify four surrounding Tx phases using the P4 [A]/[C] rule."""
    if len(phases_f1_deg) != 4 or len(phases_f2_deg) != 4:
        raise ValueError("intersection classification requires four Tx phases per tone")
    unique_f1 = sorted({round(float(value) % 360.0, 3) for value in phases_f1_deg})
    unique_f2 = sorted({round(float(value) % 360.0, 3) for value in phases_f2_deg})

    def four_quadrants(phases: list[float]) -> bool:
        return len(phases) == 4 and all(
            abs(((phases[(i + 1) % 4] - phases[i]) % 360.0) - 90.0) < 1e-3
            for i in range(4)
        )

    if len(unique_f1) == 1 and four_quadrants(unique_f2):
        return "diamond"
    if len(unique_f2) == 1 and four_quadrants(unique_f1):
        return "diamond"
    if len(unique_f1) == 2 and len(unique_f2) == 2:
        return "cross"
    return "other"


def describe() -> dict[str, Any]:
    return {
        "inputs": {"nx": "positive Tx columns", "ny": "positive Tx rows"},
        "outputs": {"phase_f1_deg": "ny x nx phase map", "phase_f2_deg": "ny x nx phase map"},
        "valid_ranges": {"phase_deg": "[0, 360)", "nx_ny": "> 0; partial 4x4 clusters allowed"},
        "TODOs": ["TODO[实测]: verify Fig.1(c) phase-map extraction against the paper source image."],
    }
