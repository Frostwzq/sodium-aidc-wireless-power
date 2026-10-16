"""Parametric Tx/Rx coil estimates for P2.

The reference Tx winding is an inverse estimate. Wheeler's square-spiral
formula and the proximity multiplier are engineering approximations, not
measurements of the Aalto prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, radians, sin, sqrt
from typing import Any

MU0_H_PER_M = 4e-7 * pi
COPPER_RESISTIVITY_OHM_M = 1.72e-8
REFERENCE_FREQUENCY_HZ = 200_000.0


@dataclass(frozen=True)
class LitzConductor:
    strands: int = 40
    strand_diameter_m: float = 0.1e-3
    proximity_eta_range: tuple[float, float] = (0.0, 2.0)
    source_label: str = "[B] proposed conductor; R_ac proximity factor is [C]"

    def validate(self) -> None:
        if self.strands < 1 or self.strand_diameter_m <= 0:
            raise ValueError("Litz strand count and diameter must be positive")
        lo, hi = self.proximity_eta_range
        if lo < 0 or hi < lo:
            raise ValueError("proximity_eta_range must satisfy 0 <= low <= high")


@dataclass(frozen=True)
class SquareSpiral:
    outer_side_m: float
    turns: int
    inner_side_m: float
    conductor: LitzConductor = LitzConductor()
    frequency_hz: float = REFERENCE_FREQUENCY_HZ

    @property
    def radial_pitch_m(self) -> float:
        if self.turns <= 1:
            return 0.0
        return (self.outer_side_m - self.inner_side_m) / (2.0 * (self.turns - 1))

    def validate(self) -> None:
        if self.outer_side_m <= 0 or self.inner_side_m <= 0:
            raise ValueError("spiral side lengths must be positive")
        if self.inner_side_m >= self.outer_side_m:
            raise ValueError("inner_side_m must be smaller than outer_side_m")
        if self.turns < 1 or self.frequency_hz <= 0:
            raise ValueError("turns and frequency must be positive")
        if self.radial_pitch_m <= 0 and self.turns > 1:
            raise ValueError("multi-turn spiral must have positive radial pitch")
        self.conductor.validate()


@dataclass(frozen=True)
class CoilEstimate:
    inductance_h: float
    inductance_uncertainty_fraction: float
    wire_length_m: float
    skin_depth_m: float
    resistance_dc_ohm: float
    resistance_ac_range_ohm: tuple[float, float]
    q_range: tuple[float, float]
    proximity_multiplier_range: tuple[float, float]
    provenance: tuple[str, ...]


def skin_depth_m(
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
    resistivity_ohm_m: float = COPPER_RESISTIVITY_OHM_M,
) -> float:
    if frequency_hz <= 0 or resistivity_ohm_m <= 0:
        raise ValueError("frequency and resistivity must be positive")
    omega = 2.0 * pi * frequency_hz
    return sqrt(2.0 * resistivity_ohm_m / (omega * MU0_H_PER_M))


def wheeler_square_spiral_h(outer_side_m: float, inner_side_m: float, turns: int) -> float:
    """Return Wheeler square-spiral inductance in henries.

    The equivalent radii are based on perimeter/(2*pi). The Wheeler estimate
    is assigned a conservative +/-25% geometry uncertainty for this square
    coil approximation.
    """
    if outer_side_m <= inner_side_m or inner_side_m <= 0 or turns < 1:
        raise ValueError("require outer_side > inner_side > 0 and turns >= 1")
    r_out_in = 4.0 * outer_side_m / (2.0 * pi * 0.0254)
    r_in_in = 4.0 * inner_side_m / (2.0 * pi * 0.0254)
    r_avg_in = (r_out_in + r_in_in) / 2.0
    winding_width_in = r_out_in - r_in_in
    inductance_u_h = (r_avg_in**2 * turns**2) / (8.0 * r_avg_in + 11.0 * winding_width_in)
    return inductance_u_h * 1e-6


def inverse_reference_spiral(
    target_inductance_h: float = 21e-6,
    outer_side_m: float = 0.157,
    turns_range: range = range(10, 25),
    conductor: LitzConductor = LitzConductor(),
) -> tuple[SquareSpiral, CoilEstimate]:
    """Find the integer-turn Wheeler estimate closest to Aalto's 21 uH value.

    Inner side is fixed to outer_side/turns as a declared full-winding design
    assumption. Actual turn count and trace geometry were not published.
    """
    if target_inductance_h <= 0 or outer_side_m <= 0:
        raise ValueError("target inductance and outer side must be positive")
    candidates: list[tuple[float, SquareSpiral]] = []
    for turns in turns_range:
        if turns < 2:
            continue
        inner_side_m = outer_side_m / turns
        coil = SquareSpiral(outer_side_m, turns, inner_side_m, conductor)
        coil.validate()
        inductance = wheeler_square_spiral_h(outer_side_m, inner_side_m, turns)
        candidates.append((abs(inductance - target_inductance_h), coil))
    if not candidates:
        raise ValueError("turns_range contains no usable multi-turn candidate")
    coil = min(candidates, key=lambda item: item[0])[1]
    return coil, estimate_coil(coil)


def estimate_coil(coil: SquareSpiral, inductance_uncertainty_fraction: float = 0.25) -> CoilEstimate:
    """Estimate L, Litz R_ac and Q, including a bounded proximity correction.

    Skin effect uses the thin-round-strand approximation. Proximity is a
    Dowell-type multiplier supplied as a range because no measured Aalto R_ac
    or winding construction was published. The multiplier is not a FEM result.
    """
    coil.validate()
    if not 0 <= inductance_uncertainty_fraction < 1:
        raise ValueError("inductance uncertainty must be in [0, 1)")
    turns = coil.turns
    mean_side = (coil.outer_side_m + coil.inner_side_m) / 2.0
    wire_length = 4.0 * mean_side * turns
    conductor = coil.conductor
    copper_area = conductor.strands * pi * (conductor.strand_diameter_m / 2.0) ** 2
    r_dc = COPPER_RESISTIVITY_OHM_M * wire_length / copper_area
    delta = skin_depth_m(coil.frequency_hz)
    strand_radius = conductor.strand_diameter_m / 2.0
    x = strand_radius / delta
    # Conservative small-strand approximation; exact bundle loss needs test data.
    k_skin = 1.0 + x * x / 12.0
    pitch = coil.radial_pitch_m
    packing = (turns * conductor.strand_diameter_m / pitch) ** 2 if pitch > 0 else 0.0
    eta_lo, eta_hi = conductor.proximity_eta_range
    k_prox_lo, k_prox_hi = 1.0 + eta_lo * packing, 1.0 + eta_hi * packing
    rac_low, rac_high = r_dc * k_skin * k_prox_lo, r_dc * k_skin * k_prox_hi
    inductance = wheeler_square_spiral_h(coil.outer_side_m, coil.inner_side_m, turns)
    omega = 2.0 * pi * coil.frequency_hz
    q_low = omega * inductance * (1.0 - inductance_uncertainty_fraction) / rac_high
    q_high = omega * inductance * (1.0 + inductance_uncertainty_fraction) / rac_low
    return CoilEstimate(
        inductance_h=inductance,
        inductance_uncertainty_fraction=inductance_uncertainty_fraction,
        wire_length_m=wire_length,
        skin_depth_m=delta,
        resistance_dc_ohm=r_dc,
        resistance_ac_range_ohm=(rac_low, rac_high),
        q_range=(q_low, q_high),
        proximity_multiplier_range=(k_prox_lo, k_prox_hi),
        provenance=(
            "[A] Aalto Tx outer footprint 157 x 157 mm and L_Tx=21 uH target",
            "[C] inverse winding: inner side=outer side/turns; Wheeler square approximation",
            "[B] candidate Litz 40 x 0.1 mm; [C] Dowell-type proximity eta range 0..2",
            "TODO[实测]: measure winding geometry and R_ac; verify L with FEM or LCR measurement",
        ),
    )


@dataclass(frozen=True)
class RectangularWinding:
    """Thin filament rectangular winding used for free-space mutual estimates."""

    outer_width_m: float
    outer_height_m: float
    turns: int = 1
    radial_pitch_m: float = 0.0

    def validate(self) -> None:
        if self.outer_width_m <= 0 or self.outer_height_m <= 0 or self.turns < 1:
            raise ValueError("winding dimensions and turns must be positive")
        if self.radial_pitch_m < 0:
            raise ValueError("radial pitch cannot be negative")
        if self.turns > 1 and min(self.outer_width_m, self.outer_height_m) <= 2 * self.radial_pitch_m * (self.turns - 1):
            raise ValueError("winding pitch leaves no positive inner opening")


@dataclass(frozen=True)
class FluxPipeRx:
    winding: RectangularWinding
    core_cross_section_m2: float | None = None
    core_window_m: tuple[float, float] | None = None
    core_relative_permeability: float | None = None
    provenance: str = "[C] equivalent rectangular winding; no core-corrected coupling without magnetic-path data"

    def validate(self) -> None:
        self.winding.validate()
        if self.core_cross_section_m2 is not None and self.core_cross_section_m2 <= 0:
            raise ValueError("core_cross_section_m2 must be positive")
        if self.core_relative_permeability is not None and self.core_relative_permeability < 1:
            raise ValueError("relative permeability must be >= 1")


def _rotate_xyz(point: tuple[float, float, float], angles_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    ax, ay, az = (radians(a) for a in angles_deg)
    y, z = y * cos(ax) - z * sin(ax), y * sin(ax) + z * cos(ax)
    x, z = x * cos(ay) + z * sin(ay), -x * sin(ay) + z * cos(ay)
    x, y = x * cos(az) - y * sin(az), x * sin(az) + y * cos(az)
    return x, y, z


def _winding_segments(
    winding: RectangularWinding,
    position_m: tuple[float, float, float],
    angles_deg: tuple[float, float, float],
    segments_per_side: int,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    winding.validate()
    if segments_per_side < 1:
        raise ValueError("segments_per_side must be >= 1")
    segments = []
    for turn in range(winding.turns):
        width = winding.outer_width_m - 2.0 * turn * winding.radial_pitch_m
        height = winding.outer_height_m - 2.0 * turn * winding.radial_pitch_m
        vertices = ((-width/2, -height/2, 0.0), (width/2, -height/2, 0.0),
                    (width/2, height/2, 0.0), (-width/2, height/2, 0.0),
                    (-width/2, -height/2, 0.0))
        for start, end in zip(vertices, vertices[1:]):
            for i in range(segments_per_side):
                t0, t1 = i / segments_per_side, (i + 1) / segments_per_side
                local0 = tuple(start[j] + (end[j] - start[j]) * t0 for j in range(3))
                local1 = tuple(start[j] + (end[j] - start[j]) * t1 for j in range(3))
                p0r, p1r = _rotate_xyz(local0, angles_deg), _rotate_xyz(local1, angles_deg)
                p0 = tuple(p0r[j] + position_m[j] for j in range(3))
                p1 = tuple(p1r[j] + position_m[j] for j in range(3))
                segments.append((p0, tuple(p1[j] - p0[j] for j in range(3))))
    return segments


def mutual_inductance_neumann_h(
    tx: RectangularWinding,
    rx: RectangularWinding,
    relative_position_m: tuple[float, float, float],
    rx_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments_per_side: int = 8,
) -> float:
    """Numerically integrate the Neumann formula for filament rectangular coils.

    This is a free-space/equivalent-winding result; it does not model ferrite
    permeability or flux-pipe concentration. Rx turn count and pitch must be
    supplied explicitly when an absolute value is needed.
    """
    tx_segments = _winding_segments(tx, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), segments_per_side)
    rx_segments = _winding_segments(rx, relative_position_m, rx_angles_deg, segments_per_side)
    total = 0.0
    for p1, dl1 in tx_segments:
        mid1 = tuple(p1[k] + dl1[k] / 2.0 for k in range(3))
        for p2, dl2 in rx_segments:
            mid2 = tuple(p2[k] + dl2[k] / 2.0 for k in range(3))
            r = tuple(mid2[k] - mid1[k] for k in range(3))
            distance = sqrt(sum(value * value for value in r))
            if distance == 0:
                raise ValueError("coils intersect; Neumann mutual integral is singular")
            total += sum(dl1[k] * dl2[k] for k in range(3)) / distance
    return MU0_H_PER_M * total / (4.0 * pi)


def tolerance_analysis(
    tx: RectangularWinding,
    rx: FluxPipeRx,
    nominal_position_m: tuple[float, float, float],
    position_tolerance_mm: float = 5.0,
    attitude_tolerance_deg: float = 5.0,
    segments_per_side: int = 8,
) -> dict[str, Any]:
    """Return baseline M and the largest relative change across signed axes."""
    rx.validate()
    if position_tolerance_mm < 0 or attitude_tolerance_deg < 0:
        raise ValueError("tolerances cannot be negative")
    base = mutual_inductance_neumann_h(tx, rx.winding, nominal_position_m,
                                       segments_per_side=segments_per_side)
    if base == 0:
        raise ValueError("nominal mutual inductance is zero; relative tolerance is undefined")
    samples: list[dict[str, Any]] = []
    p_tol = position_tolerance_mm / 1000.0
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pos = list(nominal_position_m)
            pos[axis] += sign * p_tol
            value = mutual_inductance_neumann_h(tx, rx.winding, tuple(pos),
                                                 segments_per_side=segments_per_side)
            samples.append({"kind": "position", "axis": "xyz"[axis], "offset": sign * position_tolerance_mm,
                            "m_h": value, "change_pct": (value / base - 1.0) * 100.0})
    for axis in range(3):
        for sign in (-1.0, 1.0):
            angles = [0.0, 0.0, 0.0]
            angles[axis] = sign * attitude_tolerance_deg
            value = mutual_inductance_neumann_h(tx, rx.winding, nominal_position_m, tuple(angles),
                                                 segments_per_side=segments_per_side)
            samples.append({"kind": "attitude", "axis": "xyz"[axis], "offset": sign * attitude_tolerance_deg,
                            "m_h": value, "change_pct": (value / base - 1.0) * 100.0})
    return {
        "m_nominal_h": base,
        "position_tolerance_mm": position_tolerance_mm,
        "attitude_tolerance_deg": attitude_tolerance_deg,
        "samples": samples,
        "max_abs_change_pct": max(abs(row["change_pct"]) for row in samples),
        "scope": "equivalent filament winding in free space; flux-pipe core correction not applied",
        "todos": ["TODO[实测]: Rx turns/pitch and flux-pipe magnetic-path geometry are not published."],
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": {"tx_geometry": "SquareSpiral", "rx_geometry": "FluxPipeRx", "relative_position_m": "3-vector"},
        "outputs": {"r_ac": "bounded estimate including skin and proximity factors", "l_self": "Wheeler estimate", "q_self": "bounded estimate", "m_txrx": "Neumann free-space integral"},
        "valid_ranges": {"frequency_hz": "> 0", "turns": ">= 1", "position_tolerance_mm": ">= 0", "attitude_tolerance_deg": ">= 0"},
        "TODOs": ["TODO[实测]: verify R_ac and L with prototype measurement.", "TODO[实测]: identify Rx winding and core correction."],
    }
