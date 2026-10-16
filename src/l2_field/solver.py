"""P4 synthesis and blind-spot diagnostics.

The spatial coupling kernel and 0.5 V receiver threshold are user-authorized
[C] engineering assumptions.  The baseline and proposed-pattern results are
diagnostics; this module reports unmet criteria instead of tuning thresholds.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.l2_field.array import phase_patterns

V_DC_V = 30.0
V_O_PEAK_V = 2.0 * V_DC_V / math.pi
F1_HZ = 199_000.0
F2_HZ = 201_000.0
L_TX_H = 21e-6
L_RX_H = 360e-6
L_F_H = 11.4e-6
C_TX_F = 67.2e-9
C_F_F = 55.2e-9
TX_SIDE_M = 0.157
RX_WIDTH_M = 0.255
RX_HEIGHT_M = 0.115
LOAD_INDEPENDENCE_TOLERANCE = 1e-6
V_RX_MIN_V = 0.5
RX_TURNS_ASSUMED = 40.0
BLIND_EPSILON_FRACTION = 0.05
BASE_GRID_PER_TX = 8
DENSE_GRID_PER_TX = 64
SCENARIO_LABEL = "P0-B [C] 4x4 reference field example"


def mutual_inductance_approx_h(
    dx: np.ndarray | float,
    dy: np.ndarray | float,
    side: float = 1.0,
    coil_inductance_h: float = L_TX_H,
    theta_deg: float = 0.0,
) -> np.ndarray | float:
    """Signed [C] spatial kernel from the supplied P4 model.

    x/y are in units consistent with ``side``. This qualitative kernel is not
    an FEM or P3 geometric mutual model; it intentionally preserves the
    supplied sign reversal and zero at d=0 for blind-spot falsification.
    """
    if side <= 0 or coil_inductance_h <= 0:
        raise ValueError("side and coil_inductance_h must be positive")
    dx_array, dy_array = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
    distance = np.hypot(dx_array, dy_array)
    half_side = side / 2.0
    shape = 1.0 / ((distance / half_side) ** 2 + 1.0) ** 1.5
    projection = dx_array * math.cos(math.radians(theta_deg)) - dy_array * math.sin(math.radians(theta_deg))
    # Match the supplied sign convention (zero projection chooses +1); only
    # the exact geometric center is defined as zero below.
    sign = np.where(projection >= 0.0, 1.0, -1.0)
    result = sign * (coil_inductance_h / 8.0) * shape
    result = np.where(distance < 1e-12, 0.0, result)
    if np.isscalar(dx) and np.isscalar(dy):
        return float(result)
    return result


def _tx_grid(nx: int, ny: int) -> np.ndarray:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    return np.asarray([(column + 0.5, row + 0.5) for row in range(ny) for column in range(nx)])


def _synthesize_points(
    x_points: np.ndarray,
    y_points: np.ndarray,
    tx_positions: np.ndarray,
    phases_deg: np.ndarray,
    theta_deg: float,
    chunk_size: int = 4096,
) -> np.ndarray:
    flat_x, flat_y = x_points.ravel(), y_points.ravel()
    output = np.empty(flat_x.size, dtype=float)
    phase_weight = np.exp(-1j * np.radians(phases_deg.reshape(-1)))
    cos_theta, sin_theta = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    half_side = 0.5  # normalized Tx cell size
    for start in range(0, flat_x.size, chunk_size):
        stop = min(start + chunk_size, flat_x.size)
        dx = flat_x[start:stop, None] - tx_positions[None, :, 0]
        dy = flat_y[start:stop, None] - tx_positions[None, :, 1]
        distance = np.hypot(dx, dy)
        projection = dx * cos_theta - dy * sin_theta
        signs = np.where(projection >= 0.0, 1.0, -1.0)
        kernel = signs * (L_TX_H / 8.0) / ((distance / half_side) ** 2 + 1.0) ** 1.5
        kernel[distance < 1e-12] = 0.0
        voltage = (V_O_PEAK_V / L_F_H) * kernel
        output[start:stop] = np.abs(voltage @ phase_weight)
    return output.reshape(x_points.shape)


def conventional_single_tone_map(
    nx: int = 4,
    ny: int = 4,
    samples_per_tx: int = BASE_GRID_PER_TX,
    theta_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return |sum v_k| for the traditional alternating-column x pattern."""
    if samples_per_tx < 1:
        raise ValueError("samples_per_tx must be positive")
    tx_positions = _tx_grid(nx, ny)
    phases = np.asarray([
        [0.0 if column % 2 == 0 else 180.0 for column in range(nx)]
        for _row in range(ny)
    ])
    xs = np.linspace(0.0, float(nx), nx * samples_per_tx + 1)
    ys = np.linspace(0.0, float(ny), ny * samples_per_tx + 1)
    X, Y = np.meshgrid(xs, ys)
    field = _synthesize_points(X, Y, tx_positions, phases, theta_deg)
    return X, Y, field


def blind_spot_fraction(field_v: np.ndarray, epsilon_v: float) -> tuple[float, int]:
    if epsilon_v <= 0:
        raise ValueError("epsilon_v must be positive")
    count = int(np.count_nonzero(field_v < epsilon_v))
    return count / int(field_v.size), count


def test_blind_spot_exists(
    nx: int = 4,
    ny: int = 4,
    epsilon_fraction: float = BLIND_EPSILON_FRACTION,
    dense_samples_per_tx: int = DENSE_GRID_PER_TX,
) -> dict[str, Any]:
    """Find baseline same-frequency blind spots or raise immediately."""
    if not 0 < epsilon_fraction < 1:
        raise ValueError("epsilon_fraction must be in (0, 1)")
    _, _, coarse = conventional_single_tone_map(nx, ny, BASE_GRID_PER_TX)
    nominal_v = float(np.median(coarse))
    epsilon_v = epsilon_fraction * nominal_v
    coarse_fraction, coarse_count = blind_spot_fraction(coarse, epsilon_v)
    _, _, dense = conventional_single_tone_map(nx, ny, dense_samples_per_tx)
    fraction, count = blind_spot_fraction(dense, epsilon_v)
    if count == 0:
        raise RuntimeError(
            "no same-frequency blind spot found: P4 premise is not established; aborting"
        )
    return {
        "scenario_id": SCENARIO_LABEL,
        "nx": nx,
        "ny": ny,
        "grid_per_tx": dense_samples_per_tx,
        "grid_shape": list(dense.shape),
        "nominal_median_v": nominal_v,
        "epsilon_fraction": epsilon_fraction,
        "epsilon_v": epsilon_v,
        "coarse_grid_per_tx": BASE_GRID_PER_TX,
        "coarse_grid_shape": list(coarse.shape),
        "coarse_blind_count": coarse_count,
        "coarse_point_count": int(coarse.size),
        "coarse_blind_fraction": coarse_fraction,
        "blind_count": count,
        "point_count": int(dense.size),
        "blind_fraction": fraction,
        "blind_percent": 100.0 * fraction,
        "field_map_v": dense,
        "provenance": "[C] signed qualitative mutual kernel; normalized 1 Tx pitch grid",
    }


def solve_lcc_tx_current_with_load(
    source_voltage_v: complex,
    omega_rad_s: float,
    l_f_h: float,
    l_tx_h: float,
    c_tx_f: float,
    c_f_f: float,
    mutual_h: complex,
    z_load_ohm: complex,
    series_resistance_ohm: float = 0.0,
) -> dict[str, complex]:
    """Complex nodal solution for the supplied LCC and series-reflected load."""
    if min(omega_rad_s, l_f_h, l_tx_h, c_tx_f, c_f_f) <= 0:
        raise ValueError("frequency, inductances, and capacitances must be positive")
    if z_load_ohm == 0:
        raise ValueError("load impedance cannot be zero")
    z_lf = 1j * omega_rad_s * l_f_h
    z_cf = 1j * omega_rad_s * c_f_f
    z_c1 = 1.0 / (1j * omega_rad_s * c_tx_f)
    z_ltx = series_resistance_ohm + 1j * omega_rad_s * l_tx_h
    z_ref = (omega_rad_s * abs(mutual_h)) ** 2 / z_load_ohm
    z_load_total = z_ltx + z_ref
    z_branch = z_cf + z_load_total
    z_parallel = z_c1 * z_branch / (z_c1 + z_branch)
    z_in = z_lf + z_parallel
    v_a = source_voltage_v * z_parallel / z_in
    i_o = (source_voltage_v - v_a) / z_lf
    v_b = v_a * z_load_total / z_branch
    # The winding and reflected secondary impedance are in series; their
    # branch current is V_B / (Z_Ltx + Z_ref), not V_B / Z_Ltx alone.
    i_tx = v_b / z_load_total
    return {"I_o": i_o, "I_Tx": i_tx, "V_A": v_a, "Z_in": z_in, "Z_ref": z_ref}


def test_load_independence() -> dict[str, Any]:
    """Run the non-ideal LCC load sweep and report the original 1e-6 verdict."""
    omega = 2.0 * math.pi * 200_000.0
    m_test = mutual_inductance_approx_h(0.05, 0.0, TX_SIDE_M, L_TX_H, 0.0)
    loads = [10.0, 50.0, 100.0, 1_000.0, 10_000.0]
    conditions = [
        ("ideal_no_loss", 0.0, 0.0),
        ("winding_R_0.5_ohm", 0.5, 0.0),
        ("winding_R_0.5_ohm_Lf_plus_2pct", 0.5, 0.02),
    ]
    results: list[dict[str, Any]] = []
    for label, resistance, l_f_tolerance in conditions:
        currents = [
            solve_lcc_tx_current_with_load(
                V_O_PEAK_V,
                omega,
                L_F_H * (1.0 + l_f_tolerance),
                L_TX_H,
                C_TX_F,
                C_F_F,
                m_test,
                complex(load),
                resistance,
            )
            for load in loads
        ]
        tx_magnitudes = np.asarray([abs(item["I_Tx"]) for item in currents])
        inverter_magnitudes = np.asarray([abs(item["I_o"]) for item in currents])
        tx_relative_change = float((tx_magnitudes.max() - tx_magnitudes.min()) / tx_magnitudes.mean())
        inverter_relative_change = float(
            (inverter_magnitudes.max() - inverter_magnitudes.min()) / inverter_magnitudes.mean()
        )
        results.append({
            "condition": label,
            "series_resistance_ohm": resistance,
            "l_f_relative_tolerance": l_f_tolerance,
            "load_sweep_ohm": loads,
            "i_tx_a": tx_magnitudes.tolist(),
            "i_o_a": inverter_magnitudes.tolist(),
            "i_tx_relative_change": tx_relative_change,
            "i_o_relative_change": inverter_relative_change,
            "passes_original_threshold": tx_relative_change < LOAD_INDEPENDENCE_TOLERANCE,
        })
    return {
        "scenario_id": "P0-B [Aalto reference circuit check]",
        "threshold": LOAD_INDEPENDENCE_TOLERANCE,
        "comparison_point": "Tx winding current I_Tx; complex LCC nodal solution, not an ideal current source",
        "results": results,
        "overall_pass": all(item["passes_original_threshold"] for item in results),
        "status": "pass" if all(item["passes_original_threshold"] for item in results) else "criterion_failed_as_specified",
    }


def derive_epsilon_threshold() -> dict[str, float]:
    """Derive receiver threshold from the [C] conditioner voltage floor."""
    mu0 = 4.0e-7 * math.pi
    m_min = V_RX_MIN_V * L_F_H / V_O_PEAK_V
    a_eff_min = m_min / (mu0 * RX_TURNS_ASSUMED)
    return {
        "scenario_id": SCENARIO_LABEL,
        "v_rx_min_v": V_RX_MIN_V,
        "m_min_h": m_min,
        "m_min_uh": m_min * 1e6,
        "rx_turns_assumed": RX_TURNS_ASSUMED,
        "a_eff_min_m2": a_eff_min,
        "a_eff_min_cm2": a_eff_min * 1e4,
        "epsilon_th_v": V_RX_MIN_V,
    }


def noncoherent_envelope_rms(
    amplitude_1: float,
    amplitude_2: float,
    phase_difference_deg: float = 0.0,
    samples_per_beat: int = 512,
) -> tuple[float, float]:
    """Beat-envelope RMS and minimum instantaneous envelope.

    For the supplied envelope equation, the exact full-beat RMS is
    sqrt(A1**2 + A2**2), not (A1 + A2)/2. The exact instantaneous minimum is
    abs(A1 - A2); phase is included only for finite sampled beat records.
    """
    if amplitude_1 < 0 or amplitude_2 < 0 or samples_per_beat < 1:
        raise ValueError("amplitudes must be nonnegative and samples_per_beat positive")
    rms = math.sqrt(amplitude_1**2 + amplitude_2**2)
    beat_phase = np.linspace(0.0, 2.0 * math.pi, samples_per_beat, endpoint=False)
    envelope = np.sqrt(np.maximum(
        amplitude_1**2 + amplitude_2**2
        + 2.0 * amplitude_1 * amplitude_2
        * np.cos(beat_phase + math.radians(phase_difference_deg)),
        0.0,
    ))
    exact_min = abs(amplitude_1 - amplitude_2)
    return rms, min(float(envelope.min()), exact_min)


def multifrequency_rms_map(
    nx: int,
    ny: int,
    samples_per_tx: int = DENSE_GRID_PER_TX,
    theta_deg: float = 0.0,
    swap_adjacent_clusters: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-position RMS of the two-tone beat envelope in volts."""
    if samples_per_tx < 1:
        raise ValueError("samples_per_tx must be positive")
    tx_positions = _tx_grid(nx, ny)
    phases_1, phases_2 = phase_patterns(nx, ny, swap_adjacent_clusters)
    xs = np.linspace(0.0, float(nx), nx * samples_per_tx + 1)
    ys = np.linspace(0.0, float(ny), ny * samples_per_tx + 1)
    X, Y = np.meshgrid(xs, ys)
    sum_1 = _synthesize_points(X, Y, tx_positions, phases_1, theta_deg)
    sum_2 = _synthesize_points(X, Y, tx_positions, phases_2, theta_deg)
    # Each map above is |S_n|; RMS of the envelope over the beat period.
    rms = np.sqrt(sum_1**2 + sum_2**2)
    return X, Y, rms


def test_no_blind_spot(
    nx: int = 4,
    ny: int = 4,
    epsilon_th_v: float | None = None,
    dense_samples_per_tx: int = DENSE_GRID_PER_TX,
    orientations_deg: Sequence[float] = (0.0, 45.0, 90.0),
) -> dict[str, Any]:
    """Check the noncoherent RMS map at dense spatial/orientation samples."""
    threshold = derive_epsilon_threshold()["epsilon_th_v"] if epsilon_th_v is None else epsilon_th_v
    if threshold <= 0 or not orientations_deg:
        raise ValueError("epsilon_th and orientation samples must be positive/nonempty")
    maps = [multifrequency_rms_map(nx, ny, dense_samples_per_tx, angle)[2]
            for angle in orientations_deg]
    worst_map = np.minimum.reduce(maps)
    min_index = np.unravel_index(int(np.argmin(worst_map)), worst_map.shape)
    min_value = float(worst_map[min_index])
    xs = np.linspace(0.0, float(nx), nx * dense_samples_per_tx + 1)
    ys = np.linspace(0.0, float(ny), ny * dense_samples_per_tx + 1)
    boundary_mask = np.zeros(worst_map.shape, dtype=bool)
    if nx > 4:
        for edge in range(4, nx, 4):
            index = int(round(edge * dense_samples_per_tx))
            boundary_mask[:, index] = True
    if ny > 4:
        for edge in range(4, ny, 4):
            index = int(round(edge * dense_samples_per_tx))
            boundary_mask[index, :] = True
    boundary_min = float(worst_map[boundary_mask].min()) if np.any(boundary_mask) else None
    below_count = int(np.count_nonzero(worst_map < threshold))

    # Refine Rx orientation at the spatial minimum over a 0..90 degree sweep.
    x_min, y_min = float(xs[min_index[1]]), float(ys[min_index[0]])
    fine_theta = np.arange(0.0, 90.0001, 5.0)
    fine_values = []
    for angle in fine_theta:
        _, _, one_point_map = multifrequency_rms_map_at_points(
            nx, ny, np.asarray([x_min]), np.asarray([y_min]), float(angle)
        )
        fine_values.append(float(one_point_map[0]))
    return {
        "scenario_id": SCENARIO_LABEL,
        "nx": nx,
        "ny": ny,
        "grid_per_tx": dense_samples_per_tx,
        "grid_shape": list(worst_map.shape),
        "epsilon_th_v": threshold,
        "min_rms_v": min_value,
        "below_threshold_count": below_count,
        "point_count": int(worst_map.size),
        "below_threshold_fraction": below_count / int(worst_map.size),
        "min_location_tx_units": [x_min, y_min],
        "min_orientation_coarse_deg": list(orientations_deg),
        "min_orientation_fine_sweep_deg": fine_theta.tolist(),
        "fine_sweep_min_rms_v": min(fine_values),
        "boundary_min_rms_v": boundary_min,
        "no_blind_spot": min_value >= threshold,
        "field_map_v": worst_map,
        "provenance": "[C] qualitative signed mutual kernel; beat-envelope RMS; 1/64 Tx dense grid",
    }


def multifrequency_rms_map_at_points(
    nx: int,
    ny: int,
    x_points: np.ndarray,
    y_points: np.ndarray,
    theta_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx_positions = _tx_grid(nx, ny)
    phases_1, phases_2 = phase_patterns(nx, ny, True)
    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    magnitude_1 = np.empty(x_points.size)
    magnitude_2 = np.empty(y_points.size)
    for index, (x, y) in enumerate(zip(x_points.flat, y_points.flat)):
        dx, dy = x - tx_positions[:, 0], y - tx_positions[:, 1]
        m = mutual_inductance_approx_h(dx, dy, 1.0, L_TX_H, theta_deg)
        v = (V_O_PEAK_V / L_F_H) * np.asarray(m)
        magnitude_1[index] = abs(v @ np.exp(-1j * np.radians(phases_1.ravel())))
        magnitude_2[index] = abs(v @ np.exp(-1j * np.radians(phases_2.ravel())))
    rms = np.sqrt(magnitude_1**2 + magnitude_2**2)
    return x_points, y_points, rms


def search_feasible_sizes(
    sizes: Iterable[tuple[int, int]] = ((4, 4), (5, 3), (5, 4), (6, 4), (5, 5), (6, 6), (8, 4), (8, 8)),
    epsilon_th_v: float | None = None,
    samples_per_tx: int = BASE_GRID_PER_TX,
) -> dict[str, Any]:
    """Screen declared array sizes; report the exact tested set and step."""
    threshold = derive_epsilon_threshold()["epsilon_th_v"] if epsilon_th_v is None else epsilon_th_v
    reports = []
    for nx, ny in sizes:
        maps = [multifrequency_rms_map(nx, ny, samples_per_tx, angle)[2]
                for angle in (0.0, 45.0, 90.0)]
        minimum = float(np.minimum.reduce(maps).min())
        passes_screen = minimum >= threshold
        dense_minimum = None
        passes_dense = False
        if passes_screen:
            dense_report = test_no_blind_spot(nx, ny, threshold, DENSE_GRID_PER_TX)
            dense_minimum = dense_report["min_rms_v"]
            passes_dense = dense_report["no_blind_spot"]
        reports.append({
            "nx": nx,
            "ny": ny,
            "screen_min_rms_v": minimum,
            "dense_min_rms_v": dense_minimum,
            "passes": passes_dense,
        })
    return {
        "scenario_id": SCENARIO_LABEL,
        "tested_sizes": reports,
        "feasible_sizes": [[item["nx"], item["ny"]] for item in reports if item["passes"]],
        "epsilon_th_v": threshold,
        "grid_per_tx": samples_per_tx,
        "scope": "screen at requested grid; every surviving candidate is rechecked at 1/64 Tx",
    }


def write_svg_heatmap(
    field_map: np.ndarray,
    output_path: str | Path,
    title: str,
    threshold_v: float | None = None,
) -> Path:
    """Write a dependency-free SVG map (red cells are below threshold)."""
    data = np.asarray(field_map, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("field_map must be a nonempty 2D array")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, columns = data.shape
    finite = data[np.isfinite(data)]
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 1e-15)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="950" viewBox="0 0 900 950">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="50" y="35" font-family="sans-serif" font-size="20">{title}</text>',
        f'<text x="50" y="62" font-family="sans-serif" font-size="13">field range {low:.6g}..{high:.6g} V; threshold {threshold_v if threshold_v is not None else "n/a"} V</text>',
    ]
    cell_w, cell_h = 800.0 / columns, 800.0 / rows
    for row in range(rows):
        for column in range(columns):
            value = data[rows - 1 - row, column]
            ratio = (value - low) / span
            red = int(245 - 190 * ratio)
            green = int(250 - 170 * ratio)
            blue = int(255 - 70 * ratio)
            color = "#d62728" if threshold_v is not None and value < threshold_v else f"rgb({red},{green},{blue})"
            parts.append(
                f'<rect x="{50 + column * cell_w:.3f}" y="{90 + row * cell_h:.3f}" '
                f'width="{cell_w + 0.1:.3f}" height="{cell_h + 0.1:.3f}" fill="{color}"/>'
            )
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def describe() -> dict[str, Any]:
    return {
        "inputs": {
            "array_size": "nx, ny Tx counts",
            "sample_density": "samples per normalized Tx pitch",
            "load_sweep_ohm": [10, 50, 100, 1000, 10000],
        },
        "outputs": {
            "single_tone": "blind-spot field map, count, and sampled volume fraction",
            "load_independence": "non-ideal LCC current sweep and original-threshold verdict",
            "noncoherent": "dense beat-envelope RMS map and minimum",
        },
        "valid_ranges": {
            "epsilon_fraction": "(0, 1)",
            "grid": "1/8 Tx coarse; 1/64 Tx dense",
            "orientations_deg": "at least 0, 45, 90",
        },
        "TODOs": [
            "TODO[实测/FEM]: replace the signed qualitative mutual kernel with validated flux-pipe coupling.",
            "TODO[实测]: validate 0.5 V conditioner sensitivity and 40-turn Rx assumption.",
            "TODO[原文复核]: confirm the transcribed Fig.1(c) phase map and circuit Eq.(2) against the primary paper.",
        ],
    }
