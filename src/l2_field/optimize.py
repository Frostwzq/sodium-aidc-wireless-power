"""P7 global parameter scan and multi-objective optimization.

This is a transparent engineering search around the existing P5/P4 models.
Parameters that those modules do not yet model are handled by explicit [C]
proxy terms below; the report keeps those assumptions attached to every run.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from src.config import Config
from src.l2_field.array import phase_patterns
from src.l2_field.loss import ASSUMPTIONS as P5_ASSUMPTIONS
from src.l2_field.loss import loss_breakdown
from src.l2_field.solver import (
    L_F_H as P4_L_F_H,
    L_TX_H,
    V_O_PEAK_V,
    _synthesize_points,
    _tx_grid,
)
from src.l2_field.zvs import THIRD_QUADRANT_DROP_V, zero_crossing_window_s


SCENARIOS = ("A_subzone", "B_half_rack", "C_full_rack")
SCENARIO_LABELS = {
    "A_subzone": "P0-A",
    "B_half_rack": "P0-B",
    "C_full_rack": "P0-C",
}

# The P7 ranges are retained. P5/P6 leave these system-level relationships
# undefined, so they are explicit [C] proxies rather than inferred hardware.
ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] user-authorized P7 assumptions; replace with measured/model-derived values",
    "n_tx_by_scenario": {"A_subzone": 4, "B_half_rack": 10, "C_full_rack": 24},
    "n_tx_rule": "one Tx per P0 maximum load; counts are fixed per scenario",
    "n_phase_meaning": "parallel inverter phase groups sharing scenario power",
    "n_phase_loss_model": "P_cond scales as (4/n_phase)^2; P_sw and P_ctrl scale as n_phase/4",
    "c_l_meaning": "effective field-stage capacitor; ripple-loss proxy referenced to 90 uF",
    "c_l_ripple_loss_fraction": 0.02,
    "l_f_reference_h": P4_L_F_H,
    "l_f_range_multiplier": [0.75, 1.25],
    "l_f_loss_model": "P_cond scales as (L_f_ref/L_f)^2 and P_sw as L_f_ref/L_f",
    "deadtime_natural_commutation": "P6 t_zn = 3 degree phase-to-time at each carrier frequency",
    "deadtime_excess_loss": "P6 third-quadrant conduction proxy above t_zn; literal 50..500 ns scan retained",
    "geometry": "place square 157 mm Tx cells at 157 mm pitch in the two largest Omega dimensions; no overlap",
    "field_metric": "P4 qualitative signed kernel; two-tone beat RMS on 1/8 Tx grid; min/max ratio",
    "local_refinement": "selected global compromise +/-5% on continuous variables and +/-1 integer phase",
    "selection": "equal-weight normalized distance to ideal point over P_loss and delta-f efficiency swing",
}


def _log_grid(low: float, high: float, points_per_decade: int) -> list[float]:
    intervals = max(1, math.ceil(math.log10(high / low) * points_per_decade))
    return np.geomspace(low, high, intervals + 1).astype(float).tolist()


def scan_axes() -> dict[str, list[float] | list[int]]:
    """Return the documented global grid, with inclusive endpoints."""
    return {
        "delta_f_hz": _log_grid(500.0, 8_000.0, 10),
        "c_l_uf": np.geomspace(30.0, 200.0, 8).astype(float).tolist(),
        "n_phase": list(range(3, 9)),
        "t_dead_ns": np.linspace(50.0, 500.0, 20).astype(float).tolist(),
        "l_f_uH": np.linspace(
            ASSUMPTIONS["l_f_reference_h"] * 1e6 * ASSUMPTIONS["l_f_range_multiplier"][0],
            ASSUMPTIONS["l_f_reference_h"] * 1e6 * ASSUMPTIONS["l_f_range_multiplier"][1],
            5,
        ).astype(float).tolist(),
    }


def _array_layout(config: Config, n_tx: int) -> tuple[int, int, float, float, float]:
    """Choose an aspect-matched non-overlapping grid in the two largest Omega axes."""
    side_m = 0.157
    dimensions = sorted(config.scenario.omega_m, reverse=True)
    span_x, span_y = dimensions[0], dimensions[1]
    aspect = span_x / span_y
    nx = max(1, math.ceil(math.sqrt(n_tx * aspect)))
    ny = math.ceil(n_tx / nx)
    width_m, height_m = nx * side_m, ny * side_m
    violation_m = max(0.0, width_m - span_x) + max(0.0, height_m - span_y)
    return nx, ny, width_m, height_m, violation_m


@lru_cache(maxsize=3)
def _field_uniformity(scenario_id: str) -> dict[str, Any]:
    config = Config.for_scenario(scenario_id)
    n_tx = ASSUMPTIONS["n_tx_by_scenario"][scenario_id]
    nx, ny, width_m, height_m, violation_m = _array_layout(config, n_tx)
    tx_grid = _tx_grid(nx, ny)[:n_tx]
    phases_1, phases_2 = phase_patterns(nx, ny)
    phases_1, phases_2 = phases_1.ravel()[:n_tx], phases_2.ravel()[:n_tx]
    density = 8
    xs = np.linspace(0.0, float(nx), nx * density + 1)
    ys = np.linspace(0.0, float(ny), ny * density + 1)
    x_grid, y_grid = np.meshgrid(xs, ys)
    field_1 = _synthesize_points(x_grid, y_grid, tx_grid, phases_1, 0.0)
    field_2 = _synthesize_points(x_grid, y_grid, tx_grid, phases_2, 0.0)
    rms = np.sqrt(field_1**2 + field_2**2)
    maximum = float(rms.max())
    ratio = float(rms.min() / maximum) if maximum > 0.0 else 0.0
    min_index = np.unravel_index(int(np.argmin(rms)), rms.shape)
    return {
        "n_tx": n_tx,
        "grid_nx": nx,
        "grid_ny": ny,
        "footprint_m": [width_m, height_m],
        "envelope_m": [max(sorted(config.scenario.omega_m, reverse=True)[:2]),
                        min(sorted(config.scenario.omega_m, reverse=True)[:2])],
        "geometry_violation_m": violation_m,
        "samples_per_tx": density,
        "min_phi_v": float(rms.min()),
        "max_phi_v": maximum,
        "min_location_tx_units": [float(xs[min_index[1]]), float(ys[min_index[0]])],
        "uniformity_ratio": ratio,
        "provenance": ASSUMPTIONS["field_metric"],
    }


@lru_cache(maxsize=512)
def _p5_loss(power_w: float, delta_f_hz: float) -> dict[str, Any]:
    return loss_breakdown(power_w, delta_f_hz)


def evaluate_candidate(
    scenario_id: str,
    delta_f_hz: float,
    c_l_uf: float,
    n_phase: int,
    t_dead_ns: float,
    l_f_uH: float,
) -> dict[str, Any]:
    """Evaluate one point using P5 losses plus declared P7 proxy terms."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}")
    if delta_f_hz <= 0 or c_l_uf <= 0 or n_phase < 1 or t_dead_ns <= 0 or l_f_uH <= 0:
        raise ValueError("all scan parameters must be positive")

    config = Config.for_scenario(scenario_id)
    power_w = config.scenario.p_field_target_w
    field = _field_uniformity(scenario_id)
    geometry_ok = field["geometry_violation_m"] <= 1e-12
    if not geometry_ok:
        return {
            "scenario_id": SCENARIO_LABELS[scenario_id],
            "parameters": {"delta_f_hz": delta_f_hz, "c_l_uf": c_l_uf,
                           "n_phase": n_phase, "t_dead_ns": t_dead_ns, "l_f_uH": l_f_uH,
                           "n_tx": field["n_tx"]},
            "f4_geometry_violation_m": field["geometry_violation_m"],
            "geometry_feasible": False,
        }

    p5 = _p5_loss(power_w, delta_f_hz)
    parts = {key: float(value["nominal"]) for key, value in p5["components_w"].items()}
    n_tx_factor = field["n_tx"] / float(P5_ASSUMPTIONS["number_of_tx_cores"])
    phase_factor = n_phase / 4.0
    l_f_factor = ASSUMPTIONS["l_f_reference_h"] / (l_f_uH * 1e-6)
    c_ratio = 90.0 / c_l_uf

    p_sw = parts["P_sw"] * phase_factor * l_f_factor
    p_cond = parts["P_cond"] * (4.0 / n_phase) ** 2 * l_f_factor**2
    p_mag = parts["P_mag"] * n_tx_factor
    p_ctrl = parts["P_ctrl"] * phase_factor
    p_c_l = power_w * ASSUMPTIONS["c_l_ripple_loss_fraction"] * (c_ratio - 1.0) ** 2

    f1_hz = 200_000.0 - delta_f_hz / 2.0
    f2_hz = 200_000.0 + delta_f_hz / 2.0
    t_zn = (zero_crossing_window_s(f1_hz), zero_crossing_window_s(f2_hz))
    current_a = float(p5["current_per_leg_a"]) * (4.0 / n_phase) * l_f_factor
    leg_count = int(p5["switching_legs"] * phase_factor)
    t_dead_s = t_dead_ns * 1e-9
    p_dead = sum(
        2.0 * THIRD_QUADRANT_DROP_V * current_a * leg_count * frequency_hz
        * max(t_dead_s - natural_s, 0.0)
        for frequency_hz, natural_s in zip((f1_hz, f2_hz), t_zn)
    )

    total_loss = p_sw + p_cond + p_mag + p_ctrl + p_c_l + p_dead
    return {
        "scenario_id": SCENARIO_LABELS[scenario_id],
        "parameters": {"delta_f_hz": delta_f_hz, "c_l_uf": c_l_uf,
                       "n_phase": n_phase, "t_dead_ns": t_dead_ns,
                       "l_f_uH": l_f_uH, "n_tx": field["n_tx"]},
        "f1_total_loss_w": total_loss,
        "loss_terms_w": {"P_sw": p_sw, "P_cond": p_cond, "P_mag": p_mag,
                         "P_ctrl": p_ctrl, "P_C_L_proxy": p_c_l,
                         "P_deadtime_excess": p_dead},
        "f3_uniformity_ratio": field["uniformity_ratio"],
        "f4_geometry_violation_m": field["geometry_violation_m"],
        "geometry_feasible": True,
        "field_diagnostics": field,
    }


def _efficiency_swing(
    scenario_id: str, c_l_uf: float, n_phase: int, t_dead_ns: float,
    l_f_uH: float, delta_grid_hz: Iterable[float],
) -> float:
    config = Config.for_scenario(scenario_id)
    power_w = config.scenario.p_field_target_w
    efficiencies = []
    for delta in delta_grid_hz:
        row = evaluate_candidate(scenario_id, float(delta), c_l_uf, n_phase, t_dead_ns, l_f_uH)
        efficiencies.append(power_w / (power_w + row["f1_total_loss_w"]))
    return max(efficiencies) - min(efficiencies)


def _measure(candidate: dict[str, Any], delta_grid_hz: Iterable[float]) -> dict[str, Any]:
    scenario_id = next(key for key, label in SCENARIO_LABELS.items()
                       if label == candidate["scenario_id"])
    parameters = candidate["parameters"]
    swing = _efficiency_swing(
        scenario_id, parameters["c_l_uf"], parameters["n_phase"],
        parameters["t_dead_ns"], parameters["l_f_uH"], delta_grid_hz,
    )
    result = dict(candidate)
    result["f2_max_delta_f_eff"] = swing
    return result


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fast exact front for this model, whose f3 and feasible f4 are scenario constants."""
    ordered = sorted(rows, key=lambda item: (item["f1_total_loss_w"],
                                              item["f2_max_delta_f_eff"]))
    front: list[dict[str, Any]] = []
    best_swing = math.inf
    for row in ordered:
        swing = row["f2_max_delta_f_eff"]
        if swing < best_swing - 1e-15:
            front.append(row)
            best_swing = swing
    return front


def _select_compromise(front: list[dict[str, Any]]) -> dict[str, Any]:
    if not front:
        raise ValueError("cannot select from an empty Pareto front")
    loss_lo, loss_hi = min(row["f1_total_loss_w"] for row in front), max(row["f1_total_loss_w"] for row in front)
    swing_lo, swing_hi = min(row["f2_max_delta_f_eff"] for row in front), max(row["f2_max_delta_f_eff"] for row in front)
    loss_span, swing_span = max(loss_hi - loss_lo, 1e-15), max(swing_hi - swing_lo, 1e-15)
    return min(front, key=lambda row:
               (row["f1_total_loss_w"] - loss_lo) / loss_span
               + (row["f2_max_delta_f_eff"] - swing_lo) / swing_span)


def _candidate_grid(
    scenario_id: str,
    axes: dict[str, list[float] | list[int]],
    swing_delta_grid_hz: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    field = _field_uniformity(scenario_id)
    if field["geometry_violation_m"] > 1e-12:
        return []
    config = Config.for_scenario(scenario_id)
    delta_grid = list(axes["delta_f_hz"])
    swing_grid = delta_grid if swing_delta_grid_hz is None else list(swing_delta_grid_hz)
    rows = []
    for delta in delta_grid:
        for c_l in axes["c_l_uf"]:
            for n_phase in axes["n_phase"]:
                for t_dead in axes["t_dead_ns"]:
                    for l_f in axes["l_f_uH"]:
                        candidate = evaluate_candidate(scenario_id, float(delta), float(c_l),
                                                       int(n_phase), float(t_dead), float(l_f))
                        candidate["f2_max_delta_f_eff"] = _efficiency_swing(
                            scenario_id, float(c_l), int(n_phase), float(t_dead), float(l_f), swing_grid
                        )
                        rows.append(candidate)
    return rows


def _local_axes(anchor: dict[str, Any], base_axes: dict[str, list[float] | list[int]]) -> dict[str, list[float] | list[int]]:
    p = anchor["parameters"]
    ranges = {"delta_f_hz": (500.0, 8_000.0), "c_l_uf": (30.0, 200.0),
              "t_dead_ns": (50.0, 500.0),
              "l_f_uH": (ASSUMPTIONS["l_f_reference_h"] * 1e6 * 0.75,
                         ASSUMPTIONS["l_f_reference_h"] * 1e6 * 1.25)}
    key_by_param = {"delta_f_hz": "delta_f_hz", "c_l_uf": "c_l_uf",
                    "t_dead_ns": "t_dead_ns", "l_f_uH": "l_f_uH"}
    result: dict[str, list[float] | list[int]] = {}
    for parameter, axis_key in key_by_param.items():
        value = float(p[parameter])
        lo, hi = ranges[parameter]
        points = [max(lo, value * 0.95), value, min(hi, value * 1.05)]
        result[axis_key] = sorted(set(points))
    result["n_phase"] = sorted({value for value in (p["n_phase"] - 1, p["n_phase"], p["n_phase"] + 1)
                                if 3 <= value <= 8})
    return result


def _sensitivity(scenario_id: str, anchor: dict[str, Any], delta_grid: list[float]) -> dict[str, Any]:
    p = anchor["parameters"]
    baseline_loss = anchor["f1_total_loss_w"]
    baseline_swing = anchor["f2_max_delta_f_eff"]
    rows: dict[str, Any] = {}
    for name in ("delta_f_hz", "c_l_uf", "n_phase", "t_dead_ns", "l_f_uH"):
        if name == "n_phase":
            variants = sorted({n for n in (p[name] - 1, p[name] + 1) if 3 <= n <= 8})
            note = "integer neighbor (5% is below one phase for this discrete parameter)"
        else:
            bounds = {"delta_f_hz": (500.0, 8_000.0), "c_l_uf": (30.0, 200.0),
                      "t_dead_ns": (50.0, 500.0),
                      "l_f_uH": (ASSUMPTIONS["l_f_reference_h"] * 1e6 * 0.75,
                                 ASSUMPTIONS["l_f_reference_h"] * 1e6 * 1.25)}[name]
            variants = [max(bounds[0], p[name] * 0.95), min(bounds[1], p[name] * 1.05)]
            note = "both directions, clamped to scan limits"
        evaluations = []
        for value in variants:
            parameters = dict(p)
            parameters[name] = int(value) if name == "n_phase" else float(value)
            row = evaluate_candidate(scenario_id, parameters["delta_f_hz"], parameters["c_l_uf"],
                                     parameters["n_phase"], parameters["t_dead_ns"], parameters["l_f_uH"])
            swing = _efficiency_swing(scenario_id, parameters["c_l_uf"], parameters["n_phase"],
                                      parameters["t_dead_ns"], parameters["l_f_uH"], delta_grid)
            evaluations.append({
                "value": parameters[name],
                "delta_loss_w": row["f1_total_loss_w"] - baseline_loss,
                "delta_f2_eff_swing": swing - baseline_swing,
                "delta_f3_uniformity_ratio": row["f3_uniformity_ratio"] - anchor["f3_uniformity_ratio"],
            })
        rows[name] = {"perturbation": note, "evaluations": evaluations}
    return rows


def optimize_scenario(scenario_id: str) -> dict[str, Any]:
    """Run one scenario's global grid, local refinement, Pareto selection and checks."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}")
    axes = scan_axes()
    global_rows = _candidate_grid(scenario_id, axes)
    if not global_rows:
        raise RuntimeError(f"{SCENARIO_LABELS[scenario_id]} has no geometry-feasible candidates")
    global_front = _pareto_front(global_rows)
    global_anchor = _select_compromise(global_front)

    local_axes = _local_axes(global_anchor, axes)
    local_rows = _candidate_grid(scenario_id, local_axes, axes["delta_f_hz"])
    combined_front = _pareto_front(global_rows + local_rows)
    selected = _select_compromise(combined_front)

    global_p = global_anchor["parameters"]
    final_p = selected["parameters"]
    movement = {
        key: abs(final_p[key] - global_p[key]) / max(abs(global_p[key]), 1e-12) * 100.0
        for key in ("delta_f_hz", "c_l_uf", "t_dead_ns", "l_f_uH")
    }
    max_movement = max(movement.values(), default=0.0)
    field = selected["field_diagnostics"]
    return {
        "scenario_id": SCENARIO_LABELS[scenario_id],
        "scan_axes": axes,
        "global_grid_candidate_count": len(global_rows),
        "local_refinement_axes": local_axes,
        "local_refinement_candidate_count": len(local_rows),
        "pareto_front": combined_front,
        "pareto_front_size": len(combined_front),
        "selection_strategy": ASSUMPTIONS["selection"],
        "selected_compromise": selected,
        "global_to_fine_movement_pct": movement,
        "max_fine_step_movement_pct": max_movement,
        "step_adequacy": "当前步长不足" if max_movement > 3.0 else "当前步长满足本次局部细化判据",
        "neighbor_sensitivity_5pct": _sensitivity(scenario_id, selected, axes["delta_f_hz"]),
        "geometry": field,
        "deadtime_diagnostic": {
            "literal_document_range_ns": [50.0, 500.0],
            "natural_commutation_ns_at_center": zero_crossing_window_s(200_000.0) * 1e9,
            "selected_deadtime_ns": final_p["t_dead_ns"],
            "selected_setting_exceeds_t_zn": final_p["t_dead_ns"] > zero_crossing_window_s(200_000.0) * 1e9,
            "handling": "literal P7 candidates retained; P6 excess-conduction proxy penalizes t_dead above t_zn",
        },
        "assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: identify the physical C_L network and replace its ripple-loss proxy with ESR/ripple-current data.",
            "TODO[实测]: replace n_phase and L_f current/loss scaling with the final inverter and compensation circuit model.",
            "TODO[FEM/实测]: replace the P4 qualitative mutual kernel and verify the field uniformity map.",
            "TODO[机械确认]: confirm the 157 mm Tx envelope, array plane, and available Omega mounting faces.",
        ],
    }


def optimize_all_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [optimize_scenario(scenario_id) for scenario_id in SCENARIOS],
        "global_assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: all P7 system coupling proxies remain unvalidated; current Pareto fronts are model results, not hardware optima."
        ],
    }


def describe() -> dict[str, Any]:
    axes = scan_axes()
    return {
        "inputs": ["scenario_id: A_subzone|B_half_rack|C_full_rack"],
        "outputs": ["global + local candidate search", "four objectives", "Pareto front",
                    "selected compromise", "5% neighborhood sensitivity", "fine-step movement check"],
        "scan_axes": axes,
        "global_candidates_per_scenario": math.prod(len(values) for values in axes.values()),
        "objectives": {"f1": "total loss including declared P7 proxy terms, minimize",
                       "f2": "max efficiency minus min efficiency over the scanned delta_f grid, minimize",
                       "f3": "min|Phi|/max|Phi| on the P4 two-tone RMS map, maximize",
                       "f4": "Tx layout envelope violation in metres, minimize and enforce as hard constraint"},
        "assumptions": ASSUMPTIONS,
        "TODOs": ["TODO[实测/FEM]: validate every [C] proxy before treating a selected point as a hardware optimum."],
    }
