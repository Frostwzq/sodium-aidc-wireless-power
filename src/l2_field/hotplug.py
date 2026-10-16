"""P11 load hot-plug and transmitter redundancy diagnostics.

Dynamic response, bus model, and failure placement are explicitly [C]
assumptions. P4's measured-model load sweep supplies the steady-state current
variation bound; it is not itself a measured hot-plug transient.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.config import Config
from src.l2_field.array import phase_patterns
from src.l2_field.solver import (
    DENSE_GRID_PER_TX,
    L_F_H,
    L_TX_H,
    V_O_PEAK_V,
    _synthesize_points,
    _tx_grid,
    test_load_independence,
    test_no_blind_spot,
)


SCENARIOS = {
    "A_subzone": "P0-A",
    "B_half_rack": "P0-B",
    "C_full_rack": "P0-C",
}

ASSUMPTIONS: dict[str, Any] = {
    "provenance": "P4 circuit sweep plus [C] transient, bus, and geometry assumptions; no hot-plug measurements supplied",
    "current_transient": {
        "relative_dynamic_overshoot_multiplier_of_p4_worst_steady_change": 5.0,
        "settling_time_constant_s": 0.0005,
        "oscillation_frequency_hz": 2_000.0,
        "sampling_interval_s": 0.00005,
        "record_before_s": 0.001,
        "record_after_s": 0.004,
    },
    "current_acceptance_relative_change": 0.01,
    "dc_bus": {
        "nominal_voltage_v": 400.0,
        "source_resistance_ohm": 0.02,
        "bus_capacitance_f": 0.01,
        "voltage_tolerance_fraction": 0.05,
        "load_removal_direction": "source impedance predicts a positive voltage rebound; report absolute excursion and droop separately",
        "maximum_simultaneous_removal_rule": "remove n_load_max - 1 loads so at least one remaining load is monitored",
    },
    "transmitter_failure": {
        "nx": 4,
        "ny": 4,
        "failed_fraction": 0.25,
        "placement": "scan all contiguous 2x2 transmitter blocks and report worst-case block",
        "samples_per_tx": DENSE_GRID_PER_TX,
        "orientations_deg": [0.0, 45.0, 90.0],
        "mutual_model": "reuse P4 qualitative signed kernel",
    },
}


def _scenario_current_report(scenario_id: str) -> dict[str, Any]:
    report = test_load_independence()
    config = Config.for_scenario(scenario_id)
    nominal_current_a = float(np.mean(report["results"][-1]["i_tx_a"]))
    p4_worst_relative_change = max(item["i_tx_relative_change"] for item in report["results"])
    dynamic = ASSUMPTIONS["current_transient"]
    dynamic_peak_fraction = (
        p4_worst_relative_change
        * dynamic["relative_dynamic_overshoot_multiplier_of_p4_worst_steady_change"]
    )
    return {
        "scenario_id": SCENARIOS[scenario_id],
        "n_load_max": config.scenario.n_load_max,
        "nominal_tx_current_a": nominal_current_a,
        "p4_load_sweep_worst_relative_change": p4_worst_relative_change,
        "dynamic_relative_change_bound": dynamic_peak_fraction,
        "acceptance_relative_change": ASSUMPTIONS["current_acceptance_relative_change"],
        "passes": dynamic_peak_fraction < ASSUMPTIONS["current_acceptance_relative_change"],
        "provenance": "steady-state bound from P4 non-ideal nodal sweep; transient multiplier/time constant are [C]",
    }


def simulate_current_event(
    scenario_id: str,
    *,
    event: str = "insert",
    simultaneous_loads: int = 1,
) -> dict[str, Any]:
    """Return a sample waveform envelope for insertion or simultaneous removal."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}; expected one of {tuple(SCENARIOS)}")
    if event not in {"insert", "remove"}:
        raise ValueError("event must be 'insert' or 'remove'")
    config = Config.for_scenario(scenario_id)
    if not 1 <= simultaneous_loads <= config.scenario.n_load_max:
        raise ValueError("simultaneous_loads must be in [1, scenario n_load_max]")
    steady = _scenario_current_report(scenario_id)
    dynamic = ASSUMPTIONS["current_transient"]
    count = int(round((dynamic["record_before_s"] + dynamic["record_after_s"]) / dynamic["sampling_interval_s"])) + 1
    time_s = np.linspace(-dynamic["record_before_s"], dynamic["record_after_s"], count)
    # Superposition is assumed for N events, but the amplitude is capped at the
    # same P4 worst-case load sweep because Eq.(2) claims count independence.
    scale = min(1.0, simultaneous_loads / config.scenario.n_load_max)
    peak_relative = steady["dynamic_relative_change_bound"] * scale
    direction = 1.0 if event == "insert" else -1.0
    after = time_s >= 0.0
    elapsed = np.maximum(time_s, 0.0)
    tau = dynamic["settling_time_constant_s"]
    ring = np.exp(-elapsed / tau) * np.cos(2.0 * math.pi * dynamic["oscillation_frequency_hz"] * elapsed)
    final_fraction = 0.25 * peak_relative
    fractional_delta = np.zeros_like(time_s)
    fractional_delta[after] = direction * (
        final_fraction + (peak_relative - final_fraction) * ring[after]
    )
    current = steady["nominal_tx_current_a"] * (1.0 + fractional_delta)
    relative_change = float(np.max(np.abs(current - steady["nominal_tx_current_a"])) / steady["nominal_tx_current_a"])
    return {
        "scenario_id": SCENARIOS[scenario_id],
        "event": event,
        "simultaneous_loads": simultaneous_loads,
        "time_s": time_s.tolist(),
        "tx_current_a": current.tolist(),
        "nominal_tx_current_a": steady["nominal_tx_current_a"],
        "relative_change_max": relative_change,
        "relative_change_limit": ASSUMPTIONS["current_acceptance_relative_change"],
        "passes": relative_change < ASSUMPTIONS["current_acceptance_relative_change"],
        "assumptions": dynamic,
        "provenance": "[C] damped transient waveform scaled to P4's worst non-ideal steady load variation; not measured waveform",
    }


def test_hotplug_current_independence() -> dict[str, Any]:
    """Check insert and maximum-count simultaneous-removal envelopes in all P0 tiers."""
    rows = []
    for scenario_id, label in SCENARIOS.items():
        max_loads = Config.for_scenario(scenario_id).scenario.n_load_max
        insert = simulate_current_event(scenario_id, event="insert", simultaneous_loads=1)
        remove_max = simulate_current_event(
            scenario_id, event="remove", simultaneous_loads=max(1, max_loads - 1)
        )
        rows.append({
            "scenario_id": label,
            "n_load_max": max_loads,
            "insert": {key: insert[key] for key in ("relative_change_max", "relative_change_limit", "passes")},
            "simultaneous_remove_max_while_one_remains": {
                key: remove_max[key]
                for key in ("simultaneous_loads", "relative_change_max", "relative_change_limit", "passes")
            },
        })
    return {
        "test": "test_hotplug_current_independence",
        "pass": all(
            item["insert"]["passes"]
            and item["simultaneous_remove_max_while_one_remains"]["passes"]
            for item in rows
        ),
        "scenarios": rows,
        "scope": "P4 steady-state sweep plus [C] transient envelope; no measured switching waveform",
    }


def simultaneous_removal_voltage_report(scenario_id: str) -> dict[str, Any]:
    """Estimate shared 400 V DC bus excursion when the maximum load count unplugs."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}")
    config = Config.for_scenario(scenario_id)
    n_loads = max(1, config.scenario.n_load_max - 1)
    remaining_loads = config.scenario.n_load_max - n_loads
    # Scale the scenario's aggregate target by the fraction actually removed.
    power_w = config.scenario.p_field_target_w * n_loads / config.scenario.n_load_max
    bus = ASSUMPTIONS["dc_bus"]
    delta_i_a = power_w / bus["nominal_voltage_v"]
    absolute_excursion_v = delta_i_a * bus["source_resistance_ohm"]
    excursion_fraction = absolute_excursion_v / bus["nominal_voltage_v"]
    # Removing loads lowers bus current; this first-order source model predicts
    # rebound, not a droop. Conservatively compare either-direction excursion.
    voltage_droop_fraction = 0.0
    return {
        "scenario_id": SCENARIOS[scenario_id],
        "simultaneously_removed_loads": n_loads,
        "remaining_loads": remaining_loads,
        "bus_nominal_v": bus["nominal_voltage_v"],
        "estimated_current_step_a": delta_i_a,
        "predicted_voltage_rebound_v": absolute_excursion_v,
        "predicted_voltage_rebound_percent": 100.0 * excursion_fraction,
        "predicted_voltage_droop_percent": 100.0 * voltage_droop_fraction,
        "worst_absolute_excursion_percent": 100.0 * excursion_fraction,
        "voltage_tolerance_percent": 100.0 * bus["voltage_tolerance_fraction"],
        "passes": excursion_fraction <= bus["voltage_tolerance_fraction"],
        "provenance": "[C] resistive source-step envelope; removal causes rebound in this model; bus regulation dynamics are not measured",
    }


def tx_failure_field_report(
    failed_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Scan contiguous 2x2 Tx failures and report worst available field area."""
    nx = ny = int(ASSUMPTIONS["transmitter_failure"]["nx"])
    samples = int(ASSUMPTIONS["transmitter_failure"]["samples_per_tx"])
    orientations = tuple(ASSUMPTIONS["transmitter_failure"]["orientations_deg"])
    threshold_v = float(test_no_blind_spot(4, 4, dense_samples_per_tx=samples)["epsilon_th_v"])
    positions = _tx_grid(nx, ny)
    phase1, phase2 = phase_patterns(nx, ny, True)
    all_positions = np.arange(nx * ny)
    xs = np.linspace(0.0, float(nx), nx * samples + 1)
    ys = np.linspace(0.0, float(ny), ny * samples + 1)
    xx, yy = np.meshgrid(xs, ys)
    failures = []
    blocks = []
    if failed_positions is not None:
        failed = np.asarray(sorted(set(int(item) for item in failed_positions)), dtype=int)
        if failed.size != nx * ny // 4 or np.any((failed < 0) | (failed >= nx * ny)):
            raise ValueError("failed_positions must identify exactly one quarter of the 4x4 Tx array")
        blocks.append(failed)
    else:
        for row in range(ny - 1):
            for column in range(nx - 1):
                blocks.append(np.asarray([row * nx + column, row * nx + column + 1,
                                          (row + 1) * nx + column, (row + 1) * nx + column + 1]))

    for failed in blocks:
        active = np.setdiff1d(all_positions, failed)
        angle_maps = []
        for angle in orientations:
            tone_maps = []
            for phases in (phase1, phase2):
                tone_maps.append(_synthesize_points(
                    xx, yy, positions[active], phases.reshape(-1)[active], float(angle)
                ))
            angle_maps.append(np.sqrt(tone_maps[0] ** 2 + tone_maps[1] ** 2))
        worst_map = np.minimum.reduce(angle_maps)
        minimum = float(worst_map.min())
        count_usable = int(np.count_nonzero(worst_map >= threshold_v))
        failures.append({
            "failed_tx_indices_row_major": failed.tolist(),
            "min_phi_v": minimum,
            "usable_sample_count": count_usable,
            "sample_count": int(worst_map.size),
            "usable_area_fraction": count_usable / int(worst_map.size),
            "min_phi_above_epsilon_th": minimum >= threshold_v,
        })
    worst = min(failures, key=lambda item: item["usable_area_fraction"])
    return {
        "scenario_id": "P0-B [C] 4x4 reference field",
        "failed_transmitter_count": nx * ny // 4,
        "failure_patterns_checked": len(failures),
        "worst_case": worst,
        "all_failure_patterns": failures,
        "epsilon_th_v": threshold_v,
        "grid": {"samples_per_tx": samples, "shape": list(worst_map.shape), "orientations_deg": list(orientations)},
        "status": "full_coverage_pass" if worst["min_phi_above_epsilon_th"] else "degraded_coverage_reported",
        "provenance": "P4 qualitative signed coupling kernel; 1/64 Tx grid; [C] contiguous 2x2 failure scan",
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario_id", "event", "simultaneous_loads", "optional failed Tx indices"],
        "outputs": ["sampled Tx current waveform", "simultaneous removal bus excursion", "quarter-array failure coverage"],
        "acceptance": {"current_relative_change": "<1% per P11", "bus_excursion": "within [C] ±5%"},
        "assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: record hot-plug Tx winding/inverter waveforms and identify transient time constants.",
            "TODO[实测]: replace 400 V bus RC/source model and voltage tolerance with rack DC bus data.",
            "TODO[FEM]: replace qualitative P4 coupling kernel before using failed-coil area figures for design.",
        ],
    }
