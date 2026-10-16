"""Feedback and forecast peak-shaving optimization under cycle-life limits."""

from __future__ import annotations

import itertools
import math
from typing import Any

from src.config import Config
from src.l1_buffer.load_profile import generate_load_profile
from src.l1_buffer.soc import cycle_life_consumption, internal_resistance_ohm


SCENARIOS = ("A_subzone", "B_half_rack", "C_full_rack")
SCENARIO_LABELS = {"A_subzone": "P0-A", "B_half_rack": "P0-B", "C_full_rack": "P0-C"}

ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] P8 scenario assumptions; not measurements and not written back to P0",
    "device_rated_power_w_per_load": 1_000.0,
    "battery_energy_hours_at_load_peak": 0.25,
    "initial_soc": 0.80,
    "temperature_c": 45.0,
    "charge_efficiency": 0.94,
    "discharge_efficiency": 0.94,
    "nominal_c_rate": 4.0,
    "conservative_capacity_factor": 0.80,
    "conservative_c_rate": 3.0,
    "conservative_one_way_efficiency": 0.90,
    "optimistic_capacity_factor": 1.00,
    "optimistic_c_rate": 4.0,
    "optimistic_one_way_efficiency": 0.97,
    "lifetime_budget_fraction_per_profile": 1.0 / 3650.0,
    "lifetime_budget_basis": "one representative profile per day over a 10-year service target",
    "battery_power_ramp_time_s": 120.0,
    "forecast_horizons_s": [0.0, 60.0, 180.0, 300.0, 600.0],
    "threshold_fractions_of_peak": [0.65, 0.70, 0.75, 0.80, 0.85],
    "dod_limits": [0.10, 0.20, 0.40, 0.80],
    "soc_windows": [[0.20, 0.80], [0.15, 0.85], [0.10, 0.90]],
    "fragility_or_target_tuning": "no constraints are relaxed to force 25% shaving",
}


def _scenario_energy_kwh(scenario_id: str) -> tuple[float, float]:
    config = Config.for_scenario(scenario_id)
    peak_load_w = config.scenario.n_load_max * ASSUMPTIONS["device_rated_power_w_per_load"] * config.load.p_train_peak_factor
    nominal_energy_kwh = (peak_load_w / 1_000.0) * ASSUMPTIONS["battery_energy_hours_at_load_peak"]
    return peak_load_w, nominal_energy_kwh


def _simulate(
    scenario_id: str,
    load_trace: dict[str, Any],
    strategy: str,
    threshold_w: float,
    dod_limit: float,
    soc_window: tuple[float, float],
    forecast_horizon_s: float,
    uncertainty_case: str = "nominal",
) -> dict[str, Any]:
    if strategy not in {"none", "feedback", "forecast"}:
        raise ValueError("strategy must be none, feedback, or forecast")
    config = Config.for_scenario(scenario_id)
    peak_load_w, nominal_energy_kwh = _scenario_energy_kwh(scenario_id)
    dt = float(load_trace["sample_interval_s"])
    base_rate = ASSUMPTIONS["nominal_c_rate"]
    capacity_factor = 1.0
    efficiency = ASSUMPTIONS["charge_efficiency"]
    if uncertainty_case == "conservative":
        base_rate = ASSUMPTIONS["conservative_c_rate"]
        capacity_factor = ASSUMPTIONS["conservative_capacity_factor"]
        efficiency = ASSUMPTIONS["conservative_one_way_efficiency"]
    elif uncertainty_case == "optimistic":
        base_rate = ASSUMPTIONS["optimistic_c_rate"]
        capacity_factor = ASSUMPTIONS["optimistic_capacity_factor"]
        efficiency = ASSUMPTIONS["optimistic_one_way_efficiency"]
    elif uncertainty_case != "nominal":
        raise ValueError("uncertainty_case must be nominal, conservative, or optimistic")

    effective_energy_kwh = nominal_energy_kwh * capacity_factor
    retention_curve = config.battery.cap_retention_t
    if ASSUMPTIONS["temperature_c"] in retention_curve:
        temperature_retention = retention_curve[ASSUMPTIONS["temperature_c"]]
    else:
        temperature_points = sorted(retention_curve.items())
        lower = max((point for point in temperature_points if point[0] < ASSUMPTIONS["temperature_c"]),
                    default=temperature_points[0])
        upper = min((point for point in temperature_points if point[0] > ASSUMPTIONS["temperature_c"]),
                    default=temperature_points[-1])
        fraction = ((ASSUMPTIONS["temperature_c"] - lower[0]) / (upper[0] - lower[0])
                    if upper[0] != lower[0] else 0.0)
        temperature_retention = lower[1] + fraction * (upper[1] - lower[1])
    effective_energy_kwh *= temperature_retention
    maximum_battery_power_kw = base_rate * effective_energy_kwh
    lower_soc = max(soc_window[0], 1.0 - dod_limit)
    upper_soc = min(soc_window[1], 1.0)
    if lower_soc >= upper_soc:
        return {"feasible": False, "reason": "DoD and SoC window do not overlap"}
    soc = min(max(ASSUMPTIONS["initial_soc"], lower_soc), upper_soc)
    initial_soc = soc
    soc_trace = [soc]
    grid_trace: list[float] = []
    battery_trace: list[float] = []
    request_actual_kw = 0.0
    ramp_per_step_kw = maximum_battery_power_kw * dt / ASSUMPTIONS["battery_power_ramp_time_s"]
    horizon_samples = int(math.ceil(forecast_horizon_s / dt))
    load = [float(value) / 1_000.0 for value in load_trace["p_load_w"]]
    threshold_kw = threshold_w / 1_000.0

    for index, current_load_kw in enumerate(load):
        if strategy == "none":
            request_kw = 0.0
        else:
            future_end = min(len(load), index + horizon_samples + 1)
            future_peak_kw = max(load[index:future_end], default=current_load_kw)
            anticipated = strategy == "forecast" and forecast_horizon_s > 0 and future_peak_kw > threshold_kw
            if current_load_kw > threshold_kw:
                request_kw = current_load_kw - threshold_kw
            elif anticipated:
                # Predictive pre-discharge builds converter output before the
                # measured workload crosses the threshold, subject to no-export.
                request_kw = min(future_peak_kw - threshold_kw, current_load_kw)
            else:
                # Feedback policy recharges only into headroom below its cap.
                # threshold_kw is a float (kW), not a callable: calling it raised
                # TypeError and broke every forecast-strategy simulation.
                request_kw = -min(threshold_kw - current_load_kw, maximum_battery_power_kw)

        request_kw = min(max(request_kw, -maximum_battery_power_kw), maximum_battery_power_kw)
        request_actual_kw += min(max(request_kw - request_actual_kw, -ramp_per_step_kw), ramp_per_step_kw)
        request_actual_kw = min(max(request_actual_kw, -maximum_battery_power_kw), maximum_battery_power_kw)
        if request_actual_kw > 0:
            request_actual_kw = min(request_actual_kw, current_load_kw)
            energy_limited_kw = max(0.0, (soc - lower_soc) * effective_energy_kwh * efficiency * 3_600.0 / dt)
            battery_kw = min(request_actual_kw, energy_limited_kw)
            soc -= battery_kw * dt / (3_600.0 * effective_energy_kwh * efficiency)
        elif request_actual_kw < 0:
            charge_kw = min(-request_actual_kw, max(0.0, (upper_soc - soc) * effective_energy_kwh / (efficiency * dt / 3_600.0)))
            battery_kw = -charge_kw
            soc += charge_kw * dt * efficiency / (3_600.0 * effective_energy_kwh)
        else:
            battery_kw = 0.0
        soc = min(max(soc, lower_soc), upper_soc)
        grid_kw = current_load_kw - battery_kw
        battery_trace.append(battery_kw)
        grid_trace.append(grid_kw)
        soc_trace.append(soc)

    max_load_kw = max(load, default=0.0)
    max_grid_kw = max(grid_trace, default=0.0)
    eta_shave = max(0.0, (max_load_kw - max_grid_kw) / max_load_kw) if max_load_kw else 0.0
    p_na_kw = max((abs(value) for value in battery_trace), default=0.0)
    life = cycle_life_consumption(scenario_id, soc_trace)
    life_per_eta = life["life_consumed_fraction"] / eta_shave if eta_shave > 0 else math.inf
    resistance = internal_resistance_ohm(
        scenario_id, sum(soc_trace) / len(soc_trace), ASSUMPTIONS["temperature_c"]
    )
    return {
        "feasible": True,
        "scenario_id": SCENARIO_LABELS[scenario_id],
        "strategy": strategy,
        "uncertainty_case": uncertainty_case,
        "parameters": {"threshold_w": threshold_w, "dod_limit": dod_limit,
                       "soc_window": list(soc_window), "forecast_horizon_s": forecast_horizon_s},
        "eta_shave": eta_shave,
        "P_load_peak_kw": max_load_kw,
        "P_grid_peak_kw": max_grid_kw,
        "P_Na_peak_kw": p_na_kw,
        "P_Na_rate_limit_kw": maximum_battery_power_kw,
        "rate_constraint_pass": p_na_kw <= maximum_battery_power_kw + 1e-9,
        "C_nom_kwh_assumed": nominal_energy_kwh,
        "effective_energy_kwh": effective_energy_kwh,
        "temperature_capacity_retention": temperature_retention,
        "soc_start": initial_soc,
        "soc_end": soc,
        "soc_min": min(soc_trace),
        "soc_max": max(soc_trace),
        "life_consumed_fraction": life["life_consumed_fraction"],
        "equivalent_full_cycles": life["equivalent_full_cycles"],
        "life_per_unit_eta_shave": life_per_eta,
        "lifetime_budget_fraction": ASSUMPTIONS["lifetime_budget_fraction_per_profile"],
        "lifetime_budget_pass": life["life_consumed_fraction"] <= ASSUMPTIONS["lifetime_budget_fraction_per_profile"],
        "life_detail": life,
        "r_int_diagnostic": resistance,
        "grid_power_kw": grid_trace,
        "battery_power_kw": battery_trace,
        "soc_trace": soc_trace,
        "provenance": ASSUMPTIONS["provenance"],
    }


def _candidate_results(
    scenario_id: str,
    load_trace: dict[str, Any],
    strategy: str,
    uncertainty_case: str,
) -> list[dict[str, Any]]:
    peak = float(load_trace["P_peak_W"])
    horizons = ([value for value in ASSUMPTIONS["forecast_horizons_s"] if value > 0]
                if strategy == "forecast" else [0.0])
    candidates = []
    for threshold_fraction, dod, window, horizon in itertools.product(
        ASSUMPTIONS["threshold_fractions_of_peak"], ASSUMPTIONS["dod_limits"],
        ASSUMPTIONS["soc_windows"], horizons,
    ):
        row = _simulate(scenario_id, load_trace, strategy, peak * threshold_fraction,
                        dod, tuple(window), horizon, uncertainty_case)
        if row["feasible"] and row["lifetime_budget_pass"] and row["rate_constraint_pass"]:
            candidates.append(row)
    return candidates


def _tradeoff_front(candidates: list[dict[str, Any]]) -> list[dict[str, float]]:
    ordered = sorted(candidates, key=lambda row: (-row["eta_shave"], row["life_consumed_fraction"]))
    front: list[dict[str, float]] = []
    best_life = math.inf
    for row in ordered:
        life = row["life_consumed_fraction"]
        if life < best_life - 1e-15:
            front.append({"eta_shave": row["eta_shave"],
                          "life_consumed_fraction": life,
                          "life_per_unit_eta_shave": row["life_per_unit_eta_shave"],
                          "P_grid_peak_kw": row["P_grid_peak_kw"],
                          "threshold_w": row["parameters"]["threshold_w"],
                          "dod_limit": row["parameters"]["dod_limit"],
                          "forecast_horizon_s": row["parameters"]["forecast_horizon_s"]})
            best_life = life
    return front


def optimize_scenario(scenario_id: str) -> dict[str, Any]:
    """Compare no-storage, feedback, and predictive control under P8 assumptions."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}")
    load_trace = generate_load_profile(scenario_id)
    peak_w, nominal_energy_kwh = _scenario_energy_kwh(scenario_id)
    no_storage = _simulate(scenario_id, load_trace, "none", peak_w, 0.8, (0.2, 0.8), 0.0)
    nominal_feedback = _candidate_results(scenario_id, load_trace, "feedback", "nominal")
    nominal_forecast = _candidate_results(scenario_id, load_trace, "forecast", "nominal")
    if not nominal_feedback or not nominal_forecast:
        raise RuntimeError(f"{SCENARIO_LABELS[scenario_id]}: no policy satisfies the assumed lifetime and rate constraints")
    feedback = max(nominal_feedback, key=lambda row: row["eta_shave"])
    forecast = max(nominal_forecast, key=lambda row: row["eta_shave"])
    conservative_feedback_rows = _candidate_results(scenario_id, load_trace, "feedback", "conservative")
    optimistic_feedback_rows = _candidate_results(scenario_id, load_trace, "feedback", "optimistic")
    conservative = _candidate_results(scenario_id, load_trace, "forecast", "conservative")
    optimistic = _candidate_results(scenario_id, load_trace, "forecast", "optimistic")
    if not conservative_feedback_rows or not optimistic_feedback_rows or not conservative or not optimistic:
        raise RuntimeError(f"{SCENARIO_LABELS[scenario_id]}: uncertainty envelope has no feasible predictive policy")
    conservative_feedback = max(conservative_feedback_rows, key=lambda row: row["eta_shave"])
    optimistic_feedback = max(optimistic_feedback_rows, key=lambda row: row["eta_shave"])
    conservative_best = max(conservative, key=lambda row: row["eta_shave"])
    optimistic_best = max(optimistic, key=lambda row: row["eta_shave"])
    interval = [conservative_best["eta_shave"], optimistic_best["eta_shave"]]
    life_front = _tradeoff_front(nominal_feedback + nominal_forecast)
    eta_lift_absolute = forecast["eta_shave"] - feedback["eta_shave"]
    eta_lift_relative = (eta_lift_absolute / feedback["eta_shave"]
                         if feedback["eta_shave"] > 0 else None)
    target_warning = interval[0] < 0.25
    low_shave_warning = (
        conservative_feedback["eta_shave"] < 0.10
        or conservative_best["eta_shave"] < 0.10
        or feedback["eta_shave"] < 0.10
        or forecast["eta_shave"] < 0.10
    )
    return {
        "scenario_id": SCENARIO_LABELS[scenario_id],
        "load_profile": load_trace,
        "storage_assumptions": {
            "P_device_rated_W_per_load": ASSUMPTIONS["device_rated_power_w_per_load"],
            "P_shave_reserve_kWh_assumed": nominal_energy_kwh,
            "P_shave_reserve_kWh_is_P0_value": False,
            "P_rate_C_nom_kW": ASSUMPTIONS["nominal_c_rate"] * nominal_energy_kwh,
            "temperature_c": ASSUMPTIONS["temperature_c"],
            "lifetime_budget_fraction_per_profile": ASSUMPTIONS["lifetime_budget_fraction_per_profile"],
        },
        "strategy_comparison": {
            "no_storage": no_storage,
            "feedback": feedback,
            "forecast": forecast,
        },
        "forecast_eta_shave_lift_vs_feedback": {
            "absolute_fraction": eta_lift_absolute,
            "relative_fraction": eta_lift_relative,
            "relative_percent": None if eta_lift_relative is None else 100.0 * eta_lift_relative,
        },
        "eta_shave_intervals": {
            "no_storage": [0.0, 0.0],
            "feedback_conservative_to_optimistic": [
                conservative_feedback["eta_shave"], optimistic_feedback["eta_shave"]
            ],
            "forecast_conservative_to_optimistic": interval,
        },
        "uncertainty_cases": {
            "feedback": {"conservative": conservative_feedback, "optimistic": optimistic_feedback},
            "forecast": {"conservative": conservative_best, "optimistic": optimistic_best},
        },
        "eta_shave_target_25pct": {
            "target": 0.25,
            "met_nominally": forecast["eta_shave"] >= 0.25,
            "met_conservative": interval[0] >= 0.25,
            "met_optimistic": interval[1] >= 0.25,
            "depends_on_assumptions": ["duty_cycle = 0.45", "battery power rate = 4C"],
            "warning": "WARNING: conservative eta_shave is below the 25% target; the target is assumption-dependent." if target_warning else None,
        },
        "below_10pct_warning": (
            f"WARNING: {SCENARIO_LABELS[scenario_id]} has at least one strategy/uncertainty case with eta_shave below 10%."
            if low_shave_warning else None
        ),
        "life_consumption_vs_eta_curve": life_front,
        "assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: replace the representative workload with a measured P0.3 rack trace and rated device power.",
            "TODO[实测]: populate actual battery reserve energy, pack topology, pulse resistance, and SoC/temperature resistance curve.",
            "TODO[系统约束]: replace the 10-year per-profile life budget with the service life and daily workload schedule.",
            "TODO[实测]: validate the assumed 120 s battery converter power ramp before crediting predictive pre-discharge.",
        ],
    }


def optimize_all_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [optimize_scenario(scenario_id) for scenario_id in SCENARIOS],
        "assumptions": ASSUMPTIONS,
        "P0_unresolved_values_preserved": ["P_device_rated_W", "P_shave_reserve_kWh"],
    }


def compare_strategies(scenario_id: str) -> dict[str, Any]:
    """Public P8 entry point for the strategy comparison in one scenario."""
    return optimize_scenario(scenario_id)


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario_id"],
        "outputs": ["no-storage/feedback/forecast comparison", "eta_shave uncertainty interval",
                    "SoC and life consumption", "life-versus-shaving tradeoff curve"],
        "assumptions": ASSUMPTIONS,
        "strategy_space": {
            "threshold_fraction": ASSUMPTIONS["threshold_fractions_of_peak"],
            "dod_limit": ASSUMPTIONS["dod_limits"],
            "soc_window": ASSUMPTIONS["soc_windows"],
            "forecast_horizon_s": ASSUMPTIONS["forecast_horizons_s"],
        },
        "P0_unresolved_values_preserved": ["P_device_rated_W", "P_shave_reserve_kWh"],
    }
