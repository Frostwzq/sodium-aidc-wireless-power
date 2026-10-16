"""Deterministic [C] workload traces from the P0 load contract."""

from __future__ import annotations

import math
from typing import Any

from src.config import Config


SCENARIO_LABELS = {
    "A_subzone": "P0-A",
    "B_half_rack": "P0-B",
    "C_full_rack": "P0-C",
}

ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] deterministic representative workload; P0 does not include a measured trace",
    "device_rated_power_w_per_load": 1_000.0,
    "cycle_period_s": 3_600.0,
    "sample_interval_s": 10.0,
    "train_interval_placement": "centered within each period to provide a known pre-peak window",
    "train_burst_ramp_event": "op_switch",
    "low_level_derivation": "(P_train_mean_factor - duty*P_train_peak_factor)/(1-duty)",
    "random_noise": "none; P0 jitter is retained as a separate uncertainty bound, not synthesized as noise",
}


def generate_load_profile(
    scenario_id: str,
    duration_s: float = 3_600.0,
    sample_interval_s: float = 10.0,
    cycle_period_s: float = 3_600.0,
    duty_cycle: float | None = None,
    ramp_event: str = "op_switch",
) -> dict[str, Any]:
    """Create a piecewise-constant high/low target and first-order filtered trace.

    The low target is derived from P0's mean, peak, and duty contract so that
    the requested mean and duty can coexist. All time and watt values are
    included in the returned report for reproducibility.
    """
    if scenario_id not in SCENARIO_LABELS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}")
    if min(duration_s, sample_interval_s, cycle_period_s) <= 0:
        raise ValueError("duration, sample interval, and cycle period must be positive")
    if duration_s < sample_interval_s or cycle_period_s < sample_interval_s:
        raise ValueError("duration and cycle period must be at least one sample")

    config = Config.for_scenario(scenario_id)
    load = config.load
    duty = load.duty_cycle if duty_cycle is None else duty_cycle
    if not 0 < duty < 1:
        raise ValueError("duty_cycle must lie in (0, 1)")
    if ramp_event not in load.ramp_timescale_s:
        raise ValueError(f"ramp_event must be one of {tuple(load.ramp_timescale_s)}")
    rated_power_w = config.scenario.n_load_max * ASSUMPTIONS["device_rated_power_w_per_load"]
    low_factor = (load.p_train_mean_factor - duty * load.p_train_peak_factor) / (1.0 - duty)
    if not load.p_idle_factor <= low_factor < load.p_train_mean_factor:
        raise ValueError("P0 mean/duty/peak factors imply a low level outside the declared idle-to-mean range")
    low_w = low_factor * rated_power_w
    high_w = load.p_train_peak_factor * rated_power_w
    mean_target_w = load.p_train_mean_factor * rated_power_w
    tau_s = load.ramp_timescale_s[ramp_event]
    count = int(math.floor(duration_s / sample_interval_s)) + 1
    times = [index * sample_interval_s for index in range(count)]
    high_duration_s = duty * cycle_period_s
    high_start_s = 0.5 * (cycle_period_s - high_duration_s)
    targets: list[float] = []
    filtered: list[float] = []
    previous = low_w
    for time_s in times:
        phase_s = time_s % cycle_period_s
        target = high_w if high_start_s <= phase_s < high_start_s + high_duration_s else low_w
        targets.append(target)
        if not filtered:
            previous = low_w
        if tau_s <= 0:
            current = target
        else:
            alpha = 1.0 - math.exp(-sample_interval_s / tau_s)
            current = previous + alpha * (target - previous)
        filtered.append(current)
        previous = current

    slopes_w_per_s = [
        (filtered[index] - filtered[index - 1]) / sample_interval_s
        for index in range(1, len(filtered))
    ]
    sorted_abs_slopes = sorted(abs(value) for value in slopes_w_per_s)

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        return values[min(len(values) - 1, int(round(fraction * (len(values) - 1))))]

    actual_mean_w = sum(filtered) / len(filtered)
    actual_peak_w = max(filtered)
    active_count = sum(value >= mean_target_w for value in filtered)
    actual_duty = active_count / len(filtered)
    relative_duty_deviation = abs(actual_duty - duty) / duty
    warning = relative_duty_deviation > 0.10
    return {
        "scenario_id": SCENARIO_LABELS[scenario_id],
        "rated_power_w": rated_power_w,
        "time_s": times,
        "target_power_w": targets,
        "p_load_w": filtered,
        "P_mean_W": actual_mean_w,
        "P_mean_target_W": mean_target_w,
        "P_peak_W": actual_peak_w,
        "P_low_target_W": low_w,
        "P_high_target_W": high_w,
        "ramp_rate_distribution_w_per_s": {
            "min": min(slopes_w_per_s, default=0.0),
            "median_abs": percentile(sorted_abs_slopes, 0.50),
            "p90_abs": percentile(sorted_abs_slopes, 0.90),
            "p99_abs": percentile(sorted_abs_slopes, 0.99),
            "max_abs": max(sorted_abs_slopes, default=0.0),
        },
        "duty_cycle_assumed": duty,
        "duty_cycle_actual": actual_duty,
        "duty_cycle_relative_deviation": relative_duty_deviation,
        "duty_cycle_warning": warning,
        "duty_cycle_warning_text": (
            "WARNING: measured trace duty deviates from the assumed 0.45 by more than 10%."
            if warning else None
        ),
        "duration_s": duration_s,
        "sample_interval_s": sample_interval_s,
        "cycle_period_s": cycle_period_s,
        "ramp_event": ramp_event,
        "ramp_time_constant_s": tau_s,
        "provenance": ASSUMPTIONS["provenance"],
        "assumptions": ASSUMPTIONS,
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario_id", "duration_s", "sample_interval_s", "cycle_period_s", "duty_cycle", "ramp_event"],
        "outputs": ["P_load(t)", "mean/peak power", "ramp-rate distribution", "actual duty-cycle warning"],
        "defaults": ASSUMPTIONS,
        "ramp_events_s": {
            key: Config.for_scenario("A_subzone").load.ramp_timescale_s[key]
            for key in ("epoch_switch", "phase_sync", "op_switch")
        },
    }
