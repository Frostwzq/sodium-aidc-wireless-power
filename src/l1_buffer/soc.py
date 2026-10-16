"""P0-derived SoC resistance and nonlinear cycle-life accounting."""

from __future__ import annotations

import math
from typing import Any, Iterable

from src.config import Config


ASSUMPTIONS: dict[str, Any] = {
    "provenance": "P0 lookup table and resistance formula; pulse resistance and curve interpolation are [C]",
    "r_pulse_mohm_nominal": 0.5,
    "r_pulse_mohm_uncertainty": [0.0, 2.0],
    "cycle_life_interpolation": "piecewise linear in log(cycles) versus DoD; below 10% extrapolate the first segment",
    "rainflow_method": "turning-point stack with half/full-cycle counts",
}


def internal_resistance_ohm(
    scenario_id: str,
    soc: float,
    temperature_c: float,
    r_pulse_mohm: float = ASSUMPTIONS["r_pulse_mohm_nominal"],
) -> dict[str, Any]:
    """Evaluate P0's R0*exp(-alpha*SoC)*(1+beta*(T-25)/10)+R_pulse fit."""
    if not 0 <= soc <= 1:
        raise ValueError("soc must lie in [0, 1]")
    config = Config.for_scenario(scenario_id)
    if not config.battery.t_op_min_c <= temperature_c <= config.battery.t_op_max_c:
        raise ValueError("temperature is outside the P0 battery operating range")
    if r_pulse_mohm < 0:
        raise ValueError("r_pulse_mohm cannot be negative")
    battery = config.battery
    base_mohm = battery.r0_mohm * math.exp(-battery.alpha * soc) * (
        1.0 + battery.beta * (temperature_c - 25.0) / 10.0
    )
    total_mohm = base_mohm + r_pulse_mohm
    return {
        "scenario_id": scenario_id,
        "soc": soc,
        "temperature_c": temperature_c,
        "r_base_mohm": base_mohm,
        "r_pulse_mohm": r_pulse_mohm,
        "r_int_ohm": total_mohm / 1_000.0,
        "provenance": "P0 R0/alpha/beta fit parameters are marked TODO[实测]; R_pulse is P8 [C].",
    }


def _turning_points(values: Iterable[float]) -> list[float]:
    data = [float(value) for value in values]
    if len(data) < 2:
        return data
    points = [data[0]]
    previous_delta = 0.0
    for index in range(1, len(data)):
        delta = data[index] - data[index - 1]
        if delta == 0:
            continue
        if previous_delta and delta * previous_delta < 0:
            points.append(data[index - 1])
        previous_delta = delta
    points.append(data[-1])
    return points


def rainflow_cycles(soc_trace: Iterable[float]) -> list[dict[str, float]]:
    """Return SoC cycle depths and counts (full=1, half=0.5)."""
    points = _turning_points(soc_trace)
    if any(not 0 <= value <= 1 for value in points):
        raise ValueError("SoC samples must lie in [0, 1]")
    stack: list[float] = []
    cycles: list[dict[str, float]] = []
    for point in points:
        stack.append(point)
        while len(stack) >= 3:
            older_range = abs(stack[-2] - stack[-3])
            newer_range = abs(stack[-1] - stack[-2])
            if older_range > newer_range:
                break
            if len(stack) == 3:
                cycles.append({"depth": older_range, "count": 0.5})
                stack.pop(-3)
            else:
                cycles.append({"depth": older_range, "count": 1.0})
                last = stack[-1]
                stack[-3:] = [last]
    for left, right in zip(stack, stack[1:]):
        cycles.append({"depth": abs(right - left), "count": 0.5})
    return [cycle for cycle in cycles if cycle["depth"] > 0]


def cycle_life_at_dod(scenario_id: str, dod_fraction: float) -> float:
    """Interpolate the P0 nonlinear DoD-life lookup in log-life coordinates."""
    if not 0 < dod_fraction <= 1:
        raise ValueError("dod_fraction must lie in (0, 1]")
    table = Config.for_scenario(scenario_id).battery.cycle_life_dod
    points = sorted((float(key.rstrip("%")) / 100.0, float(cycles)) for key, cycles in table.items())
    if dod_fraction <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif dod_fraction >= points[-1][0]:
        return points[-1][1]
    else:
        for lower, upper in zip(points, points[1:]):
            if lower[0] <= dod_fraction <= upper[0]:
                x0, y0 = lower
                x1, y1 = upper
                break
    ratio = (dod_fraction - x0) / (x1 - x0)
    return math.exp(math.log(y0) + ratio * (math.log(y1) - math.log(y0)))


def cycle_life_consumption(scenario_id: str, soc_trace: Iterable[float]) -> dict[str, Any]:
    """Compute EFC, rainflow damage, and life consumed per unit peak shaving."""
    values = [float(value) for value in soc_trace]
    cycles = rainflow_cycles(values)
    equivalent_full_cycles = sum(item["depth"] * item["count"] for item in cycles)
    life_damage = sum(
        item["count"] / cycle_life_at_dod(scenario_id, max(item["depth"], 0.001))
        for item in cycles
    )
    return {
        "scenario_id": scenario_id,
        "rainflow_cycles": cycles,
        "equivalent_full_cycles": equivalent_full_cycles,
        "life_consumed_fraction": life_damage,
        "cycle_life_at_cycle_depth": [
            {"depth": item["depth"], "count": item["count"],
             "reference_life_cycles": cycle_life_at_dod(scenario_id, max(item["depth"], 0.001)),
             "life_damage": item["count"] / cycle_life_at_dod(scenario_id, max(item["depth"], 0.001))}
            for item in cycles
        ],
        "provenance": ASSUMPTIONS["provenance"],
        "assumptions": ASSUMPTIONS,
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario_id", "soc", "temperature_c", "soc_trace"],
        "outputs": ["R_int(SOC,T)", "rainflow cycles", "equivalent full cycles", "normalized life consumption"],
        "cycle_life_table": Config.for_scenario("A_subzone").battery.cycle_life_dod,
        "assumptions": ASSUMPTIONS,
    }
