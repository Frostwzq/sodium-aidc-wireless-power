"""P5 rack-level thermal feasibility and power-cap estimate."""

from __future__ import annotations

from typing import Any

from src.config import Config
from src.l2_field.loss import ASSUMPTIONS, F_DELTA_BASE_HZ, loss_breakdown


THERMAL_ASSUMPTIONS = {
    "provenance": "[C] engineering assumptions inherited/extended from P2 examples",
    "tx_core_count": 16,
    "effective_surface_area_m2_per_core": 0.2,
    "convection_w_m2k": 20.0,
    "cooling_mode": "forced_air",
    "ambient_c": 45.0,
    "maximum_temperature_rise_k": 35.0,
}


def available_heat_rejection_w(
    tx_core_count: int = THERMAL_ASSUMPTIONS["tx_core_count"],
    area_per_core_m2: float = THERMAL_ASSUMPTIONS["effective_surface_area_m2_per_core"],
    convection_w_m2k: float = THERMAL_ASSUMPTIONS["convection_w_m2k"],
    allowable_rise_k: float = THERMAL_ASSUMPTIONS["maximum_temperature_rise_k"],
) -> float:
    if min(tx_core_count, area_per_core_m2, convection_w_m2k, allowable_rise_k) <= 0:
        raise ValueError("thermal geometry and limits must be positive")
    return tx_core_count * area_per_core_m2 * convection_w_m2k * allowable_rise_k


def thermal_check(
    total_loss_w: float,
    tx_core_count: int = THERMAL_ASSUMPTIONS["tx_core_count"],
    area_per_core_m2: float = THERMAL_ASSUMPTIONS["effective_surface_area_m2_per_core"],
    convection_w_m2k: float = THERMAL_ASSUMPTIONS["convection_w_m2k"],
    ambient_c: float = THERMAL_ASSUMPTIONS["ambient_c"],
    allowable_rise_k: float = THERMAL_ASSUMPTIONS["maximum_temperature_rise_k"],
) -> dict[str, Any]:
    if total_loss_w < 0:
        raise ValueError("total_loss_w cannot be negative")
    area = tx_core_count * area_per_core_m2
    heat_capacity = available_heat_rejection_w(tx_core_count, area_per_core_m2, convection_w_m2k, allowable_rise_k)
    rise = total_loss_w / (area * convection_w_m2k)
    return {
        "total_loss_w": total_loss_w,
        "total_effective_surface_area_m2": area,
        "convection_w_m2k": convection_w_m2k,
        "available_heat_rejection_w": heat_capacity,
        "temperature_rise_k": rise,
        "estimated_temperature_c": ambient_c + rise,
        "allowable_temperature_rise_k": allowable_rise_k,
        "within_thermal_capacity": total_loss_w <= heat_capacity,
        "provenance": "[C] lumped steady-state convection; assembly-level validation required",
    }


def feasible_power_upper_w(
    scenario_id: str = "C_full_rack",
    delta_f_hz: float = F_DELTA_BASE_HZ,
    heat_capacity_w: float | None = None,
    resolution_w: float = 10.0,
) -> float:
    """Scan power to account for discrete inverter-count changes."""
    if resolution_w <= 0:
        raise ValueError("resolution_w must be positive")
    config = Config.for_scenario(scenario_id)
    ceiling = config.scenario.p_field_target_w
    capacity = available_heat_rejection_w() if heat_capacity_w is None else heat_capacity_w
    if capacity <= 0:
        raise ValueError("heat_capacity_w must be positive")
    if loss_breakdown(ceiling, delta_f_hz)["total_loss_w"] <= capacity:
        return ceiling
    steps = int(ceiling // resolution_w)
    feasible_points = [
        step * resolution_w
        for step in range(steps + 1)
        if loss_breakdown(step * resolution_w, delta_f_hz)["total_loss_w"] <= capacity
    ]
    if ceiling % resolution_w:
        if loss_breakdown(ceiling, delta_f_hz)["total_loss_w"] <= capacity:
            feasible_points.append(ceiling)
    return max(feasible_points, default=0.0)


def scenario_thermal_assessment(
    scenario_id: str = "C_full_rack",
    delta_f_hz: float = F_DELTA_BASE_HZ,
    heat_capacity_w: float | None = None,
) -> dict[str, Any]:
    config = Config.for_scenario(scenario_id)
    label = {"A_subzone": "P0-A", "B_half_rack": "P0-B", "C_full_rack": "P0-C"}[scenario_id]
    power = config.scenario.p_field_target_w
    breakdown = loss_breakdown(power, delta_f_hz)
    capacity = available_heat_rejection_w() if heat_capacity_w is None else heat_capacity_w
    check = thermal_check(breakdown["total_loss_w"],
                          tx_core_count=THERMAL_ASSUMPTIONS["tx_core_count"],
                          area_per_core_m2=THERMAL_ASSUMPTIONS["effective_surface_area_m2_per_core"],
                          convection_w_m2k=THERMAL_ASSUMPTIONS["convection_w_m2k"],
                          ambient_c=THERMAL_ASSUMPTIONS["ambient_c"],
                          allowable_rise_k=THERMAL_ASSUMPTIONS["maximum_temperature_rise_k"])
    check["available_heat_rejection_w"] = capacity
    check["within_thermal_capacity"] = breakdown["total_loss_w"] <= capacity
    upper = feasible_power_upper_w(scenario_id, delta_f_hz, capacity)
    conclusion = None
    if label == "P0-C" and not check["within_thermal_capacity"]:
        conclusion = "P0-C 不可行"
    elif check["within_thermal_capacity"]:
        conclusion = f"{label} 按当前 [C] 热假设未超限"
    else:
        conclusion = f"{label} 不可行"
    return {
        "scenario_id": label,
        "field_power_w": power,
        "delta_f_hz": delta_f_hz,
        "total_loss_w": breakdown["total_loss_w"],
        "total_loss_interval_w": breakdown["total_loss_interval_w"],
        "thermal": check,
        "feasible_field_power_upper_w": upper,
        "conclusion": conclusion,
        "assumptions": THERMAL_ASSUMPTIONS,
        "TODOs": ["TODO[实测]: replace assumed rack-level h*A and 35 K limit with facility cooling allocation and measured thermal resistance."],
    }


def all_scenario_thermal_assessments(delta_f_hz: float = F_DELTA_BASE_HZ) -> list[dict[str, Any]]:
    return [scenario_thermal_assessment(sid, delta_f_hz)
            for sid in ("A_subzone", "B_half_rack", "C_full_rack")]


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario_id", "delta_f_hz", "optional heat_capacity_w"],
        "outputs": ["thermal load", "P0-C feasibility verdict", "feasible power upper bound"],
        "assumptions": THERMAL_ASSUMPTIONS,
        "valid_ranges": {"loss_w": ">=0", "h_w_m2k": ">0", "temperature_rise_k": ">0"},
        "TODOs": ["TODO[实测]: obtain rack cooling allocation and validate assembled airflow/area."],
    }
