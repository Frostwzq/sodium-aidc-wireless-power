"""P10 two-level deterministic power-budget arbitration.

All grid, battery, device portfolio, and field-sensitivity values in this
module are [C] example assumptions.  Field uniformity is enforced as a hard
feasibility constraint; it is deliberately absent from the utility weights.
"""

from __future__ import annotations

import math
from typing import Any

from src.config import Config
from src.c1_arbiter.semantic import SCENARIO_LABELS


SCENARIOS = tuple(SCENARIO_LABELS)

ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] engineering example; no grid-contract, battery dispatch, or device-priority measurements supplied",
    "grid_contract_fraction_of_p0_field_target": 0.60,
    "battery_discharge_cap_fraction_of_p0_field_target": 0.25,
    "battery_life_cost_model": "quadratic normalized discharge: (P_batt/P_batt_cap)^2",
    "battery_life_cost_weight": 0.10,
    "device_catalog": {
        "CDU": {"demand_kw": 0.55, "semantic_priority_weight": 0.75},
        "accelerator": {"demand_kw": 0.98, "semantic_priority_weight": 1.25},
        "portfolio_rule": "alternate CDU, accelerator until P0 scenario n_load_max is reached",
    },
    "utility_weights": {
        "weighted_supply_adequacy": 0.55,
        "jain_fairness": 0.25,
        "battery_peak_shave_benefit": 0.20,
    },
    "field_uniformity_hard_constraint": {
        "nominal_min_field_v": 0.80,
        "threshold_v": 0.50,
        "field_model": "0.80 * sqrt(min(zone_power_fraction))",
        "zone_assignment": "round-robin across four field zones",
        "derived_min_zone_power_fraction": (0.50 / 0.80) ** 2,
    },
    "battery_dispatch_grid_steps": 100,
    "semantic_policy": "P9 rule-baseline class labels; synthetic labels assumed correct per P9 synthetic validation",
    "baseline_without_ai": "equal per-device allocation weights, with all other constraints/objective terms unchanged",
    "ai_gain_definition": "100 * (semantic_policy_utility - equal_weight_utility) / equal_weight_utility",
}


def _portfolio(scenario_id: str) -> list[dict[str, Any]]:
    config = Config.for_scenario(scenario_id)
    catalog = ASSUMPTIONS["device_catalog"]
    devices = []
    for index in range(config.scenario.n_load_max):
        device_type = "CDU" if index % 2 == 0 else "accelerator"
        item = catalog[device_type]
        devices.append({
            "device_id": f"load_{index + 1:02d}",
            "device_type": device_type,
            "demand_kw": float(item["demand_kw"]),
            "semantic_priority_weight": float(item["semantic_priority_weight"]),
            "zone": index % 4,
        })
    return devices


def _allocate_lower_level(
    devices: list[dict[str, Any]],
    budget_kw: float,
    *,
    semantic_weights: bool,
) -> dict[str, Any] | None:
    """Allocate a fixed L2 budget by weighted progressive filling.

    Every load starts at the same declared minimum service fraction. Remaining
    power is shared according to equal or semantic weights, subject to demand
    caps. Returns ``None`` when the hard field-uniformity floor is infeasible.
    """
    floor_fraction = ASSUMPTIONS["field_uniformity_hard_constraint"]["derived_min_zone_power_fraction"]
    demands = [float(device["demand_kw"]) for device in devices]
    total_demand = sum(demands)
    min_required_kw = floor_fraction * total_demand
    if budget_kw + 1e-10 < min_required_kw:
        return None
    if budget_kw >= total_demand:
        fractions = [1.0] * len(devices)
    else:
        priority = [
            float(device["semantic_priority_weight"]) if semantic_weights else 1.0
            for device in devices
        ]
        low, high = 0.0, 2.0
        for _ in range(80):
            scale = (low + high) / 2.0
            used = sum(
                demand * min(1.0, floor_fraction + scale * weight)
                for demand, weight in zip(demands, priority)
            )
            if used > budget_kw:
                high = scale
            else:
                low = scale
        fractions = [
            min(1.0, floor_fraction + low * weight)
            for weight in priority
        ]

    allocations = [demand * fraction for demand, fraction in zip(demands, fractions)]
    zone_fractions = []
    for zone in range(4):
        indices = [index for index, device in enumerate(devices) if device["zone"] == zone]
        zone_demand = sum(demands[index] for index in indices)
        zone_power = sum(allocations[index] for index in indices)
        zone_fractions.append(zone_power / zone_demand if zone_demand else 1.0)
    field = ASSUMPTIONS["field_uniformity_hard_constraint"]
    min_field_v = field["nominal_min_field_v"] * math.sqrt(min(zone_fractions))
    field_pass = min_field_v + 1e-12 >= field["threshold_v"]
    if not field_pass:
        return None
    return {
        "allocations_kw": allocations,
        "service_fractions": fractions,
        "total_allocated_kw": sum(allocations),
        "zone_service_fractions": zone_fractions,
        "min_field_v": min_field_v,
        "field_threshold_v": field["threshold_v"],
        "field_hard_constraint_pass": True,
    }


def _jain_index(values: list[float]) -> float:
    denominator = len(values) * sum(value * value for value in values)
    if denominator <= 0:
        return 1.0
    return sum(values) ** 2 / denominator


def _evaluate_policy(scenario_id: str, *, semantic_weights: bool) -> dict[str, Any]:
    config = Config.for_scenario(scenario_id)
    devices = _portfolio(scenario_id)
    total_demand = sum(float(device["demand_kw"]) for device in devices)
    p_field_target_kw = config.scenario.p_field_target_w / 1_000.0
    grid_contract_kw = p_field_target_kw * ASSUMPTIONS["grid_contract_fraction_of_p0_field_target"]
    battery_cap_kw = p_field_target_kw * ASSUMPTIONS["battery_discharge_cap_fraction_of_p0_field_target"]
    policy_name = "semantic_weighted" if semantic_weights else "equal_weight_baseline"

    best: dict[str, Any] | None = None
    steps = ASSUMPTIONS["battery_dispatch_grid_steps"]
    for step in range(steps + 1):
        battery_kw = battery_cap_kw * step / steps
        field_budget_kw = min(total_demand, grid_contract_kw + battery_kw)
        lower = _allocate_lower_level(devices, field_budget_kw, semantic_weights=semantic_weights)
        if lower is None:
            continue
        served_kw = lower["total_allocated_kw"]
        battery_used_kw = min(battery_kw, served_kw)
        grid_draw_kw = max(0.0, served_kw - battery_used_kw)
        if grid_draw_kw > grid_contract_kw + 1e-9:
            continue

        actual_priority = [float(device["semantic_priority_weight"]) for device in devices]
        fractions = lower["service_fractions"]
        adequacy = sum(weight * fraction for weight, fraction in zip(actual_priority, fractions)) / sum(actual_priority)
        fairness = _jain_index(fractions)
        shave_benefit = min(1.0, battery_used_kw / total_demand) if total_demand else 0.0
        life_cost = (battery_used_kw / battery_cap_kw) ** 2 if battery_cap_kw else 0.0
        weights = ASSUMPTIONS["utility_weights"]
        utility = (
            weights["weighted_supply_adequacy"] * adequacy
            + weights["jain_fairness"] * fairness
            + weights["battery_peak_shave_benefit"] * shave_benefit
            - ASSUMPTIONS["battery_life_cost_weight"] * life_cost
        )
        row = {
            "scenario_id": SCENARIO_LABELS[scenario_id],
            "policy": policy_name,
            "utility": utility,
            "utility_components": {
                "weighted_supply_adequacy": adequacy,
                "jain_fairness": fairness,
                "battery_peak_shave_benefit": shave_benefit,
                "battery_life_cost_penalty": life_cost,
            },
            "l1_l2_budget": {
                "grid_contract_kw": grid_contract_kw,
                "battery_discharge_cap_kw": battery_cap_kw,
                "battery_discharge_kw": battery_used_kw,
                "l2_field_budget_kw": served_kw,
                "grid_draw_kw": grid_draw_kw,
                "grid_contract_pass": grid_draw_kw <= grid_contract_kw + 1e-9,
            },
            "field_uniformity": {
                "min_field_v": lower["min_field_v"],
                "threshold_v": lower["field_threshold_v"],
                "zone_service_fractions": lower["zone_service_fractions"],
                "hard_constraint_pass": lower["field_hard_constraint_pass"],
            },
            "device_allocation": [
                {
                    **device,
                    "allocation_kw": allocation,
                    "service_fraction": fraction,
                    "allocation_weight": (
                        device["semantic_priority_weight"] if semantic_weights else 1.0
                    ),
                }
                for device, allocation, fraction in zip(devices, lower["allocations_kw"], fractions)
            ],
        }
        # Deterministic tie-break: higher utility, then lower battery dispatch.
        if best is None or utility > best["utility"] + 1e-12 or (
            abs(utility - best["utility"]) <= 1e-12
            and battery_used_kw < best["l1_l2_budget"]["battery_discharge_kw"]
        ):
            best = row

    if best is None:
        return {
            "scenario_id": SCENARIO_LABELS[scenario_id],
            "policy": policy_name,
            "feasible": False,
            "reason": "no allocation satisfies both grid contract and hard field-uniformity threshold",
            "total_demand_kw": total_demand,
            "assumptions": ASSUMPTIONS,
        }
    best["feasible"] = True
    best["total_demand_kw"] = total_demand
    best["power_shortfall_kw"] = max(0.0, total_demand - best["l1_l2_budget"]["l2_field_budget_kw"])
    best["search"] = {"battery_dispatch_steps": steps, "step_kw": battery_cap_kw / steps}
    return best


def optimize_scenario(scenario_id: str) -> dict[str, Any]:
    """Compare P9 semantic-weighted allocation with equal-weight degradation."""
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}; expected one of {SCENARIOS}")
    semantic = _evaluate_policy(scenario_id, semantic_weights=True)
    baseline = _evaluate_policy(scenario_id, semantic_weights=False)
    if not semantic.get("feasible") or not baseline.get("feasible"):
        return {
            "scenario_id": SCENARIO_LABELS[scenario_id],
            "feasible": False,
            "semantic_policy": semantic,
            "equal_weight_baseline": baseline,
            "assumptions": ASSUMPTIONS,
        }
    gain_fraction = (semantic["utility"] - baseline["utility"]) / baseline["utility"] if baseline["utility"] else 0.0
    gain_percent = 100.0 * gain_fraction
    return {
        "scenario_id": SCENARIO_LABELS[scenario_id],
        "feasible": True,
        "semantic_policy": semantic,
        "equal_weight_baseline": baseline,
        "semantic_utility_gain_percent": gain_percent,
        "ai_verdict": "该场景不需要 AI" if gain_percent < 5.0 else "语义权重达到 5% 增益门槛",
        "ai_role": "P9 rule-baseline semantic class labels select deterministic allocation weights; no ML model is used.",
        "assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: replace [C] grid contract, battery dispatch cap, cell aging cost, and device demands.",
            "TODO[FEM/实测]: replace the [C] min-field versus zone-power proxy with validated spatial field maps.",
            "TODO[实测]: validate semantic labels and priority weights for CDU and accelerator modules.",
            "TODO[P0]: resolve P_shave_reserve_kWh and battery state/thermal limits before field deployment.",
        ],
    }


def test_degradation_without_ai() -> dict[str, Any]:
    """P10 self-check: equal weights must remain feasible without semantic input."""
    scenarios = []
    for scenario_id in SCENARIOS:
        semantic = _evaluate_policy(scenario_id, semantic_weights=True)
        baseline = _evaluate_policy(scenario_id, semantic_weights=False)
        gain = (
            100.0 * (semantic["utility"] - baseline["utility"]) / baseline["utility"]
            if semantic.get("feasible") and baseline.get("feasible") and baseline["utility"]
            else None
        )
        scenarios.append({
            "scenario_id": SCENARIO_LABELS[scenario_id],
            "ai_enabled_feasible": semantic.get("feasible", False),
            "ai_disabled_feasible": baseline.get("feasible", False),
            "ai_enabled_utility": semantic.get("utility"),
            "ai_disabled_utility": baseline.get("utility"),
            "ai_gain_percent": gain,
            "ai_needed": gain is not None and gain >= 5.0,
            "hard_field_constraint_ai_enabled": semantic.get("field_uniformity", {}).get("hard_constraint_pass", False),
            "hard_field_constraint_ai_disabled": baseline.get("field_uniformity", {}).get("hard_constraint_pass", False),
        })
    return {
        "test": "test_degradation_without_ai",
        "pass": all(item["ai_disabled_feasible"] and item["hard_field_constraint_ai_disabled"] for item in scenarios),
        "scenarios": scenarios,
        "scope": "deterministic [C] model self-check; not measured operational performance",
        "assumptions": ASSUMPTIONS,
    }


def optimize_all_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [optimize_scenario(scenario_id) for scenario_id in SCENARIOS],
        "degradation_without_ai": test_degradation_without_ai(),
        "assumptions": ASSUMPTIONS,
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario_id", "P9 semantic labels (or equal-weight degraded mode)"],
        "outputs": ["L1/L2 budget split", "per-device allocation", "hard field-uniformity result", "utility and AI gain"],
        "hard_constraints": ["P_grid_draw <= P_grid_contract", "min_field_v >= threshold_v"],
        "search": "deterministic 101-point battery-dispatch scan plus weighted progressive filling",
        "assumptions": ASSUMPTIONS,
        "P0_unresolved_values_preserved": ["P_shave_reserve_kWh"],
    }
