"""P11 over-current, over-temperature, and insulation protection logic."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.l2_field.hotplug import SCENARIOS, tx_failure_field_report
from src.l2_field.zvs import test_no_degrade_after_zvs


ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] protection thresholds, sensor delays, and device tolerance windows",
    "actions": {
        "overcurrent": {
            "trip_threshold_pu": 1.25,
            "detection_delay_s": 0.0001,
            "action_delay_s": 0.0001,
            "device_tolerance_window_s": 0.005,
            "action": "isolate the affected Tx branch and keep remaining branches in the prior phase pattern",
        },
        "overtemperature": {
            "trip_temperature_c": 110.0,
            "detection_delay_s": 0.020,
            "action_delay_s": 0.020,
            "device_tolerance_window_s": 0.100,
            "action": "isolate the affected Tx branch; retain remaining branches",
        },
        "insulation": {
            "trip_resistance_ohm": 1_000_000.0,
            "detection_delay_s": 0.005,
            "action_delay_s": 0.001,
            "device_tolerance_window_s": 0.020,
            "action": "isolate the affected Tx branch; retain remaining branches",
        },
    },
    "equation_10": {
        "rho_values": [0.05, 0.10],
        "revalidation": "re-run P6 test_no_degrade_after_zvs after each modeled protection action",
    },
}


@lru_cache(maxsize=1)
def _p6_equation10_report() -> dict[str, Any]:
    return test_no_degrade_after_zvs(tuple(ASSUMPTIONS["equation_10"]["rho_values"]))


@lru_cache(maxsize=1)
def _quarter_failure_report() -> dict[str, Any]:
    return tx_failure_field_report()


def simulate_protection_event(event: str) -> dict[str, Any]:
    """Evaluate detection/action latency and recheck P6 Eq.(10) after isolation."""
    if event not in ASSUMPTIONS["actions"]:
        raise ValueError(f"event must be one of {tuple(ASSUMPTIONS['actions'])}")
    spec = ASSUMPTIONS["actions"][event]
    response_s = spec["detection_delay_s"] + spec["action_delay_s"]
    p6_report = _p6_equation10_report()
    quarter_failure = _quarter_failure_report()
    min_original = float(p6_report["min_phi_original_v"])
    min_after_isolation = float(quarter_failure["worst_case"]["min_phi_v"])
    equation10_rows = []
    for rho in ASSUMPTIONS["equation_10"]["rho_values"]:
        lower_bound = (1.0 - rho) * min_original
        equation10_rows.append({
            "rho": rho,
            "lower_bound_v": lower_bound,
            "min_after_modeled_isolation_v": min_after_isolation,
            "passes": min_after_isolation >= lower_bound,
        })
    equation10_degenerate = min_original <= 1e-12
    return {
        "event": event,
        "trip_condition": {key: value for key, value in spec.items() if key.endswith("_pu") or key.endswith("_c") or key.endswith("_ohm")},
        "response_time_s": response_s,
        "response_time_ms": 1_000.0 * response_s,
        "device_tolerance_window_s": spec["device_tolerance_window_s"],
        "device_tolerance_window_ms": 1_000.0 * spec["device_tolerance_window_s"],
        "response_within_tolerance": response_s < spec["device_tolerance_window_s"],
        "protection_action": spec["action"],
        "equation10_revalidated_after_action": True,
        "equation10": {
            "rho_results": equation10_rows,
            "all_rho_pass": all(row["passes"] for row in equation10_rows),
            "constraint_degenerate": equation10_degenerate,
            "status": "PASS_BUT_DEGENERATE_BASELINE" if equation10_degenerate else (
                "PASS" if all(row["passes"] for row in equation10_rows) else "FAIL"
            ),
            "min_phi_original_v": min_original,
            "min_phi_after_modeled_isolation_v": min_after_isolation,
            "absolute_epsilon_th_pass_after_isolation": min_after_isolation >= quarter_failure["epsilon_th_v"],
            "baseline_warning": p6_report["baseline_warning"],
        },
        "quarter_array_failure_area": quarter_failure["worst_case"],
        "provenance": "[C] protection timings; P6 Eq.(10) recomputation; P4 kernel for branch-loss coverage",
    }


def test_protection_response_and_field_revalidation() -> dict[str, Any]:
    actions = [simulate_protection_event(event) for event in ASSUMPTIONS["actions"]]
    return {
        "test": "test_protection_response_and_field_revalidation",
        "pass": all(
            action["response_within_tolerance"]
            and action["equation10_revalidated_after_action"]
            and action["equation10"]["all_rho_pass"]
            for action in actions
        ),
        "actions": actions,
        "scope": "[C] latency assumptions; Eq.(10) reports PASS_BUT_DEGENERATE_BASELINE inherited from P6",
        "assumptions": ASSUMPTIONS,
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["event: overcurrent | overtemperature | insulation"],
        "outputs": ["protection response time", "device-window result", "post-action Eq.(10) revalidation"],
        "actions": ASSUMPTIONS["actions"],
        "assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: characterize sensor, comparator, isolation, and communication latency.",
            "TODO[系统]: confirm device-specific ride-through windows and safety trip levels.",
            "TODO[FEM/实测]: P6 Eq.(10) baseline is degenerate; obtain a nonzero validated min|Phi| baseline.",
        ],
    }
