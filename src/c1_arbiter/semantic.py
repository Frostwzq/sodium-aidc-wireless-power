"""Interpretable semantic load recognition from Tx-side current waveforms.

The reference signals and current-to-power calibration below are explicitly
synthetic [C] assumptions.  In particular, P4's ideal load-independent Tx
winding-current equation cannot by itself encode receiver power.  This module
therefore does not claim physical observability until a measured calibration
demonstrates it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from src.config import Config


SCENARIO_LABELS = {
    "A_subzone": "P0-A",
    "B_half_rack": "P0-B",
    "C_full_rack": "P0-C",
}
DEVICE_TYPES = ("CDU", "accelerator")

ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] entirely synthetic semantic waveforms; no measured labeled traces supplied",
    "sample_rate_hz": 100.0,
    "window_duration_s": 10.0,
    "current_baseline_a": 1.0,
    "current_power_gain_a_per_kw": 0.20,
    "current_noise_sigma_a": 0.0015,
    "cdu": {
        "mean_power_kw": 0.55,
        "ripple_amplitude_kw": 0.025,
        "ripple_period_s": 2.0,
    },
    "accelerator": {
        "low_power_kw": 0.50,
        "high_power_kw": 0.98,
        "burst_period_s": 2.0,
        "burst_duty": 0.45,
        "edge_ramp_s": 0.08,
    },
    "multiload_interference_sigma_a_per_other_load": 0.00035,
    "classifier_rules": {
        "accelerator_if_any": {
            "power_variance_kw2_min": 0.018,
            "max_abs_slope_kw_per_s_min": 1.0,
        },
        "otherwise": "CDU",
    },
    "failure_fallback": "maximum configured power demand; preserve unresolved P_device_rated_W as TODO",
    "accuracy_scope": "synthetic scenario test only; not field accuracy",
}


def estimate_receiver_power_w(
    tx_current_a: Iterable[float],
    *,
    current_baseline_a: float = ASSUMPTIONS["current_baseline_a"],
    current_power_gain_a_per_kw: float = ASSUMPTIONS["current_power_gain_a_per_kw"],
) -> np.ndarray:
    """Apply an externally calibrated [C] current-to-power transfer function.

    Positive current deviations map to inferred receiver power.  The linear
    transfer function is intentionally explicit and replaceable by measured
    calibration; P4 does not establish this relationship.
    """
    values = np.asarray(list(tx_current_a), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("tx_current_a must be a nonempty finite 1D waveform")
    if current_power_gain_a_per_kw <= 0:
        raise ValueError("current_power_gain_a_per_kw must be positive")
    return np.maximum(0.0, (values - current_baseline_a) / current_power_gain_a_per_kw) * 1_000.0


def extract_features(
    tx_current_a: Iterable[float],
    *,
    sample_rate_hz: float = ASSUMPTIONS["sample_rate_hz"],
    current_baseline_a: float = ASSUMPTIONS["current_baseline_a"],
    current_power_gain_a_per_kw: float = ASSUMPTIONS["current_power_gain_a_per_kw"],
) -> dict[str, float]:
    """Extract steady power, rise slope, periodicity and variance features."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    current = np.asarray(list(tx_current_a), dtype=float)
    power_kw = estimate_receiver_power_w(
        current,
        current_baseline_a=current_baseline_a,
        current_power_gain_a_per_kw=current_power_gain_a_per_kw,
    ) / 1_000.0
    # A short moving average suppresses sample noise before the ramp feature;
    # this is part of the declared [C] feature definition, not a tuned label.
    smoothing_samples = max(1, int(round(0.1 * sample_rate_hz)))
    smoothed = np.convolve(power_kw, np.ones(smoothing_samples) / smoothing_samples, mode="valid")
    slopes = np.diff(smoothed) * sample_rate_hz
    centered = smoothed - float(smoothed.mean())
    # Compare normalized autocorrelation at the assumed two-second operating cycle.
    lag = int(round(2.0 * sample_rate_hz))
    if len(centered) > lag and float(np.dot(centered, centered)) > 0:
        periodicity = float(np.dot(centered[:-lag], centered[lag:]) / np.dot(centered, centered))
    else:
        periodicity = 0.0
    return {
        "steady_power_kw": float(np.mean(power_kw)),
        "max_abs_slope_kw_per_s": float(np.max(np.abs(slopes))) if slopes.size else 0.0,
        "periodicity_2s": periodicity,
        "power_variance_kw2": float(np.var(power_kw)),
    }


def classify_features(features: dict[str, float]) -> dict[str, Any]:
    """Interpretable rule baseline; malformed features trigger conservative fallback."""
    required = {"steady_power_kw", "max_abs_slope_kw_per_s", "periodicity_2s", "power_variance_kw2"}
    if not required.issubset(features) or any(not math.isfinite(float(features[key])) for key in required):
        return {
            "device_type": None,
            "recognized": False,
            "reason": "missing_or_nonfinite_features",
            "fallback": ASSUMPTIONS["failure_fallback"],
            "power_demand_w": None,
        }
    rules = ASSUMPTIONS["classifier_rules"]["accelerator_if_any"]
    accelerator = (
        features["power_variance_kw2"] >= rules["power_variance_kw2_min"]
        or features["max_abs_slope_kw_per_s"] >= rules["max_abs_slope_kw_per_s_min"]
    )
    return {
        "device_type": "accelerator" if accelerator else "CDU",
        "recognized": True,
        "reason": "variance_or_ramp_threshold" if accelerator else "below_accelerator_thresholds",
        "features": dict(features),
        "fallback": None,
        "power_demand_w": features["steady_power_kw"] * 1_000.0,
    }


def recognize_device(
    tx_current_a: Iterable[float],
    *,
    scenario_id: str | None = None,
    sample_rate_hz: float = ASSUMPTIONS["sample_rate_hz"],
    current_baseline_a: float = ASSUMPTIONS["current_baseline_a"],
    current_power_gain_a_per_kw: float = ASSUMPTIONS["current_power_gain_a_per_kw"],
) -> dict[str, Any]:
    """Extract features and apply the rule baseline to one observed waveform."""
    try:
        features = extract_features(
            tx_current_a,
            sample_rate_hz=sample_rate_hz,
            current_baseline_a=current_baseline_a,
            current_power_gain_a_per_kw=current_power_gain_a_per_kw,
        )
    except (TypeError, ValueError):
        fallback_power_w = _fallback_power_w(scenario_id)
        return {
            "device_type": None,
            "recognized": False,
            "reason": "invalid_waveform_or_calibration",
            "fallback": ASSUMPTIONS["failure_fallback"],
            "fallback_power_demand_w": fallback_power_w,
            "power_demand_w": fallback_power_w,
        }
    return classify_features(features)


def _fallback_power_w(scenario_id: str | None) -> float | None:
    """Use the scenario's maximum configured field demand if known."""
    if scenario_id is None:
        return None
    try:
        return Config.for_scenario(scenario_id).scenario.p_field_target_w
    except ValueError:
        return None


def _synthetic_current_waveform(
    device_type: str,
    scenario_id: str,
    seed: int,
    *,
    sample_rate_hz: float = ASSUMPTIONS["sample_rate_hz"],
    duration_s: float = ASSUMPTIONS["window_duration_s"],
) -> np.ndarray:
    """Generate one synthetic [C] current trace with scenario-scaled interference."""
    if device_type not in DEVICE_TYPES:
        raise ValueError(f"device_type must be one of {DEVICE_TYPES}")
    if scenario_id not in SCENARIO_LABELS:
        raise ValueError(f"unknown scenario_id {scenario_id!r}")
    if sample_rate_hz <= 0 or duration_s < 2.0:
        raise ValueError("sample_rate_hz must be positive and duration_s at least 2 s")
    rng = np.random.default_rng(seed)
    time_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    if device_type == "CDU":
        params = ASSUMPTIONS["cdu"]
        power_kw = params["mean_power_kw"] + params["ripple_amplitude_kw"] * np.sin(
            2.0 * np.pi * time_s / params["ripple_period_s"]
        )
    else:
        params = ASSUMPTIONS["accelerator"]
        phase = np.mod(time_s, params["burst_period_s"])
        active = phase < params["burst_period_s"] * params["burst_duty"]
        ramp_samples = max(1, int(round(params["edge_ramp_s"] * sample_rate_hz)))
        envelope = active.astype(float)
        for edge in range(ramp_samples):
            weight = (edge + 1) / ramp_samples
            envelope[edge::int(round(params["burst_period_s"] * sample_rate_hz))] = np.minimum(
                envelope[edge::int(round(params["burst_period_s"] * sample_rate_hz))], weight
            )
        power_kw = params["low_power_kw"] + envelope * (params["high_power_kw"] - params["low_power_kw"])

    n_load = {"A_subzone": 4, "B_half_rack": 10, "C_full_rack": 24}[scenario_id]
    other_loads = n_load - 1
    interference_sigma = ASSUMPTIONS["multiload_interference_sigma_a_per_other_load"] * math.sqrt(other_loads)
    interference_a = rng.normal(0.0, interference_sigma, size=time_s.size)
    gain = ASSUMPTIONS["current_power_gain_a_per_kw"]
    current = ASSUMPTIONS["current_baseline_a"] + gain * power_kw + interference_a
    current += rng.normal(0.0, ASSUMPTIONS["current_noise_sigma_a"], size=time_s.size)
    return current


def _confusion_matrix(actual: list[str], predicted: list[str]) -> dict[str, dict[str, int]]:
    return {
        label: {guess: sum(a == label and p == guess for a, p in zip(actual, predicted)) for guess in DEVICE_TYPES}
        for label in DEVICE_TYPES
    }


def validate_synthetic_scenarios(
    samples_per_class: int = 200,
    base_seed: int = 910_000,
) -> dict[str, Any]:
    """Evaluate deterministic rule baseline on held-out generated traces per scenario."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    reports = []
    for scenario_index, scenario_id in enumerate(SCENARIO_LABELS):
        actual: list[str] = []
        predicted: list[str] = []
        for class_index, device_type in enumerate(DEVICE_TYPES):
            for sample_index in range(samples_per_class):
                seed = base_seed + scenario_index * 100_000 + class_index * 10_000 + sample_index
                waveform = _synthetic_current_waveform(device_type, scenario_id, seed)
                result = recognize_device(waveform, scenario_id=scenario_id)
                actual.append(device_type)
                predicted.append(result["device_type"] or "")
        correct = sum(a == p for a, p in zip(actual, predicted))
        accuracy = correct / len(actual)
        reports.append({
            "scenario_id": SCENARIO_LABELS[scenario_id],
            "samples_per_class": samples_per_class,
            "sample_count": len(actual),
            "accuracy": accuracy,
            "accuracy_percent": 100.0 * accuracy,
            "confusion_matrix_actual_rows_predicted_columns": _confusion_matrix(actual, predicted),
            "scope": "synthetic only; no field accuracy claim",
        })
    a_accuracy, c_accuracy = reports[0]["accuracy"], reports[2]["accuracy"]
    return {
        "classifier": "interpretable_rule_baseline",
        "reports": reports,
        "p0_c_vs_p0_a_accuracy_drop_percentage_points": 100.0 * (a_accuracy - c_accuracy),
        "assumptions": ASSUMPTIONS,
        "physical_observability_warning": (
            "P4 ideal load-independent Tx winding current does not encode receiver load power; "
            "the synthetic current-to-power map must be validated experimentally before use."
        ),
        "TODOs": [
            "TODO[实测]: collect labeled CDU and accelerator Tx-current records at P0-A/B/C.",
            "TODO[实测]: calibrate Tx-current to received-power transfer under Eq.(2) circuit conditions.",
            "TODO[实测]: measure multi-load coupling/interference and replace the [C] degradation model.",
            "TODO[P0]: resolve P_device_rated_W before converting fallback maximum demand to watts.",
        ],
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["tx_current_a", "sample_rate_hz", "current_baseline_a", "current_power_gain_a_per_kw"],
        "outputs": ["estimated receiver power", "steady power/ramp/periodicity/variance features", "CDU or accelerator label"],
        "classifier": "interpretable threshold baseline; no ML",
        "failure_fallback": ASSUMPTIONS["failure_fallback"],
        "assumptions": ASSUMPTIONS,
        "P0_unresolved_values_preserved": ["P_device_rated_W"],
    }
