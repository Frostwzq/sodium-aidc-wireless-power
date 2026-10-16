"""P2 Aalto-to-AIDC scaling report and verification-status contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import Config


@dataclass(frozen=True)
class EfficiencyInterval:
    lower: float
    upper: float
    provenance: str = "[C] engineering assumption; not inferred from Aalto power sweep"

    def validate(self) -> None:
        if not 0 < self.lower <= self.upper <= 1:
            raise ValueError("efficiency interval must satisfy 0 < lower <= upper <= 1")


SCENARIO_LABELS = {"A_subzone": "P0-A", "B_half_rack": "P0-B", "C_full_rack": "P0-C"}
EFFICIENCY_INTERVALS = {
    "A_subzone": EfficiencyInterval(0.85, 0.92),
    "B_half_rack": EfficiencyInterval(0.78, 0.90),
    "C_full_rack": EfficiencyInterval(0.70, 0.85),
}
VERIFIED_POWER_MAX_W = 40.0
AALTO_EFFICIENCY_ACCEPTANCE_BAND = (0.82, 0.93)


def compare_prediction_to_aalto_band(predicted_efficiency: float | None) -> dict[str, Any]:
    """Apply Aalto's position/orientation band as a constraint, not a fit target."""
    lo, hi = AALTO_EFFICIENCY_ACCEPTANCE_BAND
    if predicted_efficiency is None:
        return {"predicted_efficiency": None, "acceptance_band": (lo, hi),
                "within_band": None, "violation": None,
                "status": "no system-efficiency prediction available"}
    if not 0 < predicted_efficiency <= 1:
        raise ValueError("predicted_efficiency must lie in (0, 1]")
    violation = lo - predicted_efficiency if predicted_efficiency < lo else (
        predicted_efficiency - hi if predicted_efficiency > hi else 0.0
    )
    return {"predicted_efficiency": predicted_efficiency, "acceptance_band": (lo, hi),
            "within_band": violation == 0.0, "violation": violation,
            "status": "constraint_pass" if violation == 0.0 else "constraint_violation"}


def scenario_scaling_report(config: Config) -> dict[str, Any]:
    """Report ratio, explicit range assumption, and validation warning per scenario."""
    config.validate()
    label = SCENARIO_LABELS[config.scenario.scenario_id]
    interval = EFFICIENCY_INTERVALS[config.scenario.scenario_id]
    interval.validate()
    factor = config.scenario.p_field_target_w / VERIFIED_POWER_MAX_W
    outside = config.scenario.p_field_target_w > VERIFIED_POWER_MAX_W
    warning = (
        f"WARNING [{label}]: 超出已验证区间；{config.scenario.p_field_target_w:g} W is "
        f"{factor:g}x the Aalto 40 W maximum. No linear-efficiency claim is made."
        if outside else None
    )
    return {
        "scenario_id": label,
        "p_field_target_w": config.scenario.p_field_target_w,
        "aalto_reference_power_w": (20.0, VERIFIED_POWER_MAX_W),
        "scale_from_aalto_40w": factor,
        "outside_validated_power_range": outside,
        "warning": warning,
        "efficiency_interval_assumption": (interval.lower, interval.upper),
        "efficiency_interval_provenance": interval.provenance,
        "aalto_efficiency_acceptance_band": AALTO_EFFICIENCY_ACCEPTANCE_BAND,
        "prediction_error_pct": None,
        "validation_status": "no independent power-efficiency validation data",
        "nonlinear_factors": {
            "coil_losses": "Use 200 kHz Litz/AC-resistance model with proximity range; measure R_ac before scaling.",
            "core_saturation_temperature": "Check temperature-adjusted B_sat and thermal limits; FEM/measurement required.",
            "parasitics_common_mode": "Model array parasitics and common-mode current separately at each array scale.",
        },
    }


def all_scenario_reports() -> list[dict[str, Any]]:
    return [scenario_scaling_report(Config.for_scenario(sid)) for sid in SCENARIO_LABELS]


def describe() -> dict[str, Any]:
    return {
        "inputs": {"config": "Config.for_scenario(A_subzone|B_half_rack|C_full_rack)"},
        "outputs": {"scale_ratio": "relative to Aalto 40 W maximum", "interval": "assumption range", "warning": "超出已验证区间", "prediction_error_pct": "None without independent measurement"},
        "valid_ranges": {"efficiency_interval": "(0, 1]", "power_w": "> 0"},
        "TODOs": ["TODO[实测]: obtain position-resolved efficiency data and a power sweep for independent validation."],
    }
