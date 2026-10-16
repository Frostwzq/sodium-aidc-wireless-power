"""P2 magnetic-core saturation and thermal checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoreMaterial:
    name: str
    b_sat_25c_t: float
    relative_permeability_200khz: float
    saturation_temp_coefficient_per_k: float = 0.0015


N87_3C90 = CoreMaterial("N87 / 3C90 (MnZn power ferrite)", 0.45, 1000.0)


@dataclass(frozen=True)
class CoreGeometry:
    cross_section_m2: float
    window_width_m: float
    window_height_m: float
    effective_surface_area_m2: float

    def validate(self) -> None:
        if min(self.cross_section_m2, self.window_width_m, self.window_height_m,
               self.effective_surface_area_m2) <= 0:
            raise ValueError("core dimensions and effective surface area must be positive")


def b_sat_at_temperature_t(material: CoreMaterial = N87_3C90, ambient_c: float = 25.0) -> float:
    factor = 1.0 - material.saturation_temp_coefficient_per_k * (ambient_c - 25.0)
    if factor <= 0:
        raise ValueError("temperature correction makes B_sat non-positive")
    return material.b_sat_25c_t * factor


def saturation_check(
    b_peak_t: float | None,
    material: CoreMaterial = N87_3C90,
    ambient_c: float = 45.0,
    redline_t: float = 0.42,
) -> dict[str, Any]:
    """Check an externally computed/assumed peak core flux density.

    No P2 input maps the Aalto current to flux-pipe B_peak, so this function
    refuses to invent a core-field value when it is absent.
    """
    if redline_t <= 0:
        raise ValueError("redline_t must be positive")
    bsat_hot = b_sat_at_temperature_t(material, ambient_c)
    if b_peak_t is None:
        return {"b_peak_t": None, "b_sat_hot_t": bsat_hot, "margin_ratio": None,
                "saturated": None, "within_redline": None,
                "status": "unresolved", "todo": "TODO[实测]: compute B_peak with flux-pipe geometry or FEM."}
    if b_peak_t < 0:
        raise ValueError("b_peak_t cannot be negative")
    margin = float("inf") if b_peak_t == 0 else bsat_hot / b_peak_t
    saturated = b_peak_t >= bsat_hot
    return {"b_peak_t": b_peak_t, "b_sat_hot_t": bsat_hot, "margin_ratio": margin,
            "saturated": saturated, "within_redline": b_peak_t <= redline_t,
            "status": "saturated" if saturated else ("near_redline" if b_peak_t >= redline_t else "below_redline"),
            "todo": "TODO[实测]: validate temperature coefficient and B_peak by FEM or measurement."}


def thermal_check(
    loss_w: float | None,
    effective_surface_area_m2: float,
    convection_w_m2k: float,
    ambient_c: float = 45.0,
    coil_temp_limit_c: float = 120.0,
) -> dict[str, Any]:
    """Lumped convection estimate; h must identify natural or forced cooling."""
    if effective_surface_area_m2 <= 0 or convection_w_m2k <= 0:
        raise ValueError("surface area and convection coefficient must be positive")
    if coil_temp_limit_c <= ambient_c:
        raise ValueError("coil temperature limit must exceed ambient")
    if loss_w is None:
        return {"loss_w": None, "temperature_rise_k": None, "estimated_temp_c": None,
                "active_cooling_needed": None, "status": "unresolved",
                "todo": "TODO[实测]: provide core/coil loss and assembled cooling area."}
    if loss_w < 0:
        raise ValueError("loss_w cannot be negative")
    rise = loss_w / (convection_w_m2k * effective_surface_area_m2)
    temp = ambient_c + rise
    active = rise > 60.0 or temp > coil_temp_limit_c
    status = "active_cooling_needed" if active else ("temperature_high" if rise > 35.0 else "passive_estimate_ok")
    return {"loss_w": loss_w, "temperature_rise_k": rise, "estimated_temp_c": temp,
            "active_cooling_needed": active, "status": status,
            "cooling_mode": "natural" if convection_w_m2k <= 10 else "forced_air",
            "todo": "TODO[实测]: verify effective h, area, and loss in assembled geometry."}


def scenario_core_assessment(
    b_peak_t: float | None,
    loss_w: float | None,
    geometry: CoreGeometry,
    convection_w_m2k: float,
    ambient_c: float = 45.0,
) -> dict[str, Any]:
    geometry.validate()
    magnetic = saturation_check(b_peak_t, ambient_c=ambient_c)
    thermal = thermal_check(loss_w, geometry.effective_surface_area_m2,
                            convection_w_m2k, ambient_c=ambient_c)
    if magnetic["saturated"] is True or thermal["active_cooling_needed"] is True:
        feasible: bool | None = False
    elif magnetic["saturated"] is None or thermal["active_cooling_needed"] is None:
        feasible = None
    else:
        feasible = True
    return {"magnetic": magnetic, "thermal": thermal, "feasible": feasible,
            "conclusion": "该功率档不可行" if feasible is False else ("未定：缺少场/损耗输入" if feasible is None else "按当前假设未触发硬约束")}


def describe() -> dict[str, Any]:
    return {
        "inputs": {"core_material": "CoreMaterial", "b_peak_t": "field result or None", "loss_w": "loss estimate or None", "cooling": "area and h"},
        "outputs": {"b_sat": "temperature-adjusted saturation limit", "thermal": "lumped temperature estimate", "feasibility": "bool or None"},
        "valid_ranges": {"B_peak_t": ">= 0", "loss_w": ">= 0", "convection_w_m2k": "> 0"},
        "TODOs": ["TODO[实测]: B_peak, core loss, thermal resistance and assembled cooling area are not known."],
    }
