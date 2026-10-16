"""P5 inverter and field loss model.

This is a bounded engineering model, not a validated device simulation. Every
non-datasheet parameter is tagged [C] in ``ASSUMPTIONS``. Device anchors are
from the TI LMG5200 datasheet; switching overlap energy is derived from TI's
first-order Vin*I*tTR relation and a TI-reported 25 V/ns typical slew rate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.config import Config


F_CENTER_HZ = 200_000.0
F_DELTA_BASE_HZ = 2_000.0
V_BUS_V = 30.0
N_TX_CORES = 16
LEGS_PER_INVERTER = 8

# [C] System model and scan contract. Device limits and Qg/Coss/Rds anchors
# below are taken from the LMG5200 data sheet; transition slew is TI's forum
# answer, so the resulting Eon/Eoff are derived estimates, not measured data.
ASSUMPTIONS: dict[str, Any] = {
    "provenance": "[C] engineering model unless explicitly labeled datasheet",
    "device": "TI LMG5200, one integrated GaN half-bridge per modeled leg",
    "device_dos": "30 V bus [C] from Aalto reference; 10 A max, 15 mOhm RDS(on), Qg=3.8 nC, Coss=266 pF at 40 V: TI datasheet anchors",
    "transition_slew_v_per_ns": {"nominal": 25.0, "range": [25.0, 50.0], "source": "TI E2E engineer answer; not a datasheet switching-energy table"},
    "switching_energy_uncertainty_multiplier": [0.7, 1.5],
    "rds_on_temp_coefficient_per_k": 0.004,
    "switching_energy_temp_coefficient_per_k": 0.002,
    "junction_temperature_c": 100.0,
    "number_of_tx_cores": N_TX_CORES,
    "core_volume_m3_each": 0.001,
    "core_surface_area_m2_each": 0.2,
    "core_flux_peak_t_at_30kw": 0.1,
    "core_flux_power_exponent": 0.5,
    "n87_steinmetz": {
        "reference_frequency_hz": 100_000.0,
        "reference_flux_peak_t": 0.2,
        "reference_loss_w_m3_at_100c": 375_000.0,
        "alpha_frequency": 1.5,
        "beta_flux": 2.5,
        "temperature_multiplier_at_45c": 1.3,
        "uncertainty_multiplier": [0.5, 2.0],
        "source": "TDK N87 datasheet reference point; exponents and temperature multiplier are [C] fit assumptions",
    },
    "driver_supply_v": 5.0,
    "dsp_control_w_per_system": 15.0,
    "inverter_utilization": 0.8,
    "delta_f_scan_hz": {"min": 200.0, "max": 20_000.0, "step": 200.0},
    "power_scan_w": {"min": 0.0, "max": 30_000.0, "step": 500.0},
    "idle_power_threshold_w": 1.0,
    "zero_power_envelope_threshold_fraction": 0.05,
    "thermal_allowable_rise_k": 35.0,
    "thermal_convection_w_m2k": 20.0,
    "thermal_mode": "forced_air [C], inherited from P2's illustrative h=20 case",
}


@dataclass(frozen=True)
class Interval:
    nominal: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _check_frequency_pair(delta_f_hz: float) -> tuple[float, float]:
    if not 0 < delta_f_hz < 2 * F_CENTER_HZ:
        raise ValueError("delta_f_hz must be between 0 and twice the center frequency")
    return F_CENTER_HZ - delta_f_hz / 2.0, F_CENTER_HZ + delta_f_hz / 2.0


def _scenario(scenario_id: str) -> tuple[str, float]:
    config = Config.for_scenario(scenario_id)
    return config.scenario.scenario_id, config.scenario.p_field_target_w


def _inverter_count(power_w: float) -> int:
    if power_w <= 0:
        # Keep the power stage switching at idle; system hardware remains present.
        return _inverter_count(3_000.0)
    per_inverter_w = V_BUS_V * 10.0 * LEGS_PER_INVERTER * ASSUMPTIONS["inverter_utilization"]
    return max(1, math.ceil(power_w / per_inverter_w))


def _device_current_a(power_w: float, inverter_count: int) -> float:
    if power_w <= 0:
        return 0.0
    return power_w / (V_BUS_V * LEGS_PER_INVERTER * inverter_count)


def _energy_interval(power_w: float, current_a: float, temperature_c: float) -> tuple[Interval, Interval]:
    """Return [C] Eon/Eoff estimates per LMG5200 switch transition in joules."""
    if current_a < 0 or temperature_c <= -40 or temperature_c > 125:
        raise ValueError("current must be nonnegative and temperature within LMG5200 operating range")
    slew = ASSUMPTIONS["transition_slew_v_per_ns"]["nominal"] * 1e9
    transition_s = V_BUS_V / slew
    # TI first-order overlap estimate for both edges together; split equally.
    edge_energy = 0.5 * V_BUS_V * current_a * transition_s
    temp_factor = max(0.1, 1.0 + ASSUMPTIONS["switching_energy_temp_coefficient_per_k"] * (temperature_c - 25.0))
    nominal = edge_energy * temp_factor
    low_factor, high_factor = ASSUMPTIONS["switching_energy_uncertainty_multiplier"]
    return Interval(nominal, nominal * low_factor, nominal * high_factor), Interval(nominal, nominal * low_factor, nominal * high_factor)


def switching_energy_interval(
    current_a: float,
    temperature_c: float = ASSUMPTIONS["junction_temperature_c"],
) -> dict[str, dict[str, float]]:
    """Expose Eon/Eoff intervals per switching edge at the modeled 30 V bus."""
    on, off = _energy_interval(0.0, current_a, temperature_c)
    return {"E_on_j": on.as_dict(), "E_off_j": off.as_dict(), "provenance": "[C] derived from TI Vin*I*tTR; not directly tabulated Eon/Eoff"}


def _components(
    power_w: float,
    delta_f_hz: float,
    temperature_c: float,
    idle_inverter_count: int | None = None,
) -> dict[str, Any]:
    if power_w < 0:
        raise ValueError("power_w cannot be negative")
    f1, f2 = _check_frequency_pair(delta_f_hz)
    scenario_id, max_power_w = _scenario("C_full_rack")
    if power_w > max_power_w:
        raise ValueError("power_w must not exceed P0-C 30 kW target")
    inverter_count = (
        idle_inverter_count if power_w <= 0 and idle_inverter_count is not None
        else _inverter_count(power_w)
    )
    leg_count = inverter_count * LEGS_PER_INVERTER
    current_a = _device_current_a(power_w, inverter_count)
    eon, eoff = _energy_interval(power_w, current_a, temperature_c)

    # Two carrier groups each receive half of the eight-leg population.
    legs_per_tone = leg_count / 2.0
    p_overlap = sum(f * (eon.nominal + eoff.nominal) * legs_per_tone for f in (f1, f2))
    p_overlap_low = sum(f * (eon.lower + eoff.lower) * legs_per_tone for f in (f1, f2))
    p_overlap_high = sum(f * (eon.upper + eoff.upper) * legs_per_tone for f in (f1, f2))

    # Output-capacitance energy remains at zero load: this is a real idle
    # commutation term. Coss is the datasheet value at 40 V; linear scaling is [C].
    coss_f = 266e-12
    e_oss_j = 0.5 * coss_f * V_BUS_V**2
    p_coss = sum(2.0 * e_oss_j * f * legs_per_tone for f in (f1, f2))
    p_sw = p_overlap + p_coss
    p_sw_low = p_overlap_low + p_coss * 0.5
    p_sw_high = p_overlap_high + p_coss * 1.5

    rds_25 = 15e-3
    rds_hot = rds_25 * (1.0 + ASSUMPTIONS["rds_on_temp_coefficient_per_k"] * (temperature_c - 25.0))
    p_cond = leg_count * current_a**2 * rds_hot
    p_cond_low, p_cond_high = p_cond * 0.8, p_cond * 1.25

    # Steinmetz fit anchored to TDK N87's 100 kHz, 200 mT, 100 C point.
    st = ASSUMPTIONS["n87_steinmetz"]
    flux = ASSUMPTIONS["core_flux_peak_t_at_30kw"] * (
        power_w / max_power_w
    ) ** ASSUMPTIONS["core_flux_power_exponent"]
    volume = ASSUMPTIONS["core_volume_m3_each"] * N_TX_CORES
    p_mag_by_tone = [
        st["reference_loss_w_m3_at_100c"]
        * (f / st["reference_frequency_hz"]) ** st["alpha_frequency"]
        * (flux / st["reference_flux_peak_t"]) ** st["beta_flux"]
        * st["temperature_multiplier_at_45c"] * volume
        for f in (f1, f2)
    ]
    p_mag = sum(p_mag_by_tone)
    mag_low, mag_high = st["uncertainty_multiplier"]
    p_mag_low, p_mag_high = p_mag * mag_low, p_mag * mag_high

    # Gate-drive dissipation is accounted here, separately from overlap loss;
    # DSP/control is a [C] fixed 15 W system overhead.
    qg_c = 3.8e-9
    p_driver = sum(2.0 * qg_c * ASSUMPTIONS["driver_supply_v"] * f * legs_per_tone for f in (f1, f2))
    p_ctrl = p_driver + ASSUMPTIONS["dsp_control_w_per_system"]
    p_ctrl_low, p_ctrl_high = p_ctrl * 0.8, p_ctrl * 1.25

    parts = {
        "P_sw": Interval(p_sw, p_sw_low, p_sw_high),
        "P_cond": Interval(p_cond, p_cond_low, p_cond_high),
        "P_mag": Interval(p_mag, p_mag_low, p_mag_high),
        "P_ctrl": Interval(p_ctrl, p_ctrl_low, p_ctrl_high),
    }
    total = sum(item.nominal for item in parts.values())
    total_low = sum(item.lower for item in parts.values())
    total_high = sum(item.upper for item in parts.values())
    fractions = {name: item.nominal / total if total else 0.0 for name, item in parts.items()}
    return {
        "scenario_id": "P0-C" if power_w == max_power_w else "P0-C scaled operating point",
        "power_w": power_w,
        "delta_f_hz": delta_f_hz,
        "f1_hz": f1,
        "f2_hz": f2,
        "temperature_c": temperature_c,
        "inverter_count_assumed": inverter_count,
        "switching_legs": leg_count,
        "current_per_leg_a": current_a,
        "flux_peak_per_tone_t": flux,
        "E_on_j": eon.as_dict(),
        "E_off_j": eoff.as_dict(),
        "P_sw_idle_coss_w": p_coss,
        "components_w": {name: item.as_dict() for name, item in parts.items()},
        "component_fraction": fractions,
        "total_loss_w": total,
        "total_loss_interval_w": {"lower": total_low, "upper": total_high},
        "uncertainty_provenance": "bounded [C] intervals; switching energy derived from device transition estimate; N87 Steinmetz exponents fitted assumptions",
    }


def loss_breakdown(
    power_w: float,
    delta_f_hz: float = F_DELTA_BASE_HZ,
    temperature_c: float = ASSUMPTIONS["junction_temperature_c"],
) -> dict[str, Any]:
    """Calculate four loss terms and uncertainty intervals for an operating point."""
    return _components(power_w, delta_f_hz, temperature_c)


def idle_loss_report(
    delta_f_hz: float = F_DELTA_BASE_HZ,
    temperature_c: float = ASSUMPTIONS["junction_temperature_c"],
    scenario_id: str = "C_full_rack",
) -> dict[str, Any]:
    """Report idle loss with zero demanded field power and active switching."""
    config = Config.for_scenario(scenario_id)
    label = {"A_subzone": "P0-A", "B_half_rack": "P0-B", "C_full_rack": "P0-C"}[scenario_id]
    installed_inverters = _inverter_count(config.scenario.p_field_target_w)
    report = _components(0.0, delta_f_hz, temperature_c, installed_inverters)
    report["scenario_id"] = label
    report["installed_inverters"] = installed_inverters
    report["test_name"] = "test_idle_loss_not_zero"
    report["idle_power_threshold_w"] = ASSUMPTIONS["idle_power_threshold_w"]
    report["idle_loss_nonzero"] = report["total_loss_w"] > 0.0
    return report


def test_idle_loss_not_zero(scenario_id: str = "C_full_rack") -> dict[str, Any]:
    """Fail loudly if the modeled zero-demand interval has no inverter loss."""
    report = idle_loss_report(scenario_id=scenario_id)
    if not report["idle_loss_nonzero"]:
        raise AssertionError("P5 premise failed: zero-demand interval has zero modeled loss")
    return report


def scan_loss_surface(
    powers_w: Iterable[float] = range(0, 30_001, 500),
    delta_f_values_hz: Iterable[float] = range(200, 20_001, 200),
    temperature_c: float = ASSUMPTIONS["junction_temperature_c"],
) -> list[dict[str, Any]]:
    """Return curve data for all four component fractions over P and Δf."""
    return [_components(float(power), float(delta_f), temperature_c)
            for power in powers_w for delta_f in delta_f_values_hz]


def loss_fraction_curves(
    delta_f_step_hz: float = 200.0,
    power_step_w: float = 500.0,
    temperature_c: float = ASSUMPTIONS["junction_temperature_c"],
) -> dict[str, Any]:
    """Return line-series data for component shares versus Δf and power."""
    if delta_f_step_hz <= 0 or power_step_w <= 0:
        raise ValueError("curve steps must be positive")
    delta_count = int(math.floor((20_000.0 - 200.0) / delta_f_step_hz)) + 1
    delta_values = [200.0 + index * delta_f_step_hz for index in range(delta_count)]
    power_count = int(math.floor(30_000.0 / power_step_w)) + 1
    power_values = [index * power_step_w for index in range(power_count)]
    component_names = ("P_sw", "P_cond", "P_mag", "P_ctrl")
    by_delta_f: dict[str, Any] = {}
    for label, power in (("P0-A", 3_000.0), ("P0-B", 12_000.0), ("P0-C", 30_000.0)):
        rows = [_components(power, delta, temperature_c) for delta in delta_values]
        by_delta_f[label] = {
            "power_w": power,
            "delta_f_hz": delta_values,
            "fraction": {name: [row["component_fraction"][name] for row in rows]
                         for name in component_names},
        }
    by_power: dict[str, Any] = {}
    for delta in (200.0, 2_000.0, 10_000.0, 20_000.0):
        rows = [_components(power, delta, temperature_c) for power in power_values]
        by_power[str(int(delta))] = {
            "delta_f_hz": delta,
            "power_w": power_values,
            "fraction": {name: [row["component_fraction"][name] for row in rows]
                         for name in component_names},
        }
    return {
        "component_names": list(component_names),
        "versus_delta_f": by_delta_f,
        "versus_power": by_power,
        "provenance": "sampled model curves; ranges and steps are included in each series",
    }


def optimize_delta_f(
    power_w: float,
    delta_f_min_hz: float = 200.0,
    delta_f_max_hz: float = 20_000.0,
    delta_f_step_hz: float = 200.0,
    temperature_c: float = ASSUMPTIONS["junction_temperature_c"],
) -> dict[str, Any]:
    """Numerically scan total loss; report whether the minimum is interior.

    No artificial small-Δf penalty is inserted. With the documented two-tone
    power model, f1+f2 is constant and the Steinmetz term is convex, so an
    interior optimum is not guaranteed; that claim remains falsifiable.
    """
    if delta_f_step_hz <= 0 or delta_f_min_hz <= 0 or delta_f_max_hz < delta_f_min_hz:
        raise ValueError("invalid delta_f scan range or step")
    count = int(math.floor((delta_f_max_hz - delta_f_min_hz) / delta_f_step_hz)) + 1
    scan = [
        _components(power_w, delta_f_min_hz + index * delta_f_step_hz, temperature_c)
        for index in range(count)
    ]
    best = min(scan, key=lambda row: row["total_loss_w"])
    interior = 0 < scan.index(best) < len(scan) - 1
    return {
        "scenario_id": "P0-C" if power_w == 30_000.0 else "P0-C scaled operating point",
        "power_w": power_w,
        "scan_hz": {"min": delta_f_min_hz, "max": scan[-1]["delta_f_hz"], "step": delta_f_step_hz, "points": len(scan)},
        "optimal_delta_f_hz": best["delta_f_hz"],
        "minimum_total_loss_w": best["total_loss_w"],
        "interior_optimum_found": interior,
        "minimum_at_scan_boundary": not interior,
        "total_loss_curve_w": [(row["delta_f_hz"], row["total_loss_w"]) for row in scan],
        "claim_status": "interior optimum supported by scan" if interior else "no interior optimum in stated [C] model and scan; do not claim one",
    }


def all_scenario_loss_reports(delta_f_hz: float = F_DELTA_BASE_HZ) -> list[dict[str, Any]]:
    reports = []
    for scenario_id in ("A_subzone", "B_half_rack", "C_full_rack"):
        config = Config.for_scenario(scenario_id)
        row = loss_breakdown(config.scenario.p_field_target_w, delta_f_hz)
        row["scenario_id"] = {"A_subzone": "P0-A", "B_half_rack": "P0-B", "C_full_rack": "P0-C"}[scenario_id]
        reports.append(row)
    return reports


def describe() -> dict[str, Any]:
    return {
        "inputs": ["scenario power or power_w", "delta_f_hz", "junction temperature"],
        "outputs": ["P_sw/P_cond/P_mag/P_ctrl", "component shares", "uncertainty intervals", "Δf numeric scan"],
        "valid_ranges": {"power_w": "[0, 30000]", "delta_f_hz": "(0, 400000)", "temperature_c": "[-40, 125]"},
        "assumptions": ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: replace inferred LMG5200 switching-energy model with double-pulse Eon/Eoff over current and temperature.",
            "TODO[实测]: measure coil Rac(T), magnetic flux density, N87 core volume, and driver/DSP power.",
            "TODO[实测]: replace assumed 30 V bus, inverter count, 16 Tx-core count, and 80% utilization with final topology.",
            "TODO[实测]: validate whether any Δf-dependent commutation interaction creates an interior loss optimum.",
        ],
    }
