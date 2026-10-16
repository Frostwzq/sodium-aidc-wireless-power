"""P6 frequency-switch field and receiver-output transient estimate."""
from __future__ import annotations
import math
from typing import Any
import numpy as np

from src.l2_field.solver import V_RX_MIN_V, multifrequency_rms_map_at_points

SWITCH_ASSUMPTIONS = {
    "scenario_id": "P0-C [C] transient example",
    "start_delta_f_hz": 2_000.0,
    "end_delta_f_hz": 200.0,  # P5 scan boundary selected numerically
    "ramp_duration_s": 0.05,  # P0 op_switch timescale
    "sample_step_s": 2e-6,
    "receiver_position_tx_units": [2.5, 2.5],
    "receiver_orientation_deg": 0.0,
    "rectified_load_ohm": 100.0,  # Aalto R_L, reused as rectifier-side equivalent [C]
    "storage_capacitance_f": 470e-6,
    "maximum_voltage_droop_fraction": 0.05,
    "voltage_tolerance_window_s": 0.05,
    "receiver_minimum_voltage_v": V_RX_MIN_V,
}


def _simulate_capacitor(
    amp_1: float, amp_2: float, start_df: float, end_df: float,
    duration: float, sample_step: float, load_ohm: float, capacitance_f: float,
) -> dict[str, Any]:
    if min(amp_1, amp_2) < 0 or min(start_df, end_df, duration, sample_step, load_ohm, capacitance_f) <= 0:
        raise ValueError("amplitudes must be nonnegative and transient parameters positive")
    steps = math.ceil(duration / sample_step)
    dt = duration / steps
    tau = load_ohm * capacitance_f
    vout = amp_1 + amp_2  # starts at the rectified beat-envelope peak [C]
    initial, minimum, phase = vout, vout, 0.0
    under_band_steps = under_floor_steps = max_under_band_run = current_run = 0
    for index in range(steps):
        t_mid = (index + 0.5) * dt
        df = start_df + (t_mid / duration) * (end_df - start_df)
        phase += 2.0 * math.pi * df * dt
        envelope = math.sqrt(max(amp_1**2 + amp_2**2 + 2.0 * amp_1 * amp_2 * math.cos(phase), 0.0))
        vout = max(envelope, vout * math.exp(-dt / tau))  # ideal rectifier/storage cap [C]
        minimum = min(minimum, vout)
        if vout < initial * (1.0 - SWITCH_ASSUMPTIONS["maximum_voltage_droop_fraction"]):
            under_band_steps += 1
            current_run += 1
            max_under_band_run = max(max_under_band_run, current_run)
        else:
            current_run = 0
        if vout < SWITCH_ASSUMPTIONS["receiver_minimum_voltage_v"]:
            under_floor_steps += 1
    cycles = duration * (start_df + end_df) / 2.0
    zeros = max(0, math.floor(cycles + 0.5)) if abs(amp_1 - amp_2) < 1e-12 else 0
    droop = (initial - minimum) / initial if initial else 0.0
    max_under_band_s = max_under_band_run * dt
    return {
        "sample_count": steps,
        "time_step_s": dt,
        "electrical_time_constant_s": tau,
        "initial_rectified_voltage_v": initial,
        "minimum_rectified_voltage_v": minimum,
        "final_rectified_voltage_v": vout,
        "voltage_droop_v": initial - minimum,
        "voltage_droop_fraction": droop,
        "voltage_droop_percent": 100.0 * droop,
        "max_duration_below_5pct_band_s": max_under_band_s,
        "total_time_below_5pct_band_s": under_band_steps * dt,
        "voltage_tolerance_window_s": SWITCH_ASSUMPTIONS["voltage_tolerance_window_s"],
        "within_amplitude_tolerance": droop <= SWITCH_ASSUMPTIONS["maximum_voltage_droop_fraction"],
        "within_tolerance_time_window": max_under_band_s <= SWITCH_ASSUMPTIONS["voltage_tolerance_window_s"],
        "below_receiver_minimum_voltage_duration_s": under_floor_steps * dt,
        "instantaneous_envelope_minimum_v": abs(amp_1 - amp_2),
        "instantaneous_blind_during_switch": abs(amp_1 - amp_2) < SWITCH_ASSUMPTIONS["receiver_minimum_voltage_v"],
        "instantaneous_envelope_zero_crossings": zeros,
    }


def frequency_switch_report(
    start_delta_f_hz: float = SWITCH_ASSUMPTIONS["start_delta_f_hz"],
    end_delta_f_hz: float = SWITCH_ASSUMPTIONS["end_delta_f_hz"],
    ramp_duration_s: float = SWITCH_ASSUMPTIONS["ramp_duration_s"],
    sample_step_s: float = SWITCH_ASSUMPTIONS["sample_step_s"],
    position_tx_units: tuple[float, float] = tuple(SWITCH_ASSUMPTIONS["receiver_position_tx_units"]),
    orientation_deg: float = SWITCH_ASSUMPTIONS["receiver_orientation_deg"],
    load_ohm: float = SWITCH_ASSUMPTIONS["rectified_load_ohm"],
    capacitance_f: float = SWITCH_ASSUMPTIONS["storage_capacitance_f"],
) -> dict[str, Any]:
    """Simulate RF beat zeros and rectifier storage response during a delta-f ramp."""
    if start_delta_f_hz <= 0 or end_delta_f_hz <= 0:
        raise ValueError("start and end delta_f must be positive")
    x = np.asarray([position_tx_units[0]], dtype=float)
    y = np.asarray([position_tx_units[1]], dtype=float)
    _, _, field_rms = multifrequency_rms_map_at_points(4, 4, x, y, orientation_deg)
    # P4 phase maps differ by a global +90 degree shift and the mutual model is
    # frequency independent, so equal tone amplitudes are assumed here [C].
    tone_amp = float(field_rms[0]) / math.sqrt(2.0)
    output = _simulate_capacitor(tone_amp, tone_amp, start_delta_f_hz, end_delta_f_hz,
                                 ramp_duration_s, sample_step_s, load_ohm, capacitance_f)
    return {
        "scenario_id": SWITCH_ASSUMPTIONS["scenario_id"],
        "start_delta_f_hz": start_delta_f_hz,
        "end_delta_f_hz": end_delta_f_hz,
        "ramp_duration_s": ramp_duration_s,
        "receiver_position_tx_units": list(position_tx_units),
        "orientation_deg": orientation_deg,
        "field_rms_at_position_v": float(field_rms[0]),
        "tone_amplitudes_v": [tone_amp, tone_amp],
        "receiver_load_ohm": load_ohm,
        "receiver_capacitance_f": capacitance_f,
        "output": output,
        "conclusion": {
            "instantaneous_rf_blind": output["instantaneous_blind_during_switch"],
            "rectified_output_droop_exceeds_5pct": not output["within_amplitude_tolerance"],
            "time_outside_voltage_band_within_50ms_window": output["within_tolerance_time_window"],
            "receiver_voltage_below_0_5v": output["below_receiver_minimum_voltage_duration_s"] > 0,
        },
        "assumptions": SWITCH_ASSUMPTIONS,
        "TODOs": [
            "TODO[实测]: characterize rectifier/source impedance and receiver storage capacitance/load.",
            "TODO[设备规格]: replace assumed 5% voltage band and 50 ms tolerance window with receiver specifications.",
        ],
        "provenance": "[C] P4 equal-tone field plus ideal rectifier/capacitor model; not a transient circuit simulation",
    }


def describe() -> dict[str, Any]:
    return {
        "inputs": ["start/end delta_f", "ramp duration", "receiver position", "Rload", "Cstore"],
        "outputs": ["instantaneous beat-envelope blind intervals", "DC output droop", "duration outside assumed voltage band"],
        "assumptions": SWITCH_ASSUMPTIONS,
        "TODOs": ["TODO[实测]: validate with rectifier, regulator, and receiver storage circuit transient model."],
    }
