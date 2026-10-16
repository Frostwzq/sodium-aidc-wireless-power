"""Global configuration contract for sodium_aidc (P0/P1).

Values absent from P0 remain explicitly unresolved. Derived values that need
those inputs return ``None``; no numerical defaults are invented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    p_field_target_w: float
    n_load_max: int
    omega_m: tuple[float, float, float]
    rationale: str


@dataclass(frozen=True)
class LoadConfig:
    p_idle_factor: float
    p_train_mean_factor: float
    p_train_peak_factor: float
    duty_cycle: float
    ramp_timescale_s: dict[str, float]
    jitter_pct: float
    p_device_rated_w: float | None

    def p_rated_w(self, scenario: Scenario) -> float | None:
        if self.p_device_rated_w is None:
            return None
        return scenario.n_load_max * self.p_device_rated_w

    def profile_w(self, scenario: Scenario) -> dict[str, float] | None:
        """Return scenario-scaled load powers, or None while rating is unknown."""
        p_rated = self.p_rated_w(scenario)
        if p_rated is None:
            return None
        return {
            "P_idle_W": self.p_idle_factor * p_rated,
            "P_train_mean_W": self.p_train_mean_factor * p_rated,
            "P_train_peak_W": self.p_train_peak_factor * p_rated,
            "P_rated_W": p_rated,
        }


@dataclass(frozen=True)
class BatteryConfig:
    chemistry: str
    v_nom_v: float
    p_shave_reserve_kwh: float | None
    c_rate_max: float
    c_rate_peak_10s: float
    t_op_min_c: float
    t_op_max_c: float
    cap_retention_t: dict[int, float]
    cycle_life_dod: dict[str, int]
    r_int_model: str
    r0_mohm: float
    alpha: float
    beta: float
    r_pulse_mohm: float | None
    eta_c_dis: float
    todos: tuple[str, ...]

    @property
    def e_rated_wh(self) -> float | None:
        if self.p_shave_reserve_kwh is None:
            return None
        return self.p_shave_reserve_kwh * 1000.0


@dataclass(frozen=True)
class FieldConfig:
    f_center_hz: float
    delta_f_hz: float
    c_l_uf: float
    topology: str
    compensation: str
    rx_struct: str
    cluster: str
    phase_patterns: tuple[str, ...]


@dataclass(frozen=True)
class Reference:
    f_center_hz: float
    delta_f_hz: float
    c_l_f: float
    efficiency_range: tuple[float, float]
    sample_scale: str
    license: str
    usage_rule: str


@dataclass(frozen=True)
class ReferenceExtrapolation:
    sample_power_range_w: tuple[float, float]
    target_power_w: float
    scale_factor_from_max_sample: float
    nonlinear_factors: tuple[str, ...]
    p2_requirement: str


@dataclass(frozen=True)
class Config:
    project_name: str
    layers: tuple[str, ...]
    power_flow: tuple[str, ...]
    scenario: Scenario
    load: LoadConfig
    battery: BatteryConfig
    field: FieldConfig
    refs: dict[str, Reference]
    reference_extrapolation: ReferenceExtrapolation
    numerical_methods: dict[str, str]
    module_paths: dict[str, str]
    unresolved_parameters: dict[str, str]
    global_requirements: tuple[str, ...]

    @classmethod
    def for_scenario(cls, scenario_id: str) -> "Config":
        scenarios = {
            "A_subzone": Scenario("A_subzone", 3_000.0, 4, (0.6, 0.6, 1.0), "可实现性优先"),
            "B_half_rack": Scenario("B_half_rack", 12_000.0, 10, (1.2, 0.6, 1.2), "主流机柜形态"),
            "C_full_rack": Scenario("C_full_rack", 30_000.0, 24, (1.2, 0.6, 2.0), "论证上限"),
        }
        if scenario_id not in scenarios:
            raise ValueError(f"Unknown scenario_id {scenario_id!r}; expected one of {tuple(scenarios)}")

        config = cls(
            project_name="sodium_aidc",
            layers=("L1_grid_interface", "L1_sodium_buffer", "L2_field_array", "L2_field_rx", "C1_arbiter"),
            power_flow=("grid", "L1_sodium_buffer", "dc_bus", "L2_field_array", "load"),
            scenario=scenarios[scenario_id],
            load=LoadConfig(
                p_idle_factor=0.55,
                p_train_mean_factor=0.78,
                p_train_peak_factor=0.98,
                duty_cycle=0.45,
                ramp_timescale_s={"epoch_switch": 90.0, "phase_sync": 8.0, "op_switch": 0.05},
                jitter_pct=0.08,
                p_device_rated_w=None,
            ),
            battery=BatteryConfig(
                chemistry="Na-ion",
                v_nom_v=3.0,
                p_shave_reserve_kwh=None,
                c_rate_max=4.0,
                c_rate_peak_10s=6.0,
                t_op_min_c=-20.0,
                t_op_max_c=55.0,
                cap_retention_t={-20: 0.92, 0: 0.97, 25: 1.0, 45: 0.985},
                cycle_life_dod={"10%": 6500, "20%": 5200, "40%": 3600, "80%": 2100, "100%": 1600},
                r_int_model="R0 * exp(-alpha * SOC) * (1 + beta * (T - 25)/10) + R_pulse",
                r0_mohm=12.0,
                alpha=0.35,
                beta=0.08,
                r_pulse_mohm=None,
                eta_c_dis=0.94,
                todos=(
                    "TODO[实测]: R0/alpha/beta are P0 placeholder fit parameters.",
                    "TODO[实测]: characterize pulse resistance for pulse widths below 30 s.",
                    "TODO[实测]: confirm 6C peak rating over 10 s.",
                ),
            ),
            field=FieldConfig(
                f_center_hz=200_000.0,
                delta_f_hz=2_000.0,
                c_l_uf=90.0,
                topology="half_bridge_8leg",
                compensation="LCC",
                rx_struct="flux_pipe",
                cluster="4x4",
                phase_patterns=("diamond", "cross"),
            ),
            refs={
                "Aalto_IEEE_TIE_2024": Reference(
                    f_center_hz=200_000.0,
                    delta_f_hz=2_000.0,
                    c_l_f=90e-6,
                    efficiency_range=(0.82, 0.93),
                    sample_scale="3x2 Tx (缩比), 20-40 W",
                    license="CC BY 4.0",
                    usage_rule="参数可引用、结论可复现、原文需标注出处",
                )
            },
            reference_extrapolation=ReferenceExtrapolation(
                sample_power_range_w=(20.0, 40.0),
                target_power_w=12_000.0,
                scale_factor_from_max_sample=300.0,
                nonlinear_factors=(
                    "coil losses (skin and proximity effects) vary with frequency and size",
                    "ferrite core saturation and temperature rise",
                    "parasitics and common-mode current grow with array scale",
                ),
                p2_requirement="Report an extrapolation method and uncertainty interval; do not give a single scaled value.",
            ),
            numerical_methods={
                "field_solver": "解析磁偶极 + 数值校验",
                "mutual_inductance": "数值积分（Duann 公式）",
                "parameter_scan": "全局网格 + 局部细化",
                "optimization": "序列二次规划 / 粒子群",
                "load_profile": "分段常数 + 一阶惯性；禁止纯随机噪声",
            },
            module_paths={
                "field_solver": "src/l2_field/solver.py",
                "array": "src/l2_field/array.py",
                "rx_model": "src/l2_field/rx_model.py",
                "inverter": "src/l2_field/inverter.py",
                "loss_model": "src/l2_field/loss.py",
                "soft_switch": "src/l2_field/zvs.py",
                "load_profile": "src/l1_buffer/load_profile.py",
                "soc_model": "src/l1_buffer/soc.py",
                "peak_shave": "src/l1_buffer/peak_shave.py",
                "arbitration": "src/c1_arbiter/arbiter.py",
                "semantic_recog": "src/c1_arbiter/semantic.py",
                "integration": "src/integration/",
                "tests": "tests/",
            },
            unresolved_parameters={
                "P_device_rated_W": "Not specified in P0; P_rated_W and derived load powers are unavailable.",
                "P_shave_reserve_kWh": "Not specified in P0; battery E_rated_Wh is unavailable.",
                "R_pulse": "Not specified in P0; pulse resistance remains unresolved.",
            },
            global_requirements=(
                "Label every numerical result with scenario ID P0-A/B/C.",
                "State scan ranges and steps for every claimed optimum.",
                "Collect all TODO[实测] items in the final report.",
                "Report conclusions separately for all three scenarios.",
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ValueError if known P0 contracts are inconsistent or out of range."""
        errors: list[str] = []
        if self.scenario.scenario_id not in {"A_subzone", "B_half_rack", "C_full_rack"}:
            errors.append("scenario_id must select exactly one supported scenario")
        if len(self.scenario.omega_m) != 3 or any(value <= 0 for value in self.scenario.omega_m):
            errors.append("Omega_m must contain three positive dimensions in metres")
        if self.scenario.p_field_target_w <= 0 or self.scenario.n_load_max <= 0:
            errors.append("scenario power and maximum load count must be positive")

        load_factors = (self.load.p_idle_factor, self.load.p_train_mean_factor, self.load.p_train_peak_factor)
        if any(not 0 <= value <= 1 for value in load_factors):
            errors.append("load power factors must be within [0, 1]")
        if not 0 < self.load.duty_cycle < 1:
            errors.append("duty_cycle must be within (0, 1)")
        if not 0 <= self.load.jitter_pct < 1:
            errors.append("jitter_pct must be within [0, 1)")
        if set(self.load.ramp_timescale_s) != {"epoch_switch", "phase_sync", "op_switch"} or any(
            value <= 0 for value in self.load.ramp_timescale_s.values()
        ):
            errors.append("load ramp timescales must be positive and include the three P0 events")

        if self.battery.chemistry != "Na-ion":
            errors.append("battery chemistry must be Na-ion for this P0 contract")
        if self.battery.v_nom_v <= 0 or self.battery.c_rate_max <= 0 or self.battery.c_rate_peak_10s <= 0:
            errors.append("battery voltage and C-rate values must be positive")
        if self.battery.t_op_min_c >= self.battery.t_op_max_c:
            errors.append("battery minimum operating temperature must be below maximum")
        if any(not 0 < value <= 1 for value in self.battery.cap_retention_t.values()):
            errors.append("capacity retention values must be within (0, 1]")
        expected_dod = {"10%", "20%", "40%", "80%", "100%"}
        if set(self.battery.cycle_life_dod) != expected_dod or any(
            value <= 0 for value in self.battery.cycle_life_dod.values()
        ):
            errors.append("cycle-life DoD keys/counts must match the P0 table")
        if not 0 < self.battery.eta_c_dis <= 1:
            errors.append("battery round-trip efficiency must be within (0, 1]")

        # Unit-bearing names and conversions follow the P0 contract: W, Hz, uF/F,
        # and kWh/Wh. Scalar storage prevents combining mutually exclusive choices.
        if self.field.f_center_hz <= 0 or not 0 < self.field.delta_f_hz < self.field.f_center_hz:
            errors.append("field frequencies must be positive and frequency separation below center frequency")
        if self.field.c_l_uf <= 0:
            errors.append("C_L must be positive in microfarads")
        if self.field.topology != "half_bridge_8leg" or self.field.compensation != "LCC":
            errors.append("field topology and compensation must use one supported P0 choice each")
        if self.field.rx_struct not in {"flux_pipe", "double_d"}:
            errors.append("rx_struct must be flux_pipe or double_d")
        if set(self.field.phase_patterns) != {"diamond", "cross"}:
            errors.append("phase patterns must include diamond and cross")
        for reference in self.refs.values():
            if reference.f_center_hz <= 0 or reference.delta_f_hz <= 0 or reference.c_l_f <= 0:
                errors.append("reference frequencies and capacitance must be positive (Hz, Hz, F)")
            low, high = reference.efficiency_range
            if not 0 < low <= high <= 1:
                errors.append("reference efficiency range must lie within (0, 1]")
            # 90 uF in FIELD and 90e-6 F in REFS are dimensionally equivalent.
            if abs(reference.c_l_f * 1e6 - self.field.c_l_uf) > 1e-9:
                errors.append("FIELD C_L (uF) and reference C_L (F) are inconsistent")

        if errors:
            raise ValueError("Invalid Config:\n- " + "\n- ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Return the selected scenario and its complete P0 configuration."""
        result = asdict(self)
        result["scenario"]["omega_m"] = list(self.scenario.omega_m)
        result["field"]["phase_patterns"] = list(self.field.phase_patterns)
        result["global_requirements"] = list(self.global_requirements)
        result["battery"]["todos"] = list(self.battery.todos)
        result["refs"] = {}
        for key, value in self.refs.items():
            reference = asdict(value)
            reference["efficiency_range"] = list(value.efficiency_range)
            result["refs"][key] = reference
        result["reference_extrapolation"] = asdict(self.reference_extrapolation)
        result["reference_extrapolation"]["sample_power_range_w"] = list(
            self.reference_extrapolation.sample_power_range_w
        )
        result["reference_extrapolation"]["nonlinear_factors"] = list(
            self.reference_extrapolation.nonlinear_factors
        )
        result["load"]["p_rated_w"] = self.load.p_rated_w(self.scenario)
        result["load"]["profile_w"] = self.load.profile_w(self.scenario)
        result["battery"]["e_rated_wh"] = self.battery.e_rated_wh
        return result

    def dump_schema(self) -> dict[str, Any]:
        """Return a JSON Schema (Draft 2020-12) describing the Config payload."""
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://sodium-aidc.local/config.schema.json",
            "title": "sodium_aidc Config",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "project_name", "layers", "power_flow", "scenario", "load", "battery",
                "field", "refs", "reference_extrapolation", "numerical_methods", "module_paths", "unresolved_parameters",
                "global_requirements",
            ],
            "properties": {
                "project_name": {"const": "sodium_aidc"},
                "layers": {"type": "array", "items": {"type": "string"}},
                "power_flow": {"type": "array", "items": {"type": "string"}},
                "scenario": {"$ref": "#/$defs/scenario"},
                "load": {"$ref": "#/$defs/load"},
                "battery": {"$ref": "#/$defs/battery"},
                "field": {"$ref": "#/$defs/field"},
                "refs": {"type": "object", "additionalProperties": {"$ref": "#/$defs/reference"}},
                "reference_extrapolation": {"$ref": "#/$defs/reference_extrapolation"},
                "numerical_methods": {"type": "object", "additionalProperties": {"type": "string"}},
                "module_paths": {"type": "object", "additionalProperties": {"type": "string"}},
                "unresolved_parameters": {"type": "object", "additionalProperties": {"type": "string"}},
                "global_requirements": {"type": "array", "items": {"type": "string"}},
            },
            "$defs": {
                "scenario": {"type": "object", "required": ["scenario_id", "p_field_target_w", "n_load_max", "omega_m", "rationale"], "properties": {
                    "scenario_id": {"enum": ["A_subzone", "B_half_rack", "C_full_rack"]},
                    "p_field_target_w": {"type": "number", "exclusiveMinimum": 0, "unit": "W"},
                    "n_load_max": {"type": "integer", "minimum": 1},
                    "omega_m": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0, "unit": "m"}, "minItems": 3, "maxItems": 3},
                    "rationale": {"type": "string"}}, "additionalProperties": False},
                "load": {"type": "object", "required": ["p_idle_factor", "p_train_mean_factor", "p_train_peak_factor", "duty_cycle", "ramp_timescale_s", "jitter_pct", "p_device_rated_w"], "properties": {
                    "p_idle_factor": {"type": "number", "minimum": 0, "maximum": 1},
                    "p_train_mean_factor": {"type": "number", "minimum": 0, "maximum": 1},
                    "p_train_peak_factor": {"type": "number", "minimum": 0, "maximum": 1},
                    "duty_cycle": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
                    "ramp_timescale_s": {"type": "object", "additionalProperties": {"type": "number", "exclusiveMinimum": 0, "unit": "s"}},
                    "jitter_pct": {"type": "number", "minimum": 0, "exclusiveMaximum": 1},
                    "p_device_rated_w": {"type": ["number", "null"], "exclusiveMinimum": 0, "unit": "W"},
                    "p_rated_w": {"type": ["number", "null"], "unit": "W"},
                    "profile_w": {"type": ["object", "null"]}}, "additionalProperties": False},
                "battery": {"type": "object", "required": ["chemistry", "v_nom_v", "p_shave_reserve_kwh", "c_rate_max", "c_rate_peak_10s", "t_op_min_c", "t_op_max_c", "cap_retention_t", "cycle_life_dod", "r_int_model", "r0_mohm", "alpha", "beta", "r_pulse_mohm", "eta_c_dis", "todos", "e_rated_wh"], "properties": {
                    "chemistry": {"const": "Na-ion"}, "v_nom_v": {"type": "number", "exclusiveMinimum": 0, "unit": "V"},
                    "p_shave_reserve_kwh": {"type": ["number", "null"], "minimum": 0, "unit": "kWh"},
                    "c_rate_max": {"type": "number", "exclusiveMinimum": 0}, "c_rate_peak_10s": {"type": "number", "exclusiveMinimum": 0},
                    "t_op_min_c": {"type": "number", "unit": "degC"}, "t_op_max_c": {"type": "number", "unit": "degC"},
                    "cap_retention_t": {"type": "object", "additionalProperties": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}},
                    "cycle_life_dod": {"type": "object", "required": ["10%", "20%", "40%", "80%", "100%"], "additionalProperties": {"type": "integer", "minimum": 1}},
                    "r_int_model": {"type": "string"}, "r0_mohm": {"type": "number", "exclusiveMinimum": 0, "unit": "mohm"},
                    "alpha": {"type": "number"}, "beta": {"type": "number"}, "r_pulse_mohm": {"type": ["number", "null"], "minimum": 0, "unit": "mohm"},
                    "eta_c_dis": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "todos": {"type": "array", "items": {"type": "string"}},
                    "e_rated_wh": {"type": ["number", "null"], "unit": "Wh"}}, "additionalProperties": False},
                "field": {"type": "object", "required": ["f_center_hz", "delta_f_hz", "c_l_uf", "topology", "compensation", "rx_struct", "cluster", "phase_patterns"], "properties": {
                    "f_center_hz": {"type": "number", "exclusiveMinimum": 0, "unit": "Hz"}, "delta_f_hz": {"type": "number", "exclusiveMinimum": 0, "unit": "Hz"},
                    "c_l_uf": {"type": "number", "exclusiveMinimum": 0, "unit": "uF"}, "topology": {"const": "half_bridge_8leg"},
                    "compensation": {"const": "LCC"}, "rx_struct": {"enum": ["flux_pipe", "double_d"]}, "cluster": {"type": "string"},
                    "phase_patterns": {"type": "array", "items": {"enum": ["diamond", "cross"]}, "uniqueItems": True}}, "additionalProperties": False},
                "reference": {"type": "object", "required": ["f_center_hz", "delta_f_hz", "c_l_f", "efficiency_range", "sample_scale", "license", "usage_rule"], "properties": {
                    "f_center_hz": {"type": "number", "exclusiveMinimum": 0, "unit": "Hz"}, "delta_f_hz": {"type": "number", "exclusiveMinimum": 0, "unit": "Hz"},
                    "c_l_f": {"type": "number", "exclusiveMinimum": 0, "unit": "F"}, "efficiency_range": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}, "minItems": 2, "maxItems": 2},
                    "sample_scale": {"type": "string"}, "license": {"type": "string"}, "usage_rule": {"type": "string"}}, "additionalProperties": False},
                "reference_extrapolation": {"type": "object", "required": ["sample_power_range_w", "target_power_w", "scale_factor_from_max_sample", "nonlinear_factors", "p2_requirement"], "properties": {
                    "sample_power_range_w": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0, "unit": "W"}, "minItems": 2, "maxItems": 2},
                    "target_power_w": {"type": "number", "exclusiveMinimum": 0, "unit": "W"},
                    "scale_factor_from_max_sample": {"type": "number", "exclusiveMinimum": 0},
                    "nonlinear_factors": {"type": "array", "items": {"type": "string"}},
                    "p2_requirement": {"type": "string"}}, "additionalProperties": False},
            },
        }
        return schema

    def describe(self) -> dict[str, Any]:
        """Provide the required module contract summary."""
        return {
            "inputs": {"scenario_id": ["A_subzone", "B_half_rack", "C_full_rack"]},
            "outputs": {"config": "Config", "schema": "JSON Schema Draft 2020-12"},
            "valid_ranges": {"duty_cycle": "(0, 1)", "efficiency": "(0, 1]", "frequencies_hz": "> 0"},
            "TODOs": list(self.battery.todos) + list(self.unresolved_parameters.values()),
        }
