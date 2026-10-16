import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.l1_buffer.peak_shave import optimize_all_scenarios


result = optimize_all_scenarios()
summary = []
for scenario in result["scenarios"]:
    load = scenario["load_profile"]
    comparison = scenario["strategy_comparison"]
    conservative_feedback = scenario["uncertainty_cases"]["feedback"]["conservative"]
    optimistic_feedback = scenario["uncertainty_cases"]["feedback"]["optimistic"]
    conservative = scenario["uncertainty_cases"]["forecast"]["conservative"]
    optimistic = scenario["uncertainty_cases"]["forecast"]["optimistic"]
    feedback = comparison["feedback"]
    forecast = comparison["forecast"]
    summary.append({
        "scenario_id": scenario["scenario_id"],
        "duty_cycle_assumed": load["duty_cycle_assumed"],
        "duty_cycle_actual": load["duty_cycle_actual"],
        "duty_cycle_absolute_difference": load["duty_cycle_actual"] - load["duty_cycle_assumed"],
        "duty_cycle_relative_deviation_pct": 100.0 * load["duty_cycle_relative_deviation"],
        "duty_cycle_warning": load["duty_cycle_warning"],
        "eta_shave_intervals": scenario["eta_shave_intervals"],
        "nominal_strategy_eta": {
            "no_storage": comparison["no_storage"]["eta_shave"],
            "feedback": feedback["eta_shave"],
            "forecast": forecast["eta_shave"],
        },
        "forecast_lift_percentage_points_nominal": 100.0 * (forecast["eta_shave"] - feedback["eta_shave"]),
        "forecast_lift_percentage_points_interval": [
            100.0 * (conservative["eta_shave"] - conservative_feedback["eta_shave"]),
            100.0 * (optimistic["eta_shave"] - optimistic_feedback["eta_shave"]),
        ],
        "feedback_parameters": feedback["parameters"],
        "forecast_parameters": forecast["parameters"],
        "feedback_life_consumed_fraction": feedback["life_consumed_fraction"],
        "forecast_life_consumed_fraction": forecast["life_consumed_fraction"],
        "lifetime_budget_fraction": scenario["storage_assumptions"]["lifetime_budget_fraction_per_profile"],
        "rate_limit_kW": scenario["storage_assumptions"]["P_rate_C_nom_kW"],
        "below_10pct_warning": scenario["below_10pct_warning"],
        "eta_shave_target_25pct": scenario["eta_shave_target_25pct"],
        "target_25pct_warning": scenario["eta_shave_target_25pct"]["warning"],
    })

out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
(out_dir / "p8_strategy_comparison.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))
