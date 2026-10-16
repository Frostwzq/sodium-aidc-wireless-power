import csv
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.l2_field.optimize import optimize_all_scenarios


out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
result = optimize_all_scenarios()

summary = []
front_rows = []
for scenario in result["scenarios"]:
    summary.append({
        "scenario_id": scenario["scenario_id"],
        "global_grid_candidate_count": scenario["global_grid_candidate_count"],
        "local_refinement_candidate_count": scenario["local_refinement_candidate_count"],
        "pareto_front_size": scenario["pareto_front_size"],
        "selection_strategy": scenario["selection_strategy"],
        "selected_compromise": scenario["selected_compromise"],
        "neighbor_sensitivity_5pct": scenario["neighbor_sensitivity_5pct"],
        "global_to_fine_movement_pct": scenario["global_to_fine_movement_pct"],
        "max_fine_step_movement_pct": scenario["max_fine_step_movement_pct"],
        "step_adequacy": scenario["step_adequacy"],
        "deadtime_diagnostic": scenario["deadtime_diagnostic"],
        "geometry": scenario["geometry"],
        "pareto_front": scenario["pareto_front"],
        "assumptions": scenario["assumptions"],
        "TODOs": scenario["TODOs"],
    })
    for rank, point in enumerate(scenario["pareto_front"], start=1):
        front_rows.append({
            "scenario_id": scenario["scenario_id"],
            "pareto_rank_by_loss": rank,
            "f1_total_loss_w": point["f1_total_loss_w"],
            "f2_max_delta_f_eff": point["f2_max_delta_f_eff"],
            "f3_uniformity_ratio": point["f3_uniformity_ratio"],
            "f4_geometry_violation_m": point["f4_geometry_violation_m"],
            **point["parameters"],
        })

(out_dir / "p7_optimization_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
with (out_dir / "p7_pareto_front.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(front_rows[0]))
    writer.writeheader()
    writer.writerows(front_rows)

print(json.dumps([
    {
        "scenario_id": item["scenario_id"],
        "global_grid_candidate_count": item["global_grid_candidate_count"],
        "local_refinement_candidate_count": item["local_refinement_candidate_count"],
        "pareto_front_size": item["pareto_front_size"],
        "selected_compromise": item["selected_compromise"],
        "neighbor_sensitivity_5pct": item["neighbor_sensitivity_5pct"],
        "global_to_fine_movement_pct": item["global_to_fine_movement_pct"],
        "max_fine_step_movement_pct": item["max_fine_step_movement_pct"],
        "step_adequacy": item["step_adequacy"],
        "deadtime_diagnostic": item["deadtime_diagnostic"],
        "front": [{
            "f1_total_loss_w": row["f1_total_loss_w"],
            "f2_max_delta_f_eff": row["f2_max_delta_f_eff"],
            "f3_uniformity_ratio": row["f3_uniformity_ratio"],
            "parameters": row["parameters"],
        } for row in item["pareto_front"]],
    } for item in summary
], ensure_ascii=False))
