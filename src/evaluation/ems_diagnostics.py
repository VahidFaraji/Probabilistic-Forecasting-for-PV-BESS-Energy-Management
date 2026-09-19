import numpy as np
import pandas as pd


QUANTILE_COLUMNS = ["q_0.10", "q_0.25", "q_0.50", "q_0.75", "q_0.90"]

FORECAST_COLUMNS = {
    "unique_id", "ds", "cutoff", "y", *QUANTILE_COLUMNS,
}

SCHEDULE_COLUMNS = {
    "objective",
    "strategy",
    "unique_id",
    "ds",
    "cutoff",
    "charge_power_kw",
    "discharge_power_kw",
    "grid_energy_kwh",
    "grid_import_kwh",
    "grid_export_kwh",
    "grid_cost",
    "soc_end",
    "planned_grid_cost",
    "objective_value_component",
    "planned_objective_value_component",
}


def validate_customer_timeseries(
    data: pd.DataFrame,
    time_step_minutes: int = 30,
    tolerance: float = 1e-9,
) -> None:
    """Validate timestamps and load-PV-net-load consistency."""
    required = {"unique_id", "ds", "load", "pv", "net_load"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing time-series columns: {sorted(missing)}")

    frame = data[list(required)].copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    if frame.isna().any().any():
        raise ValueError("Customer time series cannot contain missing values.")
    if frame.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Customer timestamps must be unique within each series.")
    if not np.isfinite(frame[["load", "pv", "net_load"]].to_numpy()).all():
        raise ValueError("Load, PV, and net load must be finite.")
    if not np.allclose(
        frame["net_load"],
        frame["load"] - frame["pv"],
        atol=tolerance,
    ):
        raise ValueError("Net load must equal load minus PV.")

    expected_gap = pd.to_timedelta(int(time_step_minutes), unit="min")
    for _, series in frame.sort_values("ds").groupby("unique_id"):
        gaps = series["ds"].diff().dropna()
        if not (gaps == expected_gap).all():
            raise ValueError("Customer timestamps must follow the expected frequency.")


def rearrange_quantiles(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Sort predicted quantiles row-wise while preserving their levels."""
    missing = set(QUANTILE_COLUMNS).difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    result = forecasts.copy()
    values = result[QUANTILE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Quantile forecasts must be finite.")
    result[QUANTILE_COLUMNS] = np.sort(values, axis=1)
    return result


def calculate_daily_diagnostics(
    forecasts: pd.DataFrame,
    schedules: pd.DataFrame,
    time_step_hours: float = 0.5,
    import_price: float = 0.30,
    export_price: float = 0.08,
) -> pd.DataFrame:
    """Calculate daily forecast and realized EMS metrics."""
    forecasts = _prepare_frame(forecasts, FORECAST_COLUMNS, "forecast")
    schedules = _prepare_frame(schedules, SCHEDULE_COLUMNS, "schedule")

    forecast_metrics = _daily_forecast_metrics(
        forecasts,
        time_step_hours,
        import_price,
        export_price,
    )
    operational_metrics = _daily_operational_metrics(schedules, time_step_hours)

    daily = operational_metrics.merge(
        forecast_metrics,
        on=["unique_id", "cutoff"],
        how="left",
        validate="many_to_one",
    )
    if daily["forecast_mae_kwh"].isna().any():
        raise ValueError("Some EMS windows do not match the forecast test data.")

    perfect = (
        daily[daily["strategy"] == "perfect_foresight"]
        [[
            "objective",
            "unique_id",
            "cutoff",
            "realized_cost",
            "realized_peak_kw",
            "realized_objective",
        ]]
        .rename(columns={
            "realized_cost": "perfect_foresight_cost",
            "realized_peak_kw": "perfect_foresight_peak_kw",
            "realized_objective": "perfect_foresight_objective",
        })
    )
    daily = daily.merge(
        perfect,
        on=["objective", "unique_id", "cutoff"],
        how="left",
        validate="many_to_one",
    )
    if daily["perfect_foresight_cost"].isna().any():
        raise ValueError("Every objective must include perfect-foresight results.")

    daily["planned_realized_gap"] = daily["realized_cost"] - daily["planned_cost"]
    daily["cost_regret"] = daily["realized_cost"] - daily["perfect_foresight_cost"]
    daily["peak_regret_kw"] = (
        daily["realized_peak_kw"] - daily["perfect_foresight_peak_kw"]
    )
    daily["objective_regret"] = (
        daily["realized_objective"] - daily["perfect_foresight_objective"]
    )
    daily["cost_saving_vs_no_battery"] = (
        daily["actual_no_battery_cost"] - daily["realized_cost"]
    )
    daily["cost_saving_pct"] = np.where(
        daily["actual_no_battery_cost"] != 0,
        100 * daily["cost_saving_vs_no_battery"] / daily["actual_no_battery_cost"],
        np.nan,
    )
    daily["peak_reduction_vs_no_battery_kw"] = (
        daily["actual_peak_kw"] - daily["realized_peak_kw"]
    )
    daily["peak_reduction_pct"] = np.where(
        daily["actual_peak_kw"] > 0,
        100
        * daily["peak_reduction_vs_no_battery_kw"]
        / daily["actual_peak_kw"],
        np.nan,
    )
    daily["forecast_baseline_cost_error"] = (
        daily["actual_no_battery_cost"] - daily["forecast_no_battery_cost"]
    )
    return daily.sort_values(
        ["objective", "strategy", "unique_id", "cutoff"]
    ).reset_index(drop=True)


def calculate_relationship_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize forecast-to-operation relationships."""
    forecast_strategy = _preferred_forecast_strategy(daily)
    definitions = [
        ("forecast_mae_vs_cost_regret", "cost_only", "forecast_mae_kwh", "cost_regret"),
        ("forecast_bias_vs_planning_gap", "cost_only", "forecast_bias_kwh", "planned_realized_gap"),
        (
            "peak_timing_error_vs_peak_regret",
            "peak_only",
            "circular_peak_timing_error_hours",
            "peak_regret_kw",
        ),
    ]

    rows = []
    for name, objective, x_column, y_column in definitions:
        data = daily[
            (daily["objective"] == objective)
            & (daily["strategy"] == forecast_strategy)
        ][[x_column, y_column]].dropna()

        correlation = np.nan
        if len(data) >= 2 and data[x_column].nunique() > 1 and data[y_column].nunique() > 1:
            correlation = data[x_column].corr(data[y_column])

        rows.append({
            "relationship": name,
            "x_metric": x_column,
            "y_metric": y_column,
            "days": len(data),
            "pearson_correlation": correlation,
        })
    return pd.DataFrame(rows)


def calculate_period_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize forecast and EMS performance over the complete test period."""
    required = {
        "objective",
        "strategy",
        "realized_cost",
        "planned_cost",
        "realized_objective",
        "planned_objective",
        "cost_regret",
        "objective_regret",
        "actual_no_battery_cost",
        "actual_peak_kw",
        "realized_peak_kw",
        "grid_import_kwh",
        "grid_export_kwh",
        "charged_energy_kwh",
        "discharged_energy_kwh",
        "battery_throughput_kwh",
        "forecast_mae_kwh",
        "forecast_bias_kwh",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing daily diagnostic columns: {sorted(missing)}")

    rows = []
    for (objective, strategy), data in daily.groupby(
        ["objective", "strategy"], sort=True
    ):
        no_battery_cost = float(data["actual_no_battery_cost"].sum())
        realized_cost = float(data["realized_cost"].sum())
        rows.append({
            "objective": objective,
            "strategy": strategy,
            "days": int(data["cutoff"].nunique()),
            "no_battery_cost": no_battery_cost,
            "realized_cost": realized_cost,
            "planned_cost": float(data["planned_cost"].sum()),
            "realized_objective": float(data["realized_objective"].sum()),
            "planned_objective": float(data["planned_objective"].sum()),
            "cost_saving": no_battery_cost - realized_cost,
            "cost_saving_pct": (
                100 * (no_battery_cost - realized_cost) / no_battery_cost
                if no_battery_cost != 0 else np.nan
            ),
            "cost_regret": float(data["cost_regret"].sum()),
            "objective_regret": float(data["objective_regret"].sum()),
            "mean_no_battery_peak_kw": float(data["actual_peak_kw"].mean()),
            "mean_realized_peak_kw": float(data["realized_peak_kw"].mean()),
            "maximum_realized_peak_kw": float(data["realized_peak_kw"].max()),
            "mean_peak_reduction_kw": float(
                (data["actual_peak_kw"] - data["realized_peak_kw"]).mean()
            ),
            "grid_import_kwh": float(data["grid_import_kwh"].sum()),
            "grid_export_kwh": float(data["grid_export_kwh"].sum()),
            "charged_energy_kwh": float(data["charged_energy_kwh"].sum()),
            "discharged_energy_kwh": float(data["discharged_energy_kwh"].sum()),
            "battery_throughput_kwh": float(data["battery_throughput_kwh"].sum()),
            "mean_forecast_mae_kwh": float(data["forecast_mae_kwh"].mean()),
            "mean_forecast_bias_kwh": float(data["forecast_bias_kwh"].mean()),
        })
    return pd.DataFrame(rows)


def select_diagnostic_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Select best, representative, and worst forecast and EMS days."""
    forecast_strategy = _preferred_forecast_strategy(daily)
    median = daily[daily["strategy"] == forecast_strategy].copy()
    cost = median[median["objective"] == "cost_only"].sort_values("cutoff")
    peak = median[median["objective"] == "peak_only"].sort_values("cutoff")
    if cost.empty or peak.empty:
        raise ValueError("Daily results must include median cost_only and peak_only rows.")

    selections = []
    selections.extend(_select_cases(
        cost,
        analysis="forecast",
        metric="forecast_mae_kwh",
        best="min",
        worst="max",
    ))
    selections.extend(_select_cases(
        cost,
        analysis="cost",
        metric="cost_saving_vs_no_battery",
        best="max",
        worst="min",
    ))
    selections.extend(_select_cases(
        peak,
        analysis="peak",
        metric="peak_reduction_vs_no_battery_kw",
        best="max",
        worst="min",
    ))
    return pd.DataFrame(selections)


def compare_daily_strategies(
    daily: pd.DataFrame,
    candidate: str = "five_scenario",
    references: tuple[str, ...] = ("q50_forecast", "q75_forecast"),
) -> pd.DataFrame:
    """Compare one EMS strategy with reference strategies day by day."""
    required = {
        "objective", "strategy", "unique_id", "cutoff",
        "realized_cost", "realized_peak_kw", "realized_objective",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing daily comparison columns: {sorted(missing)}")

    candidate_data = daily[daily["strategy"] == candidate][
        [
            "objective", "unique_id", "cutoff", "realized_cost",
            "realized_peak_kw", "realized_objective",
        ]
    ].rename(columns={
        "realized_cost": "candidate_cost",
        "realized_peak_kw": "candidate_peak_kw",
        "realized_objective": "candidate_objective",
    })
    if candidate_data.empty:
        raise ValueError(f"Candidate strategy not found: {candidate}.")

    rows = []
    keys = ["objective", "unique_id", "cutoff"]
    for reference in references:
        reference_data = daily[daily["strategy"] == reference][
            [*keys, "realized_cost", "realized_peak_kw", "realized_objective"]
        ].rename(columns={
            "realized_cost": "reference_cost",
            "realized_peak_kw": "reference_peak_kw",
            "realized_objective": "reference_objective",
        })
        if reference_data.empty:
            raise ValueError(f"Reference strategy not found: {reference}.")

        paired = candidate_data.merge(
            reference_data,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        expected = len(candidate_data)
        if len(paired) != expected:
            raise ValueError(
                f"Strategy {reference} does not cover every candidate day."
            )

        paired["objective_improvement"] = (
            paired["reference_objective"] - paired["candidate_objective"]
        )
        paired["cost_improvement"] = (
            paired["reference_cost"] - paired["candidate_cost"]
        )
        paired["peak_improvement_kw"] = (
            paired["reference_peak_kw"] - paired["candidate_peak_kw"]
        )

        for objective, data in paired.groupby("objective", sort=True):
            improvement = data["objective_improvement"]
            tolerance = 1e-9
            wins = int((improvement > tolerance).sum())
            ties = int((improvement.abs() <= tolerance).sum())
            losses = int((improvement < -tolerance).sum())
            rows.append({
                "objective": objective,
                "candidate": candidate,
                "reference": reference,
                "days": len(data),
                "candidate_better_days": wins,
                "equal_days": ties,
                "candidate_worse_days": losses,
                "win_rate_pct": 100 * wins / len(data),
                "mean_objective_improvement": float(improvement.mean()),
                "median_objective_improvement": float(improvement.median()),
                "total_objective_improvement": float(improvement.sum()),
                "mean_cost_improvement": float(data["cost_improvement"].mean()),
                "mean_peak_improvement_kw": float(
                    data["peak_improvement_kw"].mean()
                ),
                "best_daily_objective_improvement": float(improvement.max()),
                "worst_daily_objective_improvement": float(improvement.min()),
            })
    return pd.DataFrame(rows)


def create_simple_stochastic_benchmark(
    daily: pd.DataFrame,
    comparisons: pd.DataFrame,
    probabilities: tuple[float, ...] = (0.175, 0.20, 0.25, 0.20, 0.175),
) -> pd.DataFrame:
    """Create a compact record of the simple five-scenario benchmark."""
    candidate = daily[daily["strategy"] == "five_scenario"].copy()
    if candidate.empty:
        raise ValueError("The five_scenario strategy is required.")
    if not np.isclose(sum(probabilities), 1.0):
        raise ValueError("Scenario probabilities must sum to one.")

    rows = []
    for objective, data in candidate.groupby("objective", sort=True):
        objective_comparison = comparisons[
            comparisons["objective"] == objective
        ]
        q50 = objective_comparison[
            objective_comparison["reference"] == "q50_forecast"
        ]
        q75 = objective_comparison[
            objective_comparison["reference"] == "q75_forecast"
        ]
        if len(q50) != 1 or len(q75) != 1:
            raise ValueError("Each objective must include q50 and q75 comparisons.")

        rows.append({
            "benchmark": "customer_001_simple_stochastic_v1",
            "unique_id": data["unique_id"].iloc[0],
            "objective": objective,
            "strategy": "five_scenario",
            "test_start": data["cutoff"].min(),
            "test_end": data["cutoff"].max(),
            "days": data["cutoff"].nunique(),
            "scenario_count": len(probabilities),
            "scenario_method": "direct_quantile_trajectories",
            "probability_method": "fixed_midpoint_masses",
            "scenario_probabilities": ",".join(map(str, probabilities)),
            "realized_cost": float(data["realized_cost"].sum()),
            "realized_objective": float(data["realized_objective"].sum()),
            "mean_daily_peak_kw": float(data["realized_peak_kw"].mean()),
            "objective_regret": float(data["objective_regret"].sum()),
            "q50_win_rate_pct": float(q50.iloc[0]["win_rate_pct"]),
            "q75_win_rate_pct": float(q75.iloc[0]["win_rate_pct"]),
            "q50_total_objective_improvement": float(
                q50.iloc[0]["total_objective_improvement"]
            ),
            "q75_total_objective_improvement": float(
                q75.iloc[0]["total_objective_improvement"]
            ),
        })
    return pd.DataFrame(rows)


def _preferred_forecast_strategy(daily: pd.DataFrame) -> str:
    strategies = set(daily["strategy"])
    if "q50_forecast" in strategies:
        return "q50_forecast"
    if "median_forecast" in strategies:
        return "median_forecast"
    raise ValueError("Daily results must include q50_forecast or median_forecast.")


def _select_cases(
    data: pd.DataFrame,
    analysis: str,
    metric: str,
    best: str,
    worst: str,
) -> list[dict]:
    valid = data.dropna(subset=[metric]).copy()
    if valid.empty:
        raise ValueError(f"Cannot select diagnostic days from empty metric: {metric}.")

    best_index = valid[metric].idxmin() if best == "min" else valid[metric].idxmax()
    worst_index = valid[metric].idxmin() if worst == "min" else valid[metric].idxmax()
    median_value = valid[metric].median()
    representative_index = (valid[metric] - median_value).abs().idxmin()

    rows = []
    for case, index in (
        ("best", best_index),
        ("representative", representative_index),
        ("worst", worst_index),
    ):
        row = valid.loc[index]
        rows.append({
            "analysis": analysis,
            "case": case,
            "metric": metric,
            "metric_value": float(row[metric]),
            "cutoff": row["cutoff"],
            "date": row["date"],
        })
    return rows


def _prepare_frame(
    data: pd.DataFrame,
    required_columns: set[str],
    name: str,
) -> pd.DataFrame:
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    prepared = data.copy()
    prepared["ds"] = pd.to_datetime(prepared["ds"])
    prepared["cutoff"] = pd.to_datetime(prepared["cutoff"])
    return prepared.sort_values(["unique_id", "cutoff", "ds"]).reset_index(drop=True)


def _daily_forecast_metrics(
    forecasts: pd.DataFrame,
    time_step_hours: float,
    import_price: float,
    export_price: float,
) -> pd.DataFrame:
    rows = []
    for (unique_id, cutoff), raw in forecasts.groupby(
        ["unique_id", "cutoff"], sort=True
    ):
        data = rearrange_quantiles(raw)
        actual = data["y"].to_numpy(dtype=float)
        median = data["q_0.50"].to_numpy(dtype=float)
        error = actual - median

        actual_peak_time = data.loc[data["y"].idxmax(), "ds"]
        forecast_peak_time = data.loc[data["q_0.50"].idxmax(), "ds"]
        timing_error = (
            forecast_peak_time - actual_peak_time
        ).total_seconds() / 3600
        circular_error = min(abs(timing_error) % 24, 24 - (abs(timing_error) % 24))

        raw_quantiles = raw[QUANTILE_COLUMNS].to_numpy(dtype=float)
        rows.append({
            "unique_id": unique_id,
            "cutoff": cutoff,
            "date": data["ds"].min().date(),
            "intervals": len(data),
            "forecast_mae_kwh": float(np.mean(np.abs(error))),
            "forecast_rmse_kwh": float(np.sqrt(np.mean(error**2))),
            "forecast_bias_kwh": float(np.mean(error)),
            "actual_peak_kwh": float(actual.max()),
            "forecast_peak_kwh": float(median.max()),
            "peak_value_error_kwh": float(median.max() - actual.max()),
            "actual_peak_kw": float(max(actual.max(), 0) / time_step_hours),
            "forecast_peak_kw": float(max(median.max(), 0) / time_step_hours),
            "actual_peak_time": actual_peak_time,
            "forecast_peak_time": forecast_peak_time,
            "peak_timing_error_hours": timing_error,
            "absolute_peak_timing_error_hours": abs(timing_error),
            "circular_peak_timing_error_hours": circular_error,
            "coverage_50": float(
                ((data["y"] >= data["q_0.25"]) & (data["y"] <= data["q_0.75"])).mean()
            ),
            "coverage_80": float(
                ((data["y"] >= data["q_0.10"]) & (data["y"] <= data["q_0.90"])).mean()
            ),
            "quantile_crossing_rate": float(
                (np.diff(raw_quantiles, axis=1) < 0).any(axis=1).mean()
            ),
            "actual_no_battery_cost": _grid_cost(actual, import_price, export_price),
            "forecast_no_battery_cost": _grid_cost(median, import_price, export_price),
        })
    return pd.DataFrame(rows)


def _daily_operational_metrics(
    schedules: pd.DataFrame,
    time_step_hours: float,
) -> pd.DataFrame:
    rows = []
    keys = ["objective", "strategy", "unique_id", "cutoff"]
    for key, data in schedules.groupby(keys, sort=True):
        objective, strategy, unique_id, cutoff = key
        balance_error = data["grid_energy_kwh"] - (
            data["grid_import_kwh"] - data["grid_export_kwh"]
        )
        rows.append({
            "objective": objective,
            "strategy": strategy,
            "unique_id": unique_id,
            "cutoff": cutoff,
            "realized_cost": float(data["grid_cost"].sum()),
            "planned_cost": float(data["planned_grid_cost"].sum()),
            "realized_objective": float(data["objective_value_component"].sum()),
            "planned_objective": float(data["planned_objective_value_component"].sum()),
            "realized_peak_kw": float(data["grid_import_kwh"].max() / time_step_hours),
            "grid_import_kwh": float(data["grid_import_kwh"].sum()),
            "grid_export_kwh": float(data["grid_export_kwh"].sum()),
            "charged_energy_kwh": float((data["charge_power_kw"] * time_step_hours).sum()),
            "discharged_energy_kwh": float((data["discharge_power_kw"] * time_step_hours).sum()),
            "battery_throughput_kwh": float(
                (data["charge_power_kw"] + data["discharge_power_kw"]).sum()
                * time_step_hours
            ),
            "minimum_soc": float(data["soc_end"].min()),
            "maximum_soc": float(data["soc_end"].max()),
            "terminal_soc": float(data.iloc[-1]["soc_end"]),
            "maximum_grid_balance_error_kwh": float(balance_error.abs().max()),
        })
    return pd.DataFrame(rows)


def _grid_cost(
    net_load_kwh: np.ndarray,
    import_price: float,
    export_price: float,
) -> float:
    grid_import = np.maximum(net_load_kwh, 0)
    grid_export = np.maximum(-net_load_kwh, 0)
    return float(import_price * grid_import.sum() - export_price * grid_export.sum())
