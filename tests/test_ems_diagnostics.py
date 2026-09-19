import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.evaluation.ems_diagnostics import (
    calculate_daily_diagnostics,
    calculate_period_summary,
    calculate_relationship_summary,
    compare_daily_strategies,
    create_simple_stochastic_benchmark,
    rearrange_quantiles,
    select_diagnostic_days,
    validate_customer_timeseries,
)
from src.visualization.ems_plots import (
    plot_battery_day,
    plot_battery_period,
    plot_daily_diagnostics,
    plot_daily_strategy_comparison,
    plot_forecast_day,
)


def make_forecasts() -> pd.DataFrame:
    timestamps = pd.date_range("2013-06-01 00:30", periods=8, freq="30min")
    actual = np.array([-1.0, -0.5, 1.0, 2.0] * 2)
    median = actual + np.array([0.1, -0.1, 0.2, -0.2] * 2)
    return pd.DataFrame({
        "unique_id": "customer_001",
        "ds": timestamps,
        "cutoff": [pd.Timestamp("2013-06-01 00:00")] * 4
        + [pd.Timestamp("2013-06-01 02:00")] * 4,
        "y": actual,
        "q_0.10": median - 0.8,
        "q_0.25": median - 0.4,
        "q_0.50": median,
        "q_0.75": median + 0.4,
        "q_0.90": median + 0.8,
    })


def make_schedules(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for objective in ("cost_only", "peak_only", "combined"):
        for strategy in ("perfect_foresight", "median_forecast"):
            for _, row in forecasts.iterrows():
                planned = row["y"] if strategy == "perfect_foresight" else row["q_0.50"]
                actual = row["y"]
                realized_cost = 0.30 * max(actual, 0) - 0.08 * max(-actual, 0)
                planned_cost = 0.30 * max(planned, 0) - 0.08 * max(-planned, 0)
                rows.append({
                    "objective": objective,
                    "strategy": strategy,
                    "unique_id": row["unique_id"],
                    "ds": row["ds"],
                    "cutoff": row["cutoff"],
                    "charge_power_kw": 0.0,
                    "discharge_power_kw": 0.0,
                    "grid_energy_kwh": actual,
                    "grid_import_kwh": max(actual, 0),
                    "grid_export_kwh": max(-actual, 0),
                    "grid_cost": realized_cost,
                    "soc_end": 0.50,
                    "planned_grid_cost": planned_cost,
                    "objective_value_component": realized_cost,
                    "planned_objective_value_component": planned_cost,
                })
    return pd.DataFrame(rows)


def make_strategy_daily() -> pd.DataFrame:
    rows = []
    objectives = ("cost_only", "peak_only", "combined")
    strategies = {
        "q50_forecast": [10.0, 12.0],
        "q75_forecast": [11.0, 11.0],
        "five_scenario": [9.0, 11.5],
    }
    for objective in objectives:
        for strategy, values in strategies.items():
            for day, objective_value in enumerate(values):
                rows.append({
                    "objective": objective,
                    "strategy": strategy,
                    "unique_id": "customer_001",
                    "cutoff": pd.Timestamp("2013-06-01") + pd.Timedelta(days=day),
                    "date": (pd.Timestamp("2013-06-01") + pd.Timedelta(days=day)).date(),
                    "realized_cost": objective_value,
                    "realized_peak_kw": objective_value / 5,
                    "realized_objective": objective_value,
                    "objective_regret": objective_value - 8.0,
                })
    return pd.DataFrame(rows)


def test_customer_timeseries_structure_and_net_load_are_validated():
    ds = pd.date_range("2012-07-01 00:30", periods=4, freq="30min")
    data = pd.DataFrame({
        "unique_id": "customer_001",
        "ds": ds,
        "load": [1.0, 1.2, 0.8, 0.6],
        "pv": [0.0, 0.1, 0.4, 0.5],
    })
    data["net_load"] = data["load"] - data["pv"]
    validate_customer_timeseries(data)
    data.loc[2, "net_load"] += 0.1
    with pytest.raises(ValueError, match="load minus PV"):
        validate_customer_timeseries(data)


def test_quantile_rearrangement_removes_crossing():
    forecasts = make_forecasts()
    forecasts.loc[0, ["q_0.10", "q_0.90"]] = [2.0, -2.0]
    fixed = rearrange_quantiles(forecasts)
    values = fixed[["q_0.10", "q_0.25", "q_0.50", "q_0.75", "q_0.90"]].to_numpy()
    assert (np.diff(values, axis=1) >= 0).all()


def test_daily_diagnostics_and_objective_regret():
    forecasts = make_forecasts()
    daily = calculate_daily_diagnostics(forecasts, make_schedules(forecasts))
    assert len(daily) == 12
    perfect = daily[daily["strategy"] == "perfect_foresight"]
    assert np.allclose(perfect["cost_regret"], 0)
    assert np.allclose(perfect["peak_regret_kw"], 0)
    assert np.allclose(perfect["objective_regret"], 0)


def test_circular_peak_timing_error_handles_day_boundary():
    timestamps = pd.date_range("2013-06-01 00:30", periods=48, freq="30min")
    actual = np.zeros(48)
    median = np.zeros(48)
    actual[0] = 3.0
    median[-1] = 3.0
    forecasts = pd.DataFrame({
        "unique_id": "customer_001",
        "ds": timestamps,
        "cutoff": pd.Timestamp("2013-06-01 00:00"),
        "y": actual,
        "q_0.10": median - 0.8,
        "q_0.25": median - 0.4,
        "q_0.50": median,
        "q_0.75": median + 0.4,
        "q_0.90": median + 0.8,
    })
    daily = calculate_daily_diagnostics(forecasts, make_schedules(forecasts))
    assert np.allclose(daily["absolute_peak_timing_error_hours"], 23.5)
    assert np.allclose(daily["circular_peak_timing_error_hours"], 0.5)


def test_relationship_summary_contains_three_checks():
    forecasts = make_forecasts()
    daily = calculate_daily_diagnostics(forecasts, make_schedules(forecasts))
    summary = calculate_relationship_summary(daily)
    assert len(summary) == 3
    assert (summary["days"] == 2).all()


def test_period_summary_covers_each_objective_and_strategy():
    forecasts = make_forecasts()
    daily = calculate_daily_diagnostics(forecasts, make_schedules(forecasts))
    summary = calculate_period_summary(daily)
    assert len(summary) == 6
    assert set(summary["objective"]) == {"cost_only", "peak_only", "combined"}
    assert set(summary["strategy"]) == {"perfect_foresight", "median_forecast"}


def test_selected_days_include_best_representative_and_worst_cases():
    forecasts = make_forecasts()
    daily = calculate_daily_diagnostics(forecasts, make_schedules(forecasts))
    selected = select_diagnostic_days(daily)
    assert len(selected) == 9
    assert set(selected["analysis"]) == {"forecast", "cost", "peak"}
    assert set(selected["case"]) == {"best", "representative", "worst"}


def test_plot_functions_return_figures():
    forecasts = make_forecasts()
    schedules = make_schedules(forecasts)
    daily = calculate_daily_diagnostics(forecasts, schedules)
    cutoff = forecasts.iloc[0]["cutoff"]
    figures = [
        plot_forecast_day(forecasts, cutoff),
        plot_battery_day(schedules, cutoff, "cost_only"),
        plot_battery_period(schedules, "cost_only"),
        plot_daily_diagnostics(daily),
    ]
    assert all(figure.axes for figure in figures)


def test_missing_forecast_column_is_rejected():
    forecasts = make_forecasts().drop(columns="q_0.90")
    with pytest.raises(ValueError, match="Missing forecast columns"):
        calculate_daily_diagnostics(forecasts, make_schedules(make_forecasts()))


def test_daily_strategy_comparison_counts_wins_and_losses():
    comparison = compare_daily_strategies(make_strategy_daily())
    assert len(comparison) == 6
    q50 = comparison[
        (comparison["objective"] == "cost_only")
        & (comparison["reference"] == "q50_forecast")
    ].iloc[0]
    assert q50["candidate_better_days"] == 2
    assert q50["win_rate_pct"] == pytest.approx(100.0)
    assert q50["total_objective_improvement"] == pytest.approx(1.5)


def test_simple_stochastic_benchmark_is_registered_for_each_objective():
    daily = make_strategy_daily()
    comparison = compare_daily_strategies(daily)
    benchmark = create_simple_stochastic_benchmark(daily, comparison)
    assert len(benchmark) == 3
    assert set(benchmark["objective"]) == {"cost_only", "peak_only", "combined"}
    assert set(benchmark["benchmark"]) == {
        "customer_001_simple_stochastic_v1"
    }
    assert (benchmark["q50_win_rate_pct"] == 100.0).all()


def test_daily_strategy_comparison_plot_returns_figure():
    figure = plot_daily_strategy_comparison(make_strategy_daily())
    assert len(figure.axes) == 6
