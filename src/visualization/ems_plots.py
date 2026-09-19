import matplotlib.pyplot as plt
import pandas as pd


STRATEGY_LABELS = {
    "perfect_foresight": "Perfect foresight",
    "median_forecast": "Median forecast",
    "q50_forecast": "Q50 EMS",
    "q75_forecast": "Q75 EMS",
    "five_scenario": "Five-scenario EMS",
}


def plot_forecast_day(
    forecasts: pd.DataFrame,
    cutoff,
):
    """Plot actual, median, and prediction intervals for one forecast day."""
    cutoff = pd.Timestamp(cutoff)
    data = forecasts.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data["cutoff"] = pd.to_datetime(data["cutoff"])
    data = data[data["cutoff"] == cutoff].sort_values("ds")
    if data.empty:
        raise ValueError(f"No forecasts found for cutoff {cutoff}.")

    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.fill_between(
        data["ds"], data["q_0.10"], data["q_0.90"],
        color="tab:blue", alpha=0.15, label="80% interval",
    )
    axis.fill_between(
        data["ds"], data["q_0.25"], data["q_0.75"],
        color="tab:blue", alpha=0.25, label="50% interval",
    )
    axis.plot(data["ds"], data["y"], color="black", label="Actual")
    axis.plot(
        data["ds"], data["q_0.50"], color="tab:blue", label="Median"
    )
    axis.set_title(f"Net-load forecast: {data['ds'].min().date()}")
    axis.set_ylabel("Net-load energy (kWh/30 min)")
    axis.set_xlabel("Time")
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.legend(ncol=4)
    axis.grid(alpha=0.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_battery_day(
    schedules: pd.DataFrame,
    cutoff,
    objective: str,
):
    """Compare battery and grid operation for one day."""
    cutoff = pd.Timestamp(cutoff)
    data = schedules.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data["cutoff"] = pd.to_datetime(data["cutoff"])
    data = data[
        (data["cutoff"] == cutoff)
        & (data["objective"] == objective)
    ].sort_values("ds")
    if data.empty:
        raise ValueError(
            f"No schedules found for {objective} at cutoff {cutoff}."
        )

    figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for strategy, strategy_data in data.groupby("strategy", sort=False):
        label = STRATEGY_LABELS.get(strategy, strategy)
        battery_power = (
            strategy_data["discharge_power_kw"]
            - strategy_data["charge_power_kw"]
        )
        axes[0].plot(strategy_data["ds"], battery_power, label=label)
        axes[1].plot(strategy_data["ds"], strategy_data["soc_end"], label=label)
        axes[2].plot(
            strategy_data["ds"],
            strategy_data["grid_import_kwh"],
            label=label,
        )
        axes[3].plot(
            strategy_data["ds"],
            strategy_data["grid_export_kwh"],
            label=label,
        )

    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_ylabel("Battery power (kW)")
    axes[0].set_title("Positive: discharge; negative: charge")
    axes[1].set_ylabel("SOC")
    axes[2].set_ylabel("Grid import (kWh)")
    axes[3].set_ylabel("Grid export (kWh)")
    axes[3].set_xlabel("Time")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle(f"{objective}: {data['ds'].min().date()}")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_daily_diagnostics(daily: pd.DataFrame):
    """Plot complete-period cost, peak, battery use, and error relationships."""
    strategies = set(daily["strategy"])
    forecast_strategy = (
        "q50_forecast" if "q50_forecast" in strategies else "median_forecast"
    )
    median = daily[daily["strategy"] == forecast_strategy].copy()
    cost = median[median["objective"] == "cost_only"].sort_values("cutoff")
    peak = median[median["objective"] == "peak_only"].sort_values("cutoff")
    perfect_cost = daily[
        (daily["strategy"] == "perfect_foresight")
        & (daily["objective"] == "cost_only")
    ].sort_values("cutoff")
    if cost.empty or peak.empty:
        raise ValueError("Daily results must include cost_only and peak_only.")

    figure, axes = plt.subplots(3, 2, figsize=(13, 11))

    axes[0, 0].plot(cost["date"], cost["actual_no_battery_cost"], label="No battery")
    axes[0, 0].plot(cost["date"], cost["perfect_foresight_cost"], label="Perfect foresight")
    forecast_label = STRATEGY_LABELS.get(forecast_strategy, forecast_strategy)
    axes[0, 0].plot(cost["date"], cost["realized_cost"], label=forecast_label)
    axes[0, 0].set_title("Daily realized cost")
    axes[0, 0].set_ylabel("Cost")

    axes[0, 1].plot(peak["date"], peak["actual_peak_kw"], label="No battery")
    axes[0, 1].plot(peak["date"], peak["perfect_foresight_peak_kw"], label="Perfect foresight")
    axes[0, 1].plot(peak["date"], peak["realized_peak_kw"], label=forecast_label)
    axes[0, 1].set_title("Daily peak import")
    axes[0, 1].set_ylabel("Power (kW)")

    median_cumulative_saving = (
        cost["actual_no_battery_cost"] - cost["realized_cost"]
    ).cumsum()
    perfect_cumulative_saving = (
        cost["actual_no_battery_cost"] - cost["perfect_foresight_cost"]
    ).cumsum()
    axes[1, 0].plot(cost["date"], perfect_cumulative_saving, label="Perfect foresight")
    axes[1, 0].plot(cost["date"], median_cumulative_saving, label=forecast_label)
    axes[1, 0].axhline(0, color="grey", linewidth=0.8)
    axes[1, 0].set_title("Cumulative cost saving")
    axes[1, 0].set_ylabel("Saving")

    axes[1, 1].plot(
        perfect_cost["date"],
        perfect_cost["battery_throughput_kwh"],
        label="Perfect foresight",
    )
    axes[1, 1].plot(
        cost["date"],
        cost["battery_throughput_kwh"],
        label=forecast_label,
    )
    axes[1, 1].set_title("Daily battery throughput")
    axes[1, 1].set_ylabel("Energy (kWh)")

    axes[2, 0].scatter(cost["forecast_mae_kwh"], cost["cost_regret"])
    axes[2, 0].set_title("Forecast MAE vs cost regret")
    axes[2, 0].set_xlabel("Forecast MAE (kWh)")
    axes[2, 0].set_ylabel("Cost regret")

    axes[2, 1].scatter(
        peak["circular_peak_timing_error_hours"],
        peak["peak_regret_kw"],
    )
    axes[2, 1].set_title("Peak-timing error vs peak regret")
    axes[2, 1].set_xlabel("Circular timing error (hours)")
    axes[2, 1].set_ylabel("Peak regret (kW)")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()

    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_battery_period(
    schedules: pd.DataFrame,
    objective: str,
):
    """Plot battery power, SOC, and cumulative grid cost over all days."""
    data = schedules.copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data = data[data["objective"] == objective].sort_values("ds")
    if data.empty:
        raise ValueError(f"No schedules found for objective {objective}.")

    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for strategy, strategy_data in data.groupby("strategy", sort=False):
        label = STRATEGY_LABELS.get(strategy, strategy)
        battery_power = (
            strategy_data["discharge_power_kw"]
            - strategy_data["charge_power_kw"]
        )
        axes[0].plot(strategy_data["ds"], battery_power, label=label, linewidth=0.8)
        axes[1].plot(
            strategy_data["ds"], strategy_data["soc_end"],
            label=label, linewidth=0.8,
        )
        axes[2].plot(
            strategy_data["ds"], strategy_data["grid_cost"].cumsum(),
            label=label, linewidth=1.0,
        )

    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_ylabel("Battery power (kW)")
    axes[0].set_title("Positive: discharge; negative: charge")
    axes[1].set_ylabel("SOC")
    axes[2].set_ylabel("Cumulative grid cost")
    axes[2].set_xlabel("Time")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle(f"Thirty-day battery operation: {objective}")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_daily_strategy_comparison(
    daily: pd.DataFrame,
    strategies: tuple[str, ...] = (
        "q50_forecast", "q75_forecast", "five_scenario"
    ),
):
    """Plot daily and cumulative objective performance by EMS strategy."""
    required = {"objective", "strategy", "date", "realized_objective"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")

    objectives = ("cost_only", "peak_only", "combined")
    figure, axes = plt.subplots(3, 2, figsize=(13, 11))

    for row, objective in enumerate(objectives):
        data = daily[
            (daily["objective"] == objective)
            & (daily["strategy"].isin(strategies))
        ].copy()
        if data.empty:
            raise ValueError(f"No daily comparison data for {objective}.")

        pivot = data.pivot(
            index="date", columns="strategy", values="realized_objective"
        ).sort_index()
        missing_strategies = set(strategies).difference(pivot.columns)
        if missing_strategies:
            raise ValueError(
                f"Missing strategies for {objective}: {sorted(missing_strategies)}"
            )

        for strategy in strategies:
            axes[row, 0].plot(
                pivot.index,
                pivot[strategy],
                label=STRATEGY_LABELS.get(strategy, strategy),
            )
        axes[row, 0].set_title(f"Daily realized objective: {objective}")
        axes[row, 0].set_ylabel("Objective value")

        for reference in ("q50_forecast", "q75_forecast"):
            improvement = (
                pivot[reference] - pivot["five_scenario"]
            ).cumsum()
            axes[row, 1].plot(
                pivot.index,
                improvement,
                label=f"Five-scenario vs {STRATEGY_LABELS[reference]}",
            )
        axes[row, 1].axhline(0, color="grey", linewidth=0.8)
        axes[row, 1].set_title(f"Cumulative objective improvement: {objective}")
        axes[row, 1].set_ylabel("Reference − five-scenario")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure
