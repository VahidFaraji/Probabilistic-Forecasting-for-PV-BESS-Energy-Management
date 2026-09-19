from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.battery.model import BatteryConfig, simulate_battery
from src.ems.deterministic import EMSConfig, optimize_deterministic_ems


QUANTILE_COLUMNS = ["q_0.10", "q_0.25", "q_0.50", "q_0.75", "q_0.90"]


def load_evaluation_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_objective_profile(ems: EMSConfig, profile: dict) -> EMSConfig:
    """Return an EMS configuration for one objective profile."""
    return replace(
        ems,
        energy_cost_weight=float(profile["energy_cost_weight"]),
        peak_penalty_per_kw=float(profile["peak_penalty_per_kw"]),
    )


def _prepare_predictions(
    predictions: pd.DataFrame,
    value_columns: list[str],
    horizon: int,
    expected_intervals: int,
    time_step_minutes: int,
) -> pd.DataFrame:
    required = ["unique_id", "ds", "cutoff", *value_columns]
    missing = set(required).difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    prepared = predictions.copy()
    if set(QUANTILE_COLUMNS).issubset(prepared.columns):
        quantiles = prepared[QUANTILE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(quantiles).all():
            raise ValueError("Quantile forecasts must be finite.")
        prepared[QUANTILE_COLUMNS] = np.sort(quantiles, axis=1)

    data = prepared[required].copy()
    data["ds"] = pd.to_datetime(data["ds"])
    data["cutoff"] = pd.to_datetime(data["cutoff"])
    data = data.sort_values(["unique_id", "cutoff", "ds"]).reset_index(drop=True)

    if len(data) != expected_intervals:
        raise ValueError(
            f"Expected {expected_intervals} test intervals, found {len(data)}."
        )
    if data.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Forecast timestamps must be unique within each series.")
    if data.isna().any().any():
        raise ValueError("Forecast test data cannot contain missing values.")
    if not np.isfinite(data[value_columns].to_numpy(dtype=float)).all():
        raise ValueError("Actual and forecast values must be finite.")

    step = pd.to_timedelta(int(time_step_minutes), unit="min")
    for (_, cutoff), window in data.groupby(["unique_id", "cutoff"], sort=True):
        if len(window) != horizon:
            raise ValueError(f"Each forecast origin must contain {horizon} intervals.")
        expected_ds = pd.date_range(cutoff + step, periods=horizon, freq=step)
        if not window["ds"].reset_index(drop=True).equals(pd.Series(expected_ds)):
            raise ValueError(
                "Each forecast window must start one time step after cutoff "
                "and continue without gaps."
            )
    return data


def prepare_forecast_test_data(
    predictions: pd.DataFrame,
    horizon: int = 48,
    expected_intervals: int = 1440,
    actual_column: str = "y",
    median_column: str = "q_0.50",
    time_step_minutes: int = 30,
) -> pd.DataFrame:
    """Prepare actual values and the median forecast for the existing benchmark."""
    data = _prepare_predictions(
        predictions,
        [actual_column, median_column],
        horizon,
        expected_intervals,
        time_step_minutes,
    )
    return data.rename(columns={
        actual_column: "actual_net_load_kwh",
        median_column: "median_forecast_kwh",
    })


def prepare_quantile_test_data(
    predictions: pd.DataFrame,
    quantile_columns: list[str] | tuple[str, ...],
    horizon: int = 48,
    expected_intervals: int = 1440,
    actual_column: str = "y",
    time_step_minutes: int = 30,
) -> pd.DataFrame:
    """Prepare actual values and selected, monotonically ordered quantiles."""
    selected = list(quantile_columns)
    if not selected:
        raise ValueError("At least one quantile column must be selected.")

    required_values = list(dict.fromkeys([actual_column, *QUANTILE_COLUMNS, *selected]))
    data = _prepare_predictions(
        predictions,
        required_values,
        horizon,
        expected_intervals,
        time_step_minutes,
    )
    return data[["unique_id", "ds", "cutoff", actual_column, *selected]].rename(
        columns={actual_column: "actual_net_load_kwh"}
    )


def evaluate_schedule_on_actual(
    actual_net_load_kwh,
    planned_schedule: pd.DataFrame,
    battery: BatteryConfig,
    ems: EMSConfig,
) -> pd.DataFrame:
    """Apply a fixed planned schedule to realized net load."""
    actual = np.asarray(actual_net_load_kwh, dtype=float)
    if len(actual) != len(planned_schedule):
        raise ValueError("Actual data and planned schedule must have equal lengths.")

    realized = simulate_battery(
        net_load_kwh=actual,
        charge_power_kw=planned_schedule["charge_power_kw"],
        discharge_power_kw=planned_schedule["discharge_power_kw"],
        config=battery,
    )
    grid = realized["grid_energy_kwh"].to_numpy()
    realized["grid_import_kwh"] = np.maximum(grid, 0)
    realized["grid_export_kwh"] = np.maximum(-grid, 0)
    realized["grid_cost"] = (
        ems.import_price_per_kwh * realized["grid_import_kwh"]
        - ems.export_price_per_kwh * realized["grid_export_kwh"]
    )

    dt = battery.time_step_hours
    realized["weighted_energy_cost"] = ems.energy_cost_weight * realized["grid_cost"]
    realized["throughput_cost"] = (
        ems.throughput_cost_per_kwh
        * (realized["charge_power_kw"] + realized["discharge_power_kw"])
        * dt
    )
    realized["peak_penalty_cost"] = 0.0
    realized.loc[0, "peak_penalty_cost"] = (
        ems.peak_penalty_per_kw * realized["grid_import_kwh"].max() / dt
    )
    realized["objective_value_component"] = (
        realized["weighted_energy_cost"]
        + realized["throughput_cost"]
        + realized["peak_penalty_cost"]
    )

    realized["planned_net_load_kwh"] = planned_schedule["net_load_kwh"].to_numpy()
    realized["planned_grid_cost"] = planned_schedule["grid_cost"].to_numpy()
    for column in (
        "weighted_energy_cost",
        "throughput_cost",
        "peak_penalty_cost",
        "objective_value_component",
    ):
        realized[f"planned_{column}"] = planned_schedule[column].to_numpy()
    return realized


def _label_schedule(
    result: pd.DataFrame,
    data: pd.DataFrame,
    unique_id: str,
    cutoff: pd.Timestamp,
    window: int,
    strategy: str,
) -> pd.DataFrame:
    result = result.copy()
    result.insert(0, "ds", data["ds"].to_numpy())
    result.insert(0, "cutoff", cutoff)
    result.insert(0, "window", window)
    result.insert(0, "strategy", strategy)
    result.insert(0, "unique_id", unique_id)
    return result


def run_forecast_ems_comparison(
    forecast_test_data: pd.DataFrame,
    battery: BatteryConfig,
    ems: EMSConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare perfect-foresight and median-forecast EMS schedules."""
    data = forecast_test_data.rename(
        columns={"median_forecast_kwh": "q50_forecast"}
    )
    return run_quantile_ems_comparison(
        data,
        battery,
        ems,
        {"median_forecast": "q50_forecast"},
    )


def run_quantile_ems_comparison(
    forecast_test_data: pd.DataFrame,
    battery: BatteryConfig,
    ems: EMSConfig,
    forecast_columns: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare perfect foresight with deterministic schedules from quantiles."""
    missing = set(forecast_columns.values()).difference(forecast_test_data.columns)
    if missing:
        raise ValueError(f"Missing selected quantile columns: {sorted(missing)}")

    schedules = []
    groups = forecast_test_data.groupby(["unique_id", "cutoff"], sort=True)
    for window, ((unique_id, cutoff), data) in enumerate(groups):
        actual = data["actual_net_load_kwh"].to_numpy(dtype=float)

        perfect_plan = optimize_deterministic_ems(actual, battery, ems)
        perfect = evaluate_schedule_on_actual(actual, perfect_plan, battery, ems)
        schedules.append(_label_schedule(
            perfect, data, unique_id, cutoff, window, "perfect_foresight"
        ))

        for strategy, column in forecast_columns.items():
            forecast = data[column].to_numpy(dtype=float)
            plan = optimize_deterministic_ems(forecast, battery, ems)
            realized = evaluate_schedule_on_actual(actual, plan, battery, ems)
            schedules.append(_label_schedule(
                realized, data, unique_id, cutoff, window, strategy
            ))

    schedules = pd.concat(schedules, ignore_index=True)
    summary = summarize_ems_comparison(
        schedules, forecast_test_data, battery, ems
    )
    return schedules, summary


def summarize_ems_comparison(
    schedules: pd.DataFrame,
    forecast_test_data: pd.DataFrame,
    battery: BatteryConfig,
    ems: EMSConfig,
) -> pd.DataFrame:
    """Calculate realized energy, peak, battery, and objective metrics."""
    actual = forecast_test_data["actual_net_load_kwh"].to_numpy(dtype=float)
    baseline_import = np.maximum(actual, 0)
    baseline_export = np.maximum(-actual, 0)
    baseline_cost = float(
        ems.import_price_per_kwh * baseline_import.sum()
        - ems.export_price_per_kwh * baseline_export.sum()
    )
    dt = battery.time_step_hours
    baseline_peak = float(baseline_import.max() / dt)
    baseline_daily_peaks = (
        forecast_test_data.assign(import_kwh=baseline_import)
        .groupby(["unique_id", "cutoff"])["import_kwh"]
        .max()
        / dt
    )
    baseline_peak_cost = ems.peak_penalty_per_kw * float(baseline_daily_peaks.sum())
    baseline_objective = ems.energy_cost_weight * baseline_cost + baseline_peak_cost
    rows = [{
        "strategy": "no_battery",
        "grid_cost": baseline_cost,
        "throughput_cost": 0.0,
        "peak_penalty_cost": baseline_peak_cost,
        "objective_value": baseline_objective,
        "grid_import_kwh": float(baseline_import.sum()),
        "grid_export_kwh": float(baseline_export.sum()),
        "maximum_peak_kw": baseline_peak,
        "mean_daily_peak_kw": float(baseline_daily_peaks.mean()),
        "charged_energy_kwh": 0.0,
        "discharged_energy_kwh": 0.0,
        "maximum_terminal_soc_error": 0.0,
        "planned_grid_cost": baseline_cost,
        "planned_objective_value": baseline_objective,
    }]

    for strategy, data in schedules.groupby("strategy", sort=False):
        terminal_soc = data.groupby(["unique_id", "cutoff"])["soc_end"].last()
        daily_peaks = data.groupby(["unique_id", "cutoff"])["grid_import_kwh"].max() / dt
        rows.append({
            "strategy": strategy,
            "grid_cost": float(data["grid_cost"].sum()),
            "throughput_cost": float(data["throughput_cost"].sum()),
            "peak_penalty_cost": float(data["peak_penalty_cost"].sum()),
            "objective_value": float(data["objective_value_component"].sum()),
            "grid_import_kwh": float(data["grid_import_kwh"].sum()),
            "grid_export_kwh": float(data["grid_export_kwh"].sum()),
            "maximum_peak_kw": float(data["grid_import_kwh"].max() / dt),
            "mean_daily_peak_kw": float(daily_peaks.mean()),
            "charged_energy_kwh": float((data["charge_power_kw"] * dt).sum()),
            "discharged_energy_kwh": float((data["discharge_power_kw"] * dt).sum()),
            "maximum_terminal_soc_error": float(
                (terminal_soc - ems.terminal_soc).abs().max()
            ),
            "planned_grid_cost": float(data["planned_grid_cost"].sum()),
            "planned_objective_value": float(
                data["planned_objective_value_component"].sum()
            ),
        })

    summary = pd.DataFrame(rows)
    summary["cost_reduction_pct"] = np.where(
        baseline_cost != 0,
        100 * (baseline_cost - summary["grid_cost"]) / baseline_cost,
        np.nan,
    )
    summary["import_reduction_pct"] = np.where(
        baseline_import.sum() != 0,
        100 * (baseline_import.sum() - summary["grid_import_kwh"])
        / baseline_import.sum(),
        np.nan,
    )
    return summary
