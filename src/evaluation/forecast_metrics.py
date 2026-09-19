from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_QUANTILE_COLUMNS = {
    0.10: "q_0.10",
    0.25: "q_0.25",
    0.50: "q_0.50",
    0.75: "q_0.75",
    0.90: "q_0.90",
}


def _finite_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted values must have the same shape.")
    if actual_values.size == 0:
        raise ValueError("Actual and predicted values cannot be empty.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Actual and predicted values must be finite.")
    return actual_values, predicted_values


def calculate_point_metrics(actual, predicted) -> dict[str, float]:
    """Calculate MAE, RMSE, and nMAE for one point forecast."""
    actual_values, predicted_values = _finite_arrays(actual, predicted)
    errors = predicted_values - actual_values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_absolute_actual = float(np.mean(np.abs(actual_values)))
    if mean_absolute_actual == 0:
        raise ValueError("nMAE is undefined when all actual values are zero.")
    return {"MAE": mae, "RMSE": rmse, "nMAE": mae / mean_absolute_actual}


def evaluate_point_forecasts(
    data: pd.DataFrame,
    forecast_columns: Sequence[str],
    actual_column: str = "net_load",
) -> pd.DataFrame:
    """Evaluate multiple point-forecast columns against one actual column."""
    required_columns = {actual_column, *forecast_columns}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    rows = []
    for forecast_column in forecast_columns:
        metrics = calculate_point_metrics(data[actual_column], data[forecast_column])
        rows.append({"model": forecast_column, **metrics})
    return pd.DataFrame(rows)


def calculate_pinball_loss(actual, predicted_quantile, quantile: float) -> float:
    """Calculate pinball loss for one quantile."""
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be between 0 and 1.")
    actual_values, predicted_values = _finite_arrays(actual, predicted_quantile)
    errors = actual_values - predicted_values
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def evaluate_quantile_forecasts(
    data: pd.DataFrame,
    actual_column: str = "y",
    quantile_columns: dict[float, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate summary and per-quantile metrics."""
    quantile_columns = quantile_columns or DEFAULT_QUANTILE_COLUMNS
    required_columns = {actual_column, *quantile_columns.values()}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    actual_values = np.asarray(data[actual_column], dtype=float)
    sorted_quantiles = sorted(quantile_columns)
    forecasts = []
    rows = []

    for quantile in sorted_quantiles:
        column = quantile_columns[quantile]
        _, predicted_values = _finite_arrays(actual_values, data[column])
        forecasts.append(predicted_values)
        empirical_probability = float(np.mean(actual_values <= predicted_values))
        pinball_loss = calculate_pinball_loss(actual_values, predicted_values, quantile)
        rows.append({
            "quantile": quantile,
            "forecast_column": column,
            "pinball_loss": pinball_loss,
            "empirical_probability": empirical_probability,
            "calibration_error": abs(empirical_probability - quantile),
        })

    quantile_metrics = pd.DataFrame(rows)
    forecast_matrix = np.column_stack(forecasts)
    crossing_rate = float(np.mean(np.any(np.diff(forecast_matrix, axis=1) < 0, axis=1)))

    quantile_values = quantile_metrics["quantile"].to_numpy(dtype=float)
    pinball_values = quantile_metrics["pinball_loss"].to_numpy(dtype=float)
    pinball_integral = np.sum(
        np.diff(quantile_values) * (pinball_values[:-1] + pinball_values[1:]) / 2
    )

    point_metrics = calculate_point_metrics(actual_values, data[quantile_columns[0.50]])
    summary = {
        "model": "nhits_quantile_median",
        **point_metrics,
        "mean_pinball_loss": float(quantile_metrics["pinball_loss"].mean()),
        "approximate_CRPS": float(2 * pinball_integral),
        "mean_calibration_error": float(quantile_metrics["calibration_error"].mean()),
        "quantile_crossing_rate": crossing_rate,
    }

    actual_range = float(actual_values.max() - actual_values.min())
    for level, (lower_quantile, upper_quantile) in {
        50: (0.25, 0.75),
        80: (0.10, 0.90),
    }.items():
        lower = data[quantile_columns[lower_quantile]].to_numpy(dtype=float)
        upper = data[quantile_columns[upper_quantile]].to_numpy(dtype=float)
        coverage = float(np.mean((actual_values >= lower) & (actual_values <= upper)))
        mean_width = float(np.mean(upper - lower))
        summary[f"coverage_{level}"] = coverage
        summary[f"mean_width_{level}"] = mean_width
        summary[f"PINAW_{level}"] = mean_width / actual_range if actual_range > 0 else float("nan")

    return pd.DataFrame([summary]), quantile_metrics
