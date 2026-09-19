from pathlib import Path

import pandas as pd
import torch
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import NHITS


def load_forecast_config(config_path: str | Path) -> dict:
    """Load forecasting settings from a YAML file."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare_neuralforecast_data(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Convert processed customer data to NeuralForecast format."""
    required_columns = {"unique_id", "ds", target_column}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    forecast_data = data[["unique_id", "ds", target_column]].copy()
    forecast_data = forecast_data.rename(columns={target_column: "y"})
    forecast_data["ds"] = pd.to_datetime(forecast_data["ds"])
    return forecast_data.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _model_settings(config: dict) -> dict:
    model_config = config["model"]
    return {
        "h": config["horizon"],
        "input_size": config["input_size"],
        "max_steps": model_config["max_steps"],
        "learning_rate": model_config["learning_rate"],
        "batch_size": model_config["batch_size"],
        "windows_batch_size": model_config["windows_batch_size"],
        "scaler_type": model_config["scaler_type"],
        "random_seed": model_config["random_seed"],
        "early_stop_patience_steps": model_config["early_stop_patience_steps"],
        "val_check_steps": model_config["val_check_steps"],
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "n_blocks": [1, 1, 1],
        "mlp_units": [[128, 128], [128, 128], [128, 128]],
        "n_pool_kernel_size": [2, 2, 1],
        "n_freq_downsample": [4, 2, 1],
    }


def build_point_nhits(config: dict) -> NHITS:
    """Build the point N-HiTS model."""
    return NHITS(loss=MAE(), valid_loss=MAE(), **_model_settings(config))


def build_quantile_nhits(config: dict) -> NHITS:
    """Build the quantile N-HiTS model."""
    levels = config["prediction_levels"]
    return NHITS(
        loss=MQLoss(level=levels),
        valid_loss=MQLoss(level=levels),
        **_model_settings(config),
    )


def _run_cross_validation(data: pd.DataFrame, config: dict, model: NHITS) -> pd.DataFrame:
    forecast_data = prepare_neuralforecast_data(data, config["target_column"])
    neural_forecast = NeuralForecast(models=[model], freq=config["frequency"])
    return neural_forecast.cross_validation(
        df=forecast_data,
        val_size=config["validation_size"],
        test_size=config["test_size"],
        step_size=config["step_size"],
        n_windows=None,
        refit=False,
    )


def _forecast_columns(predictions: pd.DataFrame) -> list[str]:
    base_columns = {"unique_id", "ds", "cutoff", "y"}
    return [column for column in predictions.columns if column not in base_columns]


def run_point_nhits_cross_validation(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Train point N-HiTS and forecast the complete test period."""
    predictions = _run_cross_validation(data, config, build_point_nhits(config))
    prediction_columns = _forecast_columns(predictions)
    if len(prediction_columns) != 1:
        raise ValueError(f"Expected one prediction column, but found {prediction_columns}.")

    predictions = predictions.rename(columns={prediction_columns[0]: "nhits"})
    output_columns = ["unique_id", "ds", "cutoff", "y", "nhits"]
    return predictions[output_columns].sort_values(["cutoff", "ds"]).reset_index(drop=True)


def _quantile_from_column(column: str, levels: list[int]) -> float | None:
    column_lower = column.lower()
    if "median" in column_lower:
        return 0.5

    for level in levels:
        tail_probability = (1 - level / 100) / 2
        if f"lo-{level}" in column_lower:
            return tail_probability
        if f"hi-{level}" in column_lower:
            return 1 - tail_probability
    return None


def _rename_quantile_columns(predictions: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    prediction_columns = _forecast_columns(predictions)
    column_quantiles = {
        column: _quantile_from_column(column, levels)
        for column in prediction_columns
    }

    unmatched = [column for column, quantile in column_quantiles.items() if quantile is None]
    if len(unmatched) == 1 and 0.5 not in column_quantiles.values():
        column_quantiles[unmatched[0]] = 0.5
        unmatched = []
    if unmatched:
        raise ValueError(f"Could not identify quantiles for columns: {unmatched}.")

    rename_columns = {
        column: f"q_{quantile:.2f}"
        for column, quantile in column_quantiles.items()
    }
    predictions = predictions.rename(columns=rename_columns)

    expected_quantiles = {0.5}
    for level in levels:
        tail_probability = (1 - level / 100) / 2
        expected_quantiles.update({tail_probability, 1 - tail_probability})
    expected_columns = [f"q_{quantile:.2f}" for quantile in sorted(expected_quantiles)]

    if not set(expected_columns).issubset(predictions.columns):
        raise ValueError(f"Expected quantile columns were not produced: {expected_columns}.")

    output_columns = ["unique_id", "ds", "cutoff", "y", *expected_columns]
    return predictions[output_columns]


def run_quantile_nhits_cross_validation(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Train quantile N-HiTS and forecast the complete test period."""
    predictions = _run_cross_validation(data, config, build_quantile_nhits(config))
    predictions = _rename_quantile_columns(predictions, config["prediction_levels"])
    return predictions.sort_values(["cutoff", "ds"]).reset_index(drop=True)


def run_quantile_nhits_validation_forecast(
    data: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Forecast historical validation data without using final test observations."""
    forecast_data = prepare_neuralforecast_data(data, config["target_column"])
    test_size = int(config["test_size"])

    historical_series = []
    for _, series in forecast_data.groupby("unique_id", sort=False):
        if len(series) <= test_size:
            raise ValueError("Insufficient data before the final test period.")
        historical_series.append(series.iloc[:-test_size])
    validation_input = pd.concat(historical_series, ignore_index=True).rename(
        columns={"y": config["target_column"]}
    )

    predictions = _run_cross_validation(
        validation_input,
        config,
        build_quantile_nhits(config),
    )
    predictions = _rename_quantile_columns(
        predictions,
        config["prediction_levels"],
    )
    return predictions.sort_values(["cutoff", "ds"]).reset_index(drop=True)
