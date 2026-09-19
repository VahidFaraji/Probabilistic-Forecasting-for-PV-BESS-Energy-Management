import pandas as pd


DAILY_LAG = 48
WEEKLY_LAG = 336


def create_seasonal_naive_forecasts(
    data: pd.DataFrame,
    target_column: str = "net_load",
) -> pd.DataFrame:
    """Create daily and weekly seasonal-naive forecasts for half-hourly data."""
    required_columns = {"unique_id", "ds", target_column}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    forecasts = data[["unique_id", "ds", target_column]].copy()
    forecasts["ds"] = pd.to_datetime(forecasts["ds"])
    forecasts = forecasts.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    target_by_customer = forecasts.groupby("unique_id")[target_column]

    forecasts["daily_naive"] = target_by_customer.shift(DAILY_LAG)
    forecasts["weekly_naive"] = target_by_customer.shift(WEEKLY_LAG)

    return forecasts
