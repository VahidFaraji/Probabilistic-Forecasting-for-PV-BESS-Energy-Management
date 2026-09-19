import numpy as np
import pandas as pd


def interval_to_minutes(interval_label: str) -> int:
    """Convert an interval-ending label to minutes after the daily start."""
    if interval_label == "0:00":
        return 24 * 60

    hour, minute = map(int, interval_label.split(":"))
    return hour * 60 + minute


def prepare_customer_timeseries(
    raw_df: pd.DataFrame,
    customer_id: int,
) -> pd.DataFrame:
    """Convert one customer's daily records to a half-hourly time series."""
    metadata_columns = [
        "Customer",
        "Generator Capacity",
        "Postcode",
        "Consumption Category",
        "date",
        "Row Quality",
    ]

    time_columns = [
        column for column in raw_df.columns
        if column not in metadata_columns
    ]

    customer_data = raw_df.loc[
        raw_df["Customer"] == customer_id
    ].copy()

    customer_long = customer_data.melt(
        id_vars=["Customer", "date", "Consumption Category"],
        value_vars=time_columns,
        var_name="interval_end",
        value_name="energy_kwh",
    )

    minute_offsets = customer_long["interval_end"].map(
        interval_to_minutes
    )

    customer_long["ds"] = (
        customer_long["date"]
        + pd.to_timedelta(minute_offsets, unit="m")
    )

    customer_ts = (
        customer_long.pivot(
            index=["Customer", "ds"],
            columns="Consumption Category",
            values="energy_kwh",
        )
        .reset_index()
    )

    customer_ts.columns.name = None

    if "CL" not in customer_ts.columns:
        customer_ts["CL"] = 0.0

    customer_ts["load"] = customer_ts["GC"] + customer_ts["CL"]
    customer_ts["pv"] = customer_ts["GG"]
    customer_ts["net_load"] = customer_ts["load"] - customer_ts["pv"]
    customer_ts["unique_id"] = f"customer_{customer_id:03d}"

    output_columns = [
        "unique_id",
        "ds",
        "GC",
        "CL",
        "GG",
        "load",
        "pv",
        "net_load",
    ]

    customer_ts = (
        customer_ts[output_columns]
        .sort_values("ds")
        .reset_index(drop=True)
    )

    return customer_ts