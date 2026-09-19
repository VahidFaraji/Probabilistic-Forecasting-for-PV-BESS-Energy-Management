import numpy as np
import pandas as pd


def calculate_battery_metrics(
    results: pd.DataFrame,
    time_step_hours: float,
) -> pd.DataFrame:
    """Calculate basic operational BESS metrics."""
    required_columns = {
        "net_load_kwh",
        "grid_energy_kwh",
        "charge_power_kw",
        "discharge_power_kw",
        "soc_end",
    }

    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if results.empty:
        raise ValueError("Battery results cannot be empty.")

    if time_step_hours <= 0:
        raise ValueError("Time step must be positive.")

    net_load = results["net_load_kwh"].to_numpy(dtype=float)
    grid_energy = results["grid_energy_kwh"].to_numpy(dtype=float)

    import_before = np.maximum(net_load, 0)
    import_after = np.maximum(grid_energy, 0)

    export_before = np.maximum(-net_load, 0)
    export_after = np.maximum(-grid_energy, 0)

    import_before_total = float(import_before.sum())
    import_after_total = float(import_after.sum())

    export_before_total = float(export_before.sum())
    export_after_total = float(export_after.sum())

    peak_before = float(import_before.max() / time_step_hours)
    peak_after = float(import_after.max() / time_step_hours)

    metrics = {
        "grid_import_before_kwh": import_before_total,
        "grid_import_after_kwh": import_after_total,
        "grid_import_reduction_pct": _percentage_reduction(
            import_before_total,
            import_after_total,
        ),
        "grid_export_before_kwh": export_before_total,
        "grid_export_after_kwh": export_after_total,
        "grid_export_reduction_pct": _percentage_reduction(
            export_before_total,
            export_after_total,
        ),
        "peak_import_before_kw": peak_before,
        "peak_import_after_kw": peak_after,
        "peak_reduction_pct": _percentage_reduction(
            peak_before,
            peak_after,
        ),
        "charged_energy_kwh": float(
            results["charge_power_kw"].sum() * time_step_hours
        ),
        "discharged_energy_kwh": float(
            results["discharge_power_kw"].sum() * time_step_hours
        ),
        "minimum_soc": float(results["soc_end"].min()),
        "maximum_soc": float(results["soc_end"].max()),
    }

    return pd.DataFrame([metrics])


def _percentage_reduction(before: float, after: float) -> float:
    """Calculate percentage reduction relative to the original value."""
    if before == 0:
        return float("nan")

    return 100 * (before - after) / before