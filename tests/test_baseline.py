import numpy as np

from src.battery import BatteryConfig, run_self_consumption_baseline
from src.evaluation.battery_metrics import calculate_battery_metrics


def make_config() -> BatteryConfig:
    return BatteryConfig(5.0, 3.0, 3.0, 0.95, 0.95, 0.10, 0.90, 0.50, 0.5)


def test_baseline_charges_and_discharges_correctly():
    result = run_self_consumption_baseline(
        net_load_kwh=[-1.0, 1.0], config=make_config()
    )
    assert result.loc[0, "charge_power_kw"] > 0
    assert result.loc[0, "discharge_power_kw"] == 0
    assert result.loc[1, "charge_power_kw"] == 0
    assert result.loc[1, "discharge_power_kw"] > 0
    assert np.allclose(result["grid_energy_kwh"], [0.0, 0.0])


def test_baseline_respects_all_battery_limits():
    net_load = np.r_[np.full(1000, -10.0), np.full(1000, 10.0)]
    result = run_self_consumption_baseline(
        net_load_kwh=net_load, config=make_config()
    )
    assert result["soc_end"].between(0.10 - 1e-8, 0.90 + 1e-8).all()
    assert result["charge_power_kw"].between(0.0, 3.0 + 1e-8).all()
    assert result["discharge_power_kw"].between(0.0, 3.0 + 1e-8).all()
    assert not (
        (result["charge_power_kw"] > 1e-8)
        & (result["discharge_power_kw"] > 1e-8)
    ).any()


def test_baseline_reduces_grid_exchange():
    result = run_self_consumption_baseline(
        net_load_kwh=[-1.0, 1.0], config=make_config()
    )
    metrics = calculate_battery_metrics(results=result, time_step_hours=0.5)
    assert metrics.loc[0, "grid_import_after_kwh"] == 0
    assert metrics.loc[0, "grid_export_after_kwh"] == 0


def test_baseline_grid_energy_balance():
    result = run_self_consumption_baseline(
        net_load_kwh=[-1.0, 0.5, 2.0], config=make_config()
    )
    expected = (
        result["net_load_kwh"]
        + 0.5 * result["charge_power_kw"]
        - 0.5 * result["discharge_power_kw"]
    )
    assert np.allclose(result["grid_energy_kwh"], expected)
