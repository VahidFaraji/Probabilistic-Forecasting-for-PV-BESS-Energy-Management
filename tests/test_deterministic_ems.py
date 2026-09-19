import numpy as np
import pytest

from src.battery import BatteryConfig
from src.ems import EMSConfig, optimize_deterministic_ems, run_deterministic_ems_windows


def make_battery() -> BatteryConfig:
    return BatteryConfig(5.0, 3.0, 3.0, 0.95, 0.95, 0.10, 0.90, 0.50, 0.5)


def make_ems(peak_penalty: float = 0.0) -> EMSConfig:
    return EMSConfig(4, "appsi_highs", 0.30, 0.08, peak_penalty, 0.001, 0.50)


def test_optimized_schedule_is_physically_feasible():
    result = optimize_deterministic_ems([-1.0, -1.0, 1.0, 1.0], make_battery(), make_ems())
    assert len(result) == 4
    assert result["soc_end"].between(0.10 - 1e-8, 0.90 + 1e-8).all()
    assert result.iloc[-1]["soc_end"] == pytest.approx(0.50)
    assert result["charge_power_kw"].between(0, 3.0 + 1e-8).all()
    assert result["discharge_power_kw"].between(0, 3.0 + 1e-8).all()
    assert np.allclose(
        result["grid_energy_kwh"],
        result["grid_import_kwh"] - result["grid_export_kwh"],
    )


def test_objective_components_are_complete():
    result = optimize_deterministic_ems(
        [-1.0, -1.0, 1.0, 1.0],
        make_battery(),
        make_ems(peak_penalty=2.0),
    )
    components = (
        result["weighted_energy_cost"]
        + result["throughput_cost"]
        + result["peak_penalty_cost"]
    )
    assert np.allclose(result["objective_value_component"], components)
    assert result["peak_penalty_cost"].sum() >= 0


def test_ems_reduces_energy_cost():
    net_load = np.array([-1.0, -1.0, 1.0, 1.0])
    result = optimize_deterministic_ems(net_load, make_battery(), make_ems())
    no_battery = 0.30 * np.maximum(net_load, 0).sum() - 0.08 * np.maximum(-net_load, 0).sum()
    assert result["grid_cost"].sum() < no_battery


def test_peak_penalty_reduces_peak_import():
    result = optimize_deterministic_ems(
        [2.0, 2.0, 0.0, 0.0], make_battery(), make_ems(peak_penalty=10.0)
    )
    assert result["grid_import_kwh"].max() / 0.5 < 4.0


def test_multiple_windows_are_combined_and_end_at_target_soc():
    result = run_deterministic_ems_windows(
        [-1.0, -1.0, 1.0, 1.0] * 2, make_battery(), make_ems()
    )
    assert len(result) == 8
    assert result["window"].nunique() == 2
    assert np.allclose(result.groupby("window")["soc_end"].last(), 0.50)


def test_wrong_horizon_is_rejected():
    with pytest.raises(ValueError, match="exactly 4"):
        optimize_deterministic_ems([1.0, 1.0], make_battery(), make_ems())
