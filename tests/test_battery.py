import numpy as np
import pytest

from src.battery.model import BatteryConfig, simulate_battery


def make_config() -> BatteryConfig:
    return BatteryConfig(5.0, 3.0, 3.0, 0.95, 0.95, 0.10, 0.90, 0.50, 0.5)


def test_idle_battery_preserves_soc_and_net_load():
    result = simulate_battery([0.4, -0.2], [0.0, 0.0], [0.0, 0.0], make_config())
    assert np.allclose(result["grid_energy_kwh"], [0.4, -0.2])
    assert np.allclose(result["soc_end"], [0.5, 0.5])


def test_charge_and_discharge_follow_energy_equation():
    result = simulate_battery([-1.0, 1.0], [1.0, 0.0], [0.0, 0.5], make_config())
    after_charge = (2.5 + 0.95 * 1.0 * 0.5) / 5.0
    after_discharge = (2.5 + 0.95 * 1.0 * 0.5 - 0.5 * 0.5 / 0.95) / 5.0
    assert result.loc[0, "soc_end"] == pytest.approx(after_charge)
    assert result.loc[1, "soc_end"] == pytest.approx(after_discharge)
    assert np.allclose(result["grid_energy_kwh"], [-0.5, 0.75])


def test_invalid_battery_power_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        simulate_battery([0.0], [-1.0], [0.0], make_config())
    with pytest.raises(ValueError, match="power limit"):
        simulate_battery([0.0], [3.1], [0.0], make_config())
    with pytest.raises(ValueError, match="simultaneously"):
        simulate_battery([0.0], [1.0], [1.0], make_config())


def test_soc_limit_violation_is_rejected():
    with pytest.raises(ValueError, match="maximum"):
        simulate_battery([0.0, 0.0], [3.0, 3.0], [0.0, 0.0], make_config())
