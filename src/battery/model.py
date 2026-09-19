from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float
    max_charge_power_kw: float
    max_discharge_power_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    min_soc: float
    max_soc: float
    initial_soc: float
    time_step_hours: float

    def validate(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("Battery capacity must be positive.")
        if self.max_charge_power_kw < 0 or self.max_discharge_power_kw < 0:
            raise ValueError("Battery power limits cannot be negative.")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("Charge efficiency must be in (0, 1].")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("Discharge efficiency must be in (0, 1].")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError("SOC limits must satisfy 0 <= min_soc < max_soc <= 1.")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("Initial SOC must lie within the SOC limits.")
        if self.time_step_hours <= 0:
            raise ValueError("Time step must be positive.")


def load_battery_config(config_path: str | Path) -> BatteryConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = BatteryConfig(**yaml.safe_load(file))
    config.validate()
    return config


def simulate_battery(
    net_load_kwh,
    charge_power_kw,
    discharge_power_kw,
    config: BatteryConfig,
) -> pd.DataFrame:
    """Simulate a fixed battery schedule using half-hourly energy balance."""
    config.validate()
    net_load = np.asarray(net_load_kwh, dtype=float)
    charge = np.asarray(charge_power_kw, dtype=float)
    discharge = np.asarray(discharge_power_kw, dtype=float)

    if net_load.ndim != 1 or not (
        net_load.shape == charge.shape == discharge.shape
    ):
        raise ValueError("Battery inputs must be one-dimensional and have equal lengths.")
    if not np.isfinite(np.column_stack([net_load, charge, discharge])).all():
        raise ValueError("Battery inputs must be finite.")
    if (charge < -1e-8).any() or (discharge < -1e-8).any():
        raise ValueError("Battery powers cannot be negative.")

    charge = np.maximum(charge, 0)
    discharge = np.maximum(discharge, 0)
    if (charge > config.max_charge_power_kw + 1e-8).any():
        raise ValueError("Charge power exceeds the battery power limit.")
    if (discharge > config.max_discharge_power_kw + 1e-8).any():
        raise ValueError("Discharge power exceeds the battery power limit.")
    if ((charge > 1e-8) & (discharge > 1e-8)).any():
        raise ValueError("The battery cannot charge and discharge simultaneously.")

    dt = config.time_step_hours
    stored = np.empty(len(net_load) + 1)
    stored[0] = config.initial_soc * config.capacity_kwh
    for index in range(len(net_load)):
        stored[index + 1] = (
            stored[index]
            + config.charge_efficiency * charge[index] * dt
            - discharge[index] * dt / config.discharge_efficiency
        )

    if stored.min() < config.min_soc * config.capacity_kwh - 1e-7:
        raise ValueError("Battery schedule violates the minimum SOC.")
    if stored.max() > config.max_soc * config.capacity_kwh + 1e-7:
        raise ValueError("Battery schedule violates the maximum SOC.")

    return pd.DataFrame({
        "net_load_kwh": net_load,
        "charge_power_kw": charge,
        "discharge_power_kw": discharge,
        "grid_energy_kwh": net_load + charge * dt - discharge * dt,
        "soc_start": stored[:-1] / config.capacity_kwh,
        "soc_end": stored[1:] / config.capacity_kwh,
    })
