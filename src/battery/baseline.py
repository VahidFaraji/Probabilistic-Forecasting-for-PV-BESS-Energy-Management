import numpy as np
import pandas as pd

from .model import BatteryConfig, simulate_battery


def run_self_consumption_baseline(
    net_load_kwh,
    config: BatteryConfig,
) -> pd.DataFrame:
    """Operate the battery to increase PV self-consumption."""
    config.validate()

    net_load = np.asarray(net_load_kwh, dtype=float)

    if net_load.ndim != 1 or net_load.size == 0:
        raise ValueError("Net load must be a non-empty one-dimensional sequence.")

    if not np.isfinite(net_load).all():
        raise ValueError("Net load must contain only finite values.")

    charge_power = np.zeros(len(net_load))
    discharge_power = np.zeros(len(net_load))

    time_step = config.time_step_hours
    stored_energy = config.initial_soc * config.capacity_kwh
    minimum_energy = config.min_soc * config.capacity_kwh
    maximum_energy = config.max_soc * config.capacity_kwh

    for index, interval_net_load in enumerate(net_load):

        # Charge using surplus PV.
        if interval_net_load < 0:
            surplus_energy = -interval_net_load
            available_capacity = max(
                maximum_energy - stored_energy,
                0.0,
            )

            charge_power[index] = min(
                surplus_energy / time_step,
                config.max_charge_power_kw,
                available_capacity
                / (config.charge_efficiency * time_step),
            )

            stored_energy += (
                config.charge_efficiency
                * charge_power[index]
                * time_step
            )

            stored_energy = min(stored_energy, maximum_energy)


        # Discharge to reduce grid import.
        elif interval_net_load > 0:
            available_energy = max(
                stored_energy - minimum_energy,
                0.0,
            )

            discharge_power[index] = min(
                interval_net_load / time_step,
                config.max_discharge_power_kw,
                available_energy
                * config.discharge_efficiency
                / time_step,
            )

            stored_energy -= (
                discharge_power[index]
                * time_step
                / config.discharge_efficiency
            )

            stored_energy = max(stored_energy, minimum_energy)
        
    return simulate_battery(
        net_load_kwh=net_load,
        charge_power_kw=charge_power,
        discharge_power_kw=discharge_power,
        config=config,
    )