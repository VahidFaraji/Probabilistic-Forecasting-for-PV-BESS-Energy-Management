from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    SolverFactory,
    Var,
    minimize,
    value,
)
from pyomo.opt import TerminationCondition

from src.battery.model import BatteryConfig, simulate_battery


@dataclass(frozen=True)
class EMSConfig:
    horizon: int
    solver: str
    import_price_per_kwh: float
    export_price_per_kwh: float
    peak_penalty_per_kw: float
    throughput_cost_per_kwh: float
    terminal_soc: float
    energy_cost_weight: float = 1.0

    def validate(self, battery: BatteryConfig) -> None:
        if self.horizon <= 0:
            raise ValueError("EMS horizon must be positive.")
        if self.import_price_per_kwh < self.export_price_per_kwh:
            raise ValueError("Import price must not be lower than export price.")
        if self.export_price_per_kwh < 0:
            raise ValueError("Electricity prices cannot be negative.")
        if min(
            self.energy_cost_weight,
            self.peak_penalty_per_kw,
            self.throughput_cost_per_kwh,
        ) < 0:
            raise ValueError("Objective weights cannot be negative.")
        if not battery.min_soc <= self.terminal_soc <= battery.max_soc:
            raise ValueError("Terminal SOC must lie within the battery SOC limits.")


def load_ems_config(config_path: str | Path) -> EMSConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        return EMSConfig(**yaml.safe_load(file))


def optimize_deterministic_ems(
    net_load_kwh,
    battery: BatteryConfig,
    ems: EMSConfig,
) -> pd.DataFrame:
    """Optimize one EMS horizon and return its physical and cost components."""
    battery.validate()
    ems.validate(battery)

    net_load = np.asarray(net_load_kwh, dtype=float)
    if net_load.ndim != 1 or len(net_load) != ems.horizon:
        raise ValueError(f"Net load must contain exactly {ems.horizon} values.")
    if not np.isfinite(net_load).all():
        raise ValueError("Net load must contain only finite values.")

    model = _build_model(net_load, battery, ems)
    solver = SolverFactory(ems.solver)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"Solver '{ems.solver}' is not available.")

    solution = solver.solve(model)
    if solution.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(
            f"EMS optimization failed: {solution.solver.termination_condition}."
        )

    charge = _values(model.charge_power, ems.horizon)
    discharge = _values(model.discharge_power, ems.horizon)
    grid_import = _values(model.grid_import, ems.horizon)
    grid_export = _values(model.grid_export, ems.horizon)

    for values in (charge, discharge, grid_import, grid_export):
        values[np.abs(values) < 1e-8] = 0.0

    results = simulate_battery(
        net_load_kwh=net_load,
        charge_power_kw=charge,
        discharge_power_kw=discharge,
        config=battery,
    )
    results["grid_import_kwh"] = grid_import
    results["grid_export_kwh"] = grid_export
    results["grid_cost"] = (
        ems.import_price_per_kwh * grid_import
        - ems.export_price_per_kwh * grid_export
    )

    dt = battery.time_step_hours
    results["weighted_energy_cost"] = ems.energy_cost_weight * results["grid_cost"]
    results["throughput_cost"] = (
        ems.throughput_cost_per_kwh * (charge + discharge) * dt
    )
    results["peak_penalty_cost"] = 0.0
    results.loc[0, "peak_penalty_cost"] = (
        ems.peak_penalty_per_kw * grid_import.max() / dt
    )
    results["objective_value_component"] = (
        results["weighted_energy_cost"]
        + results["throughput_cost"]
        + results["peak_penalty_cost"]
    )

    if not np.allclose(
        results["grid_energy_kwh"],
        grid_import - grid_export,
        atol=1e-6,
    ):
        raise RuntimeError("Optimized grid and battery balances do not match.")

    return results


def run_deterministic_ems_windows(
    net_load_kwh,
    battery: BatteryConfig,
    ems: EMSConfig,
) -> pd.DataFrame:
    """Optimize consecutive, non-overlapping EMS horizons."""
    net_load = np.asarray(net_load_kwh, dtype=float)
    if net_load.ndim != 1 or net_load.size == 0:
        raise ValueError("Net load must be a non-empty one-dimensional sequence.")
    if len(net_load) % ems.horizon != 0:
        raise ValueError("Net-load length must be divisible by the EMS horizon.")

    windows = []
    for window, start in enumerate(range(0, len(net_load), ems.horizon)):
        result = optimize_deterministic_ems(
            net_load[start:start + ems.horizon],
            battery,
            ems,
        )
        result.insert(0, "window", window)
        windows.append(result)

    return pd.concat(windows, ignore_index=True)


def _build_model(
    net_load: np.ndarray,
    battery: BatteryConfig,
    ems: EMSConfig,
) -> ConcreteModel:
    model = ConcreteModel()
    model.intervals = RangeSet(0, ems.horizon - 1)
    model.states = RangeSet(0, ems.horizon)

    model.charge_power = Var(
        model.intervals,
        domain=NonNegativeReals,
        bounds=(0, battery.max_charge_power_kw),
    )
    model.discharge_power = Var(
        model.intervals,
        domain=NonNegativeReals,
        bounds=(0, battery.max_discharge_power_kw),
    )
    model.grid_import = Var(model.intervals, domain=NonNegativeReals)
    model.grid_export = Var(model.intervals, domain=NonNegativeReals)
    model.stored_energy = Var(
        model.states,
        bounds=(
            battery.min_soc * battery.capacity_kwh,
            battery.max_soc * battery.capacity_kwh,
        ),
    )
    model.peak_import = Var(domain=NonNegativeReals)

    model.stored_energy[0].fix(battery.initial_soc * battery.capacity_kwh)
    model.stored_energy[ems.horizon].fix(
        ems.terminal_soc * battery.capacity_kwh
    )
    dt = battery.time_step_hours

    def grid_balance(model, index):
        return (
            model.grid_import[index] - model.grid_export[index]
            == net_load[index]
            + model.charge_power[index] * dt
            - model.discharge_power[index] * dt
        )

    model.grid_balance = Constraint(model.intervals, rule=grid_balance)

    def battery_state(model, index):
        return model.stored_energy[index + 1] == (
            model.stored_energy[index]
            + battery.charge_efficiency * model.charge_power[index] * dt
            - model.discharge_power[index] * dt / battery.discharge_efficiency
        )

    model.battery_state = Constraint(model.intervals, rule=battery_state)

    def peak_limit(model, index):
        return model.grid_import[index] <= model.peak_import * dt

    model.peak_limit = Constraint(model.intervals, rule=peak_limit)

    energy_cost = sum(
        ems.import_price_per_kwh * model.grid_import[index]
        - ems.export_price_per_kwh * model.grid_export[index]
        for index in model.intervals
    )
    throughput_cost = sum(
        ems.throughput_cost_per_kwh
        * (model.charge_power[index] + model.discharge_power[index])
        * dt
        for index in model.intervals
    )

    model.objective = Objective(
        expr=(
            ems.energy_cost_weight * energy_cost
            + throughput_cost
            + ems.peak_penalty_per_kw * model.peak_import
        ),
        sense=minimize,
    )
    return model


def _values(variable, length: int) -> np.ndarray:
    return np.array([value(variable[index]) for index in range(length)])
