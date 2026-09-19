import numpy as np
import pandas as pd
import pytest

from src.battery import BatteryConfig
from src.ems import (
    EMSConfig,
    apply_objective_profile,
    prepare_forecast_test_data,
    prepare_quantile_test_data,
    run_forecast_ems_comparison,
    run_quantile_ems_comparison,
)


def make_battery() -> BatteryConfig:
    return BatteryConfig(5.0, 3.0, 3.0, 0.95, 0.95, 0.10, 0.90, 0.50, 0.5)


def make_ems() -> EMSConfig:
    return EMSConfig(4, "appsi_highs", 0.30, 0.08, 0.0, 0.001, 0.50)


def make_predictions(median=None) -> pd.DataFrame:
    actual = np.array([-1.0, -1.0, 1.0, 1.0] * 2)
    median = actual if median is None else np.asarray(median)
    timestamps = pd.date_range("2013-06-01 00:30", periods=8, freq="30min")
    return pd.DataFrame({
        "unique_id": "customer_001",
        "ds": timestamps,
        "cutoff": [pd.Timestamp("2013-06-01 00:00")] * 4
        + [pd.Timestamp("2013-06-01 02:00")] * 4,
        "y": actual,
        "q_0.10": median - 0.8,
        "q_0.25": median - 0.4,
        "q_0.50": median,
        "q_0.75": median + 0.4,
        "q_0.90": median + 0.8,
    })


def prepare(predictions) -> pd.DataFrame:
    return prepare_forecast_test_data(predictions, horizon=4, expected_intervals=8)


def test_preparation_checks_cutoff_horizon_and_half_hour_alignment():
    data = prepare(make_predictions())
    assert len(data) == 8
    assert data["cutoff"].nunique() == 2
    assert not data["ds"].duplicated().any()


def test_misaligned_forecast_window_is_rejected():
    predictions = make_predictions()
    predictions.loc[2, "ds"] = predictions.loc[2, "ds"] + pd.Timedelta(minutes=10)
    with pytest.raises(ValueError, match="without gaps"):
        prepare(predictions)


def test_crossed_quantiles_are_rearranged():
    predictions = make_predictions()
    predictions.loc[0, ["q_0.10", "q_0.90"]] = [2.0, -2.0]
    data = prepare_quantile_test_data(
        predictions,
        ["q_0.50", "q_0.75", "q_0.90"],
        horizon=4,
        expected_intervals=8,
    )
    expected = np.sort(
        predictions.loc[0, ["q_0.10", "q_0.25", "q_0.50", "q_0.75", "q_0.90"]]
    )
    assert data.loc[0, "q_0.50"] == pytest.approx(expected[2])
    assert data.loc[0, "q_0.75"] == pytest.approx(expected[3])
    assert data.loc[0, "q_0.90"] == pytest.approx(expected[4])


def test_identical_median_matches_perfect_foresight():
    schedules, summary = run_forecast_ems_comparison(
        prepare(make_predictions()), make_battery(), make_ems()
    )
    values = summary.set_index("strategy")
    assert values.loc["median_forecast", "objective_value"] == pytest.approx(
        values.loc["perfect_foresight", "objective_value"]
    )
    assert len(schedules) == 16


def test_quantile_comparison_contains_all_strategies():
    data = prepare_quantile_test_data(
        make_predictions(),
        ["q_0.50", "q_0.75", "q_0.90"],
        horizon=4,
        expected_intervals=8,
    )
    schedules, summary = run_quantile_ems_comparison(
        data,
        make_battery(),
        make_ems(),
        {
            "q50_forecast": "q_0.50",
            "q75_forecast": "q_0.75",
            "q90_forecast": "q_0.90",
        },
    )
    assert set(summary["strategy"]) == {
        "no_battery",
        "perfect_foresight",
        "q50_forecast",
        "q75_forecast",
        "q90_forecast",
    }
    assert len(schedules) == 32


def test_quantile_schedule_is_evaluated_on_actual_load():
    predictions = make_predictions(median=np.zeros(8))
    data = prepare_quantile_test_data(
        predictions,
        ["q_0.50"],
        horizon=4,
        expected_intervals=8,
    )
    schedules, _ = run_quantile_ems_comparison(
        data, make_battery(), make_ems(), {"q50_forecast": "q_0.50"}
    )
    q50 = schedules[schedules["strategy"] == "q50_forecast"]
    assert np.allclose(q50["net_load_kwh"], data["actual_net_load_kwh"])
    assert np.allclose(q50["planned_net_load_kwh"], 0.0)


def test_missing_quantile_is_rejected():
    predictions = make_predictions().drop(columns="q_0.90")
    with pytest.raises(ValueError, match="Missing forecast columns"):
        prepare_quantile_test_data(
            predictions,
            ["q_0.50", "q_0.75", "q_0.90"],
            horizon=4,
            expected_intervals=8,
        )


def test_objective_profile_changes_only_objective_weights():
    base = make_ems()
    peak = apply_objective_profile(
        base, {"energy_cost_weight": 0.001, "peak_penalty_per_kw": 1.0}
    )
    assert peak.energy_cost_weight == 0.001
    assert peak.peak_penalty_per_kw == 1.0
    assert peak.import_price_per_kwh == base.import_price_per_kwh


def test_wrong_forecast_length_is_rejected():
    with pytest.raises(ValueError, match="Expected 8"):
        prepare(make_predictions().iloc[:-1])
