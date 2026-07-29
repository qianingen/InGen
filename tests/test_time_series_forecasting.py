from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from W05_Predictive_Models import (
    calculate_forecast_metrics,
    chronological_split,
    fit_arima_forecaster,
    forecast_with_confidence_interval,
    naive_persistence_forecast,
    run_adf_test,
    select_arima_order,
)


def test_chronological_split_never_shuffles_rows() -> None:
    series = pd.Series(
        np.arange(20, dtype=float),
        index=pd.RangeIndex(20),
    )

    train, test = chronological_split(series, train_fraction=0.80)

    assert train.index.tolist() == list(range(16))
    assert test.index.tolist() == list(range(16, 20))
    assert train.iloc[-1] == 15.0
    assert test.iloc[0] == 16.0


def test_naive_forecast_repeats_last_training_value() -> None:
    series = pd.Series(
        np.arange(20, dtype=float),
        index=pd.RangeIndex(20),
    )
    train, test = chronological_split(series)

    forecast = naive_persistence_forecast(train, test)

    assert forecast.index.equals(test.index)
    assert forecast.eq(train.iloc[-1]).all()


def test_calculate_forecast_metrics_reports_rmse_and_mae() -> None:
    actual = pd.Series([1.0, 2.0, 3.0])
    predicted = pd.Series([1.0, 1.0, 5.0])

    metrics = calculate_forecast_metrics(actual, predicted)

    assert np.isclose(metrics["rmse"], np.sqrt(5.0 / 3.0))
    assert np.isclose(metrics["mae"], 1.0)


def test_profile_adf_order_and_full_forecast_window() -> None:
    parquet_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "profile_cleaned_features.parquet"
    )
    battery_soc = pd.read_parquet(parquet_path)["battery_soc"].astype(float)

    raw_adf = run_adf_test(battery_soc)
    d = 0 if raw_adf.rejected else 1
    stationary = battery_soc if d == 0 else battery_soc.diff().dropna()
    order = select_arima_order(stationary, d=d).order
    train, test = chronological_split(battery_soc)
    fitted = fit_arima_forecaster(train, order)
    forecast = forecast_with_confidence_interval(
        fitted,
        steps=len(test),
        index=test.index,
    )
    naive = naive_persistence_forecast(train, test)

    arima_metrics = calculate_forecast_metrics(test, forecast.mean)
    naive_metrics = calculate_forecast_metrics(test, naive)

    assert raw_adf.p_value >= 0.05
    assert d == 1
    assert run_adf_test(stationary).p_value < 0.05
    assert order == (0, 1, 1)
    assert len(train) == 8_000
    assert len(test) == 2_000
    assert forecast.mean.shape == (2_000,)
    assert forecast.confidence_interval.shape == (2_000, 2)
    assert forecast.mean.index.equals(test.index)
    assert forecast.confidence_interval.index.equals(test.index)
    assert arima_metrics["rmse"] < naive_metrics["rmse"]
    assert arima_metrics["mae"] < naive_metrics["mae"]
