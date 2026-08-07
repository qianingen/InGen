"""Validated Week 5 battery-forecast adapter for the analytics API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BATTERY_MODEL_ORDER = (0, 1, 1)
MIN_BATTERY_HISTORY = 50
FORECAST_ALPHA = 0.05
MAX_FORECAST_HORIZON = 300


class BatteryForecastServiceError(ValueError):
    """Expected history or modeling failure suitable for an HTTP 422 response."""


@dataclass(frozen=True)
class BatteryForecastPointResult:
    """Validated numeric values for one forecast step."""

    step: int
    forecast_battery_soc: float
    lower_95: float
    upper_95: float


@dataclass(frozen=True)
class BatteryForecastResult:
    """Validated API-ready output from the Week 5 forecasting functions."""

    model_order: tuple[int, int, int]
    history_n: int
    horizon: int
    points: tuple[BatteryForecastPointResult, ...]


def build_battery_forecast(
    battery_history: Sequence[float],
    horizon: int,
) -> BatteryForecastResult:
    """Fit the validated ARIMA(0,1,1) model and forecast a fixed horizon.

    At least 50 finite observations are required. Missing and non-finite
    observations are rejected rather than silently removed so the API never
    changes the chronological series supplied by the database.
    """

    if not 1 <= horizon <= MAX_FORECAST_HORIZON:
        raise BatteryForecastServiceError(
            f"horizon must be between 1 and {MAX_FORECAST_HORIZON}"
        )

    try:
        history_values = np.asarray(list(battery_history), dtype=float)
    except (TypeError, ValueError) as exc:
        raise BatteryForecastServiceError(
            "Battery history contains missing or non-numeric values"
        ) from exc

    if history_values.ndim != 1:
        raise BatteryForecastServiceError("Battery history must be one-dimensional")
    if len(history_values) < MIN_BATTERY_HISTORY:
        raise BatteryForecastServiceError(
            f"Battery history requires at least {MIN_BATTERY_HISTORY} observations"
        )
    if not np.isfinite(history_values).all():
        raise BatteryForecastServiceError(
            "Battery history contains missing or non-finite values"
        )

    history = pd.Series(
        history_values,
        index=pd.RangeIndex(len(history_values)),
        dtype=float,
        name="battery_soc",
    )

    try:
        from W05_Predictive_Models import (
            fit_arima_forecaster,
            forecast_with_confidence_interval,
        )

        fitted_model: Any = fit_arima_forecaster(history, BATTERY_MODEL_ORDER)
        forecast = forecast_with_confidence_interval(
            fitted_model,
            horizon,
            alpha=FORECAST_ALPHA,
        )
    except (AssertionError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise BatteryForecastServiceError(
            "Battery history could not produce a forecast"
        ) from exc

    mean_values = np.asarray(forecast.mean, dtype=float)
    interval_values = np.asarray(forecast.confidence_interval, dtype=float)
    if mean_values.shape != (horizon,) or interval_values.shape != (horizon, 2):
        raise BatteryForecastServiceError(
            "Forecast output does not match the requested horizon"
        )
    if not np.isfinite(mean_values).all() or not np.isfinite(interval_values).all():
        raise BatteryForecastServiceError("Forecast output contains non-finite values")

    points: list[BatteryForecastPointResult] = []
    for index, forecast_value in enumerate(mean_values):
        lower = float(interval_values[index, 0])
        upper = float(interval_values[index, 1])
        forecast_float = float(forecast_value)
        if lower > forecast_float or forecast_float > upper:
            raise BatteryForecastServiceError(
                "Forecast confidence interval does not contain its point estimate"
            )
        points.append(
            BatteryForecastPointResult(
                step=index + 1,
                forecast_battery_soc=forecast_float,
                lower_95=lower,
                upper_95=upper,
            )
        )

    return BatteryForecastResult(
        model_order=BATTERY_MODEL_ORDER,
        history_n=len(history_values),
        horizon=horizon,
        points=tuple(points),
    )
