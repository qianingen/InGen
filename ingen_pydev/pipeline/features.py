"""Feature extraction stage for cleaned telemetry."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Any

BATTERY_WINDOW = 50
EARTH_RADIUS_M = 6_371_000.0
TORQUE_COLUMNS = [
    "wheel_torque_fl",
    "wheel_torque_fr",
    "wheel_torque_rl",
    "wheel_torque_rr",
]


def add_telemetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add battery, GPS, wheel-torque, and health-score features."""

    out = df.copy()

    out["battery_soc_roll_mean_50"] = (
        out["battery_soc"].rolling(BATTERY_WINDOW, min_periods=1).mean()
    )
    out["battery_soc_roll_std_50"] = (
        out["battery_soc"].rolling(BATTERY_WINDOW, min_periods=1).std(ddof=0)
    )

    out["cumulative_distance_m"] = _cumulative_distance_from_gps(out["lat"], out["lon"])
    out["wheel_imbalance_score"] = _wheel_imbalance_score(out)

    if "lidar_saturated" in out.columns:
        out["lidar_saturation_rate"] = (
            out["lidar_saturated"]
            .astype(float)
            .rolling(BATTERY_WINDOW, min_periods=1)
            .mean()
        )
    else:
        out["lidar_saturation_rate"] = 0.0

    out["composite_health_score"] = out["battery_soc"].clip(lower=0.0, upper=100.0) * (
        1.0 - out["lidar_saturation_rate"]
    )

    return out


def _cumulative_distance_from_gps(
    lat_series: pd.Series[Any],
    lon_series: pd.Series[Any],
) -> npt.NDArray[np.float64]:
    lat = np.radians(lat_series.astype(float).to_numpy())
    lon = np.radians(lon_series.astype(float).to_numpy())

    distances = np.zeros(len(lat), dtype=float)
    if len(lat) <= 1:
        return distances

    valid = ~(np.isnan(lat) | np.isnan(lon))
    pair_valid = valid[1:] & valid[:-1]

    delta_lat = lat[1:] - lat[:-1]
    delta_lon = lon[1:] - lon[:-1]
    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(delta_lon / 2.0) ** 2
    )
    segment_distance = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(haversine_a))
    segment_distance = np.where(pair_valid, segment_distance, 0.0)
    distances[1:] = np.cumsum(segment_distance)
    return distances


def _wheel_imbalance_score(df: pd.DataFrame) -> pd.Series[Any]:
    wheel_values = df.loc[:, TORQUE_COLUMNS]
    mean_torque = wheel_values.mean(axis=1).replace(0.0, np.nan)
    imbalance = (wheel_values.max(axis=1) - wheel_values.min(axis=1)) / mean_torque
    return imbalance.fillna(0.0)
