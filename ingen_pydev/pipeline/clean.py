"""Telemetry cleaning stage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ingen_pydev.algo.sliding_window import SlidingWindowTelemetry

LIDAR_MAX_RANGE_M = 200.0
SPIKE_COLUMNS = [
    "lidar_distance_m",
    "battery_soc",
    "wheel_torque_fl",
    "wheel_torque_fr",
    "wheel_torque_rl",
    "wheel_torque_rr",
    "ambient_temp_c",
]


def clean_telemetry(
    df: pd.DataFrame,
    window_size: int = 50,
    z_threshold: float = 3.0,
    clip_spikes: bool = False,
) -> pd.DataFrame:
    """Clean GPS dropout, LiDAR saturation, and numeric noise spikes."""

    out = df.copy()
    _clean_gps_dropout(out)
    _clean_lidar_saturation(out)
    _flag_noise_spikes(out, window_size=window_size, z_threshold=z_threshold)

    if clip_spikes:
        _clip_flagged_spikes(out, window_size=window_size, z_threshold=z_threshold)

    return out


def _clean_gps_dropout(df: pd.DataFrame) -> None:
    gps_missing = df["lat"].isna() | df["lon"].isna()
    run_id = gps_missing.ne(gps_missing.shift(fill_value=False)).cumsum()
    run_position = gps_missing.groupby(run_id).cumcount() + 1
    long_dropout = gps_missing & (run_position > 3)

    filled_gps = df[["lat", "lon"]].ffill(limit=3)
    can_fill = gps_missing & filled_gps["lat"].notna() & filled_gps["lon"].notna()

    df["gps_dropout"] = gps_missing
    df["gps_filled"] = can_fill & ~long_dropout
    df["gps_dropout_long"] = long_dropout
    df[["lat", "lon"]] = filled_gps


def _clean_lidar_saturation(df: pd.DataFrame) -> None:
    saturated = (df["lidar_distance_m"] <= 0.0) | (
        df["lidar_distance_m"] >= LIDAR_MAX_RANGE_M
    )
    df["lidar_saturated"] = saturated
    df.loc[saturated, "lidar_distance_m"] = np.nan
    df["lidar_distance_m"] = (
        df["lidar_distance_m"]
        .interpolate(method="linear", limit_direction="both")
        .clip(lower=0.0, upper=LIDAR_MAX_RANGE_M)
    )


def _flag_noise_spikes(
    df: pd.DataFrame,
    window_size: int,
    z_threshold: float,
) -> None:
    for column in SPIKE_COLUMNS:
        flags = _sliding_window_spike_flags(
            df[column],
            window_size=window_size,
            z_threshold=z_threshold,
        )
        df[f"{column}_spike"] = flags


def _sliding_window_spike_flags(
    series: pd.Series,
    window_size: int,
    z_threshold: float,
) -> pd.Series:
    detector = SlidingWindowTelemetry(
        window_size=window_size,
        anomaly_z_threshold=z_threshold,
    )
    flags: list[bool] = []

    for value in series:
        if pd.isna(value):
            flags.append(False)
            continue
        stats = detector.add(float(value))
        flags.append(bool(stats.is_anomaly))

    return pd.Series(flags, index=series.index, dtype=bool)


def _clip_flagged_spikes(
    df: pd.DataFrame,
    window_size: int,
    z_threshold: float,
) -> None:
    for column in SPIKE_COLUMNS:
        flag_column = f"{column}_spike"
        rolling_mean = df[column].rolling(window_size, min_periods=1).mean()
        rolling_std = df[column].rolling(window_size, min_periods=1).std(ddof=0)
        lower = rolling_mean - z_threshold * rolling_std
        upper = rolling_mean + z_threshold * rolling_std
        clipped = df[column].clip(lower=lower, upper=upper)
        df.loc[df[flag_column], column] = clipped.loc[df[flag_column]]
