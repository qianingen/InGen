from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ingen_pydev.pipeline.clean import clean_telemetry


def _base_df(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ms": list(range(rows)),
            "lat": [40.0 + i * 0.001 for i in range(rows)],
            "lon": [-88.0 - i * 0.001 for i in range(rows)],
            "lidar_distance_m": [10.0 + i for i in range(rows)],
            "battery_soc": [90.0 - i for i in range(rows)],
            "wheel_torque_fl": [10.0 for _ in range(rows)],
            "wheel_torque_fr": [11.0 for _ in range(rows)],
            "wheel_torque_rl": [12.0 for _ in range(rows)],
            "wheel_torque_rr": [13.0 for _ in range(rows)],
            "ambient_temp_c": [22.0 for _ in range(rows)],
        }
    )


def test_exactly_three_consecutive_gps_dropouts_are_forward_filled() -> None:
    df = _base_df()
    df.loc[1:3, ["lat", "lon"]] = np.nan

    cleaned = clean_telemetry(df)

    assert cleaned.loc[1:3, "gps_filled"].tolist() == [True, True, True]
    assert cleaned.loc[1:3, "gps_dropout_long"].tolist() == [False, False, False]
    assert cleaned.loc[1:3, "lat"].tolist() == [cleaned.loc[0, "lat"]] * 3
    assert cleaned.loc[1:3, "lon"].tolist() == [cleaned.loc[0, "lon"]] * 3


def test_fourth_consecutive_gps_dropout_is_flagged_long() -> None:
    df = _base_df(rows=7)
    df.loc[1:4, ["lat", "lon"]] = np.nan

    cleaned = clean_telemetry(df)

    assert cleaned.loc[1:3, "gps_filled"].tolist() == [True, True, True]
    assert bool(cleaned.loc[4, "gps_dropout_long"]) is True
    assert pd.isna(cleaned.loc[4, "lat"])
    assert pd.isna(cleaned.loc[4, "lon"])


def test_lidar_saturation_is_interpolated_and_flagged() -> None:
    df = _base_df(rows=5)
    df.loc[2, "lidar_distance_m"] = 0.0

    cleaned = clean_telemetry(df)

    assert bool(cleaned.loc[2, "lidar_saturated"]) is True
    lidar_value = cast(float, cleaned.loc[2, "lidar_distance_m"])
    assert lidar_value > 0.0
