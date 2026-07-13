from __future__ import annotations

import numpy as np
import pandas as pd

from ingen_pydev.pipeline.features import add_telemetry_features


def _feature_df(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ms": list(range(rows)),
            "lat": [40.0 + i * 0.00001 for i in range(rows)],
            "lon": [-88.0 for _ in range(rows)],
            "lidar_distance_m": [20.0 for _ in range(rows)],
            "battery_soc": [100.0 - i * 0.1 for i in range(rows)],
            "wheel_torque_fl": [10.0 for _ in range(rows)],
            "wheel_torque_fr": [12.0 for _ in range(rows)],
            "wheel_torque_rl": [11.0 for _ in range(rows)],
            "wheel_torque_rr": [13.0 for _ in range(rows)],
            "ambient_temp_c": [22.0 for _ in range(rows)],
            "lidar_saturated": [False for _ in range(rows)],
        }
    )


def test_battery_rolling_stats_match_pandas_ground_truth() -> None:
    df = _feature_df()

    featured = add_telemetry_features(df)

    expected_mean = df["battery_soc"].rolling(50, min_periods=1).mean()
    expected_std = df["battery_soc"].rolling(50, min_periods=1).std(ddof=0)

    assert np.allclose(featured["battery_soc_roll_mean_50"], expected_mean)
    assert np.allclose(featured["battery_soc_roll_std_50"], expected_std)


def test_cumulative_distance_is_monotonic() -> None:
    featured = add_telemetry_features(_feature_df())

    assert featured["cumulative_distance_m"].iloc[0] == 0.0
    assert featured["cumulative_distance_m"].is_monotonic_increasing
    assert featured["cumulative_distance_m"].iloc[-1] > 0.0


def test_wheel_imbalance_score_uses_max_minus_min_over_mean() -> None:
    featured = add_telemetry_features(_feature_df(rows=1))

    expected = (13.0 - 10.0) / 11.5

    assert featured.loc[0, "wheel_imbalance_score"] == expected
