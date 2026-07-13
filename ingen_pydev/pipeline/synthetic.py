"""Synthetic Aido Rover telemetry dataset generation.

The generated data is public-safe synthetic telemetry. It is not based on
private robot logs, private APIs, or internal production data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

LIDAR_MAX_RANGE_M = 200.0
DEFAULT_ROWS = 10_000
DEFAULT_SEED = 42


def generate_synthetic_telemetry(
    output_csv: str | Path,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate synthetic Aido Rover telemetry and save it as CSV.

    The output columns are:
    timestamp_ms, lat, lon, lidar_distance_m, battery_soc,
    wheel_torque_fl, wheel_torque_fr, wheel_torque_rl, wheel_torque_rr,
    ambient_temp_c.
    """

    if rows < 2:
        raise ValueError("rows must be at least 2")

    rng = np.random.default_rng(seed)

    timestamps = 1_735_689_600_000 + np.arange(rows, dtype=np.int64) * 1_000

    # Local random walk around a plausible campus/location coordinate.
    base_lat = 40.1106
    base_lon = -88.2073
    lat_steps = rng.normal(0.0, 0.000003, rows)
    lon_steps = rng.normal(0.0, 0.000003, rows)
    lat = base_lat + np.cumsum(lat_steps)
    lon = base_lon + np.cumsum(lon_steps)

    t = np.linspace(0.0, 8.0 * np.pi, rows)
    lidar_distance_m = 35.0 + 8.0 * np.sin(t) + rng.normal(0.0, 1.5, rows)
    lidar_distance_m = np.clip(lidar_distance_m, 0.5, LIDAR_MAX_RANGE_M - 1.0)

    battery_soc = 100.0 - np.linspace(0.0, 18.0, rows) + rng.normal(0.0, 0.12, rows)
    battery_soc = np.clip(battery_soc, 0.0, 100.0)

    wheel_base = 12.0 + 1.5 * np.sin(t / 2.0)
    wheel_torque_fl = wheel_base + rng.normal(0.0, 0.5, rows)
    wheel_torque_fr = wheel_base + rng.normal(0.0, 0.5, rows)
    wheel_torque_rl = wheel_base + rng.normal(0.0, 0.5, rows)
    wheel_torque_rr = wheel_base + rng.normal(0.0, 0.5, rows)

    ambient_temp_c = 22.0 + 4.0 * np.sin(t / 5.0) + rng.normal(0.0, 0.35, rows)

    df = pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "lat": lat,
            "lon": lon,
            "lidar_distance_m": lidar_distance_m,
            "battery_soc": battery_soc,
            "wheel_torque_fl": wheel_torque_fl,
            "wheel_torque_fr": wheel_torque_fr,
            "wheel_torque_rl": wheel_torque_rl,
            "wheel_torque_rr": wheel_torque_rr,
            "ambient_temp_c": ambient_temp_c,
        }
    )

    _inject_gps_dropout(df, rng, rate=0.02)
    _inject_lidar_saturation(df, rng, rate=0.01)
    _inject_battery_drops(df, rng, rate=0.03)
    _inject_noise_spikes(df, rng, rate=0.05)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _sample_indices(
    rng: np.random.Generator,
    rows: int,
    rate: float,
    start: int = 0,
) -> npt.NDArray[np.int64]:
    count = max(1, int(rows * rate))
    return rng.choice(np.arange(start, rows), size=count, replace=False)


def _inject_gps_dropout(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float,
) -> None:
    indices = _sample_indices(rng, len(df), rate)
    df.loc[indices, ["lat", "lon"]] = np.nan


def _inject_lidar_saturation(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float,
) -> None:
    indices = _sample_indices(rng, len(df), rate)
    half = len(indices) // 2
    df.loc[indices[:half], "lidar_distance_m"] = 0.0
    df.loc[indices[half:], "lidar_distance_m"] = LIDAR_MAX_RANGE_M


def _inject_battery_drops(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float,
) -> None:
    indices = _sample_indices(rng, len(df), rate, start=1)
    for index in indices:
        previous = float(df.loc[index - 1, "battery_soc"])
        drop = float(rng.uniform(21.0, 30.0))
        df.loc[index, "battery_soc"] = max(0.0, previous - drop)


def _inject_noise_spikes(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float,
) -> None:
    sensor_columns = [
        "lidar_distance_m",
        "battery_soc",
        "wheel_torque_fl",
        "wheel_torque_fr",
        "wheel_torque_rl",
        "wheel_torque_rr",
        "ambient_temp_c",
    ]
    indices = _sample_indices(rng, len(df), rate)
    for index in indices:
        column = str(rng.choice(sensor_columns))
        value = float(df.loc[index, column])
        if column == "battery_soc":
            df.loc[index, column] = np.clip(value + rng.normal(0.0, 15.0), 0.0, 100.0)
        elif column == "lidar_distance_m":
            df.loc[index, column] = np.clip(
                value + rng.normal(0.0, 35.0),
                0.0,
                LIDAR_MAX_RANGE_M,
            )
        elif column == "ambient_temp_c":
            df.loc[index, column] = value + rng.normal(0.0, 8.0)
        else:
            df.loc[index, column] = value + rng.normal(0.0, 8.0)


if __name__ == "__main__":
    generate_synthetic_telemetry("synthetic_data.csv")
