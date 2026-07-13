"""CSV ingestion and schema validation for synthetic telemetry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp_ms",
    "lat",
    "lon",
    "lidar_distance_m",
    "battery_soc",
    "wheel_torque_fl",
    "wheel_torque_fr",
    "wheel_torque_rl",
    "wheel_torque_rr",
    "ambient_temp_c",
]


def read_telemetry_csv(path: str | Path) -> pd.DataFrame:
    """Read a telemetry CSV and raise if required columns are missing."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.loc[:, REQUIRED_COLUMNS].copy()
