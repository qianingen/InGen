"""Output stage for cleaned feature tables and validation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

GPS_FLAG_COLUMNS = ["gps_dropout", "gps_filled", "gps_dropout_long"]
KNOWN_FLAG_COLUMNS = [
    *GPS_FLAG_COLUMNS,
    "lidar_saturated",
    "lidar_distance_m_spike",
    "battery_soc_spike",
    "wheel_torque_fl_spike",
    "wheel_torque_fr_spike",
    "wheel_torque_rl_spike",
    "wheel_torque_rr_spike",
    "ambient_temp_c_spike",
]


def write_pipeline_outputs(
    df: pd.DataFrame,
    parquet_path: str | Path,
    summary_json_path: str | Path,
) -> dict[str, Any]:
    """Write feature-enriched telemetry to Parquet and summary JSON."""

    parquet_file = Path(parquet_path)
    summary_file = Path(summary_json_path)
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(parquet_file, index=False)
    summary = build_validation_summary(df)

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def build_validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a validation summary from cleaning/anomaly flag columns."""

    flag_columns = [column for column in KNOWN_FLAG_COLUMNS if column in df.columns]
    if flag_columns:
        flagged_mask = df.loc[:, flag_columns].fillna(False).astype(bool).any(axis=1)
        rows_flagged = int(flagged_mask.sum())
    else:
        rows_flagged = 0

    fields = _fields_with_anomalies(df)
    return {
        "rows_processed": int(len(df)),
        "rows_flagged": rows_flagged,
        "fields_with_anomalies_detected": fields,
    }


def _fields_with_anomalies(df: pd.DataFrame) -> list[str]:
    fields: list[str] = []

    if _any_flagged(df, GPS_FLAG_COLUMNS):
        fields.append("gps")
    if _any_flagged(df, ["lidar_saturated", "lidar_distance_m_spike"]):
        fields.append("lidar_distance_m")
    if _any_flagged(df, ["battery_soc_spike"]):
        fields.append("battery_soc")
    if _any_flagged(
        df,
        [
            "wheel_torque_fl_spike",
            "wheel_torque_fr_spike",
            "wheel_torque_rl_spike",
            "wheel_torque_rr_spike",
        ],
    ):
        fields.append("wheel_torque")
    if _any_flagged(df, ["ambient_temp_c_spike"]):
        fields.append("ambient_temp_c")

    return fields


def _any_flagged(df: pd.DataFrame, columns: list[str]) -> bool:
    existing_columns = [column for column in columns if column in df.columns]
    if not existing_columns:
        return False
    return bool(df.loc[:, existing_columns].fillna(False).astype(bool).any().any())
