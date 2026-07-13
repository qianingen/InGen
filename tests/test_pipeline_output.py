from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingen_pydev.pipeline.output import build_validation_summary, write_pipeline_outputs


def _output_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ms": [1, 2],
            "lat": [40.0, 40.1],
            "lon": [-88.0, -88.1],
            "lidar_distance_m": [10.0, 11.0],
            "battery_soc": [90.0, 89.0],
            "gps_dropout": [False, True],
            "gps_filled": [False, True],
            "gps_dropout_long": [False, False],
            "lidar_saturated": [False, False],
            "battery_soc_spike": [False, False],
            "battery_soc_roll_mean_50": [90.0, 89.5],
            "battery_soc_roll_std_50": [0.0, 0.5],
            "cumulative_distance_m": [0.0, 14.0],
            "wheel_imbalance_score": [0.1, 0.2],
            "lidar_saturation_rate": [0.0, 0.0],
            "composite_health_score": [90.0, 89.0],
        }
    )


def test_validation_summary_counts_flagged_rows() -> None:
    summary = build_validation_summary(_output_df())

    assert summary["rows_processed"] == 2
    assert summary["rows_flagged"] == 1
    assert summary["fields_with_anomalies_detected"] == ["gps"]


def test_parquet_round_trip_and_summary_json(tmp_path: Path) -> None:
    parquet_path = tmp_path / "features.parquet"
    summary_path = tmp_path / "summary.json"
    df = _output_df()

    summary = write_pipeline_outputs(df, parquet_path, summary_path)
    round_trip = pd.read_parquet(parquet_path)

    assert parquet_path.exists()
    assert summary_path.exists()
    assert list(round_trip.columns) == list(df.columns)
    assert len(round_trip) == len(df)

    with summary_path.open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)

    assert saved_summary == summary
