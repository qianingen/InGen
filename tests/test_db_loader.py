from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select

from ingen_pydev.db.database import create_sqlite_engine, make_session_factory
from ingen_pydev.db.loader import load_week3_output
from ingen_pydev.db.models import Alert, Device, SensorReading, TelemetrySession


def test_load_week3_output_inserts_device_session_readings_and_alerts(
    tmp_path: Path,
) -> None:
    parquet_path, summary_path = _write_week3_inputs(tmp_path)
    db_path = tmp_path / "w04_telemetry.db"

    load_week3_output(
        parquet_path=parquet_path,
        summary_json_path=summary_path,
        db_path=db_path,
        device_id="aido_rover_001",
        device_name="Aido Rover 001",
        product_anchor="Aido Rover",
    )

    engine = create_sqlite_engine(db_path)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        device_count = session.scalar(select(func.count(Device.device_id)))
        session_count = session.scalar(select(func.count(TelemetrySession.session_id)))
        reading_count = session.scalar(select(func.count(SensorReading.reading_id)))
        alert_count = session.scalar(select(func.count(Alert.alert_id)))

        assert device_count == 1
        assert session_count == 1
        assert reading_count == 5

        # Alerts are generated from row-level anomaly flags and health thresholds.
        assert alert_count == 5


def test_loader_is_idempotent_on_replay(tmp_path: Path) -> None:
    parquet_path, summary_path = _write_week3_inputs(tmp_path)
    db_path = tmp_path / "w04_telemetry.db"

    load_week3_output(
        parquet_path=parquet_path,
        summary_json_path=summary_path,
        db_path=db_path,
        device_id="aido_rover_001",
        device_name="Aido Rover 001",
        product_anchor="Aido Rover",
    )

    load_week3_output(
        parquet_path=parquet_path,
        summary_json_path=summary_path,
        db_path=db_path,
        device_id="aido_rover_001",
        device_name="Aido Rover 001",
        product_anchor="Aido Rover",
    )

    engine = create_sqlite_engine(db_path)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        device_count = session.scalar(select(func.count(Device.device_id)))
        session_count = session.scalar(select(func.count(TelemetrySession.session_id)))
        reading_count = session.scalar(select(func.count(SensorReading.reading_id)))
        alert_count = session.scalar(select(func.count(Alert.alert_id)))

        assert device_count == 1
        assert session_count == 1
        assert reading_count == 5
        assert alert_count == 5


def test_loader_raises_when_summary_row_count_mismatches(tmp_path: Path) -> None:
    parquet_path, summary_path = _write_week3_inputs(tmp_path)
    db_path = tmp_path / "w04_telemetry.db"

    bad_summary = {
        "rows_processed": 999,
        "rows_flagged": 0,
        "fields_with_anomalies_detected": [],
    }
    summary_path.write_text(json.dumps(bad_summary), encoding="utf-8")

    with pytest.raises(ValueError, match="row count"):
        load_week3_output(
            parquet_path=parquet_path,
            summary_json_path=summary_path,
            db_path=db_path,
            device_id="aido_rover_001",
            device_name="Aido Rover 001",
            product_anchor="Aido Rover",
        )


def test_loader_raises_when_required_column_is_missing(tmp_path: Path) -> None:
    df = _make_week3_feature_df()
    df = df.drop(columns=["battery_soc"])

    parquet_path = tmp_path / "cleaned_features.parquet"
    summary_path = tmp_path / "validation_summary.json"
    db_path = tmp_path / "w04_telemetry.db"

    df.to_parquet(parquet_path, index=False)

    summary = {
        "rows_processed": len(df),
        "rows_flagged": 0,
        "fields_with_anomalies_detected": [],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="missing|required|column"):
        load_week3_output(
            parquet_path=parquet_path,
            summary_json_path=summary_path,
            db_path=db_path,
            device_id="aido_rover_001",
            device_name="Aido Rover 001",
            product_anchor="Aido Rover",
        )


def _write_week3_inputs(tmp_path: Path) -> tuple[Path, Path]:
    df = _make_week3_feature_df()

    parquet_path = tmp_path / "cleaned_features.parquet"
    summary_path = tmp_path / "validation_summary.json"

    df.to_parquet(parquet_path, index=False)

    summary = {
        "rows_processed": len(df),
        "rows_flagged": 4,
        "fields_with_anomalies_detected": [
            "gps",
            "lidar_distance_m",
            "battery_soc",
            "wheel_torque",
            "ambient_temp_c",
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    return parquet_path, summary_path


def _make_week3_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ms": [1000, 2000, 3000, 4000, 5000],
            "lat": [40.0, 40.0001, 40.0002, 40.0003, 40.0004],
            "lon": [-88.0, -88.0001, -88.0002, -88.0003, -88.0004],
            "lidar_distance_m": [10.0, 0.0, 12.0, 13.0, 14.0],
            "battery_soc": [95.0, 94.5, 70.0, 69.5, 69.0],
            "wheel_torque_fl": [10.0, 10.1, 10.2, 18.0, 10.3],
            "wheel_torque_fr": [10.0, 10.1, 10.2, 8.0, 10.3],
            "wheel_torque_rl": [10.0, 10.1, 10.2, 10.0, 10.3],
            "wheel_torque_rr": [10.0, 10.1, 10.2, 10.0, 10.3],
            "ambient_temp_c": [22.0, 22.1, 30.0, 22.3, 22.4],
            "gps_filled": [False, False, False, False, False],
            "gps_dropout_long": [True, False, False, False, False],
            "lidar_saturated": [False, True, False, False, False],
            "battery_soc_spike": [False, False, True, False, False],
            "lidar_distance_m_spike": [False, False, False, False, False],
            "ambient_temp_c_spike": [False, False, False, True, False],
            "battery_soc_roll_mean_50": [95.0, 94.75, 86.5, 82.25, 79.6],
            "battery_soc_roll_std_50": [0.0, 0.25, 11.67, 12.9, 12.67],
            "cumulative_distance_m": [0.0, 14.0, 28.0, 42.0, 56.0],
            "wheel_imbalance_score": [0.0, 0.0, 0.0, 0.87, 0.0],
            "lidar_saturation_rate": [0.0, 0.5, 0.33, 0.25, 0.2],
            "composite_health_score": [95.0, 47.25, 46.9, 52.1, 55.2],
        }
    )