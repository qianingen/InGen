"""Idempotent loader for feature-enriched Week 3 telemetry output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from ingen_pydev.db.database import (
    create_all_tables,
    create_sqlite_engine,
    make_session_factory,
)
from ingen_pydev.db.models import Alert, Device, SensorReading, TelemetrySession


READING_COLUMNS = (
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
    "gps_filled",
    "gps_dropout_long",
    "lidar_saturated",
    "battery_soc_spike",
    "lidar_distance_m_spike",
    "ambient_temp_c_spike",
    "battery_soc_roll_mean_50",
    "battery_soc_roll_std_50",
    "cumulative_distance_m",
    "wheel_imbalance_score",
    "lidar_saturation_rate",
    "composite_health_score",
)


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _upsert(session: Session, model: type[Any], values: dict[str, Any]) -> None:
    statement = insert(model).values(**values)
    update_values = {
        key: getattr(statement.excluded, key)
        for key in values
        if key not in {column.name for column in model.__table__.primary_key}
        and key != "created_at_ms"
    }
    session.execute(statement.on_conflict_do_update(
        index_elements=[column.name for column in model.__table__.primary_key],
        set_=update_values,
    ))


def _python_value(value: Any) -> Any:
    """Convert pandas/numpy scalars and missing values for SQLite."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _alert_values(
    *,
    alert_type: str,
    severity: int,
    row: dict[str, Any],
    device_id: str,
    session_id: str,
    reading_id: str,
) -> dict[str, Any]:
    return {
        # One current alert per type and load session. Replays and repeated
        # anomalous rows update that alert instead of multiplying it.
        "alert_id": _stable_id(f"{session_id}:{alert_type}"),
        "device_id": device_id,
        "session_id": session_id,
        "reading_id": reading_id,
        "alert_type": alert_type,
        "severity": severity,
        "detected_at_ms": int(row["timestamp_ms"]),
        "source": "week3_loader",
        "message": f"{alert_type} detected in Week 3 telemetry output",
    }


def load_week3_output(
    parquet_path: str | Path,
    summary_json_path: str | Path,
    db_path: str | Path,
    device_id: str = "aido_rover_001",
    device_name: str = "Aido Rover 001",
    product_anchor: str = "Aido Rover",
) -> None:
    """Validate and atomically upsert a Week 3 output into SQLite."""
    parquet = Path(parquet_path)
    summary_path = Path(summary_json_path)
    frame = pd.read_parquet(parquet)
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    rows_processed = summary.get("rows_processed")
    if not isinstance(rows_processed, int):
        raise ValueError("Validation summary requires integer rows_processed")
    if len(frame) != rows_processed:
        raise ValueError(
            "Parquet row count does not match validation summary row count"
        )

    missing = sorted(set(READING_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Week 3 output is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Week 3 output must contain at least one row")
    if frame["timestamp_ms"].isna().any():
        raise ValueError("timestamp_ms cannot contain missing values")
    if frame["timestamp_ms"].duplicated().any():
        raise ValueError("timestamp_ms must be unique within a device load")

    engine = create_sqlite_engine(db_path)
    create_all_tables(engine)
    session_factory = make_session_factory(engine)
    source_file = str(parquet.resolve())
    session_id = _stable_id(f"{device_id}:{source_file}")

    with session_factory.begin() as session:
        _upsert(
            session,
            Device,
            {
                "device_id": device_id,
                "device_name": device_name,
                "product_anchor": product_anchor,
            },
        )
        _upsert(
            session,
            TelemetrySession,
            {
                "session_id": session_id,
                "device_id": device_id,
                "source_file": source_file,
                "started_at_ms": int(frame["timestamp_ms"].min()),
                "ended_at_ms": int(frame["timestamp_ms"].max()),
                "row_count": len(frame),
            },
        )

        raw_records = cast(
            list[dict[str, Any]],
            frame.loc[:, list(READING_COLUMNS)].to_dict(orient="records"),
        )
        for raw_row in raw_records:
            row = {key: _python_value(value) for key, value in raw_row.items()}
            timestamp_ms = int(row["timestamp_ms"])
            reading_id = _stable_id(f"{device_id}:{timestamp_ms}")
            reading = dict(row)
            reading.update(
                {
                    "reading_id": reading_id,
                    "device_id": device_id,
                    "session_id": session_id,
                    "timestamp_ms": timestamp_ms,
                }
            )
            _upsert(session, SensorReading, reading)

            rules = (
                ("gps_dropout_long", bool(row["gps_dropout_long"]), 3),
                ("lidar_saturation", bool(row["lidar_saturated"]), 2),
                ("battery_soc_spike", bool(row["battery_soc_spike"]), 2),
                ("wheel_imbalance", float(row["wheel_imbalance_score"]) > 0.35, 2),
                ("low_health_score", float(row["composite_health_score"]) < 60, 3),
            )
            for alert_type, triggered, severity in rules:
                if triggered:
                    _upsert(
                        session,
                        Alert,
                        _alert_values(
                            alert_type=alert_type,
                            severity=severity,
                            row=row,
                            device_id=device_id,
                            session_id=session_id,
                            reading_id=reading_id,
                        ),
                    )

        loaded_count = session.scalar(
            select(func.count(SensorReading.reading_id)).where(
                SensorReading.device_id == device_id,
                SensorReading.session_id == session_id,
            )
        )
        if loaded_count != len(frame):
            raise ValueError("Database row count mismatch after load")
