"""Batched, idempotent loader for Week 3 telemetry output."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import pandas as pd
from sqlalchemy import Table, func, select
from sqlalchemy.dialects.sqlite import Insert, insert
from sqlalchemy.engine import Connection

from ingen_pydev.db.database import create_all_tables, create_sqlite_engine
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


@dataclass(frozen=True)
class LoadResult:
    """Summary and throughput metrics for one completed load."""

    session_id: str
    readings_processed: int
    alerts_processed: int
    elapsed_seconds: float
    rows_per_second: float


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _python_value(value: Any) -> object:
    """Convert pandas/numpy scalars and missing values for SQLite."""
    if pd.isna(value):
        return None
    converted = value.item() if hasattr(value, "item") else value
    return cast(object, converted)


def _chunks(
    rows: Sequence[dict[str, object]], batch_size: int
) -> Iterator[Sequence[dict[str, object]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _upsert_single(
    connection: Connection,
    table: Table,
    values: dict[str, object],
) -> None:
    statement = insert(table)
    primary_keys = {column.name for column in table.primary_key.columns}
    update_values = {
        key: getattr(statement.excluded, key)
        for key in values
        if key not in primary_keys and key != "created_at_ms"
    }
    connection.execute(
        statement.on_conflict_do_update(
            index_elements=list(primary_keys),
            set_=update_values,
        ),
        values,
    )


def _build_reading_rows(
    frame: pd.DataFrame,
    device_id: str,
    session_id: str,
) -> list[dict[str, object]]:
    """Materialize database-ready reading mappings before writing."""
    raw_records = cast(
        list[dict[str, Any]],
        frame.loc[:, list(READING_COLUMNS)].to_dict(orient="records"),
    )
    rows: list[dict[str, object]] = []
    for raw_record in raw_records:
        row = {key: _python_value(value) for key, value in raw_record.items()}
        timestamp_ms = int(cast(int, row["timestamp_ms"]))
        row.update(
            {
                "reading_id": _stable_id(f"{device_id}:{timestamp_ms}"),
                "device_id": device_id,
                "session_id": session_id,
                "timestamp_ms": timestamp_ms,
            }
        )
        rows.append(row)
    return rows


def _build_alert_rows(
    reading_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Materialize every row-level alert before starting database writes."""
    alerts: list[dict[str, object]] = []
    for row in reading_rows:
        reading_id = str(row["reading_id"])
        rules = (
            ("gps_dropout_long", bool(row["gps_dropout_long"]), 3),
            ("lidar_saturation", bool(row["lidar_saturated"]), 2),
            ("battery_soc_spike", bool(row["battery_soc_spike"]), 2),
            (
                "wheel_imbalance",
                float(cast(float, row["wheel_imbalance_score"])) > 0.35,
                2,
            ),
            (
                "low_health_score",
                float(cast(float, row["composite_health_score"])) < 60,
                3,
            ),
        )
        for alert_type, triggered, severity in rules:
            if not triggered:
                continue
            alerts.append(
                {
                    "alert_id": f"{reading_id}:{alert_type}",
                    "device_id": row["device_id"],
                    "session_id": row["session_id"],
                    "reading_id": reading_id,
                    "alert_type": alert_type,
                    "severity": severity,
                    "detected_at_ms": row["timestamp_ms"],
                    "source": "week3_loader",
                    "message": (f"{alert_type} detected in Week 3 telemetry output"),
                }
            )
    return alerts


def _reading_upsert_statement() -> Insert:
    reading_table = cast(Table, SensorReading.__table__)
    statement = insert(reading_table)
    immutable_columns = {
        "reading_id",
        "device_id",
        "timestamp_ms",
        "created_at_ms",
    }
    update_values = {
        column.name: getattr(statement.excluded, column.name)
        for column in reading_table.columns
        if column.name not in immutable_columns
    }
    return statement.on_conflict_do_update(
        index_elements=["device_id", "timestamp_ms"],
        set_=update_values,
    )


def _bulk_upsert_readings(
    connection: Connection,
    rows: Sequence[dict[str, object]],
    batch_size: int,
) -> None:
    """Upsert readings with one executemany call per batch."""
    statement = _reading_upsert_statement()
    for batch in _chunks(rows, batch_size):
        connection.execute(statement, list(batch))


def _alert_upsert_statement() -> Insert:
    alert_table = cast(Table, Alert.__table__)
    statement = insert(alert_table)
    immutable_columns = {"alert_id", "created_at_ms"}
    update_values = {
        column.name: getattr(statement.excluded, column.name)
        for column in alert_table.columns
        if column.name not in immutable_columns
    }
    return statement.on_conflict_do_update(
        index_elements=["alert_id"],
        set_=update_values,
    )


def _bulk_upsert_alerts(
    connection: Connection,
    rows: Sequence[dict[str, object]],
    batch_size: int,
) -> None:
    """Upsert alerts with one executemany call per batch."""
    statement = _alert_upsert_statement()
    for batch in _chunks(rows, batch_size):
        connection.execute(statement, list(batch))


def load_week3_output(
    parquet_path: str | Path,
    summary_json_path: str | Path,
    db_path: str | Path,
    device_id: str = "aido_rover_001",
    device_name: str = "Aido Rover 001",
    product_anchor: str = "Aido Rover",
    batch_size: int = 5_000,
) -> LoadResult:
    """Validate and atomically bulk-upsert a Week 3 output into SQLite."""
    started = time.perf_counter()
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

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

    source_file = str(parquet.resolve())
    session_id = _stable_id(f"{device_id}:{source_file}")
    reading_rows = _build_reading_rows(frame, device_id, session_id)
    alert_rows = _build_alert_rows(reading_rows)

    engine = create_sqlite_engine(db_path)
    try:
        create_all_tables(engine)
        with engine.begin() as connection:
            _upsert_single(
                connection,
                cast(Table, Device.__table__),
                {
                    "device_id": device_id,
                    "device_name": device_name,
                    "product_anchor": product_anchor,
                },
            )
            _upsert_single(
                connection,
                cast(Table, TelemetrySession.__table__),
                {
                    "session_id": session_id,
                    "device_id": device_id,
                    "source_file": source_file,
                    "started_at_ms": int(frame["timestamp_ms"].min()),
                    "ended_at_ms": int(frame["timestamp_ms"].max()),
                    "row_count": len(frame),
                },
            )
            _bulk_upsert_readings(connection, reading_rows, batch_size)
            _bulk_upsert_alerts(connection, alert_rows, batch_size)

            loaded_count = connection.scalar(
                select(func.count(SensorReading.reading_id)).where(
                    SensorReading.device_id == device_id,
                    SensorReading.session_id == session_id,
                )
            )
            if loaded_count != len(frame):
                raise ValueError("Database row count mismatch after load")
    finally:
        engine.dispose()

    elapsed_seconds = time.perf_counter() - started
    return LoadResult(
        session_id=session_id,
        readings_processed=len(reading_rows),
        alerts_processed=len(alert_rows),
        elapsed_seconds=elapsed_seconds,
        rows_per_second=len(reading_rows) / elapsed_seconds,
    )
