"""Typed SQLAlchemy operations for telemetry analytics."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import BigInteger, Integer, bindparam, case, func, select
from sqlalchemy.orm import Session

from ingen_pydev.db.models import Alert, Device, SensorReading, TelemetrySession


@dataclass(frozen=True)
class DeviceSummaryResult:
    """Database-backed aggregate values for one existing device."""

    device_id: str
    device_name: str
    product_anchor: str
    session_count: int
    reading_count: int
    alert_count: int
    average_battery_soc: float | None
    low_health_count: int
    gps_dropout_count: int
    latest_timestamp_ms: int | None


@dataclass(frozen=True)
class AlertResult:
    """API-ready values for one alert row."""

    alert_id: str
    device_id: str
    session_id: str
    reading_id: str | None
    alert_type: str
    severity: int
    detected_at_ms: int
    source: str
    message: str | None


@dataclass(frozen=True)
class BatteryHistoryResult:
    """Chronologically ordered battery history for one existing device."""

    device_id: str
    battery_soc: tuple[float, ...]


def get_device_summary(
    session: Session,
    device_id: str,
) -> DeviceSummaryResult | None:
    """Return one device summary, or ``None`` when the device does not exist."""

    device = session.get(Device, device_id)
    if device is None:
        return None

    session_count = session.scalar(
        select(func.count(TelemetrySession.session_id)).where(
            TelemetrySession.device_id == device_id
        )
    )
    alert_count = session.scalar(
        select(func.count(Alert.alert_id)).where(Alert.device_id == device_id)
    )

    reading_statement = select(
        func.count(SensorReading.reading_id),
        func.avg(SensorReading.battery_soc),
        func.count(case((SensorReading.composite_health_score < 60, 1))),
        func.count(case((SensorReading.gps_dropout_long.is_(True), 1))),
        func.max(SensorReading.timestamp_ms),
    ).where(SensorReading.device_id == device_id)
    (
        reading_count,
        average_battery_soc,
        low_health_count,
        gps_dropout_count,
        latest_timestamp_ms,
    ) = session.execute(reading_statement).one()

    return DeviceSummaryResult(
        device_id=device.device_id,
        device_name=device.device_name,
        product_anchor=device.product_anchor,
        session_count=int(session_count or 0),
        reading_count=int(reading_count),
        alert_count=int(alert_count or 0),
        average_battery_soc=(
            float(average_battery_soc) if average_battery_soc is not None else None
        ),
        low_health_count=int(low_health_count),
        gps_dropout_count=int(gps_dropout_count),
        latest_timestamp_ms=(
            int(latest_timestamp_ms) if latest_timestamp_ms is not None else None
        ),
    )


def count_matching_alerts(
    session: Session,
    since: int,
    device_id: str | None = None,
) -> int:
    """Count alerts at or after ``since`` with an optional exact device filter."""

    statement = select(func.count(Alert.alert_id)).where(
        Alert.detected_at_ms >= bindparam("since", type_=BigInteger)
    )
    parameters: dict[str, object] = {"since": since}
    if device_id is not None:
        statement = statement.where(Alert.device_id == bindparam("device_id"))
        parameters["device_id"] = device_id

    return int(session.scalar(statement, parameters) or 0)


def get_alert_page(
    session: Session,
    since: int,
    limit: int,
    offset: int,
    device_id: str | None = None,
) -> list[AlertResult]:
    """Return one database-paginated alert page in stable chronological order."""

    statement = (
        select(
            Alert.alert_id,
            Alert.device_id,
            Alert.session_id,
            Alert.reading_id,
            Alert.alert_type,
            Alert.severity,
            Alert.detected_at_ms,
            Alert.source,
            Alert.message,
        )
        .where(Alert.detected_at_ms >= bindparam("since", type_=BigInteger))
        .order_by(Alert.detected_at_ms.asc(), Alert.alert_id.asc())
        .limit(bindparam("limit", type_=Integer))
        .offset(bindparam("offset", type_=Integer))
    )
    parameters: dict[str, object] = {
        "since": since,
        "limit": limit,
        "offset": offset,
    }
    if device_id is not None:
        statement = statement.where(Alert.device_id == bindparam("device_id"))
        parameters["device_id"] = device_id

    rows = session.execute(statement, parameters).mappings()
    return [
        AlertResult(
            alert_id=str(row["alert_id"]),
            device_id=str(row["device_id"]),
            session_id=str(row["session_id"]),
            reading_id=(
                str(row["reading_id"]) if row["reading_id"] is not None else None
            ),
            alert_type=str(row["alert_type"]),
            severity=int(row["severity"]),
            detected_at_ms=int(row["detected_at_ms"]),
            source=str(row["source"]),
            message=str(row["message"]) if row["message"] is not None else None,
        )
        for row in rows
    ]


def get_battery_history(
    session: Session,
    device_id: str,
) -> BatteryHistoryResult | None:
    """Return stable chronological battery history, or ``None`` if absent."""

    device = session.get(Device, device_id)
    if device is None:
        return None

    statement = (
        select(SensorReading.battery_soc)
        .where(SensorReading.device_id == bindparam("device_id"))
        .order_by(SensorReading.timestamp_ms.asc(), SensorReading.reading_id.asc())
    )
    battery_soc = tuple(
        float(value) for value in session.scalars(statement, {"device_id": device_id})
    )
    return BatteryHistoryResult(device_id=device.device_id, battery_soc=battery_soc)
