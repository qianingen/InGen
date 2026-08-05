"""Typed SQLAlchemy operations for telemetry analytics."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, func, select
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
            float(average_battery_soc)
            if average_battery_soc is not None
            else None
        ),
        low_health_count=int(low_health_count),
        gps_dropout_count=int(gps_dropout_count),
        latest_timestamp_ms=(
            int(latest_timestamp_ms) if latest_timestamp_ms is not None else None
        ),
    )
