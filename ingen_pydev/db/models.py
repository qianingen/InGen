"""SQLAlchemy schema models for Week 4 telemetry database."""

from __future__ import annotations

import time

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def current_unix_ms() -> int:
    """Return current Unix time in milliseconds."""
    return int(time.time() * 1000)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class Device(Base):
    """Physical or logical device that produces telemetry or alerts."""

    __tablename__ = "device"

    device_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_name: Mapped[str] = mapped_column(String(128), nullable=False)
    product_anchor: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=current_unix_ms,
    )


class TelemetrySession(Base):
    """One telemetry collection or pipeline-loading session."""

    __tablename__ = "session"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    device_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("device.device_id"),
        nullable=False,
    )
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    started_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ended_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=current_unix_ms,
    )


class SensorReading(Base):
    """Cleaned and feature-enriched telemetry reading from Week 3 output."""

    __tablename__ = "sensor_reading"

    reading_id: Mapped[str] = mapped_column(String(160), primary_key=True)

    device_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("device.device_id"),
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("session.session_id"),
        nullable=False,
    )

    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    lidar_distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    battery_soc: Mapped[float] = mapped_column(Float, nullable=False)

    wheel_torque_fl: Mapped[float] = mapped_column(Float, nullable=False)
    wheel_torque_fr: Mapped[float] = mapped_column(Float, nullable=False)
    wheel_torque_rl: Mapped[float] = mapped_column(Float, nullable=False)
    wheel_torque_rr: Mapped[float] = mapped_column(Float, nullable=False)

    ambient_temp_c: Mapped[float] = mapped_column(Float, nullable=False)

    gps_filled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gps_dropout_long: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    lidar_saturated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    battery_soc_spike: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    lidar_distance_m_spike: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    ambient_temp_c_spike: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    battery_soc_roll_mean_50: Mapped[float] = mapped_column(Float, nullable=False)
    battery_soc_roll_std_50: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    wheel_imbalance_score: Mapped[float] = mapped_column(Float, nullable=False)
    lidar_saturation_rate: Mapped[float] = mapped_column(Float, nullable=False)
    composite_health_score: Mapped[float] = mapped_column(Float, nullable=False)

    created_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=current_unix_ms,
    )

    __table_args__ = (
        UniqueConstraint(
            "device_id",
            "timestamp_ms",
            name="uq_sensor_reading_device_timestamp",
        ),
    )


class Alert(Base):
    """Sentinel-style alert event derived from telemetry anomalies."""

    __tablename__ = "alert"

    alert_id: Mapped[str] = mapped_column(String(160), primary_key=True)

    device_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("device.device_id"),
        nullable=False,
    )
    session_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("session.session_id"),
        nullable=False,
    )
    reading_id: Mapped[str | None] = mapped_column(
        String(160),
        ForeignKey("sensor_reading.reading_id"),
        nullable=True,
    )

    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    detected_at_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=current_unix_ms,
    )