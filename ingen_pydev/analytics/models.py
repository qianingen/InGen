"""Pydantic response models for telemetry analytics endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DeviceSummaryResponse(BaseModel):
    """Aggregated telemetry scope and health indicators for one device."""

    model_config = ConfigDict(frozen=True)

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
    generated_at_ms: int


class AlertResponse(BaseModel):
    """One alert returned by the paginated alerts endpoint."""

    model_config = ConfigDict(frozen=True)

    alert_id: str
    device_id: str
    session_id: str
    reading_id: str | None
    alert_type: str
    severity: int
    detected_at_ms: int
    source: str
    message: str | None


class PaginatedAlertsResponse(BaseModel):
    """A stable page of alerts and its pagination metadata."""

    model_config = ConfigDict(frozen=True)

    items: list[AlertResponse]
    total: int
    limit: int
    offset: int
    next_offset: int | None


class BatteryForecastPoint(BaseModel):
    """One step in a battery state-of-charge forecast."""

    model_config = ConfigDict(frozen=True)

    step: int
    forecast_battery_soc: float
    lower_95: float
    upper_95: float


class BatteryForecastResponse(BaseModel):
    """Battery forecast metadata and fixed-horizon point estimates."""

    model_config = ConfigDict(frozen=True)

    device_id: str
    model_order: tuple[int, int, int]
    history_n: int
    horizon: int
    generated_at_ms: int
    points: list[BatteryForecastPoint]
