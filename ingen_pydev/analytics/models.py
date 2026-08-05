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
