"""Injected FastAPI application for cached telemetry analytics."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Response, status
from sqlalchemy.orm import Session, sessionmaker

from ingen_pydev.analytics.cache import TTLCache
from ingen_pydev.analytics.models import (
    AlertResponse,
    DeviceSummaryResponse,
    PaginatedAlertsResponse,
)
from ingen_pydev.analytics.queries import (
    DeviceSummaryResult,
    count_matching_alerts,
    get_alert_page,
    get_device_summary,
)
from ingen_pydev.db.models import current_unix_ms

logger = logging.getLogger(__name__)


def create_app(
    session_factory: sessionmaker[Session],
    cache: TTLCache[DeviceSummaryResponse],
    *,
    generated_at_ms: Callable[[], int] = current_unix_ms,
    close_resources: Callable[[], None] | None = None,
) -> FastAPI:
    """Create an API with explicitly injected database and cache dependencies."""

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if close_resources is not None:
                close_resources()

    application = FastAPI(
        title="InGen Telemetry Analytics API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get(
        "/devices/{device_id}/summary",
        response_model=DeviceSummaryResponse,
        tags=["devices"],
    )
    def read_device_summary(
        device_id: str,
        response: Response,
    ) -> DeviceSummaryResponse:
        cached = cache.get(device_id)
        if cached is not None:
            response.headers["X-Cache"] = "HIT"
            _log_cache_event(cache, "HIT", device_id)
            return cached

        response.headers["X-Cache"] = "MISS"
        _log_cache_event(cache, "MISS", device_id)
        with session_factory() as session:
            summary = get_device_summary(session, device_id)

        if summary is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found",
            )

        payload = _build_response(summary, generated_at_ms())
        cache.set(device_id, payload)
        return payload

    @application.get(
        "/alerts",
        response_model=PaginatedAlertsResponse,
        tags=["alerts"],
    )
    def read_alerts(
        since: Annotated[int, Query(ge=0)],
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
        device_id: str | None = None,
    ) -> PaginatedAlertsResponse:
        with session_factory() as session:
            total = count_matching_alerts(session, since, device_id)
            alerts = get_alert_page(session, since, limit, offset, device_id)

        page_end = offset + len(alerts)
        next_offset = page_end if page_end < total else None

        return PaginatedAlertsResponse(
            items=[
                AlertResponse(
                    alert_id=alert.alert_id,
                    device_id=alert.device_id,
                    session_id=alert.session_id,
                    reading_id=alert.reading_id,
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    detected_at_ms=alert.detected_at_ms,
                    source=alert.source,
                    message=alert.message,
                )
                for alert in alerts
            ],
            total=total,
            limit=limit,
            offset=offset,
            next_offset=next_offset,
        )

    return application


def _build_response(
    summary: DeviceSummaryResult,
    generated_at_ms: int,
) -> DeviceSummaryResponse:
    return DeviceSummaryResponse(
        device_id=summary.device_id,
        device_name=summary.device_name,
        product_anchor=summary.product_anchor,
        session_count=summary.session_count,
        reading_count=summary.reading_count,
        alert_count=summary.alert_count,
        average_battery_soc=summary.average_battery_soc,
        low_health_count=summary.low_health_count,
        gps_dropout_count=summary.gps_dropout_count,
        latest_timestamp_ms=summary.latest_timestamp_ms,
        generated_at_ms=generated_at_ms,
    )


def _log_cache_event(
    cache: TTLCache[DeviceSummaryResponse],
    event: str,
    key: str,
) -> None:
    cache_stats = cache.stats()
    logger.info(
        "device_summary_cache event=%s key=%r hits=%d misses=%d hit_rate=%.3f",
        event,
        key,
        cache_stats.hits,
        cache_stats.misses,
        cache_stats.hit_rate,
    )
