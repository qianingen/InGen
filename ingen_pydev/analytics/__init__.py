"""Cached, typed analytics services for telemetry APIs."""

from ingen_pydev.analytics.cache import (
    DEFAULT_TTL_SECONDS,
    CacheStats,
    TTLCache,
)
from ingen_pydev.analytics.models import DeviceSummaryResponse
from ingen_pydev.analytics.queries import DeviceSummaryResult, get_device_summary

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "CacheStats",
    "DeviceSummaryResponse",
    "DeviceSummaryResult",
    "TTLCache",
    "get_device_summary",
]
