"""Cached, typed analytics services for telemetry APIs."""

from ingen_pydev.analytics.cache import (
    DEFAULT_TTL_SECONDS,
    CacheStats,
    TTLCache,
)
from ingen_pydev.analytics.models import (
    AlertResponse,
    DeviceSummaryResponse,
    PaginatedAlertsResponse,
)
from ingen_pydev.analytics.queries import (
    AlertResult,
    DeviceSummaryResult,
    count_matching_alerts,
    get_alert_page,
    get_device_summary,
)

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "AlertResponse",
    "AlertResult",
    "CacheStats",
    "DeviceSummaryResponse",
    "DeviceSummaryResult",
    "PaginatedAlertsResponse",
    "TTLCache",
    "count_matching_alerts",
    "get_alert_page",
    "get_device_summary",
]
