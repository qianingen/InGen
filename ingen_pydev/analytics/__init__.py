"""Cached, typed analytics services for telemetry APIs."""

from ingen_pydev.analytics.cache import (
    DEFAULT_TTL_SECONDS,
    CacheStats,
    TTLCache,
)
from ingen_pydev.analytics.forecast import (
    BATTERY_MODEL_ORDER,
    FORECAST_ALPHA,
    MAX_FORECAST_HORIZON,
    MIN_BATTERY_HISTORY,
    BatteryForecastPointResult,
    BatteryForecastResult,
    BatteryForecastServiceError,
    build_battery_forecast,
)
from ingen_pydev.analytics.models import (
    AlertResponse,
    BatteryForecastPoint,
    BatteryForecastResponse,
    DeviceSummaryResponse,
    PaginatedAlertsResponse,
)
from ingen_pydev.analytics.queries import (
    AlertResult,
    BatteryHistoryResult,
    DeviceSummaryResult,
    count_matching_alerts,
    get_alert_page,
    get_battery_history,
    get_device_summary,
)

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "AlertResponse",
    "AlertResult",
    "BATTERY_MODEL_ORDER",
    "FORECAST_ALPHA",
    "MAX_FORECAST_HORIZON",
    "MIN_BATTERY_HISTORY",
    "BatteryForecastPoint",
    "BatteryForecastPointResult",
    "BatteryForecastResponse",
    "BatteryForecastResult",
    "BatteryForecastServiceError",
    "BatteryHistoryResult",
    "CacheStats",
    "DeviceSummaryResponse",
    "DeviceSummaryResult",
    "PaginatedAlertsResponse",
    "TTLCache",
    "build_battery_forecast",
    "count_matching_alerts",
    "get_alert_page",
    "get_battery_history",
    "get_device_summary",
]
