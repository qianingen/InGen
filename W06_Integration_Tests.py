"""Week 6 end-to-end integration tests for the telemetry analytics API.

The tests exercise the real FastAPI application factory, TTL cache,
SQLAlchemy persistence layer, file-backed SQLite database, and Week 5 ARIMA
forecasting implementation.  Each test receives a freshly seeded database.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ingen_pydev.analytics.app import create_app
from ingen_pydev.analytics.cache import TTLCache
from ingen_pydev.analytics.forecast import MIN_BATTERY_HISTORY
from ingen_pydev.analytics.models import DeviceSummaryResponse
from ingen_pydev.db.database import (
    create_all_tables,
    create_sqlite_engine,
    make_session_factory,
)
from ingen_pydev.db.models import Alert, Device, SensorReading, TelemetrySession

PRIMARY_DEVICE_ID = "integration-device"
SHORT_HISTORY_DEVICE_ID = "integration-short-history"
UNKNOWN_DEVICE_ID = "integration-missing-device"
INJECTION_DEVICE_ID = "x' OR 1=1 --"
PRIMARY_SESSION_ID = "integration-session"
SHORT_SESSION_ID = "integration-short-session"
GENERATED_AT_MS = 1_700_000_000_000
ALERT_SINCE_MS = 1_000
FORECAST_HISTORY_N = 60
FORECAST_HORIZON = 5

BATTERY_HISTORY = tuple(
    82.0 - index * 0.05 + (index % 5) * 0.02
    for index in range(FORECAST_HISTORY_N)
)
EXPECTED_ALERT_IDS = tuple(
    f"integration-alert-{number:03d}" for number in range(1, 14)
)
ALERT_SPECS = (
    ("integration-alert-002", 1_000),
    ("integration-alert-001", 1_000),
    ("integration-alert-004", 2_000),
    ("integration-alert-003", 2_000),
    ("integration-alert-006", 3_000),
    ("integration-alert-005", 3_000),
    ("integration-alert-008", 4_000),
    ("integration-alert-007", 4_000),
    ("integration-alert-010", 5_000),
    ("integration-alert-009", 5_000),
    ("integration-alert-012", 6_000),
    ("integration-alert-011", 6_000),
    ("integration-alert-013", 7_000),
)


@dataclass(frozen=True)
class IntegrationEnvironment:
    """Real HTTP client and resources for one isolated integration test."""

    client: TestClient
    cache: TTLCache[DeviceSummaryResponse]
    database_path: Path


@pytest.fixture
def integration_environment(tmp_path: Path) -> Iterator[IntegrationEnvironment]:
    """Create and seed an isolated file-backed SQLite application."""

    database_path = tmp_path / "week6-integration.db"
    engine = create_sqlite_engine(database_path)
    try:
        create_all_tables(engine)
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            _seed_database(session)

        cache = TTLCache[DeviceSummaryResponse](ttl_seconds=5, max_entries=8)
        application = create_app(
            session_factory,
            cache,
            generated_at_ms=lambda: GENERATED_AT_MS,
        )
        with TestClient(application) as client:
            yield IntegrationEnvironment(
                client=client,
                cache=cache,
                database_path=database_path,
            )
    finally:
        engine.dispose()


def test_device_summary_is_correct_then_cached(
    integration_environment: IntegrationEnvironment,
) -> None:
    """A known device crosses HTTP, cache, query, and SQLite boundaries."""

    client = integration_environment.client
    first = client.get(f"/devices/{PRIMARY_DEVICE_ID}/summary")
    second = client.get(f"/devices/{PRIMARY_DEVICE_ID}/summary")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json() == first.json()

    payload = first.json()
    assert payload == {
        "device_id": PRIMARY_DEVICE_ID,
        "device_name": "Integration Device",
        "product_anchor": "Week 6",
        "session_count": 1,
        "reading_count": FORECAST_HISTORY_N,
        "alert_count": 13,
        "average_battery_soc": pytest.approx(
            sum(BATTERY_HISTORY) / FORECAST_HISTORY_N
        ),
        "low_health_count": 4,
        "gps_dropout_count": 3,
        "latest_timestamp_ms": 100_000 + FORECAST_HISTORY_N - 1,
        "generated_at_ms": GENERATED_AT_MS,
    }
    assert integration_environment.cache.stats().hits == 1


def test_unknown_and_injection_shaped_summaries_return_404(
    integration_environment: IntegrationEnvironment,
) -> None:
    """Unknown and SQL-shaped device identifiers never expose another device."""

    client = integration_environment.client
    missing = client.get(f"/devices/{UNKNOWN_DEVICE_ID}/summary")
    encoded_injection = quote(INJECTION_DEVICE_ID, safe="")
    injection = client.get(f"/devices/{encoded_injection}/summary")

    assert missing.status_code == 404
    assert injection.status_code == 404


@pytest.mark.parametrize(
    ("offset", "expected_count", "expected_next_offset"),
    [
        (0, 5, 5),
        (5, 5, 10),
        (10, 3, None),
        (15, 0, None),
    ],
)
def test_alert_pagination_boundaries(
    integration_environment: IntegrationEnvironment,
    offset: int,
    expected_count: int,
    expected_next_offset: int | None,
) -> None:
    """Each requested offset returns the required page boundary metadata."""

    response = integration_environment.client.get(
        "/alerts",
        params={"since": ALERT_SINCE_MS, "limit": 5, "offset": offset},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == expected_count
    assert payload["total"] == 13
    assert payload["limit"] == 5
    assert payload["offset"] == offset
    assert payload["next_offset"] == expected_next_offset


def test_alert_pages_have_unique_ids_and_stable_order(
    integration_environment: IntegrationEnvironment,
) -> None:
    """Adjacent pages preserve timestamp/ID ordering without overlap."""

    client = integration_environment.client
    pages = [
        client.get(
            "/alerts",
            params={"since": ALERT_SINCE_MS, "limit": 5, "offset": offset},
        ).json()
        for offset in (0, 5, 10)
    ]
    items = [item for page in pages for item in page["items"]]
    alert_ids = [item["alert_id"] for item in items]
    ordering = [
        (item["detected_at_ms"], item["alert_id"])
        for item in items
    ]

    assert alert_ids == list(EXPECTED_ALERT_IDS)
    assert len(alert_ids) == len(set(alert_ids)) == 13
    assert ordering == sorted(ordering)


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/alerts", None),
        ("/alerts", {"since": -1}),
        ("/alerts", {"since": ALERT_SINCE_MS, "limit": 0}),
        ("/alerts", {"since": ALERT_SINCE_MS, "limit": 501}),
        ("/alerts", {"since": ALERT_SINCE_MS, "offset": -1}),
        (f"/forecast/battery/{PRIMARY_DEVICE_ID}", {"horizon": 0}),
        (f"/forecast/battery/{PRIMARY_DEVICE_ID}", {"horizon": 301}),
    ],
)
def test_invalid_query_parameters_return_422(
    integration_environment: IntegrationEnvironment,
    path: str,
    params: dict[str, int] | None,
) -> None:
    """FastAPI validates every required range before endpoint execution."""

    response = integration_environment.client.get(path, params=params)

    assert response.status_code == 422


def test_battery_forecast_uses_real_week5_arima_path(
    integration_environment: IntegrationEnvironment,
) -> None:
    """A sufficient history produces finite ordered confidence intervals."""

    response = integration_environment.client.get(
        f"/forecast/battery/{PRIMARY_DEVICE_ID}",
        params={"horizon": FORECAST_HORIZON},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["device_id"] == PRIMARY_DEVICE_ID
    assert payload["history_n"] == FORECAST_HISTORY_N
    assert payload["horizon"] == FORECAST_HORIZON
    assert payload["model_order"] == [0, 1, 1]
    assert [point["step"] for point in payload["points"]] == list(
        range(1, FORECAST_HORIZON + 1)
    )

    for point in payload["points"]:
        values = (
            point["forecast_battery_soc"],
            point["lower_95"],
            point["upper_95"],
        )
        assert all(math.isfinite(value) for value in values)
        assert point["lower_95"] <= point["forecast_battery_soc"]
        assert point["forecast_battery_soc"] <= point["upper_95"]


def test_forecast_unknown_and_insufficient_history_statuses(
    integration_environment: IntegrationEnvironment,
) -> None:
    """Forecasting distinguishes missing devices from short histories."""

    client = integration_environment.client
    missing = client.get(
        f"/forecast/battery/{UNKNOWN_DEVICE_ID}",
        params={"horizon": 3},
    )
    insufficient = client.get(
        f"/forecast/battery/{SHORT_HISTORY_DEVICE_ID}",
        params={"horizon": 3},
    )

    assert missing.status_code == 404
    assert insufficient.status_code == 422
    assert str(MIN_BATTERY_HISTORY) in insufficient.json()["detail"]


def test_injection_shaped_alert_filter_returns_empty_page(
    integration_environment: IntegrationEnvironment,
) -> None:
    """A SQL-shaped alert filter remains data and returns no other alerts."""

    response = integration_environment.client.get(
        "/alerts",
        params={"since": ALERT_SINCE_MS, "device_id": INJECTION_DEVICE_ID},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "total": 0,
        "limit": 100,
        "offset": 0,
        "next_offset": None,
    }


def _seed_database(session: Session) -> None:
    """Insert deterministic entities in foreign-key dependency order."""

    session.add_all(
        [
            Device(
                device_id=PRIMARY_DEVICE_ID,
                device_name="Integration Device",
                product_anchor="Week 6",
            ),
            Device(
                device_id=SHORT_HISTORY_DEVICE_ID,
                device_name="Short History Device",
                product_anchor="Week 6",
            ),
        ]
    )
    session.flush()

    session.add_all(
        [
            TelemetrySession(
                session_id=PRIMARY_SESSION_ID,
                device_id=PRIMARY_DEVICE_ID,
                source_file="week6-integration.parquet",
                started_at_ms=100_000,
                ended_at_ms=100_000 + FORECAST_HISTORY_N - 1,
                row_count=FORECAST_HISTORY_N,
            ),
            TelemetrySession(
                session_id=SHORT_SESSION_ID,
                device_id=SHORT_HISTORY_DEVICE_ID,
                source_file="week6-short-history.parquet",
                started_at_ms=200_000,
                ended_at_ms=200_000 + MIN_BATTERY_HISTORY - 2,
                row_count=MIN_BATTERY_HISTORY - 1,
            ),
        ]
    )
    session.flush()

    primary_readings = [
        _make_reading(
            reading_id=f"integration-reading-{index:03d}",
            device_id=PRIMARY_DEVICE_ID,
            session_id=PRIMARY_SESSION_ID,
            timestamp_ms=100_000 + index,
            battery_soc=BATTERY_HISTORY[index],
            health_score=55.0 if index < 4 else 95.0,
            gps_dropout=index < 3,
        )
        for index in range(FORECAST_HISTORY_N)
    ]
    short_readings = [
        _make_reading(
            reading_id=f"integration-short-reading-{index:03d}",
            device_id=SHORT_HISTORY_DEVICE_ID,
            session_id=SHORT_SESSION_ID,
            timestamp_ms=200_000 + index,
            battery_soc=70.0 - index * 0.05,
            health_score=95.0,
            gps_dropout=False,
        )
        for index in range(MIN_BATTERY_HISTORY - 1)
    ]
    session.add_all(primary_readings + short_readings)
    session.flush()

    session.add_all(
        [
            Alert(
                alert_id=alert_id,
                device_id=PRIMARY_DEVICE_ID,
                session_id=PRIMARY_SESSION_ID,
                reading_id=f"integration-reading-{index:03d}",
                alert_type="integration_alert",
                severity=2,
                detected_at_ms=detected_at_ms,
                source="week6_integration",
                message=f"Controlled alert {alert_id}",
            )
            for index, (alert_id, detected_at_ms) in enumerate(
                reversed(ALERT_SPECS)
            )
        ]
    )
    session.commit()


def _make_reading(
    *,
    reading_id: str,
    device_id: str,
    session_id: str,
    timestamp_ms: int,
    battery_soc: float,
    health_score: float,
    gps_dropout: bool,
) -> SensorReading:
    """Build one complete Week 4 reading without duplicating any schema."""

    return SensorReading(
        reading_id=reading_id,
        device_id=device_id,
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        lat=40.0,
        lon=-88.0,
        lidar_distance_m=10.0,
        battery_soc=battery_soc,
        wheel_torque_fl=10.0,
        wheel_torque_fr=10.0,
        wheel_torque_rl=10.0,
        wheel_torque_rr=10.0,
        ambient_temp_c=22.0,
        gps_filled=False,
        gps_dropout_long=gps_dropout,
        lidar_saturated=False,
        battery_soc_spike=False,
        lidar_distance_m_spike=False,
        ambient_temp_c_spike=False,
        battery_soc_roll_mean_50=battery_soc,
        battery_soc_roll_std_50=0.0,
        cumulative_distance_m=0.0,
        wheel_imbalance_score=0.0,
        lidar_saturation_rate=0.0,
        composite_health_score=health_score,
    )
