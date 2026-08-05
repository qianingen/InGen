from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from ingen_pydev.analytics.app import create_app
from ingen_pydev.analytics.cache import TTLCache
from ingen_pydev.analytics.models import DeviceSummaryResponse
from ingen_pydev.db.models import Alert, Device, SensorReading, TelemetrySession

DEVICE_ID = "summary_test_device"
GENERATED_AT_BASE_MS = 1_700_000_000_000


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass(frozen=True)
class APIContext:
    client: TestClient
    cache: TTLCache[DeviceSummaryResponse]
    clock: FakeClock


@pytest.fixture
def summary_session_factory(
    mutable_seeded_session_factory: sessionmaker[Session],
) -> sessionmaker[Session]:
    with mutable_seeded_session_factory() as session:
        _insert_summary_device(session)
        session.commit()
    return mutable_seeded_session_factory


@pytest.fixture
def api_context(
    summary_session_factory: sessionmaker[Session],
) -> Iterator[APIContext]:
    clock = FakeClock()
    cache = TTLCache[DeviceSummaryResponse](
        ttl_seconds=5,
        max_entries=8,
        clock=clock,
    )
    application = create_app(
        summary_session_factory,
        cache,
        generated_at_ms=lambda: GENERATED_AT_BASE_MS + int(clock() * 1_000),
    )
    with TestClient(application) as client:
        yield APIContext(client=client, cache=cache, clock=clock)


def test_known_device_returns_correct_summary(api_context: APIContext) -> None:
    response = api_context.client.get(f"/devices/{DEVICE_ID}/summary")

    assert response.status_code == 200
    assert response.json() == {
        "device_id": DEVICE_ID,
        "device_name": "Summary Test Device",
        "product_anchor": "Test Product",
        "session_count": 2,
        "reading_count": 3,
        "alert_count": 2,
        "average_battery_soc": 80.0,
        "low_health_count": 2,
        "gps_dropout_count": 2,
        "latest_timestamp_ms": 3_000,
        "generated_at_ms": GENERATED_AT_BASE_MS,
    }


def test_first_request_is_miss_and_second_is_identical_hit(
    api_context: APIContext,
) -> None:
    first = api_context.client.get(f"/devices/{DEVICE_ID}/summary")
    second = api_context.client.get(f"/devices/{DEVICE_ID}/summary")

    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json() == first.json()


def test_expired_request_is_a_new_miss(api_context: APIContext) -> None:
    first = api_context.client.get(f"/devices/{DEVICE_ID}/summary")
    api_context.clock.advance(5)
    expired = api_context.client.get(f"/devices/{DEVICE_ID}/summary")

    assert first.headers["X-Cache"] == "MISS"
    assert expired.headers["X-Cache"] == "MISS"
    assert expired.json()["generated_at_ms"] > first.json()["generated_at_ms"]
    assert api_context.cache.stats().expirations == 1


def test_unknown_device_is_not_cached(api_context: APIContext) -> None:
    first = api_context.client.get("/devices/missing-device/summary")
    second = api_context.client.get("/devices/missing-device/summary")

    assert first.status_code == 404
    assert second.status_code == 404
    assert api_context.cache.stats().hits == 0
    assert api_context.cache.stats().misses == 2


def test_injection_shaped_device_id_returns_404(api_context: APIContext) -> None:
    injection_id = "x' OR 1=1 --"

    response = api_context.client.get(f"/devices/{injection_id}/summary")

    assert response.status_code == 404


def test_existing_device_without_readings_returns_zero_and_null_aggregates(
    summary_session_factory: sessionmaker[Session],
) -> None:
    with summary_session_factory() as session:
        session.add(
            Device(
                device_id="empty_device",
                device_name="Empty Device",
                product_anchor="Test Product",
            )
        )
        session.commit()

    cache = TTLCache[DeviceSummaryResponse](ttl_seconds=5, max_entries=8)
    application = create_app(
        summary_session_factory,
        cache,
        generated_at_ms=lambda: GENERATED_AT_BASE_MS,
    )
    with TestClient(application) as client:
        response = client.get("/devices/empty_device/summary")

    assert response.status_code == 200
    assert response.headers["X-Cache"] == "MISS"
    assert response.json() == {
        "device_id": "empty_device",
        "device_name": "Empty Device",
        "product_anchor": "Test Product",
        "session_count": 0,
        "reading_count": 0,
        "alert_count": 0,
        "average_battery_soc": None,
        "low_health_count": 0,
        "gps_dropout_count": 0,
        "latest_timestamp_ms": None,
        "generated_at_ms": GENERATED_AT_BASE_MS,
    }


def _insert_summary_device(session: Session) -> None:
    session.add(
        Device(
            device_id=DEVICE_ID,
            device_name="Summary Test Device",
            product_anchor="Test Product",
        )
    )
    session.add_all(
        [
            TelemetrySession(
                session_id="summary_session_1",
                device_id=DEVICE_ID,
                source_file="summary-1.parquet",
                started_at_ms=1_000,
                ended_at_ms=3_000,
                row_count=3,
            ),
            TelemetrySession(
                session_id="summary_session_2",
                device_id=DEVICE_ID,
                source_file="summary-2.parquet",
                started_at_ms=4_000,
                ended_at_ms=4_000,
                row_count=0,
            ),
        ]
    )
    readings = [
        _make_reading("summary-reading-1", 1_000, 90.0, 59.0, True),
        _make_reading("summary-reading-2", 2_000, 80.0, 60.0, False),
        _make_reading("summary-reading-3", 3_000, 70.0, 40.0, True),
    ]
    session.add_all(readings)
    session.add_all(
        [
            Alert(
                alert_id="summary-alert-1",
                device_id=DEVICE_ID,
                session_id="summary_session_1",
                reading_id="summary-reading-1",
                alert_type="gps_dropout_long",
                severity=3,
                detected_at_ms=1_000,
                source="test",
                message=None,
            ),
            Alert(
                alert_id="summary-alert-2",
                device_id=DEVICE_ID,
                session_id="summary_session_1",
                reading_id="summary-reading-3",
                alert_type="low_health_score",
                severity=3,
                detected_at_ms=3_000,
                source="test",
                message=None,
            ),
        ]
    )


def _make_reading(
    reading_id: str,
    timestamp_ms: int,
    battery_soc: float,
    health_score: float,
    gps_dropout: bool,
) -> SensorReading:
    return SensorReading(
        reading_id=reading_id,
        device_id=DEVICE_ID,
        session_id="summary_session_1",
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
