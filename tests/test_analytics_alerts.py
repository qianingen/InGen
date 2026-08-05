from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ingen_pydev.analytics.app import create_app
from ingen_pydev.analytics.cache import TTLCache
from ingen_pydev.analytics.models import DeviceSummaryResponse
from ingen_pydev.db.database import create_all_tables, make_session_factory
from ingen_pydev.db.models import Alert, Device, SensorReading, TelemetrySession

SQLiteEngineFactory = Callable[[str | Path], Engine]

SINCE = 1_000
DEVICE_A = "alerts-device-a"
DEVICE_B = "alerts-device-b"
EXPECTED_ALERT_IDS = [f"alert-{number:03d}" for number in range(1, 14)]
ALERT_SPECS = [
    ("alert-002", DEVICE_A, 1_000),
    ("alert-001", DEVICE_B, 1_000),
    ("alert-004", DEVICE_A, 2_000),
    ("alert-003", DEVICE_A, 2_000),
    ("alert-006", DEVICE_B, 3_000),
    ("alert-005", DEVICE_A, 3_000),
    ("alert-008", DEVICE_B, 4_000),
    ("alert-007", DEVICE_A, 4_000),
    ("alert-010", DEVICE_B, 5_000),
    ("alert-009", DEVICE_A, 5_000),
    ("alert-012", DEVICE_B, 6_000),
    ("alert-011", DEVICE_A, 6_000),
    ("alert-013", DEVICE_A, 7_000),
]


@pytest.fixture
def alerts_session_factory(
    tmp_path: Path,
    sqlite_engine_factory: SQLiteEngineFactory,
) -> sessionmaker[Session]:
    database_path = tmp_path / "alerts.db"
    engine = sqlite_engine_factory(database_path)
    create_all_tables(engine)
    factory = make_session_factory(engine)

    with factory() as session:
        _insert_alert_test_data(session)
        session.commit()

    return factory


@pytest.fixture
def alerts_client(
    alerts_session_factory: sessionmaker[Session],
) -> Iterator[TestClient]:
    cache = TTLCache[DeviceSummaryResponse](ttl_seconds=5, max_entries=8)
    application = create_app(alerts_session_factory, cache)
    with TestClient(application) as client:
        yield client


def test_first_full_page(alerts_client: TestClient) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "limit": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["alert_id"] for item in payload["items"]] == EXPECTED_ALERT_IDS[:5]
    assert payload == {
        "items": payload["items"],
        "total": 13,
        "limit": 5,
        "offset": 0,
        "next_offset": 5,
    }
    assert payload["items"][0] == {
        "alert_id": "alert-001",
        "device_id": DEVICE_B,
        "session_id": "session-b",
        "reading_id": "reading-alert-001",
        "alert_type": "test_alert",
        "severity": 1,
        "detected_at_ms": 1_000,
        "source": "test",
        "message": "Alert alert-001",
    }


def test_second_full_page(alerts_client: TestClient) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "limit": 5, "offset": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["alert_id"] for item in payload["items"]] == EXPECTED_ALERT_IDS[5:10]
    assert payload["total"] == 13
    assert payload["next_offset"] == 10


def test_final_partial_page(alerts_client: TestClient) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "limit": 5, "offset": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["alert_id"] for item in payload["items"]] == EXPECTED_ALERT_IDS[10:]
    assert payload["total"] == 13
    assert payload["next_offset"] is None
    assert payload["items"][-1]["reading_id"] is None
    assert payload["items"][-1]["message"] is None


def test_empty_page_beyond_end(alerts_client: TestClient) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "limit": 5, "offset": 15},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "total": 13,
        "limit": 5,
        "offset": 15,
        "next_offset": None,
    }


def test_adjacent_pages_have_no_duplicate_alert_ids(
    alerts_client: TestClient,
) -> None:
    first = alerts_client.get(
        "/alerts", params={"since": SINCE, "limit": 5, "offset": 0}
    ).json()
    second = alerts_client.get(
        "/alerts", params={"since": SINCE, "limit": 5, "offset": 5}
    ).json()

    first_ids = {item["alert_id"] for item in first["items"]}
    second_ids = {item["alert_id"] for item in second["items"]}
    assert first_ids.isdisjoint(second_ids)


def test_ordering_is_deterministic_by_timestamp_then_alert_id(
    alerts_client: TestClient,
) -> None:
    first = alerts_client.get("/alerts", params={"since": SINCE, "limit": 13}).json()
    second = alerts_client.get("/alerts", params={"since": SINCE, "limit": 13}).json()

    first_order = [
        (item["detected_at_ms"], item["alert_id"]) for item in first["items"]
    ]
    second_order = [
        (item["detected_at_ms"], item["alert_id"]) for item in second["items"]
    ]
    assert first_order == sorted(first_order)
    assert second_order == first_order
    assert [alert_id for _, alert_id in first_order] == EXPECTED_ALERT_IDS


def test_since_lower_bound_is_inclusive(alerts_client: TestClient) -> None:
    response = alerts_client.get("/alerts", params={"since": 3_000})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 9
    assert payload["items"][0]["detected_at_ms"] == 3_000
    assert [item["alert_id"] for item in payload["items"]] == EXPECTED_ALERT_IDS[4:]


def test_device_id_filtering(alerts_client: TestClient) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "device_id": DEVICE_A},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 8
    assert [item["alert_id"] for item in payload["items"]] == [
        "alert-002",
        "alert-003",
        "alert-004",
        "alert-005",
        "alert-007",
        "alert-009",
        "alert-011",
        "alert-013",
    ]
    assert {item["device_id"] for item in payload["items"]} == {DEVICE_A}


def test_nonexistent_device_filter_returns_empty_page(
    alerts_client: TestClient,
) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "device_id": "missing-device"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    assert response.json()["next_offset"] is None


def test_no_matching_timestamp_range_returns_empty_page(
    alerts_client: TestClient,
) -> None:
    response = alerts_client.get("/alerts", params={"since": 8_000})

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    assert response.json()["next_offset"] is None


def test_injection_shaped_device_id_is_only_data(alerts_client: TestClient) -> None:
    response = alerts_client.get(
        "/alerts",
        params={"since": SINCE, "device_id": "x' OR 1=1 --"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0


def test_missing_since_returns_422(alerts_client: TestClient) -> None:
    response = alerts_client.get("/alerts")

    assert response.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"since": -1},
        {"since": SINCE, "limit": 0},
        {"since": SINCE, "limit": 501},
        {"since": SINCE, "offset": -1},
    ],
)
def test_out_of_range_parameters_return_422(
    alerts_client: TestClient,
    params: dict[str, int],
) -> None:
    response = alerts_client.get("/alerts", params=params)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"since": "not-an-integer"},
        {"since": str(SINCE), "limit": "not-an-integer"},
        {"since": str(SINCE), "offset": "not-an-integer"},
    ],
)
def test_non_integer_parameters_return_422(
    alerts_client: TestClient,
    params: dict[str, str],
) -> None:
    response = alerts_client.get("/alerts", params=params)

    assert response.status_code == 422


def _insert_alert_test_data(session: Session) -> None:
    session.add_all(
        [
            Device(
                device_id=DEVICE_A,
                device_name="Alerts Device A",
                product_anchor="Test Product",
            ),
            Device(
                device_id=DEVICE_B,
                device_name="Alerts Device B",
                product_anchor="Test Product",
            ),
        ]
    )
    session.flush()

    session.add_all(
        [
            TelemetrySession(
                session_id="session-a",
                device_id=DEVICE_A,
                source_file="alerts-a.parquet",
                started_at_ms=SINCE,
                ended_at_ms=7_000,
                row_count=8,
            ),
            TelemetrySession(
                session_id="session-b",
                device_id=DEVICE_B,
                source_file="alerts-b.parquet",
                started_at_ms=SINCE,
                ended_at_ms=6_000,
                row_count=5,
            ),
        ]
    )
    session.flush()

    session.add_all(
        [
            _make_reading(
                alert_id,
                device_id,
                detected_at_ms * 100 + int(alert_id.removeprefix("alert-")),
            )
            for alert_id, device_id, detected_at_ms in ALERT_SPECS
        ]
    )
    session.flush()

    session.add_all(
        [
            Alert(
                alert_id=alert_id,
                device_id=device_id,
                session_id=_session_id(device_id),
                reading_id=(None if alert_id == "alert-013" else f"reading-{alert_id}"),
                alert_type="test_alert",
                severity=1,
                detected_at_ms=detected_at_ms,
                source="test",
                message=None if alert_id == "alert-013" else f"Alert {alert_id}",
            )
            for alert_id, device_id, detected_at_ms in reversed(ALERT_SPECS)
        ]
    )


def _make_reading(
    alert_id: str,
    device_id: str,
    timestamp_ms: int,
) -> SensorReading:
    return SensorReading(
        reading_id=f"reading-{alert_id}",
        device_id=device_id,
        session_id=_session_id(device_id),
        timestamp_ms=timestamp_ms,
        lat=40.0,
        lon=-88.0,
        lidar_distance_m=10.0,
        battery_soc=80.0,
        wheel_torque_fl=10.0,
        wheel_torque_fr=10.0,
        wheel_torque_rl=10.0,
        wheel_torque_rr=10.0,
        ambient_temp_c=22.0,
        gps_filled=False,
        gps_dropout_long=False,
        lidar_saturated=False,
        battery_soc_spike=False,
        lidar_distance_m_spike=False,
        ambient_temp_c_spike=False,
        battery_soc_roll_mean_50=80.0,
        battery_soc_roll_std_50=0.0,
        cumulative_distance_m=0.0,
        wheel_imbalance_score=0.0,
        lidar_saturation_rate=0.0,
        composite_health_score=100.0,
    )


def _session_id(device_id: str) -> str:
    return "session-a" if device_id == DEVICE_A else "session-b"
