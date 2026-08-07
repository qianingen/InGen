from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import ingen_pydev.analytics.app as analytics_app
from ingen_pydev.analytics.app import create_app
from ingen_pydev.analytics.cache import TTLCache
from ingen_pydev.analytics.forecast import (
    BATTERY_MODEL_ORDER,
    MIN_BATTERY_HISTORY,
    BatteryForecastPointResult,
    BatteryForecastResult,
    BatteryForecastServiceError,
    build_battery_forecast,
)
from ingen_pydev.analytics.models import DeviceSummaryResponse
from ingen_pydev.db.database import create_all_tables, make_session_factory
from ingen_pydev.db.models import Device, SensorReading, TelemetrySession

SQLiteEngineFactory = Callable[[str | Path], Engine]

SUFFICIENT_DEVICE = "forecast-device"
INSUFFICIENT_DEVICE = "short-history-device"
HISTORY_N = 60
GENERATED_AT_MS = 1_700_000_000_000
EXPECTED_HISTORY = [
    82.0 - index * 0.05 + (index % 5) * 0.02 for index in range(HISTORY_N)
]


@pytest.fixture
def forecast_session_factory(
    tmp_path: Path,
    sqlite_engine_factory: SQLiteEngineFactory,
) -> sessionmaker[Session]:
    database_path = tmp_path / "forecast.db"
    engine = sqlite_engine_factory(database_path)
    create_all_tables(engine)
    factory = make_session_factory(engine)

    with factory() as session:
        _insert_forecast_test_data(session)
        session.commit()

    return factory


@pytest.fixture
def forecast_client(
    forecast_session_factory: sessionmaker[Session],
) -> Iterator[TestClient]:
    cache = TTLCache[DeviceSummaryResponse](ttl_seconds=5, max_entries=8)
    application = create_app(
        forecast_session_factory,
        cache,
        generated_at_ms=lambda: GENERATED_AT_MS,
    )
    with TestClient(application) as client:
        yield client


def test_valid_forecast_uses_real_week5_arima_path(
    forecast_client: TestClient,
) -> None:
    response = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["device_id"] == SUFFICIENT_DEVICE
    assert payload["model_order"] == [0, 1, 1]
    assert payload["history_n"] == HISTORY_N
    assert payload["horizon"] == 5
    assert payload["generated_at_ms"] == GENERATED_AT_MS
    assert len(payload["points"]) == 5
    assert [point["step"] for point in payload["points"]] == list(range(1, 6))

    for point in payload["points"]:
        numeric_values = (
            point["forecast_battery_soc"],
            point["lower_95"],
            point["upper_95"],
        )
        assert all(math.isfinite(value) for value in numeric_values)
        assert point["lower_95"] <= point["forecast_battery_soc"]
        assert point["forecast_battery_soc"] <= point["upper_95"]


def test_horizon_one_works(forecast_client: TestClient) -> None:
    response = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 1},
    )

    assert response.status_code == 200
    assert response.json()["horizon"] == 1
    assert [point["step"] for point in response.json()["points"]] == [1]


def test_horizon_300_and_default_use_requested_boundaries(
    forecast_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analytics_app, "build_battery_forecast", _fake_forecast)

    maximum = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 300},
    )
    default = forecast_client.get(f"/forecast/battery/{SUFFICIENT_DEVICE}")

    assert maximum.status_code == 200
    assert maximum.json()["horizon"] == 300
    assert len(maximum.json()["points"]) == 300
    assert maximum.json()["points"][-1]["step"] == 300
    assert default.status_code == 200
    assert default.json()["horizon"] == 60
    assert len(default.json()["points"]) == 60


@pytest.mark.parametrize("horizon", [0, 301, "not-an-integer"])
def test_invalid_horizon_returns_422(
    forecast_client: TestClient,
    horizon: int | str,
) -> None:
    response = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": horizon},
    )

    assert response.status_code == 422


def test_unknown_device_returns_404(forecast_client: TestClient) -> None:
    response = forecast_client.get(
        "/forecast/battery/missing-device",
        params={"horizon": 3},
    )

    assert response.status_code == 404


def test_insufficient_history_returns_422(forecast_client: TestClient) -> None:
    response = forecast_client.get(
        f"/forecast/battery/{INSUFFICIENT_DEVICE}",
        params={"horizon": 3},
    )

    assert response.status_code == 422
    assert str(MIN_BATTERY_HISTORY) in response.json()["detail"]


def test_database_supplies_history_in_chronological_order(
    forecast_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_history: list[float] = []

    def capture_history(
        battery_history: Sequence[float],
        horizon: int,
    ) -> BatteryForecastResult:
        observed_history.extend(battery_history)
        return _fake_forecast(battery_history, horizon)

    monkeypatch.setattr(analytics_app, "build_battery_forecast", capture_history)
    response = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 2},
    )

    assert response.status_code == 200
    assert observed_history == pytest.approx(EXPECTED_HISTORY)


@pytest.mark.parametrize("invalid_value", [None, np.nan, np.inf, -np.inf])
def test_adapter_rejects_missing_and_nonfinite_history(
    invalid_value: float | None,
) -> None:
    history: list[float | None] = [80.0] * MIN_BATTERY_HISTORY
    history[-1] = invalid_value

    with pytest.raises(BatteryForecastServiceError, match="missing or non-finite"):
        build_battery_forecast(cast(Sequence[float], history), 1)


def test_repeated_forecasts_are_numerically_stable(
    forecast_client: TestClient,
) -> None:
    first = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 3},
    )
    second = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 3},
    )

    assert first.status_code == second.status_code == 200
    first_points = first.json()["points"]
    second_points = second.json()["points"]
    for first_point, second_point in zip(first_points, second_points, strict=True):
        assert first_point["step"] == second_point["step"]
        assert first_point["forecast_battery_soc"] == pytest.approx(
            second_point["forecast_battery_soc"], rel=1e-9, abs=1e-9
        )
        assert first_point["lower_95"] == pytest.approx(
            second_point["lower_95"], rel=1e-9, abs=1e-9
        )
        assert first_point["upper_95"] == pytest.approx(
            second_point["upper_95"], rel=1e-9, abs=1e-9
        )


def test_unexpected_model_failure_returns_500(
    forecast_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_forecast(
        _battery_history: Sequence[float],
        _horizon: int,
    ) -> BatteryForecastResult:
        raise RuntimeError("unexpected model failure")

    monkeypatch.setattr(analytics_app, "build_battery_forecast", fail_forecast)
    response = forecast_client.get(
        f"/forecast/battery/{SUFFICIENT_DEVICE}",
        params={"horizon": 2},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Battery forecast failed"


def _fake_forecast(
    battery_history: Sequence[float],
    horizon: int,
) -> BatteryForecastResult:
    points = tuple(
        BatteryForecastPointResult(
            step=step,
            forecast_battery_soc=75.0,
            lower_95=74.0,
            upper_95=76.0,
        )
        for step in range(1, horizon + 1)
    )
    return BatteryForecastResult(
        model_order=BATTERY_MODEL_ORDER,
        history_n=len(battery_history),
        horizon=horizon,
        points=points,
    )


def _insert_forecast_test_data(session: Session) -> None:
    session.add_all(
        [
            Device(
                device_id=SUFFICIENT_DEVICE,
                device_name="Forecast Device",
                product_anchor="Test Product",
            ),
            Device(
                device_id=INSUFFICIENT_DEVICE,
                device_name="Short History Device",
                product_anchor="Test Product",
            ),
        ]
    )
    session.flush()

    session.add_all(
        [
            TelemetrySession(
                session_id="forecast-session",
                device_id=SUFFICIENT_DEVICE,
                source_file="forecast.parquet",
                started_at_ms=1_000,
                ended_at_ms=1_000 + HISTORY_N - 1,
                row_count=HISTORY_N,
            ),
            TelemetrySession(
                session_id="short-session",
                device_id=INSUFFICIENT_DEVICE,
                source_file="short.parquet",
                started_at_ms=2_000,
                ended_at_ms=2_000 + MIN_BATTERY_HISTORY - 2,
                row_count=MIN_BATTERY_HISTORY - 1,
            ),
        ]
    )
    session.flush()

    sufficient_readings = [
        _make_reading(
            reading_id=f"forecast-reading-{index:03d}",
            device_id=SUFFICIENT_DEVICE,
            session_id="forecast-session",
            timestamp_ms=1_000 + index,
            battery_soc=EXPECTED_HISTORY[index],
        )
        for index in range(HISTORY_N)
    ]
    insufficient_readings = [
        _make_reading(
            reading_id=f"short-reading-{index:03d}",
            device_id=INSUFFICIENT_DEVICE,
            session_id="short-session",
            timestamp_ms=2_000 + index,
            battery_soc=70.0 - index * 0.05,
        )
        for index in range(MIN_BATTERY_HISTORY - 1)
    ]
    session.add_all(list(reversed(sufficient_readings + insufficient_readings)))
    session.flush()


def _make_reading(
    *,
    reading_id: str,
    device_id: str,
    session_id: str,
    timestamp_ms: int,
    battery_soc: float,
) -> SensorReading:
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
        gps_dropout_long=False,
        lidar_saturated=False,
        battery_soc_spike=False,
        lidar_distance_m_spike=False,
        ambient_temp_c_spike=False,
        battery_soc_roll_mean_50=battery_soc,
        battery_soc_roll_std_50=0.0,
        cumulative_distance_m=0.0,
        wheel_imbalance_score=0.0,
        lidar_saturation_rate=0.0,
        composite_health_score=100.0,
    )
