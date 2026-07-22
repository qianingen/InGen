from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import insert

from ingen_pydev.db.database import (
    create_all_tables,
    create_sqlite_engine,
    make_session_factory,
)
from ingen_pydev.db.models import Device, SensorReading, TelemetrySession


def test_schema_creates_tables() -> None:
    engine = create_sqlite_engine(":memory:")
    create_all_tables(engine)

    session_factory = make_session_factory(engine)

    with session_factory() as session:
        device = Device(
            device_id="aido_rover_001",
            device_name="Aido Rover 001",
            product_anchor="Aido Rover",
        )
        session.add(device)
        session.commit()

        result = session.get(Device, "aido_rover_001")

        assert result is not None
        assert result.device_name == "Aido Rover 001"


def test_sensor_reading_unique_device_timestamp_constraint() -> None:
    engine = create_sqlite_engine(":memory:")
    create_all_tables(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        device = Device(
            device_id="aido_rover_001",
            device_name="Aido Rover 001",
            product_anchor="Aido Rover",
        )
        telemetry_session = TelemetrySession(
            session_id="session_001",
            device_id="aido_rover_001",
            source_file="outputs/cleaned_features.parquet",
            started_at_ms=1000,
            ended_at_ms=2000,
            row_count=2,
        )

        session.add(device)
        session.add(telemetry_session)
        session.commit()

        reading_1 = _make_reading(
            reading_id="aido_rover_001:1000",
            device_id="aido_rover_001",
            session_id="session_001",
            timestamp_ms=1000,
        )
        reading_2 = _make_reading(
            reading_id="aido_rover_001:1000_duplicate",
            device_id="aido_rover_001",
            session_id="session_001",
            timestamp_ms=1000,
        )

        session.add(reading_1)
        session.commit()

        session.add(reading_2)

        with pytest.raises(IntegrityError):
            session.commit()


def test_sensor_reading_foreign_key_constraint() -> None:
    engine = create_sqlite_engine(":memory:")
    create_all_tables(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        bad_reading = _make_reading(
            reading_id="missing_device:1000",
            device_id="missing_device",
            session_id="missing_session",
            timestamp_ms=1000,
        )

        session.add(bad_reading)

        with pytest.raises(IntegrityError):
            session.commit()


def test_not_null_constraint_for_device_name() -> None:
    engine = create_sqlite_engine(":memory:")
    create_all_tables(engine)

    with engine.begin() as connection:
        with pytest.raises(IntegrityError):
            connection.execute(
                insert(Device).values(
                    device_id="aido_rover_001",
                    device_name=None,
                    product_anchor="Aido Rover",
                )
            )


def _make_reading(
    reading_id: str,
    device_id: str,
    session_id: str,
    timestamp_ms: int,
) -> SensorReading:
    return SensorReading(
        reading_id=reading_id,
        device_id=device_id,
        session_id=session_id,
        timestamp_ms=timestamp_ms,
        lat=40.0,
        lon=-88.0,
        lidar_distance_m=10.0,
        battery_soc=90.0,
        wheel_torque_fl=10.0,
        wheel_torque_fr=11.0,
        wheel_torque_rl=10.5,
        wheel_torque_rr=10.2,
        ambient_temp_c=22.0,
        gps_filled=False,
        gps_dropout_long=False,
        lidar_saturated=False,
        battery_soc_spike=False,
        lidar_distance_m_spike=False,
        ambient_temp_c_spike=False,
        battery_soc_roll_mean_50=90.0,
        battery_soc_roll_std_50=0.0,
        cumulative_distance_m=0.0,
        wheel_imbalance_score=0.1,
        lidar_saturation_rate=0.0,
        composite_health_score=90.0,
    )
