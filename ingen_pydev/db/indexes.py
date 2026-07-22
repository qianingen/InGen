from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


INDEX_SQL: tuple[str, ...] = (
    """
    CREATE INDEX IF NOT EXISTS idx_sensor_reading_device_time
    ON sensor_reading(device_id, timestamp_ms)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sensor_reading_wheel_imbalance
    ON sensor_reading(wheel_imbalance_score DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sensor_reading_timestamp
    ON sensor_reading(timestamp_ms)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_sensor_reading_health_score
    ON sensor_reading(composite_health_score)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_alert_severity_time
    ON alert(severity DESC, detected_at_ms DESC)
    """,
)


def apply_optimization_indexes(engine: Engine) -> None:
    with engine.begin() as connection:
        for index_sql in INDEX_SQL:
            connection.execute(text(index_sql))
