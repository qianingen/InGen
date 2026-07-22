from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class NamedQuery:
    name: str
    sql: str


ANALYTIC_QUERIES: tuple[NamedQuery, ...] = (
    NamedQuery(
        name="mean_battery_soc_per_device_hour",
        sql="""
        SELECT
            device_id,
            timestamp_ms / 3600000 AS hour_bucket,
            AVG(battery_soc) AS mean_battery_soc
        FROM sensor_reading
        GROUP BY device_id, hour_bucket
        ORDER BY device_id, hour_bucket
        """,
    ),
    NamedQuery(
        name="top_10_wheel_imbalance",
        sql="""
        SELECT
            reading_id,
            device_id,
            timestamp_ms,
            wheel_imbalance_score
        FROM sensor_reading
        ORDER BY wheel_imbalance_score DESC, timestamp_ms ASC
        LIMIT 10
        """,
    ),
    NamedQuery(
        name="anomaly_flagged_rows_per_day",
        sql="""
        SELECT
            timestamp_ms / 86400000 AS day_bucket,
            COUNT(*) AS anomaly_count
        FROM sensor_reading
        WHERE gps_dropout_long = 1
           OR lidar_saturated = 1
           OR battery_soc_spike = 1
           OR lidar_distance_m_spike = 1
           OR ambient_temp_c_spike = 1
        GROUP BY day_bucket
        ORDER BY day_bucket
        """,
    ),
    NamedQuery(
        name="low_health_score_readings",
        sql="""
        SELECT
            reading_id,
            device_id,
            timestamp_ms,
            composite_health_score
        FROM sensor_reading
        WHERE composite_health_score < 60
        ORDER BY composite_health_score ASC, timestamp_ms ASC
        LIMIT 50
        """,
    ),
    NamedQuery(
        name="alerts_by_severity_and_time",
        sql="""
        SELECT
            alert_id,
            device_id,
            alert_type,
            severity,
            detected_at_ms
        FROM alert
        WHERE severity >= 1
        ORDER BY severity DESC, detected_at_ms DESC, alert_id ASC
        LIMIT 100
        """,
    ),
)


def get_analytic_queries() -> tuple[NamedQuery, ...]:
    return ANALYTIC_QUERIES


def run_query(engine: Engine, query: NamedQuery) -> list[tuple[object, ...]]:
    with engine.connect() as connection:
        rows = connection.execute(text(query.sql)).all()

    return [tuple(row) for row in rows]


def explain_query_plan(engine: Engine, query: NamedQuery) -> list[tuple[object, ...]]:
    explain_sql = f"EXPLAIN QUERY PLAN {query.sql}"

    with engine.connect() as connection:
        rows = connection.execute(text(explain_sql)).all()

    return [tuple(row) for row in rows]
