from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingen_pydev.db.database import create_sqlite_engine
from ingen_pydev.db.indexes import apply_optimization_indexes
from ingen_pydev.db.loader import load_week3_output
from ingen_pydev.db.queries import (
    explain_query_plan,
    get_analytic_queries,
    run_query,
)


def test_all_five_analytic_queries_are_defined() -> None:
    queries = get_analytic_queries()

    assert len(queries) == 5

    query_names = {query.name for query in queries}

    assert query_names == {
        "mean_battery_soc_per_device_hour",
        "top_10_wheel_imbalance",
        "anomaly_flagged_rows_per_day",
        "low_health_score_readings",
        "alerts_by_severity_and_time",
    }


def test_query_results_are_identical_before_and_after_indexing(
    tmp_path: Path,
) -> None:
    parquet_path, summary_path = _write_week3_inputs(tmp_path)
    db_path = tmp_path / "w04_telemetry.db"

    load_week3_output(
        parquet_path=parquet_path,
        summary_json_path=summary_path,
        db_path=db_path,
        device_id="aido_rover_001",
        device_name="Aido Rover 001",
        product_anchor="Aido Rover",
    )

    engine = create_sqlite_engine(db_path)
    queries = get_analytic_queries()

    results_before = {query.name: run_query(engine, query) for query in queries}

    plans_before = {query.name: explain_query_plan(engine, query) for query in queries}

    apply_optimization_indexes(engine)

    results_after = {query.name: run_query(engine, query) for query in queries}

    plans_after = {query.name: explain_query_plan(engine, query) for query in queries}

    assert results_before == results_after

    for query in queries:
        assert plans_before[query.name]
        assert plans_after[query.name]


def test_index_application_is_repeatable(tmp_path: Path) -> None:
    parquet_path, summary_path = _write_week3_inputs(tmp_path)
    db_path = tmp_path / "w04_telemetry.db"

    load_week3_output(
        parquet_path=parquet_path,
        summary_json_path=summary_path,
        db_path=db_path,
        device_id="aido_rover_001",
        device_name="Aido Rover 001",
        product_anchor="Aido Rover",
    )

    engine = create_sqlite_engine(db_path)

    apply_optimization_indexes(engine)
    apply_optimization_indexes(engine)

    queries = get_analytic_queries()

    for query in queries:
        result = run_query(engine, query)
        plan = explain_query_plan(engine, query)

        assert isinstance(result, list)
        assert plan


def _write_week3_inputs(tmp_path: Path) -> tuple[Path, Path]:
    df = _make_week3_feature_df()

    parquet_path = tmp_path / "cleaned_features.parquet"
    summary_path = tmp_path / "validation_summary.json"

    df.to_parquet(parquet_path, index=False)

    summary = {
        "rows_processed": len(df),
        "rows_flagged": 5,
        "fields_with_anomalies_detected": [
            "gps",
            "lidar_distance_m",
            "battery_soc",
            "wheel_torque",
            "ambient_temp_c",
        ],
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    return parquet_path, summary_path


def _make_week3_feature_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ms": [1000, 2000, 3000, 4000, 5000],
            "lat": [40.0, 40.0001, 40.0002, 40.0003, 40.0004],
            "lon": [-88.0, -88.0001, -88.0002, -88.0003, -88.0004],
            "lidar_distance_m": [10.0, 0.0, 12.0, 13.0, 14.0],
            "battery_soc": [95.0, 94.5, 70.0, 69.5, 69.0],
            "wheel_torque_fl": [10.0, 10.1, 10.2, 18.0, 10.3],
            "wheel_torque_fr": [10.0, 10.1, 10.2, 8.0, 10.3],
            "wheel_torque_rl": [10.0, 10.1, 10.2, 10.0, 10.3],
            "wheel_torque_rr": [10.0, 10.1, 10.2, 10.0, 10.3],
            "ambient_temp_c": [22.0, 22.1, 30.0, 22.3, 22.4],
            "gps_filled": [False, False, False, False, False],
            "gps_dropout_long": [True, False, False, False, False],
            "lidar_saturated": [False, True, False, False, False],
            "battery_soc_spike": [False, False, True, False, False],
            "lidar_distance_m_spike": [False, False, False, False, False],
            "ambient_temp_c_spike": [False, False, False, True, False],
            "battery_soc_roll_mean_50": [95.0, 94.75, 86.5, 82.25, 79.6],
            "battery_soc_roll_std_50": [0.0, 0.25, 11.67, 12.9, 12.67],
            "cumulative_distance_m": [0.0, 14.0, 28.0, 42.0, 56.0],
            "wheel_imbalance_score": [0.0, 0.0, 0.0, 0.87, 0.0],
            "lidar_saturation_rate": [0.0, 0.5, 0.33, 0.25, 0.2],
            "composite_health_score": [95.0, 47.25, 46.9, 52.1, 55.2],
        }
    )
