from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingen_pydev.modeling.anomaly import (
    ANOMALY_GROUPS,
    CLASSIFIER_FEATURES,
    CLASSIFIER_LABELS,
    FEATURE_LEAKAGE_AUDIT,
    LABEL_SOURCE_FLAGS,
    LEAKAGE_EXCLUDED_DERIVED_FEATURES,
    build_anomaly_type,
    build_classifier_dataset,
)


def _empty_flags(rows: int) -> dict[str, list[bool]]:
    return {flag: [False] * rows for flag in LABEL_SOURCE_FLAGS}


def test_build_anomaly_type_assigns_mutually_exclusive_labels() -> None:
    values = _empty_flags(rows=8)
    values["gps_dropout"][1] = True
    values["lidar_saturated"][2] = True
    values["battery_soc_spike"][3] = True
    values["wheel_torque_fr_spike"][4] = True
    values["ambient_temp_c_spike"][5] = True
    values["gps_filled"][6] = True
    values["battery_soc_spike"][6] = True
    values["lidar_distance_m_spike"][7] = True
    values["wheel_torque_rr_spike"][7] = True
    values["ambient_temp_c_spike"][7] = True

    labels = build_anomaly_type(pd.DataFrame(values))

    assert labels.tolist() == [
        "normal",
        "gps",
        "lidar",
        "battery",
        "wheel_torque",
        "ambient_temp",
        "overlap",
        "overlap",
    ]
    assert labels.name == "anomaly_type"
    assert labels.notna().all()


def test_build_anomaly_type_has_no_group_priority() -> None:
    values = _empty_flags(rows=1)
    values["gps_dropout"][0] = True
    values["battery_soc_spike"][0] = True

    labels = build_anomaly_type(pd.DataFrame(values))

    assert labels.iloc[0] == "overlap"


def test_build_anomaly_type_reports_all_missing_flag_columns() -> None:
    missing_flags = {
        "gps_dropout",
        "battery_soc_spike",
        "ambient_temp_c_spike",
    }
    values = {
        flag: [False]
        for flag in LABEL_SOURCE_FLAGS
        if flag not in missing_flags
    }

    with pytest.raises(ValueError, match="Missing anomaly flag columns") as error:
        build_anomaly_type(pd.DataFrame(values))

    for missing_flag in missing_flags:
        assert missing_flag in str(error.value)


def test_build_anomaly_type_rejects_missing_flag_values() -> None:
    values: dict[str, list[object]] = {
        flag: list(entries) for flag, entries in _empty_flags(rows=1).items()
    }
    values["lidar_saturated"] = [None]

    with pytest.raises(ValueError, match="contain missing values"):
        build_anomaly_type(pd.DataFrame(values))


def test_profile_parquet_has_expected_mutually_exclusive_counts() -> None:
    parquet_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "profile_cleaned_features.parquet"
    )
    df = pd.read_parquet(parquet_path)

    labels = build_anomaly_type(df)
    counts = labels.value_counts().to_dict()

    assert counts == {
        "normal": 8_816,
        "wheel_torque": 374,
        "battery": 297,
        "gps": 189,
        "lidar": 182,
        "ambient_temp": 95,
        "overlap": 47,
    }
    assert int(labels.isin(CLASSIFIER_LABELS).sum()) == 1_137
    assert len(labels) == 10_000


def test_profile_classifier_dataset_has_expected_scope() -> None:
    parquet_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "profile_cleaned_features.parquet"
    )
    df = pd.read_parquet(parquet_path)

    features, target = build_classifier_dataset(df)

    assert features.shape == (1_137, len(CLASSIFIER_FEATURES))
    assert len(target) == 1_137
    assert set(target.unique()) == set(CLASSIFIER_LABELS)
    assert "normal" not in target.values
    assert "overlap" not in target.values


def test_classifier_features_exclude_all_label_sources() -> None:
    assert set(CLASSIFIER_FEATURES).isdisjoint(LABEL_SOURCE_FLAGS)
    assert set(CLASSIFIER_FEATURES).isdisjoint(
        LEAKAGE_EXCLUDED_DERIVED_FEATURES
    )
    assert set(ANOMALY_GROUPS) == set(CLASSIFIER_LABELS)
    assert "wheel_imbalance_score" in CLASSIFIER_FEATURES
    assert FEATURE_LEAKAGE_AUDIT["lidar_saturation_rate"][0] is False
    assert FEATURE_LEAKAGE_AUDIT["composite_health_score"][0] is False
    assert FEATURE_LEAKAGE_AUDIT["wheel_imbalance_score"][0] is True


def test_build_classifier_dataset_reports_missing_features() -> None:
    values = _empty_flags(rows=2)
    frame = pd.DataFrame(values)

    with pytest.raises(ValueError, match="Missing classifier feature columns") as error:
        build_classifier_dataset(frame)

    assert "battery_soc" in str(error.value)
