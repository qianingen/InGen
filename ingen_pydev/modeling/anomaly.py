"""Leakage-aware anomaly labels and classifier features."""

from __future__ import annotations

from typing import Any

import pandas as pd

ANOMALY_GROUPS: dict[str, tuple[str, ...]] = {
    "gps": (
        "gps_dropout",
        "gps_filled",
        "gps_dropout_long",
    ),
    "lidar": (
        "lidar_saturated",
        "lidar_distance_m_spike",
    ),
    "battery": ("battery_soc_spike",),
    "wheel_torque": (
        "wheel_torque_fl_spike",
        "wheel_torque_fr_spike",
        "wheel_torque_rl_spike",
        "wheel_torque_rr_spike",
    ),
    "ambient_temp": ("ambient_temp_c_spike",),
}

CLASSIFIER_LABELS: tuple[str, ...] = tuple(ANOMALY_GROUPS)
LABEL_SOURCE_FLAGS: tuple[str, ...] = tuple(
    flag for flags in ANOMALY_GROUPS.values() for flag in flags
)

# timestamp_ms is an ordering key, not a record-level predictor. The two excluded
# derived features below transitively use lidar_saturated, a target source:
#   lidar_saturation_rate <- lidar_saturated
#   composite_health_score <- battery_soc, lidar_saturation_rate
# wheel_imbalance_score uses only the four raw wheel-torque measurements.
CLASSIFIER_FEATURES: tuple[str, ...] = (
    "lat",
    "lon",
    "lidar_distance_m",
    "battery_soc",
    "wheel_torque_fl",
    "wheel_torque_fr",
    "wheel_torque_rl",
    "wheel_torque_rr",
    "ambient_temp_c",
    "battery_soc_roll_mean_50",
    "battery_soc_roll_std_50",
    "cumulative_distance_m",
    "wheel_imbalance_score",
)

LEAKAGE_EXCLUDED_DERIVED_FEATURES: tuple[str, ...] = (
    "lidar_saturation_rate",
    "composite_health_score",
)

FEATURE_LEAKAGE_AUDIT: dict[str, tuple[bool, str]] = {
    "composite_health_score": (
        False,
        "Depends on lidar_saturation_rate, which is computed from " "lidar_saturated.",
    ),
    "lidar_saturation_rate": (
        False,
        "Computed directly from the target-source flag lidar_saturated.",
    ),
    "wheel_imbalance_score": (
        True,
        "Computed only from the four raw wheel-torque measurements.",
    ),
}


def build_anomaly_type(df: pd.DataFrame) -> pd.Series[Any]:
    """Return one mutually exclusive anomaly label for every input row.

    Rows with no active anomaly group are ``normal``. Rows with exactly one
    active group receive that group name. Rows with two or more active groups
    receive ``overlap``. Because a group name is assigned only when exactly
    one group is active, the implementation has no cross-group priority rule.
    """

    missing = sorted(set(LABEL_SOURCE_FLAGS).difference(df.columns))
    if missing:
        raise ValueError(f"Missing anomaly flag columns: {missing}")

    source_flags = df.loc[:, list(LABEL_SOURCE_FLAGS)]
    columns_with_nulls = source_flags.columns[source_flags.isna().any()].tolist()
    if columns_with_nulls:
        raise ValueError(
            "Anomaly flag columns contain missing values: " f"{columns_with_nulls}"
        )

    group_activity = pd.DataFrame(
        {
            group: source_flags.loc[:, list(flags)].astype(bool).any(axis=1)
            for group, flags in ANOMALY_GROUPS.items()
        },
        index=df.index,
    )
    active_group_count = group_activity.sum(axis=1)

    anomaly_type = pd.Series(
        "normal",
        index=df.index,
        dtype="string",
        name="anomaly_type",
    )
    single_group = active_group_count.eq(1)
    anomaly_type.loc[single_group] = (
        group_activity.loc[single_group].idxmax(axis=1).astype("string")
    )
    anomaly_type.loc[active_group_count.ge(2)] = "overlap"

    return anomaly_type


def build_classifier_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series[Any]]:
    """Return features and targets for single-category anomaly rows only."""

    missing_features = sorted(set(CLASSIFIER_FEATURES).difference(df.columns))
    if missing_features:
        raise ValueError(f"Missing classifier feature columns: {missing_features}")

    anomaly_type = build_anomaly_type(df)
    classifier_rows = anomaly_type.isin(CLASSIFIER_LABELS)
    features = df.loc[classifier_rows, list(CLASSIFIER_FEATURES)].copy()
    target = anomaly_type.loc[classifier_rows].copy()

    return features, target
