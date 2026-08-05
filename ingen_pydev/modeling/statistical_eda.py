"""Reusable statistical EDA helpers for Week 5 telemetry analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS_FIELDS: tuple[str, ...] = (
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
    "lidar_saturation_rate",
    "composite_health_score",
)

OUTLIER_REFERENCE_FLAGS: dict[str, str] = {
    "lidar_distance_m": "lidar_distance_m_spike",
    "battery_soc": "battery_soc_spike",
    "wheel_torque_fl": "wheel_torque_fl_spike",
    "wheel_torque_fr": "wheel_torque_fr_spike",
    "wheel_torque_rl": "wheel_torque_rl_spike",
    "wheel_torque_rr": "wheel_torque_rr_spike",
    "ambient_temp_c": "ambient_temp_c_spike",
}

CORRELATION_RESULT_COLUMNS: tuple[str, ...] = (
    "variable_x",
    "variable_y",
    "method",
    "coefficient",
    "raw_p_value",
    "bh_adjusted_p_value",
    "n",
)

NORMALITY_RESULT_COLUMNS: tuple[str, ...] = (
    "variable",
    "mean",
    "median",
    "standard_deviation",
    "skewness",
    "normality_statistic",
    "normality_p_value",
    "normality_p_value_report",
    "n",
)

OUTLIER_RESULT_COLUMNS: tuple[str, ...] = (
    "variable",
    "reference_flag",
    "method",
    "detector_flagged_count",
    "reference_flagged_count",
    "intersection_count",
    "union_count",
    "jaccard_similarity",
    "n",
)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    required = tuple(columns)
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _numeric_values(df: pd.DataFrame, column: str) -> np.ndarray[Any, Any]:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError(f"Column {column!r} contains missing or non-finite values")
    return values


def format_p_value(p_value: float, threshold: float = 0.001) -> str:
    """Format very small p-values without presenting them as zero."""

    if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must be finite and between 0 and 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if p_value < threshold:
        return f"p < {threshold:.3f}"
    return f"p = {p_value:.3f}"


def calculate_distribution_statistics(
    df: pd.DataFrame,
    columns: Sequence[str] = CONTINUOUS_FIELDS,
) -> pd.DataFrame:
    """Calculate descriptive statistics and D'Agostino K-squared tests."""

    _require_columns(df, columns)
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = _numeric_values(df, column)
        if len(values) < 8:
            raise ValueError(
                f"Column {column!r} needs at least 8 values for normality testing"
            )
        normality = stats.normaltest(values, nan_policy="raise")
        p_value = float(normality.pvalue)
        rows.append(
            {
                "variable": column,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "skewness": float(stats.skew(values, bias=False)),
                "normality_statistic": float(normality.statistic),
                "normality_p_value": p_value,
                "normality_p_value_report": format_p_value(p_value),
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows, columns=NORMALITY_RESULT_COLUMNS)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray[Any, Any]:
    """Return Benjamini-Hochberg adjusted p-values in input order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return np.array([], dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must be finite and between 0 and 1")

    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    ranks = np.arange(1, len(values) + 1, dtype=float)
    adjusted_ranked = ranked * len(values) / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return cast(np.ndarray[Any, Any], adjusted)


def calculate_pairwise_correlations(
    df: pd.DataFrame,
    columns: Sequence[str] = CONTINUOUS_FIELDS,
    methods: Sequence[str] = ("pearson", "spearman"),
) -> pd.DataFrame:
    """Calculate pairwise correlations and within-method BH corrections."""

    _require_columns(df, columns)
    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicates")
    unsupported = sorted(set(methods).difference({"pearson", "spearman"}))
    if unsupported:
        raise ValueError(f"Unsupported correlation methods: {unsupported}")

    numeric = {column: _numeric_values(df, column) for column in columns}
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows: list[dict[str, Any]] = []
        for left_index, variable_x in enumerate(columns):
            for variable_y in columns[left_index + 1 :]:
                x_values = numeric[variable_x]
                y_values = numeric[variable_y]
                if method == "pearson":
                    result = stats.pearsonr(x_values, y_values)
                else:
                    result = stats.spearmanr(x_values, y_values)
                method_rows.append(
                    {
                        "variable_x": variable_x,
                        "variable_y": variable_y,
                        "method": method,
                        "coefficient": float(result.statistic),
                        "raw_p_value": float(result.pvalue),
                        "n": int(len(x_values)),
                    }
                )

        adjusted = benjamini_hochberg([row["raw_p_value"] for row in method_rows])
        for row, adjusted_p_value in zip(method_rows, adjusted, strict=True):
            row["bh_adjusted_p_value"] = float(adjusted_p_value)
        rows.extend(method_rows)

    return pd.DataFrame(rows, columns=CORRELATION_RESULT_COLUMNS)


def iqr_outlier_mask(
    series: pd.Series[Any],
    *,
    multiplier: float = 1.5,
) -> pd.Series[Any]:
    """Return a Boolean mask for values outside the Tukey IQR fences."""

    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("series contains missing or non-finite values")
    first_quartile = float(numeric.quantile(0.25))
    third_quartile = float(numeric.quantile(0.75))
    iqr = third_quartile - first_quartile
    lower = first_quartile - multiplier * iqr
    upper = third_quartile + multiplier * iqr
    return ((numeric < lower) | (numeric > upper)).rename("iqr_outlier")


def zscore_outlier_mask(
    series: pd.Series[Any],
    *,
    threshold: float = 3.0,
) -> pd.Series[Any]:
    """Return a Boolean mask for absolute population z-scores over threshold."""

    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("series contains missing or non-finite values")
    standard_deviation = float(numeric.std(ddof=0))
    if standard_deviation == 0.0:
        return pd.Series(False, index=series.index, name="zscore_outlier")
    z_scores = (numeric - float(numeric.mean())) / standard_deviation
    return z_scores.abs().gt(threshold).rename("zscore_outlier")


def jaccard_similarity(
    left: pd.Series[Any] | np.ndarray[Any, Any],
    right: pd.Series[Any] | np.ndarray[Any, Any],
) -> float:
    """Calculate Jaccard similarity between two Boolean detector masks."""

    left_values = np.asarray(left, dtype=bool)
    right_values = np.asarray(right, dtype=bool)
    if left_values.shape != right_values.shape:
        raise ValueError("Jaccard masks must have identical shapes")
    if left_values.ndim != 1:
        raise ValueError("Jaccard masks must be one-dimensional")
    union_count = int(np.logical_or(left_values, right_values).sum())
    if union_count == 0:
        return 1.0
    intersection_count = int(np.logical_and(left_values, right_values).sum())
    return float(intersection_count / union_count)


def compare_outlier_detectors(
    df: pd.DataFrame,
    reference_flags: Mapping[str, str] = OUTLIER_REFERENCE_FLAGS,
) -> pd.DataFrame:
    """Compare global outlier rules with sliding-window reference flags."""

    required = [column for pair in reference_flags.items() for column in pair]
    _require_columns(df, required)

    rows: list[dict[str, Any]] = []
    for variable, reference_flag in reference_flags.items():
        reference = df[reference_flag].astype(bool)
        detector_masks = {
            "IQR (1.5 × IQR)": iqr_outlier_mask(df[variable]),
            "|z-score| > 3": zscore_outlier_mask(df[variable]),
        }
        for method, detector in detector_masks.items():
            intersection = detector & reference
            union = detector | reference
            rows.append(
                {
                    "variable": variable,
                    "reference_flag": reference_flag,
                    "method": method,
                    "detector_flagged_count": int(detector.sum()),
                    "reference_flagged_count": int(reference.sum()),
                    "intersection_count": int(intersection.sum()),
                    "union_count": int(union.sum()),
                    "jaccard_similarity": jaccard_similarity(
                        detector,
                        reference,
                    ),
                    "n": int(len(df)),
                }
            )
    return pd.DataFrame(rows, columns=OUTLIER_RESULT_COLUMNS)
