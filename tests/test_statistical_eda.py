from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ingen_pydev.modeling.statistical_eda import (
    CORRELATION_RESULT_COLUMNS,
    NORMALITY_RESULT_COLUMNS,
    benjamini_hochberg,
    calculate_distribution_statistics,
    calculate_pairwise_correlations,
    compare_outlier_detectors,
    format_p_value,
    iqr_outlier_mask,
    jaccard_similarity,
    zscore_outlier_mask,
)


def _statistical_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x_values = rng.normal(size=64)
    return pd.DataFrame(
        {
            "x": x_values,
            "y": 2.0 * x_values + rng.normal(scale=0.2, size=64),
            "z": rng.uniform(-2.0, 2.0, size=64),
        }
    )


def test_normality_result_structure() -> None:
    result = calculate_distribution_statistics(
        _statistical_frame(),
        columns=("x", "y", "z"),
    )

    assert tuple(result.columns) == NORMALITY_RESULT_COLUMNS
    assert result["variable"].tolist() == ["x", "y", "z"]
    assert result["n"].eq(64).all()
    assert result["normality_p_value"].between(0.0, 1.0).all()
    assert result["normality_statistic"].notna().all()


def test_correlations_include_raw_p_values_adjusted_p_values_and_n() -> None:
    result = calculate_pairwise_correlations(
        _statistical_frame(),
        columns=("x", "y", "z"),
    )

    assert tuple(result.columns) == CORRELATION_RESULT_COLUMNS
    assert len(result) == 6
    assert set(result["method"]) == {"pearson", "spearman"}
    assert result["raw_p_value"].between(0.0, 1.0).all()
    assert result["bh_adjusted_p_value"].between(0.0, 1.0).all()
    assert result["n"].eq(64).all()


def test_benjamini_hochberg_matches_known_result() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])

    assert np.allclose(adjusted, [0.02, 0.04, 0.04, 0.008])
    assert (adjusted >= np.array([0.01, 0.04, 0.03, 0.002])).all()


def test_iqr_detection_flags_extreme_value() -> None:
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 100.0])

    mask = iqr_outlier_mask(series)

    assert mask.tolist() == [False, False, False, False, True]


def test_zscore_detection_uses_absolute_threshold() -> None:
    series = pd.Series([0.0] * 100 + [100.0])

    mask = zscore_outlier_mask(series)

    assert int(mask.sum()) == 1
    assert bool(mask.iloc[-1])


def test_jaccard_computation_and_empty_set_convention() -> None:
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])

    assert jaccard_similarity(left, right) == pytest.approx(1.0 / 3.0)
    assert jaccard_similarity(
        np.zeros(4, dtype=bool),
        np.zeros(4, dtype=bool),
    ) == 1.0


def test_outlier_comparison_reports_counts_and_jaccard() -> None:
    frame = pd.DataFrame(
        {
            "sensor": [0.0] * 100 + [100.0],
            "sensor_spike": [False] * 100 + [True],
        }
    )

    result = compare_outlier_detectors(
        frame,
        reference_flags={"sensor": "sensor_spike"},
    )

    assert len(result) == 2
    assert result["reference_flagged_count"].eq(1).all()
    assert result["intersection_count"].eq(1).all()
    assert result["union_count"].eq(1).all()
    assert result["jaccard_similarity"].eq(1.0).all()
    assert result["n"].eq(101).all()


@pytest.mark.parametrize(
    "operation",
    [
        lambda frame: calculate_distribution_statistics(
            frame,
            columns=("missing",),
        ),
        lambda frame: calculate_pairwise_correlations(
            frame,
            columns=("x", "missing"),
        ),
        lambda frame: compare_outlier_detectors(
            frame,
            reference_flags={"x": "missing_flag"},
        ),
    ],
)
def test_missing_columns_raise_clear_errors(operation: object) -> None:
    frame = _statistical_frame()

    with pytest.raises(ValueError, match="Missing required columns"):
        operation(frame)  # type: ignore[operator]


def test_statistical_outputs_are_deterministic() -> None:
    frame = _statistical_frame()

    first_normality = calculate_distribution_statistics(
        frame,
        columns=("x", "y", "z"),
    )
    second_normality = calculate_distribution_statistics(
        frame,
        columns=("x", "y", "z"),
    )
    first_correlations = calculate_pairwise_correlations(
        frame,
        columns=("x", "y", "z"),
    )
    second_correlations = calculate_pairwise_correlations(
        frame,
        columns=("x", "y", "z"),
    )

    assert_frame_equal(first_normality, second_normality, check_exact=True)
    assert_frame_equal(first_correlations, second_correlations, check_exact=True)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -0.1, 1.1])
def test_p_value_formatting_rejects_invalid_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="p_value must be finite"):
        format_p_value(bad_value)

    assert format_p_value(0.00001) == "p < 0.001"


@pytest.mark.parametrize("threshold", [0.0, 1.0, -1.0])
def test_p_value_formatting_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between"):
        format_p_value(0.5, threshold=threshold)

    assert format_p_value(0.5) == "p = 0.500"


def test_distribution_statistics_reject_short_and_nonfinite_columns() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        calculate_distribution_statistics(
            pd.DataFrame({"x": np.arange(7, dtype=float)}),
            columns=("x",),
        )

    nonfinite = pd.DataFrame({"x": [1.0] * 7 + [np.inf]})
    with pytest.raises(ValueError, match="missing or non-finite"):
        calculate_distribution_statistics(nonfinite, columns=("x",))

    assert len(nonfinite) == 8


def test_benjamini_hochberg_validates_shape_range_and_empty_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        benjamini_hochberg([[0.1, 0.2]])  # type: ignore[list-item]

    for invalid in ([np.nan], [np.inf], [-0.1], [1.1]):
        with pytest.raises(ValueError, match="finite and between"):
            benjamini_hochberg(invalid)

    result = benjamini_hochberg([])
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


def test_correlations_reject_duplicate_columns_and_unknown_methods() -> None:
    frame = _statistical_frame()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        calculate_pairwise_correlations(frame, columns=("x", "x"))

    with pytest.raises(ValueError, match="Unsupported correlation methods"):
        calculate_pairwise_correlations(
            frame,
            columns=("x", "y"),
            methods=("kendall",),
        )

    assert frame.columns.tolist() == ["x", "y", "z"]


@pytest.mark.parametrize("multiplier", [0.0, -1.0])
def test_iqr_rejects_invalid_multiplier(multiplier: float) -> None:
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="multiplier must be positive"):
        iqr_outlier_mask(series, multiplier=multiplier)

    assert series.tolist() == [1.0, 2.0, 3.0]


def test_iqr_rejects_missing_and_nonfinite_values() -> None:
    for invalid in (None, np.inf):
        series = pd.Series([1.0, invalid, 3.0])
        with pytest.raises(ValueError, match="missing or non-finite"):
            iqr_outlier_mask(series)

    assert len(series) == 3


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_zscore_rejects_invalid_threshold(threshold: float) -> None:
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="threshold must be positive"):
        zscore_outlier_mask(series, threshold=threshold)

    assert not series.empty


def test_zscore_handles_constant_data_and_rejects_nonfinite_values() -> None:
    constant = pd.Series([3.0, 3.0, 3.0], index=[4, 5, 6])

    mask = zscore_outlier_mask(constant)

    assert mask.index.equals(constant.index)
    assert not mask.any()
    with pytest.raises(ValueError, match="missing or non-finite"):
        zscore_outlier_mask(pd.Series([1.0, np.nan]))


def test_jaccard_rejects_mismatched_and_multidimensional_masks() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        jaccard_similarity([True], [True, False])  # type: ignore[arg-type]

    left = np.array([[True, False]])
    right = np.array([[False, True]])
    with pytest.raises(ValueError, match="one-dimensional"):
        jaccard_similarity(left, right)

    assert left.shape == right.shape == (1, 2)
