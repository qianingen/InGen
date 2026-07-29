from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from W05_Predictive_Models import (
    ClassifierExperimentResult,
    load_classifier_dataset,
    make_stratified_split,
    run_classifier_experiment,
)
from ingen_pydev.modeling.anomaly import (
    CLASSIFIER_FEATURES,
    CLASSIFIER_LABELS,
    LABEL_SOURCE_FLAGS,
    LEAKAGE_EXCLUDED_DERIVED_FEATURES,
)


@pytest.fixture(scope="module")
def classifier_data() -> tuple[pd.DataFrame, pd.Series[Any]]:
    return load_classifier_dataset()


@pytest.fixture(scope="module")
def classifier_result() -> ClassifierExperimentResult:
    return run_classifier_experiment()


def test_stratified_split_preserves_class_proportions(
    classifier_data: tuple[pd.DataFrame, pd.Series[Any]],
) -> None:
    X, y = classifier_data

    X_train, X_test, y_train, y_test = make_stratified_split(X, y)

    overall_proportions = y.value_counts(normalize=True).sort_index()
    train_proportions = y_train.value_counts(normalize=True).sort_index()
    test_proportions = y_test.value_counts(normalize=True).sort_index()

    assert len(X_train) == len(y_train) == 909
    assert len(X_test) == len(y_test) == 228
    assert X_train.index.isin(X_test.index).sum() == 0
    assert (train_proportions - overall_proportions).abs().max() < 0.01
    assert (test_proportions - overall_proportions).abs().max() < 0.01


def test_all_five_classes_are_present_in_data_and_both_splits(
    classifier_data: tuple[pd.DataFrame, pd.Series[Any]],
) -> None:
    X, y = classifier_data

    _, _, y_train, y_test = make_stratified_split(X, y)

    expected = set(CLASSIFIER_LABELS)
    assert set(y.unique()) == expected
    assert set(y_train.unique()) == expected
    assert set(y_test.unique()) == expected
    assert "normal" not in y.values
    assert "overlap" not in y.values


def test_classifier_features_have_no_leakage_columns(
    classifier_data: tuple[pd.DataFrame, pd.Series[Any]],
) -> None:
    X, _ = classifier_data

    assert tuple(X.columns) == CLASSIFIER_FEATURES
    assert set(X.columns).isdisjoint(LABEL_SOURCE_FLAGS)
    assert set(X.columns).isdisjoint(LEAKAGE_EXCLUDED_DERIVED_FEATURES)


def test_confusion_matrix_has_fixed_five_class_shape(
    classifier_result: ClassifierExperimentResult,
) -> None:
    result = classifier_result

    assert result.selected_evaluation.confusion_matrix.shape == (5, 5)
    assert result.selected_evaluation.confusion_matrix.index.tolist() == list(
        CLASSIFIER_LABELS
    )
    assert result.selected_evaluation.confusion_matrix.columns.tolist() == list(
        CLASSIFIER_LABELS
    )
    assert int(result.selected_evaluation.confusion_matrix.to_numpy().sum()) == 228


def test_classification_report_covers_every_class(
    classifier_result: ClassifierExperimentResult,
) -> None:
    result = classifier_result
    report = result.selected_evaluation.classification_report

    assert set(CLASSIFIER_LABELS).issubset(report.index)
    assert {"macro avg", "weighted avg"}.issubset(report.index)
    assert {
        "precision",
        "recall",
        "f1-score",
        "support",
    }.issubset(report.columns)
    assert report.reindex(CLASSIFIER_LABELS)["support"].sum() == 228


def test_classifier_experiment_is_deterministic(
    classifier_result: ClassifierExperimentResult,
) -> None:
    first = classifier_result
    second = run_classifier_experiment()

    assert first.selected_model_name == second.selected_model_name
    assert_frame_equal(first.cv_results, second.cv_results, check_exact=True)
    assert_frame_equal(
        first.selected_evaluation.classification_report,
        second.selected_evaluation.classification_report,
        check_exact=True,
    )
    assert_frame_equal(
        first.selected_evaluation.confusion_matrix,
        second.selected_evaluation.confusion_matrix,
        check_exact=True,
    )
    assert_frame_equal(
        first.holdout_comparison,
        second.holdout_comparison,
        check_exact=True,
    )


def test_single_class_input_is_rejected() -> None:
    rows = 20
    X = pd.DataFrame(
        0.0,
        index=pd.RangeIndex(rows),
        columns=CLASSIFIER_FEATURES,
    )
    y = pd.Series(["gps"] * rows, index=X.index, dtype="string")

    with pytest.raises(ValueError, match="at least two classes"):
        make_stratified_split(X, y)
