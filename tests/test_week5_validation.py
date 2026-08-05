from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import W05_Predictive_Models as predictive
from ingen_pydev.modeling.anomaly import (
    CLASSIFIER_FEATURES,
    CLASSIFIER_LABELS,
)


def _classifier_data(rows_per_class: int = 3) -> tuple[pd.DataFrame, pd.Series[Any]]:
    labels = [label for label in CLASSIFIER_LABELS for _ in range(rows_per_class)]
    index = pd.RangeIndex(len(labels))
    features = pd.DataFrame(
        np.arange(len(labels) * len(CLASSIFIER_FEATURES), dtype=float).reshape(
            len(labels),
            len(CLASSIFIER_FEATURES),
        ),
        index=index,
        columns=CLASSIFIER_FEATURES,
    )
    target = pd.Series(labels, index=index, dtype="string")
    return features, target


@pytest.mark.parametrize(
    ("series", "message"),
    [
        (pd.Series([1.0] * 19 + [None]), "missing or non-numeric"),
        (pd.Series([1.0] * 19 + [np.inf]), "non-finite"),
        (pd.Series(np.arange(19, dtype=float)), "at least 20"),
        (
            pd.Series(
                np.arange(20, dtype=float),
                index=[*range(19), -1],
            ),
            "chronologically ordered",
        ),
        (pd.Series([4.0] * 20), "must contain variation"),
    ],
)
def test_adf_rejects_invalid_and_constant_series(
    series: pd.Series[Any],
    message: str,
) -> None:
    original = series.copy()

    with pytest.raises(ValueError, match=message):
        predictive.run_adf_test(series)

    assert_series_equal(series, original)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_chronological_split_rejects_invalid_fractions(fraction: float) -> None:
    series = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="between 0 and 1"):
        predictive.chronological_split(series, train_fraction=fraction)

    assert series.tolist() == [1.0, 2.0]


def test_chronological_split_requires_rows_in_both_windows() -> None:
    series = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="at least one train and test row"):
        predictive.chronological_split(series, train_fraction=0.1)

    assert len(series) == 2


def test_chronological_split_rejects_duplicate_boundary_index() -> None:
    series = pd.Series([1.0, 2.0], index=[0, 0])

    with pytest.raises(AssertionError, match="overlap"):
        predictive.chronological_split(series)

    assert series.index.tolist() == [0, 0]


def test_arima_order_diagnostics_reject_invalid_d_and_short_series() -> None:
    valid_length = pd.Series(np.arange(40, dtype=float))

    with pytest.raises(ValueError, match="d must be 0 or 1"):
        predictive.select_arima_order(valid_length, d=2)

    with pytest.raises(ValueError, match="too short"):
        predictive.select_arima_order(
            pd.Series(np.arange(20, dtype=float)),
            d=0,
            max_order=10,
        )

    assert len(valid_length) == 40


@pytest.mark.parametrize(
    ("acf_lag_one", "pacf_lag_one", "expected_order"),
    [
        (0.10, 0.20, (1, 0, 0)),
        (0.20, 0.10, (0, 0, 1)),
    ],
)
def test_arima_order_fallback_uses_stronger_first_lag(
    monkeypatch: pytest.MonkeyPatch,
    acf_lag_one: float,
    pacf_lag_one: float,
    expected_order: tuple[int, int, int],
) -> None:
    def fake_acf(
        _series: pd.Series[Any],
        *,
        nlags: int,
        fft: bool,
    ) -> np.ndarray[Any, Any]:
        assert fft is True
        values = np.zeros(nlags + 1)
        values[0] = 1.0
        values[1] = acf_lag_one
        return values

    def fake_pacf(
        _series: pd.Series[Any],
        *,
        nlags: int,
        method: str,
    ) -> np.ndarray[Any, Any]:
        assert method == "ywm"
        values = np.zeros(nlags + 1)
        values[0] = 1.0
        values[1] = pacf_lag_one
        return values

    monkeypatch.setattr("statsmodels.tsa.stattools.acf", fake_acf)
    monkeypatch.setattr("statsmodels.tsa.stattools.pacf", fake_pacf)

    result = predictive.select_arima_order(
        pd.Series(np.arange(40, dtype=float)),
        d=0,
        nlags=10,
    )

    assert result.order == expected_order
    assert result.acf_cutoff == 0
    assert result.pacf_cutoff == 0


@pytest.mark.parametrize(
    "order",
    [
        (1, 2, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (True, 0, 1),
        (1.0, 0, 1),
        (1, 0),
    ],
)
def test_fit_arima_rejects_invalid_orders(order: tuple[Any, ...]) -> None:
    train = pd.Series(np.linspace(1.0, 2.0, 20))

    with pytest.raises(ValueError, match="non-negative integer p and q"):
        predictive.fit_arima_forecaster(train, order)

    assert [train.iloc[0], train.iloc[-1]] == [1.0, 2.0]


class _FakePrediction:
    def __init__(self, mean: object, interval: object) -> None:
        self.predicted_mean = mean
        self._interval = interval

    def conf_int(self, *, alpha: float) -> object:
        assert 0.0 < alpha < 1.0
        return self._interval


class _FakeFittedModel:
    def __init__(self, prediction: _FakePrediction) -> None:
        self.prediction = prediction
        self.requested_steps: int | None = None

    def get_forecast(self, *, steps: int) -> _FakePrediction:
        self.requested_steps = steps
        return self.prediction


@pytest.mark.parametrize(
    ("steps", "alpha", "index", "message"),
    [
        (0, 0.05, None, "steps must be positive"),
        (2, 0.0, None, "alpha must be between 0 and 1"),
        (2, 0.05, pd.Index([1]), "index length must equal steps"),
    ],
)
def test_forecast_validates_horizon_alpha_and_index(
    steps: int,
    alpha: float,
    index: pd.Index[Any] | None,
    message: str,
) -> None:
    model = _FakeFittedModel(_FakePrediction([1.0, 2.0], [[0.0, 2.0]] * 2))

    with pytest.raises(ValueError, match=message):
        predictive.forecast_with_confidence_interval(
            model,
            steps,
            alpha=alpha,
            index=index,
        )

    assert model.requested_steps is None


@pytest.mark.parametrize(
    ("mean", "interval", "message"),
    [
        ([1.0], [[0.0, 2.0], [1.0, 3.0]], "full horizon"),
        ([1.0, 2.0], [[0.0, 2.0]], r"shape \(steps, 2\)"),
    ],
)
def test_forecast_rejects_model_output_with_wrong_shape(
    mean: object,
    interval: object,
    message: str,
) -> None:
    model = _FakeFittedModel(_FakePrediction(mean, interval))

    with pytest.raises(AssertionError, match=message):
        predictive.forecast_with_confidence_interval(model, 2)

    assert model.requested_steps == 2


def test_forecast_uses_range_index_when_none_is_supplied() -> None:
    model = _FakeFittedModel(
        _FakePrediction(
            [1.0, 2.0],
            [[0.0, 2.0], [1.0, 3.0]],
        )
    )

    result = predictive.forecast_with_confidence_interval(model, 2)

    assert result.mean.index.equals(pd.RangeIndex(2))
    assert result.confidence_interval.columns.tolist() == ["lower_95", "upper_95"]


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        (pd.Series([1.0]), pd.Series([1.0, 2.0]), "identical shapes"),
        (pd.Series([], dtype=float), pd.Series([], dtype=float), "non-empty"),
        (
            pd.DataFrame([[1.0, 2.0]]),
            pd.DataFrame([[1.0, 2.0]]),
            "one-dimensional",
        ),
    ],
)
def test_forecast_metrics_reject_invalid_shapes(
    actual: pd.Series[Any] | pd.DataFrame,
    predicted: pd.Series[Any] | pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predictive.calculate_forecast_metrics(
            cast(Any, actual),
            cast(Any, predicted),
        )

    assert actual.shape != () and predicted.shape != ()


def test_naive_forecast_rejects_empty_training_series() -> None:
    train = pd.Series([], dtype=float)
    test = pd.Series([1.0], index=[5])

    with pytest.raises(ValueError, match="at least 1 observation"):
        predictive.naive_persistence_forecast(train, test)

    assert test.index.tolist() == [5]


@pytest.mark.parametrize("test_size", [0.0, 1.0])
def test_stratified_split_rejects_invalid_test_size(test_size: float) -> None:
    features, target = _classifier_data()

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        predictive.make_stratified_split(features, target, test_size=test_size)

    assert len(features) == len(target) == 15


def test_classifier_input_validates_alignment_classes_counts_and_missing_values() -> (
    None
):
    features, target = _classifier_data()

    with pytest.raises(ValueError, match="same number of rows"):
        predictive._validate_classifier_input(features.iloc[:-1], target)

    with pytest.raises(ValueError, match="must not be empty"):
        predictive._validate_classifier_input(features.iloc[:0], target.iloc[:0])

    shifted = target.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(ValueError, match="indices must align"):
        predictive._validate_classifier_input(features, shifted)

    unexpected = target.replace({"ambient_temp": "unknown"})
    with pytest.raises(ValueError, match="exactly these classes"):
        predictive._validate_classifier_input(features, unexpected)

    sparse_features, sparse_target = _classifier_data(rows_per_class=2)
    sparse_target = sparse_target.drop(
        sparse_target[sparse_target.eq(CLASSIFIER_LABELS[0])].index[0]
    )
    sparse_features = sparse_features.loc[sparse_target.index]
    with pytest.raises(ValueError, match="at least two rows"):
        predictive._validate_classifier_input(sparse_features, sparse_target)

    missing = features.copy()
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="must not contain missing"):
        predictive._validate_classifier_input(missing, target)

    assert tuple(features.columns) == CLASSIFIER_FEATURES


@pytest.mark.parametrize(
    ("leak_column", "message"),
    [
        ("gps_dropout", "Target-source flags leaked"),
        ("composite_health_score", "Target-derived features leaked"),
    ],
)
def test_classifier_feature_audit_identifies_leak_type(
    leak_column: str,
    message: str,
) -> None:
    features, _ = _classifier_data()
    leaked = features.rename(columns={CLASSIFIER_FEATURES[0]: leak_column})

    with pytest.raises(AssertionError, match=message):
        predictive._assert_leakage_safe_features(leaked)

    assert leak_column in leaked.columns


def test_classifier_feature_audit_requires_exact_allowlist() -> None:
    features, _ = _classifier_data()
    incomplete = features.drop(columns=[CLASSIFIER_FEATURES[-1]])

    with pytest.raises(AssertionError, match="exactly match"):
        predictive._assert_leakage_safe_features(incomplete)

    assert len(incomplete.columns) == len(CLASSIFIER_FEATURES) - 1


def test_cross_validation_validates_fold_count_and_class_support() -> None:
    features, target = _classifier_data(rows_per_class=2)

    with pytest.raises(ValueError, match="at least 2"):
        predictive.cross_validate_candidates(features, target, cv_folds=1)

    with pytest.raises(ValueError, match="at least cv_folds"):
        predictive.cross_validate_candidates(features, target, cv_folds=3)

    assert target.value_counts().eq(2).all()


def test_cv_selection_validates_structure_and_uses_stable_tie_break() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        predictive.select_model_from_cv(pd.DataFrame({"model": ["a"]}))

    empty = pd.DataFrame(
        columns=["model", "candidate_order", "cv_mean_macro_f1"],
    )
    with pytest.raises(ValueError, match="must not be empty"):
        predictive.select_model_from_cv(empty)

    tied = pd.DataFrame(
        {
            "model": ["later", "earlier"],
            "candidate_order": [1, 0],
            "cv_mean_macro_f1": [0.8, 0.8],
        }
    )
    assert predictive.select_model_from_cv(tied) == "earlier"


def _synthetic_experiment_result() -> predictive.ClassifierExperimentResult:
    report = pd.DataFrame(
        {
            "precision": [1.0],
            "recall": [1.0],
            "f1-score": [1.0],
            "support": [1.0],
        },
        index=pd.Index(["macro avg"], name="label"),
    )
    matrix = pd.DataFrame([[1]], index=["gps"], columns=["gps"])
    selected = predictive.ClassifierEvaluation(
        name="Selected model",
        classification_report=report,
        confusion_matrix=matrix,
        accuracy=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
    )
    baseline = predictive.ClassifierEvaluation(
        name="Dummy most-frequent",
        classification_report=report.copy(),
        confusion_matrix=matrix.copy(),
        accuracy=0.2,
        macro_f1=0.1,
        weighted_f1=0.1,
    )
    return predictive.ClassifierExperimentResult(
        train_n=10,
        test_n=5,
        cv_results=pd.DataFrame(
            {
                "model": ["Selected model"],
                "candidate_order": [0],
                "cv_mean_macro_f1": [0.9],
            }
        ),
        selected_model_name="Selected model",
        selected_evaluation=selected,
        baseline_evaluation=baseline,
        holdout_comparison=pd.DataFrame(
            {
                "model": ["Selected model", "Dummy most-frequent"],
                "macro_f1": [1.0, 0.1],
            }
        ),
    )


def test_run_pipeline_saves_reproducible_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _classifier_data()
    synthetic_result = _synthetic_experiment_result()
    observed_paths: list[Path] = []

    def fake_load(path: Path) -> tuple[pd.DataFrame, pd.Series[Any]]:
        observed_paths.append(path)
        return features, target

    monkeypatch.setattr(predictive, "load_classifier_dataset", fake_load)
    monkeypatch.setattr(
        predictive,
        "_run_classifier_experiment_from_data",
        lambda X, y: synthetic_result,
    )

    first_result, first_paths, class_counts = predictive.run_pipeline(
        "input.parquet",
        tmp_path,
    )
    first_bytes = {name: path.read_bytes() for name, path in first_paths.items()}
    _, second_paths, second_counts = predictive.run_pipeline(
        "input.parquet",
        tmp_path,
    )

    assert first_result is synthetic_result
    assert set(first_paths) == {
        "cv_results",
        "selected_classification_report",
        "selected_confusion_matrix",
        "baseline_classification_report",
        "baseline_confusion_matrix",
        "holdout_comparison",
    }
    assert all(
        path.parent == tmp_path and path.is_file() for path in first_paths.values()
    )
    assert first_bytes == {
        name: path.read_bytes() for name, path in second_paths.items()
    }
    assert class_counts.to_dict() == second_counts.to_dict()
    assert observed_paths == [Path("input.parquet"), Path("input.parquet")]


def test_main_prints_evaluation_and_saved_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _synthetic_experiment_result()
    output_path = tmp_path / "cv_results.csv"
    output_path.write_text("model\nSelected model\n", encoding="utf-8")
    counts = pd.Series(
        [3] * len(CLASSIFIER_LABELS),
        index=CLASSIFIER_LABELS,
    )
    monkeypatch.setattr(
        predictive,
        "run_pipeline",
        lambda: (result, {"cv_results": output_path}, counts),
    )

    predictive.main()
    output = capsys.readouterr().out

    assert "Classifier dataset: n = 15, features = 13" in output
    assert "Train n: 10; test n: 5" in output
    assert "Selected from CV only: Selected model" in output
    assert "Dummy most-frequent" in output
    assert f"cv_results: {output_path}" in output
