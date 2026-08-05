"""Week 5 anomaly classification and telemetry forecasting helpers.

The classification experiment answers a deliberately narrow question:

    Given that a telemetry record is a single-category anomaly, which of the
    five anomaly categories does it belong to?

Normal and overlapping-anomaly rows are therefore excluded before the
stratified train/test split.  Target-source flags and derived features that
depend on those flags are never included in the feature matrix.

The time-series functions implement a separate, chronological battery-SoC
forecast. The ADF p-value decides whether d is 0 or 1; ACF and PACF diagnostics
then select a small candidate ARIMA order without using a shuffled split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ingen_pydev.modeling.anomaly import (
    CLASSIFIER_FEATURES,
    CLASSIFIER_LABELS,
    LABEL_SOURCE_FLAGS,
    LEAKAGE_EXCLUDED_DERIVED_FEATURES,
    build_anomaly_type,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
EXPECTED_INPUT_ROWS = 10_000
EXPECTED_CLASSIFIER_ROWS = 1_137
PARQUET_PATH = (
    Path(__file__).resolve().parent / "outputs" / "profile_cleaned_features.parquet"
)


@dataclass(frozen=True)
class ClassifierEvaluation:
    """One model's single held-out-test evaluation."""

    name: str
    classification_report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    accuracy: float
    macro_f1: float
    weighted_f1: float


@dataclass(frozen=True)
class ClassifierExperimentResult:
    """Training-only selection evidence and held-out evaluations."""

    train_n: int
    test_n: int
    cv_results: pd.DataFrame
    selected_model_name: str
    selected_evaluation: ClassifierEvaluation
    baseline_evaluation: ClassifierEvaluation
    holdout_comparison: pd.DataFrame


@dataclass(frozen=True)
class ADFTestResult:
    """Augmented Dickey-Fuller test output."""

    statistic: float
    p_value: float
    nobs: int
    used_lag: int
    critical_values: dict[str, float]
    rejected: bool


@dataclass(frozen=True)
class ARIMAOrderDiagnostics:
    """ACF/PACF evidence and selected ARIMA order."""

    order: tuple[int, int, int]
    acf_values: np.ndarray[Any, np.dtype[np.float64]]
    pacf_values: np.ndarray[Any, np.dtype[np.float64]]
    significance_threshold: float
    acf_cutoff: int | None
    pacf_cutoff: int | None


@dataclass(frozen=True)
class ForecastResult:
    """Point forecast and confidence interval over one forecast horizon."""

    mean: pd.Series[Any]
    confidence_interval: pd.DataFrame


def _as_numeric_series(
    series: pd.Series[Any],
    name: str,
    *,
    min_observations: int = 1,
) -> pd.Series[Any]:
    """Return a finite float series while preserving its chronological index."""

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.isna().any():
        raise ValueError(f"{name} contains missing or non-numeric values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values")
    if len(numeric) < min_observations:
        raise ValueError(
            f"{name} must contain at least {min_observations} observations"
        )
    if not numeric.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be chronologically ordered")
    return numeric


def run_adf_test(
    series: pd.Series[Any],
    *,
    alpha: float = 0.05,
) -> ADFTestResult:
    """Run an ADF test whose p-value can be used to choose d."""

    from statsmodels.tsa.stattools import adfuller

    clean = _as_numeric_series(
        series,
        "ADF series",
        min_observations=20,
    )
    if clean.nunique() < 2:
        raise ValueError("ADF series must contain variation")
    result = adfuller(clean.to_numpy(), autolag="AIC")
    critical_values = {str(level): float(value) for level, value in result[4].items()}
    return ADFTestResult(
        statistic=float(result[0]),
        p_value=float(result[1]),
        nobs=int(result[3]),
        used_lag=int(result[2]),
        critical_values=critical_values,
        rejected=bool(float(result[1]) < alpha),
    )


def chronological_split(
    series: pd.Series[Any],
    *,
    train_fraction: float = 0.80,
) -> tuple[pd.Series[Any], pd.Series[Any]]:
    """Split an ordered series into an earlier train and later test window."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    clean = _as_numeric_series(
        series,
        "Forecast series",
        min_observations=2,
    )
    split_at = int(len(clean) * train_fraction)
    if split_at <= 0 or split_at >= len(clean):
        raise ValueError("Split must leave at least one train and test row")
    train = clean.iloc[:split_at].copy()
    test = clean.iloc[split_at:].copy()
    if train.index[-1] >= test.index[0]:
        raise AssertionError("Chronological train/test windows overlap")
    return train, test


def _diagnostic_cutoff(
    values: np.ndarray[Any, np.dtype[np.float64]],
    threshold: float,
    max_order: int,
) -> int | None:
    """Return a short-lag cutoff, or None when correlations tail off."""

    for lag in range(1, max_order + 1):
        if abs(float(values[lag])) <= threshold:
            return lag - 1
    return None


def select_arima_order(
    stationary_series: pd.Series[Any],
    *,
    d: int,
    nlags: int = 40,
    max_order: int = 5,
) -> ARIMAOrderDiagnostics:
    """Use ACF/PACF cutoff behavior to choose a compact ARIMA order.

    A short PACF cutoff supplies p and a short ACF cutoff supplies q. If a
    diagnostic remains significant through ``max_order``, it is treated as a
    tail rather than a finite cutoff and contributes order zero.
    """

    from statsmodels.tsa.stattools import acf, pacf

    if d not in (0, 1):
        raise ValueError("d must be 0 or 1")
    clean = _as_numeric_series(
        stationary_series,
        "Stationary series",
        min_observations=20,
    )
    safe_nlags = min(nlags, len(clean) // 2 - 1)
    if safe_nlags < max_order:
        raise ValueError("Series is too short for requested ACF/PACF diagnostics")
    acf_values = np.asarray(acf(clean, nlags=safe_nlags, fft=True), dtype=float)
    pacf_values = np.asarray(
        pacf(clean, nlags=safe_nlags, method="ywm"),
        dtype=float,
    )
    threshold = float(1.96 / np.sqrt(len(clean)))
    acf_cutoff = _diagnostic_cutoff(acf_values, threshold, max_order)
    pacf_cutoff = _diagnostic_cutoff(pacf_values, threshold, max_order)
    p = pacf_cutoff if pacf_cutoff is not None and pacf_cutoff != 0 else 0
    q = acf_cutoff if acf_cutoff is not None and acf_cutoff != 0 else 0

    if p == 0 and q == 0:
        if abs(float(pacf_values[1])) >= abs(float(acf_values[1])):
            p = 1
        else:
            q = 1

    return ARIMAOrderDiagnostics(
        order=(p, d, q),
        acf_values=acf_values,
        pacf_values=pacf_values,
        significance_threshold=threshold,
        acf_cutoff=acf_cutoff,
        pacf_cutoff=pacf_cutoff,
    )


def fit_arima_forecaster(
    train: pd.Series[Any],
    order: tuple[int, int, int],
) -> Any:
    """Fit an ARIMA model to the chronological training window."""

    from statsmodels.tsa.arima.model import ARIMA

    clean_train = _as_numeric_series(
        train,
        "ARIMA training series",
        min_observations=20,
    )
    if (
        len(order) != 3
        or any(not isinstance(value, int) or isinstance(value, bool) for value in order)
        or order[0] < 0
        or order[1] not in (0, 1)
        or order[2] < 0
    ):
        raise ValueError(
            "order must contain non-negative integer p and q with d in {0, 1}"
        )
    model = ARIMA(
        clean_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit()


def forecast_with_confidence_interval(
    fitted_model: Any,
    steps: int,
    *,
    index: pd.Index[Any] | None = None,
    alpha: float = 0.05,
) -> ForecastResult:
    """Forecast exactly ``steps`` rows and return a two-column interval."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if index is None:
        forecast_index: pd.Index[Any] = pd.RangeIndex(steps)
    else:
        if len(index) != steps:
            raise ValueError("Forecast index length must equal steps")
        forecast_index = index

    prediction = fitted_model.get_forecast(steps=steps)
    mean_values = np.asarray(prediction.predicted_mean, dtype=float)
    interval_values = np.asarray(prediction.conf_int(alpha=alpha), dtype=float)
    if mean_values.shape != (steps,):
        raise AssertionError("Forecast mean does not cover the full horizon")
    if interval_values.shape != (steps, 2):
        raise AssertionError("Confidence interval must have shape (steps, 2)")

    mean = pd.Series(mean_values, index=forecast_index, name="forecast")
    confidence_interval = pd.DataFrame(
        interval_values,
        index=forecast_index,
        columns=["lower_95", "upper_95"],
    )
    return ForecastResult(
        mean=mean,
        confidence_interval=confidence_interval,
    )


def calculate_forecast_metrics(
    actual: pd.Series[Any],
    predicted: pd.Series[Any],
) -> dict[str, float]:
    """Calculate RMSE and MAE over two equal-length forecast arrays."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have identical shapes")
    if actual_values.ndim != 1 or actual_values.size == 0:
        raise ValueError("Forecast metrics require non-empty one-dimensional data")
    errors = actual_values - predicted_values
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
    }


def naive_persistence_forecast(
    train: pd.Series[Any],
    test: pd.Series[Any],
) -> pd.Series[Any]:
    """Forecast every test row as the final observed training value."""

    clean_train = _as_numeric_series(train, "Naive training series")
    clean_test = _as_numeric_series(test, "Naive test series")
    return pd.Series(
        float(clean_train.iloc[-1]),
        index=clean_test.index,
        name="naive_persistence",
    )


def load_classifier_dataset(
    parquet_path: Path = PARQUET_PATH,
) -> tuple[pd.DataFrame, pd.Series[Any]]:
    """Load the Week 3/4 data and select single-category anomaly records."""

    df = pd.read_parquet(parquet_path)
    if len(df) != EXPECTED_INPUT_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_INPUT_ROWS:,} telemetry rows, got {len(df):,}"
        )

    anomaly_type = build_anomaly_type(df)
    if len(anomaly_type) != len(df) or anomaly_type.isna().any():
        raise AssertionError("Every telemetry row must have exactly one label")

    classifier_mask = anomaly_type.isin(CLASSIFIER_LABELS)
    X = df.loc[classifier_mask, list(CLASSIFIER_FEATURES)].copy()
    y = anomaly_type.loc[classifier_mask].copy()

    _assert_leakage_safe_features(X)
    if len(X) != EXPECTED_CLASSIFIER_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_CLASSIFIER_ROWS:,} single-category anomalies, "
            f"got {len(X):,}"
        )
    if len(X) != len(y):
        raise AssertionError("Feature and target row counts differ")
    if set(y.unique()) != set(CLASSIFIER_LABELS):
        raise AssertionError(
            "Classifier target must contain exactly the five single-anomaly "
            f"labels: {CLASSIFIER_LABELS}"
        )
    if y.isin(("normal", "overlap")).any():
        raise AssertionError("normal and overlap must be excluded from training")
    if X.isna().any().any():
        raise AssertionError("Classifier features must not contain missing values")

    return X, y


def _assert_leakage_safe_features(X: pd.DataFrame) -> None:
    """Assert that X matches the centrally audited feature allow-list."""

    label_flag_leaks = sorted(set(X.columns).intersection(LABEL_SOURCE_FLAGS))
    if label_flag_leaks:
        raise AssertionError(
            f"Target-source flags leaked into classifier features: {label_flag_leaks}"
        )

    derived_leaks = sorted(
        set(X.columns).intersection(LEAKAGE_EXCLUDED_DERIVED_FEATURES)
    )
    if derived_leaks:
        raise AssertionError(
            f"Target-derived features leaked into classifier features: {derived_leaks}"
        )

    actual_features = tuple(X.columns)
    if actual_features != CLASSIFIER_FEATURES:
        raise AssertionError(
            "Feature matrix must exactly match CLASSIFIER_FEATURES; "
            f"got {actual_features}"
        )


def make_stratified_split(
    X: pd.DataFrame,
    y: pd.Series[Any],
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[Any], pd.Series[Any]]:
    """Create a deterministic stratified holdout split."""

    from sklearn.model_selection import train_test_split

    _validate_classifier_input(X, y)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if set(y_train.unique()) != set(CLASSIFIER_LABELS):
        raise AssertionError("Training split does not contain all classifier labels")
    if set(y_test.unique()) != set(CLASSIFIER_LABELS):
        raise AssertionError("Test split does not contain all classifier labels")
    _assert_leakage_safe_features(X_train)
    _assert_leakage_safe_features(X_test)
    return X_train, X_test, y_train, y_test


def _validate_classifier_input(
    X: pd.DataFrame,
    y: pd.Series[Any],
) -> None:
    """Validate class coverage, alignment, and the feature allow-list."""

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if len(X) == 0:
        raise ValueError("Classifier input must not be empty")
    if not X.index.equals(y.index):
        raise ValueError("X and y indices must align")
    unique_labels = set(y.unique())
    if len(unique_labels) < 2:
        raise ValueError("Classifier target must contain at least two classes")
    if unique_labels != set(CLASSIFIER_LABELS):
        raise ValueError(
            "Classifier target must contain exactly these classes: "
            f"{CLASSIFIER_LABELS}"
        )
    if y.value_counts().min() < 2:
        raise ValueError("Every classifier class needs at least two rows")
    if X.isna().any().any():
        raise ValueError("Classifier features must not contain missing values")
    _assert_leakage_safe_features(X)


def build_candidate_classifiers() -> dict[str, Any]:
    """Return the three fixed, untuned classifier candidates."""

    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Dummy most-frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series[Any],
    *,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Rank fixed candidates using training-only stratified macro-F1."""

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    _validate_classifier_input(X_train, y_train)
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if int(y_train.value_counts().min()) < cv_folds:
        raise ValueError("Every class must have at least cv_folds training rows")
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    rows: list[dict[str, Any]] = []
    for candidate_order, (name, model) in enumerate(
        build_candidate_classifiers().items()
    ):
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="f1_macro",
            cv=splitter,
            n_jobs=1,
        )
        row: dict[str, Any] = {
            "model": name,
            "candidate_order": candidate_order,
            "cv_mean_macro_f1": float(np.mean(scores)),
            "cv_std_macro_f1": float(np.std(scores, ddof=0)),
            "cv_folds": int(cv_folds),
        }
        for fold_number, score in enumerate(scores, start=1):
            row[f"fold_{fold_number}_macro_f1"] = float(score)
        rows.append(row)
    return pd.DataFrame(rows)


def select_model_from_cv(cv_results: pd.DataFrame) -> str:
    """Select the highest mean macro-F1 candidate without test-set access."""

    required = {"model", "candidate_order", "cv_mean_macro_f1"}
    missing = sorted(required.difference(cv_results.columns))
    if missing:
        raise ValueError(f"CV results are missing columns: {missing}")
    if cv_results.empty:
        raise ValueError("CV results must not be empty")
    ranked = cv_results.sort_values(
        ["cv_mean_macro_f1", "candidate_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["model"])


def evaluate_classifier(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series[Any],
    y_test: pd.Series[Any],
) -> ClassifierEvaluation:
    """Fit a classifier and return fixed-label holdout metrics."""

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report_dict = classification_report(
        y_test,
        predictions,
        labels=list(CLASSIFIER_LABELS),
        target_names=list(CLASSIFIER_LABELS),
        output_dict=True,
        zero_division=0,
    )
    report = pd.DataFrame(report_dict).transpose()
    report.index.name = "label"
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(CLASSIFIER_LABELS),
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(CLASSIFIER_LABELS, name="actual"),
        columns=pd.Index(CLASSIFIER_LABELS, name="predicted"),
    )
    return ClassifierEvaluation(
        name=name,
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(cast(float, report.loc["macro avg", "f1-score"])),
        weighted_f1=float(cast(float, report.loc["weighted avg", "f1-score"])),
        classification_report=report,
        confusion_matrix=matrix_df,
    )


def _run_classifier_experiment_from_data(
    X: pd.DataFrame,
    y: pd.Series[Any],
) -> ClassifierExperimentResult:
    """Run selection and evaluation on an already validated feature set."""

    from sklearn.base import clone

    X_train, X_test, y_train, y_test = make_stratified_split(X, y)
    cv_results = cross_validate_candidates(X_train, y_train)
    selected_name = select_model_from_cv(cv_results)
    candidates = build_candidate_classifiers()

    selected_evaluation = evaluate_classifier(
        selected_name,
        clone(candidates[selected_name]),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    baseline_name = "Dummy most-frequent"
    baseline_evaluation = evaluate_classifier(
        baseline_name,
        clone(candidates[baseline_name]),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    holdout_comparison = pd.DataFrame(
        [
            {
                "model": selected_evaluation.name,
                "accuracy": selected_evaluation.accuracy,
                "macro_f1": selected_evaluation.macro_f1,
                "weighted_f1": selected_evaluation.weighted_f1,
                "selection_role": "Selected by training CV",
            },
            {
                "model": baseline_evaluation.name,
                "accuracy": baseline_evaluation.accuracy,
                "macro_f1": baseline_evaluation.macro_f1,
                "weighted_f1": baseline_evaluation.weighted_f1,
                "selection_role": "Predefined baseline",
            },
        ]
    )
    return ClassifierExperimentResult(
        train_n=len(X_train),
        test_n=len(X_test),
        cv_results=cv_results,
        selected_model_name=selected_name,
        selected_evaluation=selected_evaluation,
        baseline_evaluation=baseline_evaluation,
        holdout_comparison=holdout_comparison,
    )


def run_classifier_experiment(
    parquet_path: Path = PARQUET_PATH,
) -> ClassifierExperimentResult:
    """Select by training CV, then evaluate once on the held-out test set."""

    X, y = load_classifier_dataset(parquet_path)
    return _run_classifier_experiment_from_data(X, y)


def save_classifier_outputs(
    result: ClassifierExperimentResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save reproducible CV, report, matrix, and baseline comparison tables."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "cv_results": result.cv_results,
        "selected_classification_report": (
            result.selected_evaluation.classification_report
        ),
        "selected_confusion_matrix": result.selected_evaluation.confusion_matrix,
        "baseline_classification_report": (
            result.baseline_evaluation.classification_report
        ),
        "baseline_confusion_matrix": result.baseline_evaluation.confusion_matrix,
        "holdout_comparison": result.holdout_comparison,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=True, float_format="%.17g")
        paths[name] = path
    return paths


def run_pipeline(
    input_path: str | Path = PARQUET_PATH,
    output_dir: str | Path = Path("outputs") / "classifier_evaluation",
) -> tuple[
    ClassifierExperimentResult,
    dict[str, Path],
    pd.Series[Any],
]:
    """Run the complete classifier workflow and persist reproducible tables."""

    X, y = load_classifier_dataset(Path(input_path))
    result = _run_classifier_experiment_from_data(X, y)
    output_paths = save_classifier_outputs(result, output_dir)
    class_counts = y.value_counts().reindex(CLASSIFIER_LABELS)
    return result, output_paths, class_counts


def _print_dataset_summary(
    result: ClassifierExperimentResult,
    class_counts: pd.Series[Any],
) -> None:
    total_rows = result.train_n + result.test_n
    print(
        f"Classifier dataset: n = {total_rows:,}, features = {len(CLASSIFIER_FEATURES)}"
    )
    print("Scope: five mutually exclusive single-category anomalies")
    print("Excluded targets: normal, overlap")
    print("\nClass counts:")
    print(class_counts.to_string())
    print("\nLeakage-safe features:")
    print(list(CLASSIFIER_FEATURES))


def _print_evaluation(evaluation: ClassifierEvaluation) -> None:
    print(f"\n{'=' * 72}\n{evaluation.name}")
    print(f"Accuracy: {evaluation.accuracy:.4f}")
    print(f"Macro-F1: {evaluation.macro_f1:.4f}")
    print(f"Weighted-F1: {evaluation.weighted_f1:.4f}")
    print("\nClassification report:")
    print(evaluation.classification_report.round(4).to_string())
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(evaluation.confusion_matrix.to_string())


def main() -> None:
    """Run training-only selection and final held-out evaluation."""

    result, output_paths, class_counts = run_pipeline()
    _print_dataset_summary(result, class_counts)
    print(f"\nTrain n: {result.train_n:,}; test n: {result.test_n:,}")
    print("\nTraining-only CV comparison:")
    print(result.cv_results.round(4).to_string(index=False))
    print(f"\nSelected from CV only: {result.selected_model_name}")
    _print_evaluation(result.selected_evaluation)
    _print_evaluation(result.baseline_evaluation)
    print("\nSaved outputs:")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
