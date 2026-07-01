import pytest

from ingen_pydev.algo.sliding_window import SlidingWindowTelemetry


def test_add_single_value_returns_basic_stats() -> None:
    window = SlidingWindowTelemetry(window_size=5)

    stats = window.add(10.0)

    assert stats.count == 1
    assert stats.mean == pytest.approx(10.0)
    assert stats.std == pytest.approx(0.0)
    assert stats.z_score == pytest.approx(0.0)
    assert stats.is_anomaly is False


def test_rolling_mean_updates_when_window_slides() -> None:
    window = SlidingWindowTelemetry(window_size=3)

    window.add(10.0)
    window.add(20.0)
    stats = window.add(30.0)

    assert stats.count == 3
    assert stats.mean == pytest.approx(20.0)

    # Window should now slide from [10, 20, 30] to [20, 30, 40]
    stats = window.add(40.0)

    assert stats.count == 3
    assert stats.mean == pytest.approx(30.0)


def test_rolling_std_is_computed_from_running_sums() -> None:
    window = SlidingWindowTelemetry(window_size=3)

    window.add(10.0)
    window.add(11.0)
    stats = window.add(12.0)

    # For [10, 11, 12]:
    # mean = 11
    # variance = ((100 + 121 + 144) / 3) - 11^2
    # variance = 365/3 - 121 = 0.666...
    assert stats.mean == pytest.approx(11.0)
    assert stats.std == pytest.approx((2 / 3) ** 0.5)


def test_anomaly_detected_with_large_z_score() -> None:
    window = SlidingWindowTelemetry(window_size=5, anomaly_z_threshold=3.0)

    for value in [10.0, 11.0, 10.0, 11.0, 10.0]:
        window.add(value)

    stats = window.add(100.0)

    assert stats.z_score > 3.0
    assert stats.is_anomaly is True


def test_non_anomaly_when_value_is_close_to_window_mean() -> None:
    window = SlidingWindowTelemetry(window_size=5, anomaly_z_threshold=3.0)

    for value in [10.0, 11.0, 10.0, 11.0, 10.0]:
        window.add(value)

    stats = window.add(10.5)

    assert abs(stats.z_score) < 3.0
    assert stats.is_anomaly is False


def test_invalid_window_size_raises_error() -> None:
    with pytest.raises(ValueError):
        SlidingWindowTelemetry(window_size=0)


def test_invalid_anomaly_threshold_raises_error() -> None:
    with pytest.raises(ValueError):
        SlidingWindowTelemetry(window_size=5, anomaly_z_threshold=0.0)
