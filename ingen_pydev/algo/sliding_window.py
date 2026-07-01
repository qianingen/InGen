from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WindowStats:
    count: int
    mean: float
    std: float
    z_score: float
    is_anomaly: bool


class SlidingWindowTelemetry:
    """Fixed-size sliding window for numeric telemetry streams.

    This is a lightweight stream preprocessing utility.
    It maintains rolling mean, rolling standard deviation, and z-score
    in O(1) time per new reading.
    """

    def __init__(self, window_size: int, anomaly_z_threshold: float = 3.0) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if anomaly_z_threshold <= 0:
            raise ValueError("anomaly_z_threshold must be positive")

        self.window_size = window_size
        self.anomaly_z_threshold = anomaly_z_threshold
        self._window: deque[float] = deque()
        self._running_sum = 0.0
        self._running_sum_sq = 0.0

    def add(self, value: float) -> WindowStats:
        """Add one new telemetry reading and return updated window stats."""

        previous_mean = self.mean
        previous_std = self.std

        if len(self._window) == self.window_size:
            old_value = self._window.popleft()
            self._running_sum -= old_value
            self._running_sum_sq -= old_value * old_value

        self._window.append(value)
        self._running_sum += value
        self._running_sum_sq += value * value

        if previous_std > 0:
            z_score = (value - previous_mean) / previous_std
        else:
            z_score = 0.0

        return WindowStats(
            count=len(self._window),
            mean=self.mean,
            std=self.std,
            z_score=z_score,
            is_anomaly=abs(z_score) >= self.anomaly_z_threshold,
        )

    @property
    def mean(self) -> float:
        if not self._window:
            return 0.0
        return self._running_sum / len(self._window)

    @property
    def std(self) -> float:
        if not self._window:
            return 0.0

        mean = self.mean
        variance = (self._running_sum_sq / len(self._window)) - (mean * mean)
        variance = max(0.0, variance)
        return math.sqrt(variance)