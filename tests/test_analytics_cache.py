from __future__ import annotations

import math

import pytest

from ingen_pydev.analytics.cache import TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_initial_get_is_a_miss() -> None:
    cache = TTLCache[str](ttl_seconds=5, max_entries=2)

    assert cache.get("missing") is None
    assert cache.stats().misses == 1


def test_set_then_get_is_a_hit() -> None:
    cache = TTLCache[str](ttl_seconds=5, max_entries=2)

    cache.set("device", "summary")

    assert cache.get("device") == "summary"
    assert cache.stats().hits == 1


def test_expired_value_is_never_returned_without_sleep() -> None:
    clock = FakeClock()
    cache = TTLCache[str](ttl_seconds=5, max_entries=2, clock=clock)
    cache.set("device", "summary")

    clock.advance(5)

    assert cache.get("device") is None
    assert cache.stats().expirations == 1
    assert cache.stats().misses == 1


def test_clear_removes_values() -> None:
    cache = TTLCache[str](ttl_seconds=5, max_entries=2)
    cache.set("device", "summary")

    cache.clear()

    assert cache.get("device") is None


def test_eviction_is_deterministic_least_recently_used() -> None:
    cache = TTLCache[str](ttl_seconds=5, max_entries=2)
    cache.set("first", "one")
    cache.set("second", "two")
    assert cache.get("first") == "one"

    cache.set("third", "three")

    assert cache.get("second") is None
    assert cache.get("first") == "one"
    assert cache.get("third") == "three"
    assert cache.stats().evictions == 1


def test_stats_report_all_counters_and_hit_rate() -> None:
    clock = FakeClock()
    cache = TTLCache[str](ttl_seconds=5, max_entries=1, clock=clock)
    assert cache.get("missing") is None
    cache.set("first", "one")
    assert cache.get("first") == "one"
    clock.advance(5)
    cache.set("second", "two")
    cache.set("third", "three")

    stats = cache.stats()

    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.expirations == 1
    assert stats.evictions == 1
    assert stats.hit_rate == 0.5


def test_empty_cache_hit_rate_is_zero() -> None:
    cache = TTLCache[str](ttl_seconds=5, max_entries=1)

    assert cache.stats().hit_rate == 0.0


@pytest.mark.parametrize("ttl", [0, -1, math.inf, math.nan, True, "5"])
def test_invalid_ttl_raises_value_error(ttl: object) -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        TTLCache[str](ttl_seconds=ttl, max_entries=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_entries", [0, -1, True, 1.5, "2"])
def test_invalid_max_entries_raises_value_error(max_entries: object) -> None:
    with pytest.raises(ValueError, match="max_entries"):
        TTLCache[str](
            ttl_seconds=5,
            max_entries=max_entries,  # type: ignore[arg-type]
        )
