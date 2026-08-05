"""Small thread-safe bounded TTL cache for analytics responses."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class CacheStats:
    """Cumulative cache counters and their derived hit rate."""

    hits: int
    misses: int
    expirations: int
    evictions: int
    hit_rate: float


@dataclass(frozen=True)
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """A deterministic least-recently-used cache with per-entry TTLs."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a finite positive number")
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries <= 0
        ):
            raise ValueError("max_entries must be a positive integer")

        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = max_entries
        self._clock = clock or time.monotonic
        self._entries: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._evictions = 0

    def get(self, key: str) -> T | None:
        """Return a live cached value and record a hit or miss."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.expires_at <= self._clock():
                del self._entries[key]
                self._expirations += 1
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        """Insert or refresh a value, evicting the least-recently-used entry."""

        with self._lock:
            now = self._clock()
            self._remove_expired(now)
            if key in self._entries:
                del self._entries[key]
            elif len(self._entries) >= self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

            self._entries[key] = _CacheEntry(
                value=value,
                expires_at=now + self._ttl_seconds,
            )

    def clear(self) -> None:
        """Remove every cached value while preserving cumulative statistics."""

        with self._lock:
            self._entries.clear()

    def stats(self) -> CacheStats:
        """Return an atomic snapshot of cumulative cache statistics."""

        with self._lock:
            requests = self._hits + self._misses
            hit_rate = self._hits / requests if requests else 0.0
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                expirations=self._expirations,
                evictions=self._evictions,
                hit_rate=hit_rate,
            )

    def _remove_expired(self, now: float) -> None:
        expired_keys = [
            key for key, entry in self._entries.items() if entry.expires_at <= now
        ]
        for key in expired_keys:
            del self._entries[key]
            self._expirations += 1
