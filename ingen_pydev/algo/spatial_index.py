from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpatialItem:
    item_id: str
    x: float
    y: float
    kind: str | None = None


class SpatialIndex2D:
    """Sparse grid spatial index for nearby-entity lookup.

    This is not a full map, SLAM system, or navigation mesh.
    It only accelerates radius queries over known point-like entities such as
    robots, alerts, charging stations, task locations, and tagged obstacles.
    """

    def __init__(self, cell_size: float) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")

        self.cell_size = cell_size
        self._buckets: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._items: dict[str, SpatialItem] = {}

    def insert(
        self,
        item_id: str,
        x: float,
        y: float,
        kind: str | None = None,
    ) -> None:
        if item_id in self._items:
            raise ValueError(f"item_id already exists: {item_id}")

        item = SpatialItem(item_id=item_id, x=x, y=y, kind=kind)
        cell = self._cell_for(x, y)

        self._items[item_id] = item
        self._buckets[cell].add(item_id)

    def remove(self, item_id: str) -> None:
        item = self._items.pop(item_id)
        cell = self._cell_for(item.x, item.y)

        self._buckets[cell].remove(item_id)

        if not self._buckets[cell]:
            del self._buckets[cell]

    def update(self, item_id: str, x: float, y: float) -> None:
        old_item = self._items[item_id]
        old_cell = self._cell_for(old_item.x, old_item.y)
        new_cell = self._cell_for(x, y)

        if old_cell != new_cell:
            self._buckets[old_cell].remove(item_id)
            if not self._buckets[old_cell]:
                del self._buckets[old_cell]
            self._buckets[new_cell].add(item_id)

        self._items[item_id] = SpatialItem(
            item_id=item_id,
            x=x,
            y=y,
            kind=old_item.kind,
        )

    def query_radius(
        self,
        x: float,
        y: float,
        radius: float,
        kind: str | None = None,
    ) -> list[SpatialItem]:
        if radius < 0:
            raise ValueError("radius must be non-negative")

        radius_sq = radius * radius
        results: list[SpatialItem] = []

        min_cx = math.floor((x - radius) / self.cell_size)
        max_cx = math.floor((x + radius) / self.cell_size)
        min_cy = math.floor((y - radius) / self.cell_size)
        max_cy = math.floor((y + radius) / self.cell_size)

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                for item_id in self._buckets.get((cx, cy), set()):
                    item = self._items[item_id]

                    if kind is not None and item.kind != kind:
                        continue

                    dx = item.x - x
                    dy = item.y - y

                    if dx * dx + dy * dy <= radius_sq:
                        results.append(item)

        return results

    def __len__(self) -> int:
        return len(self._items)

    def _cell_for(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor(x / self.cell_size),
            math.floor(y / self.cell_size),
        )
