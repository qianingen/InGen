import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    priority: int
    deadline: float
    created_at: float
    payload: dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    def __init__(self) -> None:
        self._heap: list[tuple[int, float, float, str, Task]] = []

    def add_task(self, task: Task) -> None:
        heapq.heappush(
            self._heap,
            (-task.priority, task.deadline, task.created_at, task.task_id, task),
        )

    def pop_next(self, now: float) -> Task | None:
        while self._heap:
            _, deadline, _, _, task = heapq.heappop(self._heap)

            if deadline >= now:
                return task

            # expired task: skip it

        return None

    def __len__(self) -> int:
        return len(self._heap)
