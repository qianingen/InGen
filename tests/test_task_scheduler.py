from ingen_pydev.algo.task_scheduler import Task, TaskScheduler


def test_pop_next_returns_highest_priority_task() -> None:
    scheduler = TaskScheduler()

    low = Task(
        task_id="routine_patrol",
        priority=10,
        deadline=100.0,
        created_at=1.0,
    )
    high = Task(
        task_id="gas_alert",
        priority=90,
        deadline=100.0,
        created_at=2.0,
    )

    scheduler.add_task(low)
    scheduler.add_task(high)

    task = scheduler.pop_next(now=0.0)

    assert task is not None
    assert task.task_id == "gas_alert"


def test_deadline_breaks_priority_tie() -> None:
    scheduler = TaskScheduler()

    later = Task(
        task_id="later_deadline",
        priority=50,
        deadline=100.0,
        created_at=1.0,
    )
    earlier = Task(
        task_id="earlier_deadline",
        priority=50,
        deadline=50.0,
        created_at=2.0,
    )

    scheduler.add_task(later)
    scheduler.add_task(earlier)

    task = scheduler.pop_next(now=0.0)

    assert task is not None
    assert task.task_id == "earlier_deadline"


def test_created_at_breaks_priority_and_deadline_tie() -> None:
    scheduler = TaskScheduler()

    newer = Task(
        task_id="newer",
        priority=50,
        deadline=100.0,
        created_at=2.0,
    )
    older = Task(
        task_id="older",
        priority=50,
        deadline=100.0,
        created_at=1.0,
    )

    scheduler.add_task(newer)
    scheduler.add_task(older)

    task = scheduler.pop_next(now=0.0)

    assert task is not None
    assert task.task_id == "older"


def test_expired_tasks_are_skipped() -> None:
    scheduler = TaskScheduler()

    expired = Task(
        task_id="expired",
        priority=100,
        deadline=10.0,
        created_at=1.0,
    )
    valid = Task(
        task_id="valid",
        priority=50,
        deadline=100.0,
        created_at=2.0,
    )

    scheduler.add_task(expired)
    scheduler.add_task(valid)

    task = scheduler.pop_next(now=20.0)

    assert task is not None
    assert task.task_id == "valid"


def test_pop_next_returns_none_when_no_valid_tasks_exist() -> None:
    scheduler = TaskScheduler()

    expired = Task(
        task_id="expired",
        priority=100,
        deadline=10.0,
        created_at=1.0,
    )

    scheduler.add_task(expired)

    task = scheduler.pop_next(now=20.0)

    assert task is None


def test_pop_next_returns_none_when_empty() -> None:
    scheduler = TaskScheduler()

    task = scheduler.pop_next(now=0.0)

    assert task is None


def test_payload_is_preserved() -> None:
    scheduler = TaskScheduler()

    task = Task(
        task_id="inspect_zone_a",
        priority=80,
        deadline=100.0,
        created_at=1.0,
        payload={"target": [12.5, 8.0], "robot_id": "rover_01"},
    )

    scheduler.add_task(task)
    result = scheduler.pop_next(now=0.0)

    assert result is not None
    assert result.payload["target"] == [12.5, 8.0]
    assert result.payload["robot_id"] == "rover_01"


def test_len_tracks_heap_size_for_pending_tasks() -> None:
    scheduler = TaskScheduler()

    scheduler.add_task(
        Task(
            task_id="task_1",
            priority=10,
            deadline=100.0,
            created_at=1.0,
        )
    )
    scheduler.add_task(
        Task(
            task_id="task_2",
            priority=20,
            deadline=100.0,
            created_at=2.0,
        )
    )

    assert len(scheduler) == 2

    scheduler.pop_next(now=0.0)

    assert len(scheduler) == 1
