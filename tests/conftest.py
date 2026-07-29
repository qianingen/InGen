from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine

from ingen_pydev.db.database import create_sqlite_engine

SQLiteEngineFactory = Callable[[str | Path], Engine]


@pytest.fixture
def sqlite_engine_factory() -> Iterator[SQLiteEngineFactory]:
    """Create engines that are explicitly disposed after each database test."""

    engines: list[Engine] = []

    def factory(db_path: str | Path) -> Engine:
        engine = create_sqlite_engine(db_path)
        engines.append(engine)
        return engine

    try:
        yield factory
    finally:
        for engine in reversed(engines):
            engine.dispose()
