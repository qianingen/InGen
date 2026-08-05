from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ingen_pydev.db.database import create_sqlite_engine, make_session_factory
from ingen_pydev.db.loader import load_week3_output

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


@pytest.fixture(scope="session")
def seeded_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one file-backed source database from the tracked profile output."""

    project_root = Path(__file__).resolve().parents[1]
    database_dir = tmp_path_factory.mktemp("analytics-source-db")
    database_path = database_dir / "telemetry.db"
    load_week3_output(
        parquet_path=project_root / "outputs" / "profile_cleaned_features.parquet",
        summary_json_path=(
            project_root / "outputs" / "profile_validation_summary.json"
        ),
        db_path=database_path,
    )
    return database_path


@pytest.fixture(scope="session")
def seeded_session_factory(
    seeded_db_path: Path,
) -> Iterator[sessionmaker[Session]]:
    """Provide read-only test sessions and dispose their engine at teardown."""

    engine = create_sqlite_engine(seeded_db_path)
    try:
        yield make_session_factory(engine)
    finally:
        engine.dispose()


@pytest.fixture
def mutable_seeded_session_factory(
    tmp_path: Path,
    seeded_db_path: Path,
) -> Iterator[sessionmaker[Session]]:
    """Copy the shared source database before tests perform mutations."""

    database_path = tmp_path / "telemetry.db"
    shutil.copy2(seeded_db_path, database_path)
    engine = create_sqlite_engine(database_path)
    try:
        yield make_session_factory(engine)
    finally:
        engine.dispose()
