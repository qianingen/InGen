"""Database helpers for local SQLite development."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ingen_pydev.db.models import Base


def create_sqlite_engine(db_path: str | Path = "w04_telemetry.db") -> Engine:
    """Create a SQLite engine with foreign-key enforcement enabled."""

    if str(db_path) == ":memory:":
        database_url = "sqlite+pysqlite:///:memory:"
    else:
        database_url = f"sqlite+pysqlite:///{Path(db_path)}"

    engine = create_engine(database_url, future=True)

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(
        dbapi_connection: Any,
        _connection_record: Any,
    ) -> None:
        if isinstance(dbapi_connection, sqlite3.Connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_all_tables(engine: Engine) -> None:
    """Create all schema tables."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a SQLAlchemy session factory."""
    return sessionmaker(bind=engine, future=True)