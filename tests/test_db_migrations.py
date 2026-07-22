from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def _run_alembic(
    project_root: Path,
    database_url: str,
    *arguments: str,
) -> None:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url

    subprocess.run(
        [sys.executable, "-m", "alembic", *arguments],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_alembic_upgrade_downgrade_and_reupgrade(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "migration_test.db"
    database_url = f"sqlite:///{db_path.as_posix()}"

    _run_alembic(project_root, database_url, "upgrade", "head")

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }

    assert {
        "device",
        "session",
        "sensor_reading",
        "alert",
        "alembic_version",
    }.issubset(tables)

    _run_alembic(project_root, database_url, "downgrade", "base")
    _run_alembic(project_root, database_url, "upgrade", "head")
    _run_alembic(project_root, database_url, "upgrade", "head")