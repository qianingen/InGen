"""Runnable entrypoint for the Week 6 telemetry analytics API."""

from __future__ import annotations

import os
from pathlib import Path

from ingen_pydev.analytics.app import create_app
from ingen_pydev.analytics.cache import DEFAULT_TTL_SECONDS, TTLCache
from ingen_pydev.analytics.models import DeviceSummaryResponse
from ingen_pydev.db.database import create_sqlite_engine, make_session_factory

DATABASE_PATH = Path(os.getenv("INGEN_ANALYTICS_DB_PATH", "w04_telemetry.db"))

engine = create_sqlite_engine(DATABASE_PATH)
session_factory = make_session_factory(engine)
summary_cache = TTLCache[DeviceSummaryResponse](
    ttl_seconds=DEFAULT_TTL_SECONDS,
    max_entries=1_024,
)
app = create_app(
    session_factory,
    summary_cache,
    close_resources=engine.dispose,
)


def main() -> None:
    """Serve the environment-configured analytics application."""

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
