from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingen_pydev.pipeline.ingest import REQUIRED_COLUMNS, read_telemetry_csv


def test_read_valid_telemetry_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "valid.csv"
    df = pd.DataFrame({column: [1.0] for column in REQUIRED_COLUMNS})
    df["timestamp_ms"] = [1_735_689_600_000]
    df.to_csv(csv_path, index=False)

    result = read_telemetry_csv(csv_path)

    assert list(result.columns) == REQUIRED_COLUMNS
    assert len(result) == 1


def test_missing_required_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    columns = [column for column in REQUIRED_COLUMNS if column != "battery_soc"]
    df = pd.DataFrame({column: [1.0] for column in columns})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        read_telemetry_csv(csv_path)
