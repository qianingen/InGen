"""Pipeline integration entrypoint for Week 3 telemetry processing."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingen_pydev.pipeline.clean import clean_telemetry
from ingen_pydev.pipeline.features import add_telemetry_features
from ingen_pydev.pipeline.ingest import read_telemetry_csv
from ingen_pydev.pipeline.output import write_pipeline_outputs
from ingen_pydev.pipeline.synthetic import generate_synthetic_telemetry


@dataclass(frozen=True)
class PipelineResult:
    """Runtime metadata returned by a pipeline run."""

    rows_processed: int
    elapsed_seconds: float
    stage_times_seconds: dict[str, float]
    validation_summary: dict[str, Any]


def run_pipeline(
    input_csv: str | Path = "synthetic_data.csv",
    output_parquet: str | Path = "outputs/cleaned_features.parquet",
    summary_json: str | Path = "outputs/validation_summary.json",
    generate_if_missing: bool = True,
) -> PipelineResult:
    """Run ingest, clean, feature, and output stages end to end."""

    input_path = Path(input_csv)
    if generate_if_missing and not input_path.exists():
        generate_synthetic_telemetry(input_path)

    stage_times: dict[str, float] = {}
    start_total = time.perf_counter()

    start = time.perf_counter()
    df = read_telemetry_csv(input_path)
    stage_times["ingest"] = time.perf_counter() - start

    start = time.perf_counter()
    cleaned = clean_telemetry(df)
    stage_times["clean"] = time.perf_counter() - start

    start = time.perf_counter()
    featured = add_telemetry_features(cleaned)
    stage_times["features"] = time.perf_counter() - start

    start = time.perf_counter()
    summary = write_pipeline_outputs(featured, output_parquet, summary_json)
    stage_times["output"] = time.perf_counter() - start

    elapsed = time.perf_counter() - start_total
    return PipelineResult(
        rows_processed=len(featured),
        elapsed_seconds=elapsed,
        stage_times_seconds=stage_times,
        validation_summary=summary,
    )


def profile_pipeline(
    input_csv: str | Path = "synthetic_data.csv",
    output_parquet: str | Path = "outputs/cleaned_features.parquet",
    summary_json: str | Path = "outputs/validation_summary.json",
) -> str:
    """Profile the full pipeline and return a cProfile text report."""

    profiler = cProfile.Profile()
    profiler.enable()
    run_pipeline(input_csv, output_parquet, summary_json)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(15)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 3 telemetry pipeline.")
    parser.add_argument("--input", default="synthetic_data.csv")
    parser.add_argument("--parquet", default="outputs/cleaned_features.parquet")
    parser.add_argument("--summary", default="outputs/validation_summary.json")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(args.input, args.parquet, args.summary)
    print(f"Rows processed: {result.rows_processed}")
    print(f"Elapsed seconds: {result.elapsed_seconds:.4f}")
    print(f"Stage times: {result.stage_times_seconds}")
    print(f"Validation summary: {result.validation_summary}")

    if args.profile:
        print(profile_pipeline(args.input, args.parquet, args.summary))


if __name__ == "__main__":
    main()
