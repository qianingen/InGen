# Wk-03 DevLog

## Week 3 Focus

Week 3 implements a synthetic telemetry ingest, clean, feature, and output pipeline for an Aido Rover-style sensor stream. The project remains public-safe: the dataset is synthetic and the code does not use private robot logs, internal APIs, or confidential production data.

## What I Built

I added a modular pipeline under `ingen_pydev/pipeline/`:

```text
ingen_pydev/pipeline/
  synthetic.py
  ingest.py
  clean.py
  features.py
  output.py
  run_pipeline.py
```

The root-level `W03_Telemetry_Pipeline.py` re-exports the main pipeline functions as a Week 3 deliverable entry point.

## Dataset Generation

`synthetic.py` generates `synthetic_data.csv` with 10,000 timestamped rows and the following columns:

```text
timestamp_ms
lat
lon
lidar_distance_m
battery_soc
wheel_torque_fl
wheel_torque_fr
wheel_torque_rl
wheel_torque_rr
ambient_temp_c
```

The generator injects GPS dropout, LiDAR saturation, battery sudden drops, and random sensor noise spikes to create a realistic dirty-input problem for the pipeline.

## Pipeline Stages

### Ingest

`ingest.py` reads the CSV and validates the required schema. Missing required columns are treated as fatal schema errors.

### Clean

`clean.py` handles GPS dropout, LiDAR saturation, and sensor noise spikes. GPS dropout is forward-filled for up to 3 consecutive missing values. The 4th consecutive dropout and beyond are flagged as long dropout. LiDAR saturation values are replaced and interpolated. Noise spikes are flagged using the Week 2 sliding-window anomaly detector.

### Features

`features.py` computes battery rolling mean and rolling standard deviation, cumulative GPS distance, wheel imbalance score, LiDAR saturation rate, and composite health score.

### Output

`output.py` writes the cleaned and feature-enriched table to Parquet and creates a validation summary JSON with rows processed, rows flagged, and fields with anomalies detected.

## Tests

I added pytest coverage for all four stages:

```text
tests/test_pipeline_ingest.py
tests/test_pipeline_clean.py
tests/test_pipeline_features.py
tests/test_pipeline_output.py
```

Key test cases include exactly 3 consecutive GPS dropouts, the 4th consecutive GPS dropout flag, battery rolling statistics compared against pandas rolling ground truth, validation summary row counting, and Parquet round-trip output.

## cProfile Note

The full pipeline is designed to process 10,000 rows in under 2 seconds on a normal local development environment. Profiling with `cProfile` should be run with:

```bash
python -m ingen_pydev.pipeline.run_pipeline --profile
```

In this implementation, the slowest stage is expected to be the cleaning stage because it applies the Week 2 sliding-window detector row-by-row across several sensor columns. Feature extraction is mostly vectorized with pandas and NumPy, so it should remain fast even with cumulative GPS distance calculation.

## Boundary

This is not a robot intelligence system. It is a data pipeline that prepares noisy telemetry for downstream ML, alerting, or monitoring.
