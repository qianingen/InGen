# Wk-01 DevLog

## What I Built

This week I built the Week 1 Python development scaffold for the internship project. The repository now contains an installable Python package structure, `pyproject.toml`, `pytest`, `black`, `flake8`, strict `mypy`, `pre-commit`, and a GitHub Actions CI workflow.

The first test is intentionally small. It checks that the `ingen_pydev` package imports correctly and that the Python version is 3.11 or newer. This matches the Week 1 goal: write the first test and prove the scaffold works before building the first real module.

## Design Decision

The main design decision was to keep Week 1 focused on infrastructure rather than feature code. Since later weeks add algorithms, telemetry pipelines, evaluation tooling, computer vision, and RAG modules, the repo needs a clean structure first. A green CI check gives a stable base for future work.

I also decided to frame the landscape brief as public-safe assumptions and proposed Python module ideas. The available public product material is high-level, so the brief should not pretend to describe internal InGen architecture. Instead, it should translate public platform narratives into reasonable Python developer surfaces.

## Issue Encountered

The main setup issue was Python tooling configuration. I had to fix the project configuration and pre-commit behavior before the checks passed. After resolving those issues, GitHub Actions showed a passing check on the repository.

## Most Important Physical AI Constraint

The most important constraint I would need to respect when writing Python code for a physical AI system is predictable latency under noisy real-world data streams.

Physical AI modules may process telemetry, video frames, spatial events, task queues, or retrieved context before downstream decisions are made. If a module is slow, inconsistent, or unclear about its inputs and outputs, the system may make decisions using stale or invalid information.

That means future modules should have clear schemas, type hints, unit tests, edge-case handling, runtime benchmarks, and Big-O comments for non-trivial functions. Correctness matters, but predictable behavior under realistic input size matters just as much.

## Next Step

The next step is Week 2: building the first real algorithm modules. The planned modules are a sliding-window telemetry aggregator, a spatial index for zone queries, and a priority task scheduler. Each should include tests, type hints, Big-O documentation, and benchmark comparisons against a naïve baseline.
