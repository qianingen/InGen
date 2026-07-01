## Status

### Week 1: Python Development Scaffold ✅

Week 1 established the base Python development environment:

- Python 3.11 package structure
- `pyproject.toml`
- `pytest`
- `black`
- `flake8`
- strict `mypy`
- `pre-commit`
- GitHub Actions CI
- first scaffold test

### Week 2: Algorithm Module ✅

Week 2 adds three bounded algorithm primitives under `ingen_pydev/algo/`:

- sliding-window telemetry preprocessing
- sparse 2D spatial index
- priority/deadline task scheduler

## Repository Structure

```text
ingen_pydev/
  __init__.py
  algo/
    __init__.py
    sliding_window.py
    spatial_index.py
    task_scheduler.py

tests/
  test_scaffold.py
  test_sliding_window.py
  test_spatial_index.py
  test_task_scheduler.py

weekly/
  Wk-01-DevLog.md
  Wk-02-DevLog.md

W02_Algo_Module.py
W02_Algo_Benchmarks.ipynb
pyproject.toml