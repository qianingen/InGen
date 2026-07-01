# Wk-02 DevLog

## Week 2 Focus

Week 2 moved from landscape analysis into implementation. The goal was to build a small physical-AI algorithm library with three bounded, testable Python utilities:

1. sliding-window telemetry preprocessing
2. sparse 2D spatial entity lookup
3. priority/deadline task scheduling

These modules are not full robot intelligence systems. They are middleware primitives that could support a future robot decision loop.

## What I Completed

I implemented three algorithm modules under `ingen_pydev/algo/`:

```text
ingen_pydev/algo/
  sliding_window.py
  spatial_index.py
  task_scheduler.py
```

I also added a Week 2 module entry point:

```text
W02_Algo_Module.py
```

The entry point re-exports the main Week 2 classes so the deliverable has one simple import surface.

## Module 1: Sliding Window Telemetry

The sliding-window module processes a numeric telemetry stream and returns rolling statistics.

It maintains:

```text
deque window
running sum
running sum of squares
```

This allows each new reading to update mean, standard deviation, z-score, and anomaly status without rescanning the full window.

The module should be understood as telemetry preprocessing, not a full AMDC or STUM model. It produces lightweight health/anomaly signals that downstream modules could use.

Complexity:

```text
Naive recomputation: O(n·k)
Optimized rolling update: O(n) total, O(1) per reading
```

## Module 2: Sparse 2D Spatial Index

The spatial-index module provides nearby-entity lookup over point-like entities in a shared local 2D coordinate frame.

It assumes upstream systems already provide coordinates. It does not perform localization, SLAM, GPS conversion, 3D mapping, or navigation.

The module stores sparse entities such as:

```text
robots
alerts
task locations
charging stations
tagged obstacles
```

It uses grid buckets to avoid scanning every item for every radius query.

Complexity:

```text
Brute force query: O(n)
Sparse grid query: O(c + m)
```

where `c` is the number of candidate items in checked cells and `m` is the number of returned results.

## Module 3: Priority Task Scheduler

The task scheduler manages tasks that already have priority and deadline values assigned by upstream logic.

It does not understand language and does not decompose high-level tasks. It simply chooses the next valid task using deterministic rules:

```text
1. higher priority first
2. earlier deadline breaks priority ties
3. earlier created_at breaks deadline ties
4. expired tasks are skipped
```

The scheduler uses a heap instead of sorting all pending tasks every time.

Complexity:

```text
Naive sort-every-pop: O(n log n) per pop
Heap scheduler: O(log n) per add/pop
```

## Testing

I added pytest coverage for the three modules:

```text
tests/test_sliding_window.py
tests/test_spatial_index.py
tests/test_task_scheduler.py
```

The tests cover normal behavior, edge cases, invalid inputs, updates/removals, expiration behavior, and priority/deadline ordering.

## Key Technical Insight

The most important Week 2 insight is that physical-AI systems do not only need large models. They also need simple, reliable, deterministic middleware around streaming data, spatial lookup, and task dispatch.

These modules are intentionally small, but they are useful because they are:

```text
bounded
typed
testable
benchmarkable
low-latency
easy to reason about
```

## Most Valuable Complexity Reduction

The strongest complexity reduction was the sliding-window telemetry module.

The naive version recomputes statistics by scanning the full window after every new reading. The optimized version keeps running sums and updates the window in constant time.

This changes the total processing cost from `O(n·k)` to `O(n)`, which matters for real-time telemetry streams.

## Confidentiality Boundary

This Week 2 implementation does not reproduce internal robot models, internal platform code, private APIs, customer data, or confidential system behavior.

The modules are public-safe algorithmic primitives built with synthetic/public-safe assumptions.

## Next Step

The next useful step is to connect these primitives into a small simulated robot decision loop:

```text
telemetry stream
→ anomaly/health signal
→ nearby entity query
→ task dispatch
→ high-level action decision
→ audit log
```

That would turn the three separate utilities into a coherent mini physical-AI compute-cycle simulator.
