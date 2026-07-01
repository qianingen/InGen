# W01 Physical AI Python Developer Landscape

## 1. Scope and Confidentiality Boundary

This brief is a public-safe Python developer landscape. The provided product and platform materials were used only to understand general engineering direction. This document does not include internal source code, private architecture diagrams, screenshots, customer data, confidential performance metrics, proprietary API details, or direct excerpts from internal documents.

The goal is not to reproduce InGen’s internal implementation. The goal is to identify where a Python developer can build useful public-safe modules using synthetic data, public datasets, and open-source tooling.

The guiding question for Week 1 is:

> Given a physical-AI platform direction, what kinds of data would a Python developer likely handle, and what small public-safe modules could be built around that data?

## 2. Platform-Level Engineering Map

The clearest platform-level abstraction is a physical-AI compute cycle.

A robot-like system receives sensor data, calibrates noisy readings, estimates uncertainty, updates task state, selects or simulates an action, checks safety constraints, and records an audit log.

A public-safe simulator can represent this as:

```text
synthetic robot sensor packet
→ calibration / residual calculation
→ uncertainty scoring
→ task-state update
→ candidate action selection
→ safety gate
→ audit log
```

This does not require access to real robot data or internal models. It only requires a clean Python representation of the data flow.

| Stage               | Engineering role                            | Python data handled                                       | Public-safe module candidate |
| ------------------- | ------------------------------------------- | --------------------------------------------------------- | ---------------------------- |
| Sensor ingestion    | Structure robot-like input data             | timestamp, sensor readings, battery, task ID, event flags | `schemas.py`                 |
| Calibration         | Estimate and correct noisy readings         | raw value, expected value, residual, corrected value      | `calibration.py`             |
| Uncertainty scoring | Decide whether the state is reliable enough | residuals, stale time, confidence, threshold flags        | `uncertainty.py`             |
| Task state          | Track the current task or subtask           | task ID, current phase, preconditions, status             | `task_state.py`              |
| Action selection    | Select or simulate next action              | state, task, uncertainty, candidate action                | `policy_stub.py`             |
| Safety gate         | Accept, reject, or escalate action          | action, rule checks, rejection reason                     | `safety_gate.py`             |
| Audit log           | Record the decision path                    | input state, uncertainty, action, safety result           | `audit_log.py`               |

## 3. PIC Model Class to Python Module Mapping

The following table converts the PIC-style model classes into public-safe Python module candidates. This is an engineering abstraction, not a claim about internal implementation.

| PIC class | Engineering meaning                                         | Python module candidate           |
| --------- | ----------------------------------------------------------- | --------------------------------- |
| HTD-IRL   | Break a high-level task into smaller executable task states | `task_graph.py` / `task_state.py` |
| AMDC      | Calibrate or correct noisy sensor readings                  | `calibration.py`                  |
| STUM      | Estimate uncertainty from sensor noise and state staleness  | `uncertainty.py`                  |
| GRPO      | Select actions or skills from state/task/reward context     | `policy_stub.py`                  |
| SEOM      | Reject unsafe actions and record safety reasons             | `safety_gate.py`                  |
| CRL-MRS   | Coordinate multiple robot agents or fleet tasks             | `fleet_coordination.py`           |

For Week 2+, the implementation should not attempt to reproduce full internal AI models. A better public-safe approach is to simulate the engineering contract: structured state in, uncertainty-aware decision out, safety checked, audit logged.

## 4. Product Developer Surface

The product files suggest different Python developer surfaces. For Week 1, the goal is not to deeply implement each product. The goal is to identify what type of data each product direction implies.

| Product           | Product role                             | Likely Python data handled                                                                   | Public-safe module candidate                                  | Priority   |
| ----------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------- |
| Origami / PIC 2.0 | Physical-AI platform layer               | sensor state, task state, uncertainty score, candidate action, safety result, audit log      | mini PIC compute-cycle simulator                              | Highest    |
| Aido Rover        | Outdoor inspection and security robot    | telemetry, LiDAR-like readings, location, battery, environmental readings, inspection events | sensor pipeline + uncertainty + anomaly/audit log             | High       |
| Aido Humanoid     | Bipedal humanoid / manipulation platform | task queue, actuator state, grasp/action plans, safety events, failure logs                  | task/action safety simulator or manipulation failure analyzer | High       |
| Sentinel Prime AI | Security intelligence platform           | video events, detection confidence, false alerts, evidence clips, audit logs                 | detection-event evaluator with uncertainty and audit trail    | Medium     |
| Fari              | Eldercare / healthcare companion         | session logs, health events, fall/medication/emotion records, escalation decisions           | care-event safety/evaluation harness                          | Medium-low |
| Senpai            | Education companion robot                | curriculum data, student interactions, learning progress, tutoring responses                 | tutoring-response evaluator                                   | Medium-low |
| Carry & Go        | Indoor delivery robot                    | delivery tasks, route events, building/elevator state, payload, battery, timing records      | delivery-task scheduler                                       | Low-medium |

## 5. Real-Time / Physical AI Complexity Primer

Physical-AI software is different from ordinary backend or notebook code because it may sit close to sensor streams, action decisions, and safety constraints.

The first constraint is predictable latency. Sensor validation, residual calculation, uncertainty scoring, and safety checking should be bounded, testable, and fast. Slow batch analytics, visualization, and report generation should be separated from the fast path.

The second constraint is uncertainty-aware behavior. A physical-AI system should not only output an action. It should know when the input state is unreliable, when to delay, and when to escalate to a safer fallback.

The third constraint is safety gating. A robot-like system should not execute an action only because a policy selected it. It should check the action against safety constraints before execution.

The fourth constraint is auditability. Every action-like decision should leave a record: observed state, uncertainty score, selected action, safety result, and reason for acceptance or rejection.

## 6. Recommended Direction

The recommended next direction is a public-safe mini PIC compute-cycle simulator anchored on Rover and Humanoid-style data.

The simulator would take synthetic robot sensor packets as input and produce:

```text
calibrated state
uncertainty score
candidate action
safety result
audit log
```

A possible module layout:

```text
ingen_pydev/
  pic_cycle/
    schemas.py
    calibration.py
    uncertainty.py
    task_state.py
    policy_stub.py
    safety_gate.py
    audit_log.py
```

This direction is stronger than disconnected algorithm exercises because it creates one runnable system with clear input, output, tests, benchmarks, and demo value. It also stays aligned with the platform context while avoiding confidential implementation details.