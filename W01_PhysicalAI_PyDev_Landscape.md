# W01 Physical AI Python Developer Landscape

## Scope Note

This brief is based on public-facing product descriptions, open-source tooling, and public-safe assumptions only. It does not describe InGen internal architecture, private source code, proprietary APIs, customer data, or confidential technical documentation.

Where this brief discusses technical structure, it is framed as a reasonable design hypothesis or proposed Python module candidate. The goal is not to reconstruct InGen’s actual implementation, but to translate public product narratives into concrete Python development surfaces that could be explored with public datasets, synthetic data, and open-source libraries.

## 1. Public Product Narrative to Python Developer Surface

The Week 1 development question is: given a high-level Physical AI product narrative, what kinds of data would a Python engineer likely handle, and what module would be useful to build first?

For a Physical AI system, the useful Python surface is usually not the product slogan itself. The useful surface is the data moving through the system: telemetry records, video frames, task queues, retrieved text chunks, model outputs, evaluation logs, and benchmark results. A strong Python module should make one of those data flows easier to ingest, validate, process, test, or evaluate.

## 2. Platform Developer Briefs

### Fari

Based on public descriptions, Fari can be treated as an eldercare companion platform with a likely need for conversational memory, domain knowledge retrieval, and response evaluation. A Python engineer would likely handle text queries, retrieved context chunks, embeddings, conversation turns, safety/evaluation labels, generated responses, and token/cost logs.

A proposed Python module would be a public-safe RAG pipeline for eldercare-style question answering: document chunking, embedding, retrieval, prompt assembly, response generation, and faithfulness/cost reporting. This module would not use private user data; it would use public or synthetic eldercare-style documents and mock conversations.

### Senpai

Based on public descriptions, Senpai can be treated as an educational robot or tutoring assistant. A Python engineer would likely handle curriculum documents, student questions, expected answer ranges, tutoring scenarios, response-quality scores, and evaluation records.

A proposed Python module would be a tutoring evaluation harness that loads public/synthetic learning scenarios, generates or receives answers, and scores them for correctness, helpfulness, grounding, and clarity. This connects naturally to YAML scenario loading, RAG retrieval, LLM judge outputs, and cost-aware evaluation.

### Sentinel Prime AI

Based on public descriptions, Sentinel Prime AI can be treated as a security/computer-vision platform. A Python engineer would likely handle video frames, image arrays, timestamps, keyframe indices, motion masks, alert events, bounding regions, and latency measurements.

A proposed Python module would be an OpenCV utility library for keyframe extraction, motion-change detection, and image preprocessing on public outdoor/security datasets. The module should report both accuracy-style metrics and latency per frame, because security vision systems need useful detections without excessive processing delay.

### Aido Rover

Based on public descriptions, Aido Rover can be treated as an outdoor mobile robotics platform. A Python engineer would likely handle timestamped telemetry streams such as GPS-like coordinates, distance readings, battery state, wheel or drive status, ambient temperature, patrol events, and anomaly flags.

A proposed Python module would be a telemetry ingestion and validation pipeline that reads synthetic rover-style data, validates schema, handles missing or noisy readings, computes rolling features, and exports clean data for downstream anomaly detection or benchmarking. This is a good primary development anchor because telemetry data can be simulated safely without internal robot access.

### Aido Humanoid

Based on public descriptions, Aido Humanoid can be treated as a bipedal or humanoid robotics research platform where sequencing and prioritizing actions matters. A Python engineer would likely handle task IDs, priorities, deadlines, worker or actuator availability, action queues, scheduling decisions, and state-transition logs.

A proposed Python module would be a priority task scheduler that selects the highest-priority non-expired task using a heap-based data structure. This module is useful as a public-safe algorithm exercise because it can be tested without hardware while still connecting to real robotics constraints such as bounded scheduling time and predictable behavior under load.

## 3. Origami / PIC 2.0 Model-Class Mapping

The following mapping is conceptual. It identifies Python processing patterns and open-source libraries that could support public-safe experiments around PIC 2.0-style model classes. It is not a claim about InGen’s internal implementation.

| PIC 2.0 Class | Python Processing Pattern                               | Possible Open-Source Library                     | Proposed Module Candidate                                                                                |
| ------------- | ------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| GRPO          | Reinforcement learning / policy optimization            | PyTorch                                          | `pic/grpo_policy.py` for policy experiment scaffolding, reward logging, and simple policy-update tests   |
| STUM          | Streaming temporal update over sensor-like data         | `collections.deque`, `river`, or pandas          | `pic/stum_stream.py` for rolling state updates, anomaly flags, and time-window features                  |
| SEOM          | Spatial environment object modeling and spatial queries | `shapely`, `scipy.spatial`, or custom grid index | `pic/seom_spatial.py` for zones, points, radius queries, and patrol-space utilities                      |
| AMDC          | Action/mission decision control                         | `networkx`, `heapq`, or `scipy.optimize`         | `pic/amdc_planner.py` for decision graphs, constraints, and mission-state transitions                    |
| HTD-IRL       | Trajectory learning / inverse reinforcement learning    | PyTorch                                          | `pic/htd_irl.py` for public trajectory datasets, reward-shaping experiments, and demonstration analysis  |
| CRL-MRS       | Multi-agent coordination and message passing            | `asyncio`, ROS 2 `rclpy`, or `pydantic` messages | `pic/crl_mrs.py` for robot-agent message schemas, async event passing, and coordination-state simulation |

## 4. Complexity Primer for Real-Time Physical AI Systems

Physical AI software has different constraints from normal web or notebook code because it may sit near sensor streams and decision loops. A slow Python function is not only inconvenient; it can cause downstream modules to work with stale data. For this reason, the first engineering principle is predictable latency.

The second principle is bounded complexity. If a module processes a stream of 100,000 telemetry records, an accidental O(n²) implementation can become unusable as input size grows. Sliding-window algorithms, spatial indexes, priority queues, and vectorized operations matter because they keep runtime behavior predictable.

The third principle is robustness to messy data. Physical systems receive missing, noisy, delayed, or saturated sensor readings. A useful Python module should validate schemas, handle missing values explicitly, flag anomalies, and include tests for edge cases instead of assuming perfect input.

The fourth principle is reproducibility. Since this project is public-safe and open-source, every result should be reproducible from a clean install and a one-command test run. This makes the work reviewable and prevents hidden local-environment dependencies from becoming part of the project.

## 5. Week 1 Engineering Takeaways

The Week 1 scaffold establishes the engineering frame before real modules are built. The repository now has package structure, dependency management, formatting, linting, strict type checking, pre-commit hooks, tests, and GitHub Actions CI.

The main design decision is to separate public product narrative from implementation claims. The platform descriptions provide useful anchors, but the proposed modules are framed as public-safe developer candidates rather than descriptions of InGen’s internal architecture.

The next technical step is the Week 2 algorithm library: sliding-window telemetry aggregation, spatial indexing, and priority task scheduling. These modules should be type-hinted, tested, benchmarked against naïve baselines, and documented with Big-O complexity notes.

## 6. References to Add

Add exact links to the public sources actually used before final submission:

* Public InGen product pages for Fari, Senpai, Sentinel Prime AI, Aido Rover/Humanoid, and Origami/PIC 2.0
* Python engineering style reference such as Google Python Style Guide or Hypermodern Python
* ROS 2 Python or robotics middleware overview
* Public robotics/sensor-stream reference on data rates, latency, or real-time constraints
* Public paper or article on embedded/real-time algorithm design
* Public datasets or dataset organizations considered for later weeks, such as robotics, outdoor video, or security-camera datasets
