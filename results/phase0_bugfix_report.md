# Phase 0: Bug-Fix and Baseline Verification Report

This report documents the identification, diagnosis, code resolution, and verification for the ten defects identified in the Phase 0 calibration of the MAPF conflict prioritization codebase.

---

### Bug 1: `validate_proxy.py` Never Computed Spearman Correlation
- **Diagnosis**: In `validate_proxy.py`, the Spearman correlation was guarded by `if len(set(q_rollout_costs)) > 1:`. Because all rollout costs hit the 300-expansion budget cap and received identical penalty costs (450), the set cardinality was always 1. The code bypassed the correlation calculation entirely, leaving `map_spearmans` empty. The report stated "Spearman = 0.0000", but no correlation had ever been evaluated.
- **Fix**: Updated `validate_proxy.py` to check both `len(set(q_proxy_scores)) > 1` and `len(set(q_rollout_costs)) > 1`. If zero variance is observed in either signal, the query correlation is explicitly recorded as `float('nan')` ("undefined"), and the evaluation summary reports the fraction of zero-variance queries rather than falsifying a zero correlation.
- **Verification**: On zero-variance queries, `NaN (undefined — zero variance in signals)` is reported, preventing silent truncation or misleading zero-correlation claims.

---

### Bug 2: `proxy_labeler.py` Ignored `max_candidates_per_query`
- **Diagnosis**: The `max_candidates_per_query` parameter was absent from the function signature of `run_proxy_labeling` and was never referenced in the candidate-selection loop. Consequently, every conflict detected at a constraint tree (CT) node was evaluated as a candidate (~15 per node), violating the design budget of 3–5 candidates per query.
- **Fix**: Added `max_candidates_per_query: int = 5` to `run_proxy_labeling`. Before feature extraction and labeling, candidates within the active regime are prioritized by their estimated cost increase (`delta_f_res` / cardinal tier) and truncated to the top-$N$ candidates. The same cap was built into `fast_dataset_builder.py`.
- **Verification**: Output CSV candidate group sizes are strictly bounded by `max_candidates_per_query`, stabilizing query generation time.

---

### Bug 3: Synthetic "MVC Delta" Was a Binary Indicator, Not a Graph Metric
- **Diagnosis**: In both `proxy_labeler.py` and `validate_proxy.py`, the code computed `p_mvc_delta = 1.0 if c_cls == 'cardinal' else 0.0`. This is merely an indicator of conflict classification rather than the change in Minimum Vertex Cover (MVC) size resulting from resolving the conflict.
- **Fix**: Implemented `_compute_real_mvc_delta(conflict, c_cls, classified_conflicts, parent_mvc)` in `proxy_labeler.py`. It removes the resolved conflict edge from the cardinal conflict graph $G_{CF}$ and invokes `ConflictGraph(child_edges).compute_mvc()` to compute the true structural delta $\text{MVC}(G_{CF}^{\text{child}}) - \text{MVC}(G_{CF}^{\text{parent}})$.
- **Verification**: Inspected computed `p_mvc_delta` values; they range across $\{ -1.0, 0.0, +1.0 \}$ depending on graph connectivity rather than always evaluating to 1.0 or 0.0.

---

### Bug 4: `validate_proxy.py` Truncated After 1 Query Per Map
- **Diagnosis**: An unconditional `break` statement at the end of the query processing loop in `validate_proxy.py` terminated search after the very first multi-conflict node of each scenario. Across 5 scenarios, this yielded at most 5 queries per map instead of the intended target of 15 queries.
- **Fix**: Removed the premature unconditional `break` statement. The loop now continues popping nodes from `open_list` until `map_queries >= target_queries_per_map` is reached.
- **Verification**: Running `validate_proxy.py` now harvests the full target number of queries per map across multiple search depths.

---

### Bug 5: `Paris_1_256` Returned `NO_ROOT_PATH` on All Scenarios
- **Diagnosis**: On the large 256×256 `Paris_1_256.map` (47,240 passable cells), Space-Time A* without a precomputed reverse-BFS 2D distance heuristic expanded tens of thousands of space-time states due to Manhattan distance local minima, stalling and terminating with `NO_ROOT_PATH`. Furthermore, MovingAI scenario files were missing from `data/scen-random/`. Diagnostic test `mapf/diagnose_paris.py` proved that when exact reverse-BFS heuristic tables (`compute_bfs_h_table`) are used, single-agent A* finds optimal 177-step paths on Paris in under 0.001 seconds.
- **Resolution**: While the low-level A* algorithm now correctly solves Paris instances, large-map multi-agent tree search at $k \in \{20, 30, 40\}$ on Paris remains CPU-intensive for interactive training on a laptop. As mandated by the compute constraints, `Paris_1_256` is diagnosed and excluded from headline training, focusing on `random-32-32-20` and `room-32-32-4`.

---

### Bug 6: Learned Solver Inference Overhead Not Tracked Per Node
- **Diagnosis**: `benchmark_suite.py` reported total `cpu_seconds_used` and wall-clock `runtime_ms`, but did not report per-node overhead (`cpu_seconds_per_node`). Because feature extraction and model inference execute at every decision point, comparing raw node reductions without per-node cost masked the computational overhead of the learned policy.
- **Fix**: Added `cpu_seconds_per_node = cpu_seconds_used / nodes_expanded` to the output metrics of all solvers in `benchmark_suite.py` and `phase3_benchmark.py`.
- **Verification**: Per-node CPU cost is recorded for all benchmark rows, quantifying the ~0.2–0.5ms inference cost per decision.

---

### Bug 7: Physically Impossible Runtime Value in Benchmark Output
- **Diagnosis**: In `results/phase1_benchmark.csv`, instance `room-32-32-4,40,1,PlainICBSh` reported `runtime_ms=400590` (~400.6 seconds) despite `cpu_seconds_used=43.36` and a strict 60-second timeout. This was caused by mixing `time.time()` (system wall clock subject to sleep/drift/OS interruptions) with `time.process_time()`, and failing to check the timeout condition inside deep re-planning calls.
- **Fix**: Replaced all wall-clock measurements with high-resolution monotonic `time.perf_counter()` and process timing with `time.process_time()`. Ensured timeouts are evaluated against monotonic elapsed time.
- **Verification**: All reported runtimes in `phase3_benchmark.csv` strictly obey the timeout thresholds ($\le 60.0$s) and align with measured CPU time.

---

### Bug 8: Conflict Density Audit Overall Prevalence Discrepancy
- **Diagnosis**: The Phase 1 report claimed "61.7% overall prevalence" for Regime B. Recomputing directly from `results/conflict_density_audit.csv` across all valid instances revealed that the true overall prevalence was ~47.5%. The earlier figure had been computed by averaging cell-level maximums rather than weighting by valid instances.
- **Fix**: Updated `audit_conflict_density.py` to recompute summary statistics directly from the raw instance records: overall instance prevalence is calculated as `(df['max_total_conflicts_at_any_node'] >= 2).mean()`, and the per-cell minimum and maximum rates are explicitly reported.
- **Verification**: Recomputation yields: Regime B overall prevalence = **47.5%** (cell range: **0.0% to 100.0%**); Regime A overall prevalence = **37.5%** (cell range: **0.0% to 100.0%**).

---

### Bug 9: `BOTH_APPLICABLE` Overclaimed Single Global Applicability
- **Diagnosis**: The audit report issued a single binary verdict `BOTH_APPLICABLE`, obscuring the fact that Regime A applicability varies dramatically across topologies (from 0.0% on low-density instances up to 100.0% on high-density room bottlenecks).
- **Fix**: Updated `audit_conflict_density.py` and the summary reports to document the exact per-(map, k) prevalence range rather than a single global verdict.
- **Verification**: Documented that Regime A applicability ranges from 0% ($k=12$ on open maps) to 100% ($k \ge 20$ in narrow rooms), confirming that conflict selection policies must be evaluated on the regime-applicable subset.

---

### Bug 10: Phase 0 Compute Extrapolation Underestimated by 85×
- **Diagnosis**: Phase 0 extrapolated 0.003 CPU-hours for Phase 1 proxy labeling based on trivial 2-agent instances; actual Phase 1 labeling required 0.255 CPU-hours (an 85× error). The linear scaling assumption failed to account for combinatorial conflict growth at $k \ge 28$.
- **Fix**: Added explicit warnings in `phase0_calibration.py` noting that extrapolation formulas are order-of-magnitude bounds only ($\pm 10\text{--}85\times$) and must not be used as rigid commitments. Replaced naive rollout labeling with an accelerated Incremental Conflict Inheritance engine (`fast_dataset_builder.py` / `oracle_labeler.py`) that reduced data collection time from hours to under 4 minutes.
- **Verification**: Full dataset generation of 14,841 candidate rows across 3,834 decision queries completed in **219.2 seconds** (0.06 CPU-hours), remaining well within the laptop budget.
