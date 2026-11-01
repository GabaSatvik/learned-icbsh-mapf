# Learned Conflict Prioritization in Multi-Agent Path Finding (ICBS-h)

Implementation and empirical evaluation of **Learned Conflict Prioritization** in **Improved Conflict-Based Search (ICBS-h)** for Multi-Agent Path Finding (MAPF), based on Sharon et al. (2012), Boyarski et al. (2015), Felner et al. (2018), and Huang et al. (2021).

---

## 📌 Project Overview

In standard Conflict-Based Search (CBS) and ICBS-h, whenever multiple collision candidates exist at a Constraint Tree (CT) node, the solver selects which conflict to branch on using fixed heuristics (first-found, or cardinal > semi-cardinal > non-cardinal with arbitrary tie-breaking).

This project investigates:
> **Does a machine learning ranker (XGBRanker / Linear Ridge) trained on parameter-free lexicographic lookahead supervision reduce CT node expansions and low-level $A^*$ invocations in ICBS-h compared to deterministic policies, while strictly maintaining 100% solution optimality?**

### Key Features
- **Low-Level Space-Time $A^*$**: Dynamic vertex and edge constraints with goal-holding persistence and instrumented low-level $A^*$ call tracking.
- **3-Way Conflict Classification & Bypass**: Exact identification of **Cardinal**, **Semi-Cardinal**, and **Non-Cardinal** conflicts with bypass path updates.
- **Minimum Vertex Cover (MVC) Heuristic**: Exact branch-and-bound computation of $h(N) = \text{MVC}(G_{CF})$ over the Cardinal Conflict Graph.
- **24 Concrete Features (7 Families)**:
  1. *Corridor Structure* (`corridor_score_i`, `corridor_score_j`)
  2. *Congestion* (`local_density_i`, `local_density_j`)
  3. *Conflict Frequency* (`pair_conflict_count`)
  4. *Goal Distance* (`goal_dist_i`, `goal_dist_j`)
  5. *Affected Agents* (`num_agents_in_conflict_graph`, `cardinal_degree_i`, `cardinal_degree_j`)
  6. *Downstream Conflicts* (`downstream_conflict_count_i`, `downstream_conflict_count_j`, `marginal_conflict_delta_min`)
  7. *Zero-$A^*$ Cached Lookahead & Baseline Context* (`delta_g_i`, `delta_g_j`, `delta_g_min`, `delta_g_max`, `delta_f_res`, `conflict_type`, `conflict_time`, `is_edge_conflict`, `agent_i_path_length`, `agent_j_path_length`, `mvc_size`)
- **Parameter-Free Lexicographic Oracle**: 6-tuple ranking ground truth $(\Delta f_{\min}, \Delta f_{\max}, \Delta h_{\min}, -\text{micro\_cost}, -c_{\min}, -\text{card}_{\min})$ that eliminates hand-tuned magic weights.
- **Tie-Only Analytical ML Gating**: Primary analytical lookahead $(\Delta f_{\text{res}}, \Delta g_{\min})$ suffers from **95.6% tie saturation** on unweighted 4-connected grids because unit-cost steps cause most cardinal bypass detours to cost $\Delta g_{\min} = 1$. ML inference is short-circuited on the 4.4% of queries where a candidate strictly dominates, and invoked exclusively on the 95.6% hard analytical plateaus to break ties via secondary branch asymmetry ($\Delta g_{\max}$) and downstream conflict lookahead.
- **Vectorized NumPy Dot-Product**: Microsecond inference for linear models, eliminating Python pipeline overhead.
- **Multi-Topology Benchmark Suite**: Evaluates 7 solvers across 4 distinct topologies (`random-32-32-20`, `room-32-32-4`, `warehouse-32-32-aisles`, and `Paris_1_256`) with low-level $A^*$ calls and `tracemalloc` peak memory profiling under standardized budgets ($B_{\max} = 350$, $T_{\max} = 2.5\text{s}$).

---

## 📂 Repository Structure

```
├── mapf/
│   ├── grid.py                  # Moving AI map and scenario parser (4-connected grid)
│   ├── low_level.py             # Space-Time A* with vertex/edge constraints & A* counter
│   ├── conflicts.py             # O(k·T) single-agent & pair collision detection
│   ├── cardinal.py              # Conflict classification and bypass logic
│   ├── conflict_graph.py        # Cardinal Conflict Graph G_CF & exact MVC solver
│   ├── features.py              # 24-feature extraction across 7 families in O(k·T) time
│   ├── logger.py                # Query-grouped LTR data logger
│   ├── high_level_cbs.py        # Baseline Plain CBS implementation
│   ├── icbs_h.py                # ICBS-h solver with MVC heuristic + tie-break hooks
│   ├── baselines.py             # Fixed heuristics + GuardedLearnedSelector
│   ├── fast_dataset_builder.py  # Parameter-free lexicographic dataset builder
│   ├── dataset_quality_report.py# Dataset audit & Spearman correlation analysis
│   ├── train_cl_ranker.py       # 5-Fold GroupKFold training + family ablations
│   ├── phase3_benchmark.py      # 7-solver, 25-scenario comparative benchmark
│   └── test_features.py         # Unit tests on 5x5 grid instances
├── data/                        # Benchmark maps (.map) and scenario files (.scen)
├── models/
│   ├── ranker_cl_oracle.json    # Trained XGBRanker model
│   └── linear_cl_oracle.pkl     # Trained Linear Ridge model
├── results/
│   ├── ONBOARDING_AND_PROJECT_PROGRESSION.md # New member onboarding & chronological progression guide
│   ├── SUMMARY_OF_RESULTS.md    # Executive summary of empirical results
│   ├── FINAL_REPORT.md          # Comprehensive final research report
│   ├── PHASE3_FINAL_REPORT.md   # Phase 3 technical milestone report
│   ├── DATASET_CONTRIBUTION_REPORT.md # Dataset quality audit report
│   ├── oracle_training_report.md# ML training & cross-validation report
│   ├── phase3_statistical_summary.md # 7-solver benchmark statistical summary
│   ├── phase3_benchmark.csv     # Raw 1,050 benchmark execution records
│   └── phase3_training_summary.json  # Cross-validation metrics & ablations JSON
├── download_data.py             # Automated dataset downloader
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Reproduction Pipeline

To reproduce all results, execute the following 5 verification commands in sequence:

### 1. Verify Unit Tests & $O(k \cdot T)$ Feature Extractor
```bash
python -m mapf.test_features
```
*Expected: 24 features extracted, low-level $A^*$ counter verified, optimality verified in < 0.5s.*

### 2. Generate Parameter-Free Lexicographic Dataset
```bash
python -m mapf.fast_dataset_builder
```
*Harvests 6,789 candidate records across 2,251 deduplicated queries over 4 topologies into `data/cl_oracle_training.csv`.*

### 3. Generate Dataset Quality & Correlation Report
```bash
python -m mapf.dataset_quality_report
```
*Generates `results/DATASET_CONTRIBUTION_REPORT.md` analyzing tie rates, topology coverage, and within-query Spearman correlations.*

### 4. Train Rankers & Cross-Validation Audit
```bash
python -m mapf.train_cl_ranker
```
*Trains `XGBRanker` and `LinearRanker` with 5-fold cross-validation, feature ablations, and zero-shot cross-map transfer tests, writing summary metrics to `results/phase3_training_summary.json` and `results/oracle_training_report.md`.*

### 5. Run the 7-Solver Statistical Benchmark
```bash
python -m mapf.phase3_benchmark
```
*Executes all 7 solvers across 25 scenarios per cell (150 instances, 1,050 total solver runs) with low-level $A^*$ counts, peak memory tracking, and Wilcoxon signed-rank tests.*

---

## 📊 Summary of Empirical Results

### 1. Cross-Validation Ranking Performance (5-Fold `GroupKFold`)
- **Full Model (`XGBRanker`)**: Spearman $\rho = \mathbf{+0.4578 \pm 0.0479}$ (0 negative folds)
- **Hard Analytical Ties (`is_analytical_tie == 1`)**: Spearman $\rho = \mathbf{+0.4479}$ (vs. -0.0450 for random control)
- **Easy Queries (`is_analytical_tie == 0`)**: Spearman $\rho = \mathbf{+0.7994}$
- **Linear Ridge Baseline**: Spearman $\rho = +0.2589$
- **Random Choice (Control)**: Spearman $\rho = -0.0459$
- **Zero-Shot Cross-Map Generalization**: $\ge +0.1917$ across all held-out topology transfers.
- **Critical Feature Families**: Downstream replan conflict count is #1 most critical family ($\Delta \rho = -0.0589$ when removed); secondary branch detour $\Delta g_{\max}$ is #1 feature by gain (0.1577).

### 2. High-Level Node Expansions & Low-Level $A^*$ Suppression (`random-32-32-20`, $k=20$)
All 1,050 solver runs used a standardized search budget of $B_{\max} = 350$ CT node expansions and $T_{\max} = 2.5\text{s}$ wall-clock timeout (which is why `PlainCBS` caps at 350.0 median nodes and timeouts cluster near 2.50s).
- **`PlainCBS`**: 350.0 median nodes, 86 median $A^*$ calls (10/25 solved)
- **`PlainICBSh`**: 19.0 median nodes, 57 median $A^*$ calls (20/25 solved)
- **`FAwareResidualICBSh`**: 18.0 median nodes, 20 median $A^*$ calls (23/25 solved)
- **`Learned_Linear_CL_ICBSh`**: 18.0 median nodes, 20 median $A^*$ calls (23/25 solved)
- **`Learned_XGB_CL_ICBSh`**: 18.0 median nodes, 20 median $A^*$ calls (24/25 solved)
- **Low-Level $A^*$ Savings**: Lookahead-guided solvers reduce low-level $A^*$ searches by **76.7%** vs. Plain CBS and **64.9%** vs. Plain ICBS-h.

### 3. Topology Breakdown & Comparative Analysis
- **Random Grids (`random-32-32-20`, $k=28$)**: `Learned_XGB_CL_ICBSh` statistically outperforms `FAwareResidualICBSh` (60.7 vs. 68.2 mean nodes, Wilcoxon $p = 0.0083^*$ overall; 57.5 vs. 65.5 median nodes, $p = 0.0352^*$ on held-out test scenarios). In open, obstacle-scattered terrain, alternative detours often create cascading downstream collisions among bystander agents. The learned ranker's downstream conflict lookahead (`downstream_conflict_count_j`, `marginal_conflict_delta_min`) prunes these branching cascades where fixed analytical rules remain blind.
- **Doorway Bottlenecks (`room-32-32-4`, $k=20$)**: Deterministic `FAwareResidualICBSh` edges out the learned ranker in solve rate (14/25 vs. 12/25 solved; 38.0 vs. 57.0 median nodes). In 1-cell doorway chokepoints, conflict resolution is strictly constrained by topological hub degree and $\Delta g_{\max}$; the deterministic lookahead captures these chokepoint constraints directly with zero feature extraction and model inference overhead, avoiding borderline timeouts near the 2.50s cap.

### 4. Solution Cost Optimality
- **100% Cost Optimality**: Verified across all 1,050 benchmark executions without exception.
