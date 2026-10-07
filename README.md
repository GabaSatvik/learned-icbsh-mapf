# Learned Conflict Prioritization in Multi-Agent Path Finding (ICBS-h)

Implementation and empirical evaluation of **Learned Conflict Prioritization** in **Improved Conflict-Based Search (ICBS-h)** for Multi-Agent Path Finding (MAPF), based on Sharon et al. (2012), Boyarski et al. (2015), and Felner et al. (2018).

---

## 📌 Project Overview

In standard Conflict-Based Search (CBS) and ICBS-h, whenever multiple collision candidates exist at a Constraint Tree (CT) node, the solver selects which conflict to branch on using fixed heuristics (first-found, or cardinal > semi-cardinal > non-cardinal with arbitrary tie-breaking).

This project investigates:
> **Does a machine learning ranker (LambdaMART / XGBRanker) over per-conflict spatial and temporal features reduce CT node expansions compared to fixed-rule policies, while strictly maintaining 100% solution optimality?**

### Key Features
- **Low-Level Space-Time $A^*$**: Dynamic vertex and edge constraints with goal-holding persistence and precomputed reverse BFS heuristics.
- **3-Way Conflict Classification**: Exact identification of **Cardinal**, **Semi-Cardinal**, and **Non-Cardinal** conflicts with bypass path updates.
- **Minimum Vertex Cover (MVC) Heuristic**: Exact branch-and-bound computation of $h(N) = \text{MVC}(G_{CF})$ over the Cardinal Conflict Graph.
- **18 Concrete Features**: Spanning all 6 feature families:
  1. *Corridor Structure* (Distance-transform clearance)
  2. *Congestion* (Local spatio-temporal agent density)
  3. *Conflict Frequency* (Pairwise historical collision tracking)
  4. *Goal Distance* (True static shortest-path distance)
  5. *Affected Agents* (Graph degrees in $G_{CF}$)
  6. *Downstream Conflicts* (One-step lookahead replan conflicts)
  7. *Baseline State Context* (Conflict type, MVC size, collision time, edge flag, path lengths)
- **Learning-to-Rank Pipeline**: Query-grouped XGBoost `rank:pairwise` model trained on candidate conflict decision points.
- **Rigorous Verification & Benchmarking**: Tested across Moving AI benchmark maps (`random-32-32-20`, `room-32-32-4`, `Paris_1_256`).

---

## 📂 Repository Structure

```
├── mapf/
│   ├── grid.py                  # Moving AI map and scenario parser (4-connected grid)
│   ├── low_level.py             # Space-Time A* with vertex/edge constraints
│   ├── conflicts.py             # Collision detection (vertex and edge conflicts)
│   ├── cardinal.py              # Conflict classification and bypass logic
│   ├── conflict_graph.py        # Cardinal Conflict Graph G_CF & exact MVC solver
│   ├── features.py              # 18-feature extraction across 6 families
│   ├── logger.py                # Query-grouped LTR data logger
│   ├── high_level_cbs.py        # Baseline Plain CBS implementation
│   ├── icbs_h.py                # Full ICBS-h solver with MVC heuristic + LTR hooks
│   ├── phase0_calibration.py    # Compute calibration & budget extrapolation
│   ├── audit_conflict_density.py# Conflict density empirical sweep
│   ├── proxy_labeler.py         # Surrogate proxy label generator (no rollout cost)
│   ├── validate_proxy.py        # Linchpin proxy vs rollout Spearman check
│   ├── train_ranker.py          # 5-Fold GroupKFold training + family ablations
│   ├── benchmark_suite.py       # 4-solver comparative benchmark runner
│   ├── phase1_gate.py           # Pre-registered gate decision evaluator
│   └── test_features.py         # Unit tests on 5x5 grid instances
├── data/                        # Benchmark maps (.map) and scenario files (.scen)
├── models/                      # Saved trained models & feature ablation checkpoints
├── results/
│   ├── FINAL_REPORT.md          # Comprehensive research paper & milestone review
│   ├── PHASE1_INCONCLUSIVE_REPORT.md # Diagnostic gate analysis
│   ├── conflict_density_audit.csv # Empirical density audit results
│   ├── proxy_validation.csv     # Proxy vs rollout correlation logs
│   ├── phase1_benchmark.csv     # Benchmark suite results across all solvers
│   └── environment.txt          # Hardware, OS, and library version disclosure
├── download_data.py             # Automated dataset downloader
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/GabaSatvik/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Download Benchmark Data
Download the Moving AI benchmark maps and scenarios:
```bash
python download_data.py
```

### 3. Run Unit Tests & Sanity Check
Verify feature extraction and solver correctness:
```bash
python -m mapf.test_features
python -m mapf.phase0_calibration
```

### 4. Run Conflict-Density Audit
Sweep benchmark maps to evaluate multi-conflict prevalence:
```bash
python -m mapf.audit_conflict_density
```

### 5. Train the Learning-to-Rank Model
Generate proxy data and train the ranker with 5-fold cross-validation and feature-family ablations:
```bash
python -m mapf.proxy_labeler --regime B
python -m mapf.train_ranker
```

### 6. Run Benchmark Evaluation & Gate Decision
Evaluate all 4 solvers side-by-side (`PlainCBS`, `PlainICBSh`, `RandomRankerICBSh`, `LearnedICBSh_proxy`):
```bash
python -m mapf.benchmark_suite
python -m mapf.phase1_gate
```

---

## 📊 Summary of Results

| Map | Agents ($k$) | CBS Solved | ICBS-h Solved | Median Node $\Delta$ (Learned vs Plain ICBS-h) | Optimality Preserved |
|---|---|---|---|---|---|
| `random-32-32-20` | 20 | 5 / 5 | 5 / 5 | **$\pm 0.0$** | **100% Match** |
| `random-32-32-20` | 28 | 3 / 5 | 3 / 5 | **-8.0 (Learned wins)** | **100% Match** |
| `random-32-32-20` | 40 | 1 / 5 | 1 / 5 | **-8.0 (Learned wins)** | **100% Match** |
| `room-32-32-4` | 20 | 4 / 5 | 4 / 5 | +15.0 (Plain wins) | **100% Match** |
| `room-32-32-4` | 28 | 0 / 5 (Timeout) | 0 / 5 (Timeout) | **-27.0 (Learned wins)** | — |
| `room-32-32-4` | 40 | 0 / 5 (Timeout) | 0 / 5 (Timeout) | +4.0 (Plain wins) | — |

- **Optimality:** 100% agreement across all solved instances (Zero cost violations).
- **Feature Importance:** Corridor clearance (14.8%) and downstream collisions (10.5%) contributed the highest predictive signal beyond conflict type.
- **Gate Verdict:** Evaluated as **`CASE_INCONCLUSIVE`** because micro-rollout budget capping ($B=300$) prevented variance in proxy validation. Full details and roadmap are documented in [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md).

---

## 📚 References
1. Sharon, G., Stern, R., Felner, A., & Sturtevant, N. R. (2012). *Conflict-based search for optimal multi-agent pathfinding*. Artificial Intelligence.
2. Boyarski, E., Felner, A., Stern, R., Sharon, G., Tolpin, D., Betzalel, O., & Shimony, E. (2015). *ICBS: Improved Conflict-Based Search for Multi-Agent Pathfinding*. IJCAI.
3. Felner, A., Li, J., Boyarski, E., Ma, H., Cohen, L., Kumar, T. K. S., & Koenig, S. (2018). *Adding Heuristics to Conflict-Based Search for Multi-Agent Path Finding*. ICAPS.
4. Stern, R., et al. (2019). *Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks*. SoCS.
5. Sturtevant, N. R. (2012). *Benchmarks for Grid-Based Pathfinding*. IEEE Transactions on Computational Intelligence and AI in Games.
