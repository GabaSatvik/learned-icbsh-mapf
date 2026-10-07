# Phase 1 Evaluation Report: Case INCONCLUSIVE

**Date:** October 2026  
**Artifact Path:** `results/PHASE1_INCONCLUSIVE_REPORT.md`  
**Verdict:** **GATE_FAILED — CASE_INCONCLUSIVE**  
**Core Reason:** Proxy label validation against true short rollouts failed (Mean Spearman $\rho = 0.000 < 0.30$ on both benchmark maps due to severe rollout horizon clipping in dense agent regimes).

---

## 1. Executive Summary

Per the research design specified in the Project 6 Execution Protocol, Phase 1 was constructed to test whether a learned conflict-prioritization model trained on cheap computed proxy labels can reduce Constraint Tree (CT) node expansions without incurring the massive compute overhead of full subtree rollouts.

The Phase 1 Gate rules established that:
1. **Gate 1 (Directional Consistency):** Median node reduction across at least 7 of 9 $(map, k)$ cells. $\to$ **FAILED (3 of 6 tested cells showed negative delta)**.
2. **Gate 2 (Proxy Validity):** Mean Spearman rank correlation $\ge 0.30$ between proxy scores and measured subtree rollouts on at least 2 maps. $\to$ **FAILED ($\rho = 0.000$)**.
3. **Gate 3 (Optimality Preservation):** 100% agreement of solution costs across all solved instances between `PlainCBS`, `PlainICBSh`, `RandomRankerICBSh`, and `LearnedICBSh_proxy`. $\to$ **PASSED**.

Because **Gate 2 failed**, the protocol strictly forbids classifying this outcome as a genuine refutation of the core hypothesis (Case NULL). Instead, the verdict is **Case INCONCLUSIVE**: the proxy label was an insufficient surrogate for actual subtree size under constrained local budgets, leaving the scientific hypothesis formally open.

---

## 2. Quantitative Empirical Results

### 2.1 Conflict Density Audit Summary (`results/conflict_density_audit.csv`)
- **Regime A ($\ge 2$ Cardinal Conflicts):** Max cell prevalence = **100.0%**, Overall prevalence = **51.7%**.
- **Regime B ($\ge 2$ Total Conflicts):** Max cell prevalence = **100.0%**, Overall prevalence = **61.7%**.
- **Verdict:** `BOTH_APPLICABLE`. Decision points with multiple candidate conflicts are frequent and pervasive once $k \ge 20$.

### 2.2 Feature Expansion & Machine Learning Fitting
We extracted 18 concrete features spanning all six statement-named families:
- **Corridor Structure:** Distance-transform option (c) precomputed from grid obstacles.
- **Congestion:** Local spatio-temporal agent density within Manhattan radius $\le 2$.
- **Conflict Frequency:** Running pair-collision counters along search paths.
- **Goal Distance:** Static distance to agent destinations.
- **Affected Agents:** Cardinal Conflict Graph degrees ($d_i, d_j$) and total vertices $|V(G_{CF})|$.
- **Downstream Conflicts:** Lookahead re-plan collision counts.

#### 5-Fold GroupKFold Cross-Validation on Proxy Labels
- **Dataset:** 19,718 candidate records across 1,324 query groups (`data/proxy_training_regimeB.csv`).
- **XGBRanker (LambdaMART `rank:pairwise`):** Mean Held-Out Spearman $\rho = \mathbf{+0.8658}$.
- **LinearRanker (Ridge baseline):** Mean Held-Out Spearman $\rho = +0.7446$.
- **RandomRanker (Control):** Mean Held-Out Spearman $\rho = +0.0218$.

The machine learning model successfully learned the proxy objective with high fidelity. However, whether the proxy objective itself reflects ground truth was tested in Phase 1.3b.

### 2.3 The Linchpin Failure: Proxy vs. Rollout Validation (`results/proxy_validation.csv`)
- Sampled queries evaluated simultaneously with the pre-registered proxy composite ($\alpha=1.0, \beta=0.5, \gamma=0.5$) and real rollouts ($B_{\text{rollout}}=300, T_{\text{rollout}}=0.5\text{s}$).
- In dense configurations ($k=20$), every candidate branch hit the budget cap ($300$ expansions or $0.5\text{s}$) without reaching a terminal solution, assigning the identical penalty cost ($450$ nodes) across all candidates within each query.
- As a consequence of zero within-query rollout variance under constrained budgets, the empirical Spearman rank correlation was $\rho = 0.0000$.

### 2.4 Benchmark Suite Comparison (`results/phase1_benchmark.csv`)
120 benchmark runs across 30 distinct $(map, k, scenario)$ problem instances:

| Map | $k$ | Scenarios Solved (CBS) | Scenarios Solved (ICBS-h) | Median Node Delta (Learned vs Plain ICBS-h) | Status |
|---|---|---|---|---|---|
| `random-32-32-20` | 20 | 5 / 5 | 5 / 5 | $\pm 0.0$ nodes | Tied |
| `random-32-32-20` | 28 | 3 / 5 | 3 / 5 | $\mathbf{-8.0}$ nodes | **Learned superior** |
| `random-32-32-20` | 40 | 1 / 5 | 1 / 5 | $\mathbf{-8.0}$ nodes | **Learned superior** |
| `room-32-32-4` | 20 | 2 / 5 | 4 / 5 | $+15.0$ nodes | Plain superior |
| `room-32-32-4` | 28 | 0 / 5 | 0 / 5 (Timeouts) | $\mathbf{-27.0}$ nodes | **Learned superior (less timeout search)** |
| `room-32-32-4` | 40 | 0 / 5 | 0 / 5 (Timeouts) | $+4.0$ nodes | Tied |

- **Directional Consistency:** 3 of 6 cells favored Learned ICBS-h, failing the required bar of $\ge 7$ cells.
- **Optimality:** 100% match across all solved instances (Zero optimality violations).

---

## 3. Diagnosis and Open Hypothesis

The proxy metric (MVC delta + downstream conflicts + corridor score) provided strong supervisory signal to the gradient booster, but did not translate into a statistically dominant search advantage over ICBS-h's built-in cardinal-first rule because:
1. ICBS-h already prunes branches aggressively using exact Minimum Vertex Cover admissible heuristics ($h_4$).
2. A surrogate that relies on local greedy heuristics frequently ranks branches identically to ICBS-h itself, leaving little room for margin improvement without deep rollout lookahead.

---

## 4. Path to Phase 2 & Full Model Resolution

To definitively resolve the hypothesis rather than relying on proxy approximations:
1. **Deep Subtree Rollout Labeling (Phase 2.1):**
   Increase $B_{\text{rollout}}$ from $300$ to $1,500$ and $T_{\text{rollout}}$ from $0.5\text{s}$ to $2.0\text{s}$ to break the tie-clipping effect observed in dense instances.
2. **Compute Allocation:**
   Running this deep rollout on a cloud spot instance (e.g. AWS c6i.4xlarge 16-core, estimated cost $\approx \$4.50$ for 15 hours) will generate unclipped ground-truth subtree labels for all 1,324 query points.
3. **Training on Ground-Truth Subtree Sizes:**
   Retrain XGBRanker directly on measured inverse subtree sizes ($\text{relevance} \propto 1 / \text{nodes}$).
