# MAPF-CT-Rank: Multi-Fidelity Constraint Tree Conflict Prioritization Dataset

> **Generated automatically by `mapf/dataset_quality_report.py`**

---

## 1. Executive Dataset Summary

| Metric | Value |
|---|---|
| Total Candidate Rows | `6,789` |
| Total Decision Queries | `2,251` |
| Mean Candidates per Query | `3.02` |
| Mean Labeling Time per Query | `187.13 ms` |
| Hard Analytical Ties (is_analytical_tie == 1) | `2,202 (97.8%)` |
| Analytically Solved Queries (is_analytical_tie == 0) | `49 (2.2%)` |
| Ground-Truth Label Columns | `9` |

---

## 2. Coverage Breakdown by Map and Agent Count (*k*)

| Map | *k* | Scenarios | Queries | Candidate Rows | Mean Tree Depth | Mean Cardinals/Node |
|---|---:|---:|---:|---:|---:|---:|
| `Paris_1_256` | 20 | 10 | 18 | 55 | 2.1 | 4.87 |
| `Paris_1_256` | 30 | 10 | 50 | 178 | 5.9 | 8.71 |
| `Paris_1_256` | 40 | 10 | 53 | 208 | 6.5 | 21.33 |
| `random-32-32-20` | 16 | 20 | 162 | 403 | 6.2 | 4.75 |
| `random-32-32-20` | 20 | 20 | 200 | 555 | 7.4 | 6.38 |
| `random-32-32-20` | 24 | 20 | 206 | 638 | 8.6 | 9.50 |
| `random-32-32-20` | 28 | 20 | 222 | 757 | 10.9 | 13.06 |
| `room-32-32-4` | 12 | 18 | 198 | 505 | 5.8 | 5.78 |
| `room-32-32-4` | 16 | 20 | 229 | 673 | 7.7 | 8.53 |
| `room-32-32-4` | 20 | 20 | 233 | 778 | 8.6 | 13.28 |
| `room-32-32-4` | 24 | 20 | 229 | 881 | 10.1 | 20.98 |
| `warehouse-32-32-aisles` | 12 | 13 | 69 | 158 | 3.2 | 3.62 |
| `warehouse-32-32-aisles` | 16 | 15 | 113 | 267 | 5.6 | 4.23 |
| `warehouse-32-32-aisles` | 20 | 15 | 130 | 337 | 7.2 | 6.16 |
| `warehouse-32-32-aisles` | 24 | 15 | 139 | 396 | 8.5 | 8.84 |

---

## 3. Discriminative Power: Within-Query Tie Saturation

> A *tied query* is one where all candidates have identical label values,
> making it impossible to learn a ranking. Proxy labels are catastrophically
> tied; oracle labels are nearly tie-free.

| Supervision Signal | Column | Tied Queries | Discriminative Queries |
|---|---|---:|---:|
| Phase 1 Proxy Formula | `y_proxy_score` | **4.9%** | 95.1% |
| True Δ-MVC (h_child − h_parent) | `y_oracle_dh_min` | **82.5%** | 17.5% |
| Zero-A* Residual Δf | `y_residual_df_min` | **95.6%** | 4.4% |
| Exact 1-Step Δf_min (Huang 2021) | `y_oracle_df_min` | **76.1%** | 23.9% |
| Exact 1-Step Δf_max | `y_oracle_df_max` | **56.8%** | 43.2% |
| Child Conflicts Min | `y_child_conflicts` | **41.7%** | 58.3% |
| Micro-Rollout Cost (B=30) | `y_micro_rollout` | **54.3%** | 45.7% |
| Hybrid Composite Oracle | `y_composite` | **25.5%** | 74.5% |
| Parameter-Free Lexicographic Rank | `y_lex_rank_norm` | **24.6%** | 75.4% |

---

## 4. Within-Query Spearman ρ Correlation Matrix

> *Higher ρ = more similar ranking. Proxy's near-zero ρ with oracle proves
> why Phase 1 failed. Oracle signals correlate strongly with each other.*

| Signal | Phase 1 Proxy Formula | True Δ-MVC (h_child − h_parent) | Zero-A* Residual Δf | Exact 1-Step Δf_min (Huang 2021) | Exact 1-Step Δf_max | Child Conflicts Min | Micro-Rollout Cost (B=30) | Hybrid Composite Oracle | Parameter-Free Lexicographic Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Phase 1 Proxy Formula** | `+1.000` | `-0.338` | `-0.219` | `-0.170` | `-0.389` | `-0.060` | `+0.146` | `-0.181` | `-0.135` |
| **True Δ-MVC (h_child − h_parent)** | `-0.338` | `+1.000` | `-0.123` | `+0.913` | `+0.124` | `+0.029` | `-0.244` | `+0.560` | `+0.635` |
| **Zero-A* Residual Δf** | `-0.219` | `-0.123` | `+1.000` | `+0.507` | `+0.635` | `-0.288` | `-0.799` | `+0.659` | `+0.695` |
| **Exact 1-Step Δf_min (Huang 2021)** | `-0.170` | `+0.913` | `+0.507` | `+1.000` | `+0.061` | `+0.307` | `-0.470` | `+0.704` | `+0.883` |
| **Exact 1-Step Δf_max** | `-0.389` | `+0.124` | `+0.635` | `+0.061` | `+1.000` | `-0.251` | `-0.557` | `+0.857` | `+0.760` |
| **Child Conflicts Min** | `-0.060` | `+0.029` | `-0.288` | `+0.307` | `-0.251` | `+1.000` | `+0.130` | `-0.329` | `-0.272` |
| **Micro-Rollout Cost (B=30)** | `+0.146` | `-0.244` | `-0.799` | `-0.470` | `-0.557` | `+0.130` | `+1.000` | `-0.711` | `-0.727` |
| **Hybrid Composite Oracle** | `-0.181` | `+0.560` | `+0.659` | `+0.704` | `+0.857` | `-0.329` | `-0.711` | `+1.000` | `+0.949` |
| **Parameter-Free Lexicographic Rank** | `-0.135` | `+0.635` | `+0.695` | `+0.883` | `+0.760` | `-0.272` | `-0.727` | `+0.949` | `+1.000` |

---

## 5. Feature-to-Oracle Rank Correlation (24 Features vs `y_lex_rank_norm`)

> Within-query Spearman ρ between each feature and the Oracle Target,
> reported both across **All Queries** and specifically on **Hard Analytical Tie Queries** (`is_analytical_tie == 1`).

| Rank | Feature | ρ (All Queries) | ρ (Hard Ties Only) | Direction |
|---:|---|---:|---:|---|
| 1 | `delta_g_max` | `+0.6969` | `+0.6880` | ↑ Higher = better branch |
| 2 | `delta_f_res` | `+0.6955` | `+0.7806` | ↑ Higher = better branch |
| 3 | `delta_f_res` | `+0.6955` | `+0.7806` | ↑ Higher = better branch |
| 4 | `delta_g_j` | `+0.3364` | `+0.3102` | ↑ Higher = better branch |
| 5 | `conflict_time` | `+0.2154` | `+0.2015` | ↑ Higher = better branch |
| 6 | `goal_dist_j` | `-0.1548` | `-0.1381` | ↓ Lower = better branch |
| 7 | `pair_conflict_count` | `+0.1537` | `+0.1378` | ↑ Higher = better branch |
| 8 | `downstream_conflict_count_j` | `+0.1388` | `+0.1463` | ↑ Higher = better branch |
| 9 | `goal_dist_i` | `-0.1118` | `-0.0992` | ↓ Lower = better branch |
| 10 | `delta_g_i` | `+0.0855` | `+0.0835` | ↑ Higher = better branch |
| 11 | `is_edge_conflict` | `-0.0604` | `-0.0557` | ↓ Lower = better branch |
| 12 | `agent_i_path_length` | `+0.0579` | `+0.0407` | ↑ Higher = better branch |
| 13 | `cardinal_degree_j` | `+0.0548` | `-0.0088` | ↑ Higher = better branch |
| 14 | `marginal_conflict_delta_min` | `-0.0419` | `-0.0585` | ↓ Lower = better branch |
| 15 | `downstream_conflict_count_i` | `+0.0369` | `+0.0275` | ↑ Higher = better branch |
| 16 | `corridor_score_j` | `-0.0363` | `-0.0337` | ↓ Lower = better branch |
| 17 | `cardinal_degree_i` | `+0.0349` | `+0.1385` | ↑ Higher = better branch |
| 18 | `agent_j_path_length` | `-0.0338` | `-0.0270` | ↓ Lower = better branch |
| 19 | `corridor_score_i` | `-0.0272` | `-0.0229` | ↓ Lower = better branch |
| 20 | `local_density_j` | `+0.0112` | `+0.0213` | ↑ Higher = better branch |
| 21 | `local_density_i` | `-0.0060` | `+0.0027` | ↓ Lower = better branch |
| 22 | `num_agents_in_conflict_graph` | N/A | N/A | — |
| 23 | `conflict_type` | N/A | N/A | — |
| 24 | `mvc_size` | N/A | N/A | — |
| 25 | `delta_g_min` | N/A | N/A | — |

---

## 6. Hard Analytical Tie Query Analysis

A Hard Analytical Tie query is one where >= 2 candidates share the identical maximum
(delta_f_res, delta_g_min) pair, so zero-A* analytical lookahead cannot discriminate.
These are exactly the queries where machine learning adds value beyond analytical rules.

| Metric | Value |
|---|---|
| Total Queries | `2,251` |
| Hard Analytical Tie Queries (is_analytical_tie == 1) | `2,202` |
| Easy Queries (analytically resolved, is_analytical_tie == 0) | `49` |
| % Hard Queries | `97.8%` |


---

## 7. Proxy Failure Diagnosis

- **Phase 1 Proxy ρ with oracle:** `-0.1808` (on 1631 non-tied queries)
- **Exact 1-Step Oracle ρ with composite:** `+0.7043` (on 539 non-tied queries)
- **Proxy tie rate:** `4.9%` — majority of queries are uniformative
- **Oracle tie rate:** `76.1%` — nearly all queries are discriminative

> **Conclusion:** The Phase 1 proxy correlates poorly with true branch quality (ρ=-0.181), while the 1-step oracle is discriminative and actionable.

---

## 8. Methodology Notes

### Label Definitions

| Column | Definition | Oracle Cost |
|---|---|---|
| `y_proxy_score` | −(1·cardinal + 0.5·downstream + 0.5/(corridor+1)) | 0 extra A* calls |
| `y_oracle_dh_min` | min(h_left, h_right) − h_parent | 0 extra (from classification) |
| `y_residual_df_min` | min(Δcost_i, Δcost_j) from alt_cost | 0 extra A* calls |
| `y_oracle_df_min` | min(f_left, f_right) − f_parent [Huang 2021] | via incremental inheritance |
| `y_oracle_df_max` | max(f_left, f_right) − f_parent | same |
| `y_child_conflicts` | min(\|C_left\|, \|C_right\|) | O(k·T) incremental scan |
| `y_micro_rollout` | nodes expanded from (child_left, child_right) until Δf≥+2 | ~8ms in-place |
| `y_composite` | 25·Δf_min + 6·Δf_max + 4·Δh_min − 0.5·micro − 0.15·conflicts | sum of above |
| `y_khalil_binary` | 1 if y_composite ≥ 80th percentile at this node | derived |

### Key Algorithmic Accelerations

1. **AStarMemoizer**: ~80% cache hit rate across all branching decisions per instance
2. **Incremental Conflict Inheritance**: O(1) for untouched agent pairs; only 1–3 new A* calls per branch
3. **O(k·T) Conflict Scan**: detect_conflicts_for_agent() vs O(k²·T) full scan
4. **Bounded A* horizon**: cost + 12 instead of width+height+50; 10× fewer stalled searches
5. **In-place micro-rollout**: starts from child nodes (~8ms) vs root re-solve (~4800ms)
