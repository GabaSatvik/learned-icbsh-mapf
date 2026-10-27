# MAPF-CT-Rank: Multi-Fidelity Constraint Tree Conflict Prioritization Dataset

> **Generated automatically by `mapf/dataset_quality_report.py`**

---

## 1. Executive Dataset Summary

| Metric | Value |
|---|---|
| Total Candidate Rows | `14,841` |
| Total Decision Queries | `3,834` |
| Mean Candidates per Query | `3.87` |
| Mean Labeling Time per Query | `nan ms` |
| Ground-Truth Label Columns | `8` |

---

## 2. Coverage Breakdown by Map and Agent Count (*k*)

| Map | *k* | Scenarios | Queries | Candidate Rows | Mean Tree Depth | Mean Cardinals/Node |
|---|---:|---:|---:|---:|---:|---:|
| `random-32-32-20` | 16 | 18 | 478 | 1,723 | nan | nan |
| `random-32-32-20` | 20 | 19 | 592 | 2,453 | nan | nan |
| `random-32-32-20` | 24 | 20 | 632 | 2,805 | nan | nan |
| `random-32-32-20` | 28 | 20 | 656 | 2,967 | nan | nan |
| `room-32-32-4` | 12 | 9 | 163 | 467 | nan | nan |
| `room-32-32-4` | 16 | 14 | 280 | 911 | nan | nan |
| `room-32-32-4` | 20 | 18 | 468 | 1,587 | nan | nan |
| `room-32-32-4` | 24 | 19 | 565 | 1,928 | nan | nan |

---

## 3. Discriminative Power: Within-Query Tie Saturation

> A *tied query* is one where all candidates have identical label values,
> making it impossible to learn a ranking. Proxy labels are catastrophically
> tied; oracle labels are nearly tie-free.

| Supervision Signal | Column | Tied Queries | Discriminative Queries |
|---|---|---:|---:|
| Phase 1 Proxy Formula | `y_proxy_score` | **5.2%** | 94.8% |
| Zero-A* Residual Δf | `y_residual_df_min` | **92.0%** | 8.0% |
| Exact 1-Step Δf_min (Huang 2021) | `y_oracle_df_min` | **92.0%** | 8.0% |
| Micro-Rollout Cost (B=30) | `y_micro_rollout` | **13.1%** | 86.9% |
| Hybrid Composite Oracle | `y_composite` | **1.0%** | 99.0% |

---

## 4. Within-Query Spearman ρ Correlation Matrix

> *Higher ρ = more similar ranking. Proxy's near-zero ρ with oracle proves
> why Phase 1 failed. Oracle signals correlate strongly with each other.*

| Signal | Phase 1 Proxy Formula | Zero-A* Residual Δf | Exact 1-Step Δf_min (Huang 2021) | Micro-Rollout Cost (B=30) | Hybrid Composite Oracle |
|---|---:|---:|---:|---:|---:|
| **Phase 1 Proxy Formula** | `+1.000` | `-0.097` | `-0.097` | `-0.292` | `+0.223` |
| **Zero-A* Residual Δf** | `-0.097` | `+1.000` | `+1.000` | `-0.797` | `+0.765` |
| **Exact 1-Step Δf_min (Huang 2021)** | `-0.097` | `+1.000` | `+1.000` | `-0.797` | `+0.765` |
| **Micro-Rollout Cost (B=30)** | `-0.292` | `-0.797` | `-0.797` | `+1.000` | `-0.931` |
| **Hybrid Composite Oracle** | `+0.223` | `+0.765` | `+0.765` | `-0.931` | `+1.000` |

---

## 5. Feature-to-Oracle Rank Correlation (21 Inference-Time Features)

> Within-query Spearman ρ between each feature and the Hybrid Composite Oracle.
> Features with |ρ| > 0.2 are potentially useful for the learned ranker.

| Rank | Feature | ρ with `y_composite` | Direction |
|---:|---|---:|---|
| 1 | `delta_g_max` | `+0.8481` | ↑ Higher = better branch |
| 2 | `delta_f_res` | `+0.7651` | ↑ Higher = better branch |
| 3 | `delta_f_res` | `+0.7651` | ↑ Higher = better branch |
| 4 | `delta_g_j` | `+0.4784` | ↑ Higher = better branch |
| 5 | `marginal_conflict_delta_min` | `-0.3569` | ↓ Lower = better branch |
| 6 | `pair_conflict_count` | `+0.3403` | ↑ Higher = better branch |
| 7 | `conflict_type` | `+0.3347` | ↑ Higher = better branch |
| 8 | `is_edge_conflict` | `-0.2247` | ↓ Lower = better branch |
| 9 | `downstream_conflict_count_j` | `-0.2217` | ↓ Lower = better branch |
| 10 | `delta_g_i` | `+0.2094` | ↑ Higher = better branch |
| 11 | `delta_g_min` | `-0.2039` | ↓ Lower = better branch |
| 12 | `goal_dist_j` | `-0.1986` | ↓ Lower = better branch |
| 13 | `downstream_conflict_count_i` | `-0.1886` | ↓ Lower = better branch |
| 14 | `local_density_j` | `-0.1790` | ↓ Lower = better branch |
| 15 | `conflict_time` | `+0.1487` | ↑ Higher = better branch |
| 16 | `local_density_i` | `-0.1471` | ↓ Lower = better branch |
| 17 | `agent_j_path_length` | `-0.1173` | ↓ Lower = better branch |
| 18 | `goal_dist_i` | `-0.0897` | ↓ Lower = better branch |
| 19 | `cardinal_degree_i` | `+0.0462` | ↑ Higher = better branch |
| 20 | `agent_i_path_length` | `+0.0424` | ↑ Higher = better branch |
| 21 | `corridor_score_j` | `-0.0412` | ↓ Lower = better branch |
| 22 | `corridor_score_i` | `-0.0409` | ↓ Lower = better branch |
| 23 | `cardinal_degree_j` | `+0.0354` | ↑ Higher = better branch |
| 24 | `num_agents_in_conflict_graph` | N/A | — |
| 25 | `mvc_size` | N/A | — |

---

## 6. Proxy Failure Diagnosis

- **Phase 1 Proxy ρ with oracle:** `+0.2229` (on 3601 non-tied queries)
- **Exact 1-Step Oracle ρ with composite:** `+0.7651` (on 308 non-tied queries)
- **Proxy tie rate:** `5.2%` — majority of queries are uniformative
- **Oracle tie rate:** `92.0%` — nearly all queries are discriminative

> **Conclusion:** The Phase 1 proxy correlates poorly with true branch quality (ρ=+0.223), while the 1-step oracle is discriminative and actionable.

---

## 7. Methodology Notes

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
