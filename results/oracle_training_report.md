# Oracle Ranker Training Report

*Generated in 19.7s*

---

## 1. Cross-Validation Results

| Model | Mean Spearman ρ | Std | Mean NDCG@3 |
|---|---:|---:|---:|
| XGBRanker | `+0.7031` | `0.0411` | `0.9067` |
| LinearRanker | `+0.4906` | `0.0173` | `0.8441` |
| RandomRanker | `+0.0027` | `0.0164` | `0.5789` |

**Intermediate Gate (0.3):** ✅ PASSED
  - XGBRanker held-out Spearman = `0.7031`

---

## 2. Feature-Family Ablation

| Family Dropped | Features | Ablated Spearman | Full Spearman | Δ |
|---|---:|---:|---:|---:|
| (Full model) | 19 | — | `+0.7031` | — |
| `affected_agents` | 3 | `+0.6408` | `+0.7031` | `-0.0623` |
| `downstream` | 2 | `+0.6914` | `+0.7031` | `-0.0116` |
| `delta_f_res` | 1 | `+0.6982` | `+0.7031` | `-0.0048` |
| `goal_distance` | 2 | `+0.6985` | `+0.7031` | `-0.0045` |
| `corridor` | 2 | `+0.7012` | `+0.7031` | `-0.0018` |
| `conflict_frequency` | 1 | `+0.7025` | `+0.7031` | `-0.0005` |
| `congestion` | 2 | `+0.7033` | `+0.7031` | `+0.0002` |

---

## 3. Feature Importances (Full Model)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `pair_conflict_count` | `0.1391` |
| 2 | `conflict_time` | `0.1156` |
| 3 | `is_edge_conflict` | `0.0803` |
| 4 | `conflict_type` | `0.0672` |
| 5 | `num_agents_in_conflict_graph` | `0.0625` |
| 6 | `agent_j_path_length` | `0.0607` |
| 7 | `delta_f_res` | `0.0515` |
| 8 | `cardinal_degree_i` | `0.0514` |
| 9 | `goal_dist_i` | `0.0482` |
| 10 | `goal_dist_j` | `0.0473` |
| 11 | `agent_i_path_length` | `0.0449` |
| 12 | `cardinal_degree_j` | `0.0441` |
| 13 | `mvc_size` | `0.0372` |
| 14 | `corridor_score_j` | `0.0293` |
| 15 | `local_density_i` | `0.0289` |
| 16 | `downstream_conflict_count_j` | `0.0276` |
| 17 | `downstream_conflict_count_i` | `0.0242` |
| 18 | `local_density_j` | `0.0210` |
| 19 | `corridor_score_i` | `0.0191` |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Dataset | `data/cl_oracle_training.csv` |
| Rows | `14,841` |
| Queries | `3,834` |
| Features | `19` |
| CV Strategy | GroupKFold(5) by (map, k, scenario) |
| Training target | `y_khalil_binary` (top-20% by y_composite) |
| Evaluation target | `y_composite` (continuous oracle) |
| XGBoost objective | `rank:pairwise` |
| Model path | `models/oracle_ranker.json` |
| Total time | `19.7s` |