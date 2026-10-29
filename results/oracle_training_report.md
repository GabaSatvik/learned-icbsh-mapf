# CL-ICBS-h Oracle Training & Cross-Validation Report

> Generated automatically by `mapf/train_cl_ranker.py`

---

## 1. Executive Training Summary

| Metric | Value |
|---|---|
| Training Samples (Candidate Rows) | `6,789` |
| Decision Queries | `2,251` |
| Features Used | `24` |
| 5-Fold CV XGBRanker (Overall Mean ρ) | `+0.4578 (±0.0479)` |
| 5-Fold CV LinearRanker (Ridge) | `+0.2589` |
| 5-Fold CV RandomRanker (Control) | `+-0.0459` |
| Hard Analytical Tie Mean ρ (is_analytical_tie == 1) | `+0.4480` |
| Easy Query Mean ρ (is_analytical_tie == 0) | `+0.7994` |

---

## 2. 5-Fold Cross-Validation Performance Breakdown

| Fold | XGBRanker (Overall) | XGBRanker (Hard Ties) | XGBRanker (Easy) | Linear (Ridge) | Random (Control) |
|---|---:|---:|---:|---:|---:|
| Fold 1 | `+0.4359` | `+0.4222` | `+0.9556` | `+0.1845` | `+-0.0413` |
| Fold 2 | `+0.3952` | `+0.3893` | `+0.6831` | `+0.2715` | `+-0.0847` |
| Fold 3 | `+0.4773` | `+0.4655` | `+0.9500` | `+0.2574` | `+-0.0014` |
| Fold 4 | `+0.5380` | `+0.5333` | `+0.6905` | `+0.3342` | `+-0.0254` |
| Fold 5 | `+0.4424` | `+0.4294` | `+0.7176` | `+0.2469` | `+-0.0767` |

---

## 3. Zero-Shot Cross-Map Generalization Transfer

| Train Map Topology | Held-Out Test Topology | Spearman ρ |
|---|---|---:|
| `Paris_1_256` | Held-Out Maps | `+0.1917` |
| `random-32-32-20` | Held-Out Maps | `+0.2805` |
| `room-32-32-4` | Held-Out Maps | `+0.3372` |
| `warehouse-32-32-aisles` | Held-Out Maps | `+0.2217` |

---

## 4. 7-Family Feature Ablation Study

| Feature Family Removed | Excluded Features | 5-Fold CV Spearman ρ | Δ vs Full Model | Impact |
|---|---|---:|---:|---|
| **Full 24-Feature Model** | *(None)* | `+0.4578` | `0.0000` | Reference Baseline |
| `no_corridor` | ['corridor_score_i', 'corridor_score_j'] | `+0.4599` | `+0.0022` | Neutral / Redundant |
| `no_congestion` | ['local_density_i', 'local_density_j'] | `+0.4482` | `-0.0096` | Minor Signal |
| `no_conflict_frequency` | ['pair_conflict_count'] | `+0.4217` | `-0.0361` | Critical Loss |
| `no_goal_distance` | ['goal_dist_i', 'goal_dist_j'] | `+0.4460` | `-0.0118` | Critical Loss |
| `no_affected_agents` | ['num_agents_in_conflict_graph', 'cardinal_degree_i', 'cardinal_degree_j', 'mvc_vertex_cover_membership'] | `+0.4488` | `-0.0090` | Minor Signal |
| `no_downstream` | ['downstream_conflict_count_i', 'downstream_conflict_count_j', 'marginal_conflict_delta_min'] | `+0.3989` | `-0.0589` | Critical Loss |
| `no_cached_lookahead` | ['delta_g_i', 'delta_g_j', 'delta_g_min', 'delta_g_max', 'delta_f_res'] | `+0.4229` | `-0.0349` | Critical Loss |
