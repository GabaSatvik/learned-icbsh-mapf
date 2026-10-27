"""
mapf/train_oracle_ranker.py
============================
Trains an XGBRanker on oracle-labeled data from data/mapf_ct_dataset.csv.
Uses y_khalil_binary (top-20% binary label from y_composite oracle) as the training target.

Pipeline:
  1. Load oracle dataset
  2. GroupKFold(5) keyed on (map_name, k, scenario_id) — split by instance, not by row
  3. Train XGBRanker + LinearRanker + RandomRanker controls
  4. Report per-fold held-out Spearman ρ and NDCG@3
  5. Intermediate gate: if mean Spearman < 0.3 on held-out data, report and exit
  6. Feature-family ablation (7 ablations: 6 original families + delta_f_res)
  7. Retrain full model on all data and save to models/oracle_ranker.json

Run: python -m mapf.train_oracle_ranker
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

DATASET_CSV = "data/mapf_ct_dataset.csv"
MODEL_OUT = "models/oracle_ranker.json"
REPORT_OUT = "results/oracle_training_report.md"

# Feature families for ablation
FEATURE_FAMILIES = {
    'corridor':           ['corridor_score_i', 'corridor_score_j'],
    'congestion':         ['local_density_i', 'local_density_j'],
    'conflict_frequency': ['pair_conflict_count'],
    'goal_distance':      ['goal_dist_i', 'goal_dist_j'],
    'affected_agents':    ['num_agents_in_conflict_graph', 'cardinal_degree_i', 'cardinal_degree_j'],
    'downstream':         ['downstream_conflict_count_i', 'downstream_conflict_count_j'],
    'delta_f_res':        ['delta_f_res'],          # new zero-A* residual feature
}

BASELINE_FEATURES = [
    'conflict_type', 'mvc_size', 'conflict_time', 'is_edge_conflict',
    'agent_i_path_length', 'agent_j_path_length', 'is_head_on', 'mdd_width_approx',
]

ALL_FEATURES = (
    BASELINE_FEATURES +
    [f for fam_feats in FEATURE_FAMILIES.values() for f in fam_feats]
)


def compute_query_spearman(
    df: pd.DataFrame, score_col: str, label_col: str = 'y_composite'
) -> Tuple[float, int]:
    """Within-query Spearman ρ between predicted scores and oracle label."""
    rhos = []
    for _, grp in df.groupby('query_id'):
        if len(grp) < 2:
            continue
        a = grp[score_col].values
        b = grp[label_col].values
        if np.std(a) > 1e-8 and np.std(b) > 1e-8:
            r, _ = spearmanr(a, b)
            if not np.isnan(r):
                rhos.append(float(r))
    return (float(np.mean(rhos)) if rhos else float('nan')), len(rhos)


def compute_ndcg_at_k(
    df: pd.DataFrame, score_col: str, label_col: str = 'y_khalil_binary', k: int = 3
) -> float:
    """Mean NDCG@k across all queries."""
    ndcgs = []
    for _, grp in df.groupby('query_id'):
        if len(grp) < 2:
            continue
        order = np.argsort(-grp[score_col].values)
        labels = grp[label_col].values
        ranked_labels = labels[order[:k]]
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ranked_labels))
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_labels))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


class RandomRanker:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return self.rng.random(len(X))


class LinearRanker:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()

    def fit(self, X, y, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


def make_xgb_ranker(n_estimators: int = 200) -> xgb.XGBRanker:
    return xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@3",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )


def train_and_evaluate(
    csv_path: str = DATASET_CSV,
    model_out: str = MODEL_OUT,
    report_out: str = REPORT_OUT,
    n_folds: int = 5,
    spearman_gate: float = 0.3,
) -> Dict:
    print("=" * 60)
    print("Oracle Ranker Training Pipeline")
    print("=" * 60)
    t0 = time.perf_counter()

    # ── Load data ──
    print(f"\n[1] Loading dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset not found. Run: python -m mapf.fast_dataset_builder")
        return {}

    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows | {df['query_id'].nunique():,} queries")

    # Standardize column aliases
    if 'oracle_score' in df.columns and 'y_composite' not in df.columns:
        df['y_composite'] = df['oracle_score']
    if 'binary_label' in df.columns and 'y_khalil_binary' not in df.columns:
        df['y_khalil_binary'] = df['binary_label']
    if 'df_min' in df.columns and 'y_oracle_df_min' not in df.columns:
        df['y_oracle_df_min'] = df['df_min']

    # Filter to rows with valid labels
    df = df.dropna(subset=['y_composite', 'y_khalil_binary'])
    # Only keep queries with >= 2 candidates (otherwise no learning signal)
    query_sizes = df.groupby('query_id').size()
    valid_queries = query_sizes[query_sizes >= 2].index
    df = df[df['query_id'].isin(valid_queries)].copy()
    print(f"  After filtering: {len(df):,} rows | {df['query_id'].nunique():,} queries")

    # Build feature matrix
    avail_features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (will be skipped): {missing}")

    X = df[avail_features].fillna(0.0).values
    y_binary = df['y_khalil_binary'].values.astype(float)
    y_composite = df['y_composite'].values.astype(float)

    # Group keys: (map_name, k, scenario_id) → integer group
    df['_group_key'] = df['map_name'].astype(str) + "_k" + df['k'].astype(str) + "_s" + df['scenario_id'].astype(str)
    group_keys = df['_group_key'].values
    unique_groups = list(dict.fromkeys(group_keys))
    group_int = np.array([unique_groups.index(g) for g in group_keys])

    # XGBoost requires group sizes in order (sorted by qid)
    # We use query_id as the XGBoost group identifier
    df['_qid_int'] = pd.factorize(df['query_id'])[0]
    qid = df['_qid_int'].values

    # ── Cross-validation ──
    print(f"\n[2] GroupKFold({n_folds}) cross-validation keyed on (map, k, scenario_id)...")
    gkf = GroupKFold(n_splits=n_folds)

    fold_results = {
        'XGBRanker':    {'spearman': [], 'ndcg3': []},
        'LinearRanker': {'spearman': [], 'ndcg3': []},
        'RandomRanker': {'spearman': [], 'ndcg3': []},
    }

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_binary, groups=group_int)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_binary[train_idx]
        df_te = df.iloc[test_idx].copy()

        # XGBoost: needs group sizes sorted by qid within training set
        tr_df = df.iloc[train_idx][['_qid_int']].copy()
        tr_df['_orig_idx'] = np.arange(len(train_idx))
        tr_sorted = tr_df.sort_values('_qid_int')
        X_tr_xgb = X_tr[tr_sorted['_orig_idx'].values]
        y_tr_xgb = y_binary[train_idx][tr_sorted['_orig_idx'].values]
        train_qsizes = tr_sorted.groupby('_qid_int').size().values

        xgb_model = make_xgb_ranker(n_estimators=200)
        xgb_model.fit(X_tr_xgb, y_tr_xgb, group=train_qsizes, verbose=False)
        df_te['_xgb_score'] = xgb_model.predict(X_te)

        lin_model = LinearRanker()
        lin_model.fit(X_tr, y_tr)
        df_te['_lin_score'] = lin_model.predict(X_te)

        rand_model = RandomRanker(seed=42 + fold_idx)
        rand_model.fit(X_tr, y_tr)
        df_te['_rand_score'] = rand_model.predict(X_te)

        for model_name, score_col in [
            ('XGBRanker', '_xgb_score'),
            ('LinearRanker', '_lin_score'),
            ('RandomRanker', '_rand_score'),
        ]:
            rho, n = compute_query_spearman(df_te, score_col, 'y_composite')
            ndcg = compute_ndcg_at_k(df_te, score_col, 'y_khalil_binary', k=3)
            fold_results[model_name]['spearman'].append(rho)
            fold_results[model_name]['ndcg3'].append(ndcg)

        xgb_rho = fold_results['XGBRanker']['spearman'][-1]
        print(f"  Fold {fold_idx+1}/{n_folds}: XGB Spearman={xgb_rho:+.4f}, "
              f"NDCG@3={fold_results['XGBRanker']['ndcg3'][-1]:.4f}")

    # ── Report CV results ──
    print("\n[3] Cross-Validation Summary:")
    cv_summary = {}
    for model_name, res in fold_results.items():
        valid_rhos = [r for r in res['spearman'] if not np.isnan(r)]
        mean_rho = float(np.mean(valid_rhos)) if valid_rhos else float('nan')
        std_rho = float(np.std(valid_rhos)) if valid_rhos else float('nan')
        mean_ndcg = float(np.mean(res['ndcg3']))
        cv_summary[model_name] = {
            'mean_spearman': mean_rho, 'std_spearman': std_rho,
            'mean_ndcg3': mean_ndcg, 'n_valid_folds': len(valid_rhos)
        }
        print(f"  {model_name:15s}: Spearman={mean_rho:+.4f}±{std_rho:.4f}, NDCG@3={mean_ndcg:.4f}")

    # ── Intermediate gate ──
    xgb_mean_rho = cv_summary['XGBRanker']['mean_spearman']
    print(f"\n[4] Intermediate Gate: Spearman >= {spearman_gate}")
    if np.isnan(xgb_mean_rho) or xgb_mean_rho < spearman_gate:
        print(f"  [GATE FAILED] XGBRanker Spearman = {xgb_mean_rho:.4f} < {spearman_gate}")
        print(f"  The features are insufficient to predict the oracle ranking.")
        print(f"  Stopping before benchmark phase. See report for feature importances.")
        gate_passed = False
    else:
        print(f"  [GATE PASSED] XGBRanker Spearman = {xgb_mean_rho:.4f} >= {spearman_gate}")
        gate_passed = True

    # ── Feature-family ablation ──
    print("\n[5] Feature-Family Ablation Study...")
    ablation_results = {}

    for fam_name, fam_feats in FEATURE_FAMILIES.items():
        ablated_features = [f for f in avail_features if f not in fam_feats]
        if not ablated_features:
            continue

        X_abl = df[ablated_features].fillna(0.0).values
        fold_rhos_abl = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_abl, y_binary, groups=group_int)):
            X_tr_a, X_te_a = X_abl[train_idx], X_abl[test_idx]
            y_tr_a = y_binary[train_idx]
            df_te_a = df.iloc[test_idx].copy()

            tr_df_a = df.iloc[train_idx][['_qid_int']].copy()
            tr_df_a['_orig_idx'] = np.arange(len(train_idx))
            tr_sorted_a = tr_df_a.sort_values('_qid_int')
            X_tr_abl_sorted = X_tr_a[tr_sorted_a['_orig_idx'].values]
            y_tr_abl_sorted = y_binary[train_idx][tr_sorted_a['_orig_idx'].values]
            train_qsizes_a = tr_sorted_a.groupby('_qid_int').size().values

            # Use same n_estimators for fair ablation comparison (Bug: old code used 60 vs 100)
            abl_model = make_xgb_ranker(n_estimators=200)
            abl_model.fit(X_tr_abl_sorted, y_tr_abl_sorted, group=train_qsizes_a, verbose=False)
            df_te_a['_abl_score'] = abl_model.predict(X_te_a)
            rho, _ = compute_query_spearman(df_te_a, '_abl_score', 'y_composite')
            fold_rhos_abl.append(rho)

        valid_abl = [r for r in fold_rhos_abl if not np.isnan(r)]
        mean_abl = float(np.mean(valid_abl)) if valid_abl else float('nan')
        delta = mean_abl - xgb_mean_rho
        ablation_results[fam_name] = {'mean_spearman': mean_abl, 'delta': delta, 'n_features': len(fam_feats)}
        print(f"  Drop {fam_name:20s}: Spearman={mean_abl:+.4f} (Delta={delta:+.4f})")

    # ── Train full model on all data ──
    print("\n[6] Training full XGBRanker on all data...")
    full_df_sorted = df.sort_values('_qid_int')
    X_full = df[avail_features].fillna(0.0).values[full_df_sorted.index.map(
        lambda i: df.index.get_loc(i) if i in df.index else 0
    )]
    X_full = full_df_sorted[avail_features].fillna(0.0).values
    y_full = full_df_sorted['y_khalil_binary'].values.astype(float)
    full_qsizes = full_df_sorted.groupby('_qid_int').size().values

    full_model = make_xgb_ranker(n_estimators=200)
    full_model.fit(X_full, y_full, group=full_qsizes, verbose=False)

    os.makedirs(os.path.dirname(model_out) if os.path.dirname(model_out) else '.', exist_ok=True)
    full_model.save_model(model_out)
    print(f"  Model saved to {model_out}")

    # Feature importances
    importances = dict(zip(avail_features, full_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # ── Write training report ──
    print(f"\n[7] Writing training report to {report_out}...")
    elapsed = time.perf_counter() - t0

    lines = [
        "# Oracle Ranker Training Report",
        "",
        f"*Generated in {elapsed:.1f}s*",
        "",
        "---",
        "",
        "## 1. Cross-Validation Results",
        "",
        "| Model | Mean Spearman ρ | Std | Mean NDCG@3 |",
        "|---|---:|---:|---:|",
    ]
    for model_name, res in cv_summary.items():
        lines.append(
            f"| {model_name} | `{res['mean_spearman']:+.4f}` | `{res['std_spearman']:.4f}` | `{res['mean_ndcg3']:.4f}` |"
        )

    lines += [
        "",
        f"**Intermediate Gate ({spearman_gate:.1f}):** {'✅ PASSED' if gate_passed else '❌ FAILED'}",
        f"  - XGBRanker held-out Spearman = `{xgb_mean_rho:.4f}`",
        "",
        "---",
        "",
        "## 2. Feature-Family Ablation",
        "",
        "| Family Dropped | Features | Ablated Spearman | Full Spearman | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.append(f"| (Full model) | {len(avail_features)} | — | `{xgb_mean_rho:+.4f}` | — |")
    for fam_name, res in sorted(ablation_results.items(), key=lambda x: x[1]['delta']):
        lines.append(
            f"| `{fam_name}` | {res['n_features']} | `{res['mean_spearman']:+.4f}` "
            f"| `{xgb_mean_rho:+.4f}` | `{res['delta']:+.4f}` |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Feature Importances (Full Model)",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]
    for rank_i, (fname, imp) in enumerate(top_features, 1):
        lines.append(f"| {rank_i} | `{fname}` | `{imp:.4f}` |")

    lines += [
        "",
        "---",
        "",
        "## 4. Training Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Dataset | `{csv_path}` |",
        f"| Rows | `{len(df):,}` |",
        f"| Queries | `{df['query_id'].nunique():,}` |",
        f"| Features | `{len(avail_features)}` |",
        f"| CV Strategy | GroupKFold({n_folds}) by (map, k, scenario) |",
        f"| Training target | `y_khalil_binary` (top-20% by y_composite) |",
        f"| Evaluation target | `y_composite` (continuous oracle) |",
        f"| XGBoost objective | `rank:pairwise` |",
        f"| Model path | `{model_out}` |",
        f"| Total time | `{elapsed:.1f}s` |",
    ]

    os.makedirs(os.path.dirname(report_out) if os.path.dirname(report_out) else '.', exist_ok=True)
    with open(report_out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {report_out}")

    # Also save JSON summary
    summary = {
        'gate_passed': gate_passed,
        'xgb_mean_spearman': xgb_mean_rho,
        'spearman_gate_threshold': spearman_gate,
        'cv_summary': cv_summary,
        'ablation_results': ablation_results,
        'top_features': {k: float(v) for k, v in top_features[:10]},
        'n_rows': len(df),
        'n_queries': int(df['query_id'].nunique()),
        'n_features': len(avail_features),
        'elapsed_seconds': elapsed,
    }
    json_path = report_out.replace('.md', '.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary saved to {json_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DATASET_CSV)
    parser.add_argument("--model-out", default=MODEL_OUT)
    parser.add_argument("--report-out", default=REPORT_OUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gate", type=float, default=0.3)
    args = parser.parse_args()

    result = train_and_evaluate(
        csv_path=args.csv,
        model_out=args.model_out,
        report_out=args.report_out,
        n_folds=args.folds,
        spearman_gate=args.gate,
    )
    if result:
        gate = result.get('gate_passed', False)
        print(f"\n{'[PROCEED TO BENCHMARK]' if gate else '[STOP — GATE FAILED]'}")
