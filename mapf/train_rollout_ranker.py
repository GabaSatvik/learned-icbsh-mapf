"""
Train Ranker on REAL Rollout Labels (Phase 2.1)
===============================================
Loads data/rollout_training_regimeB.csv (real subtree rollout labels, B_rollout=1500)
Runs 5-fold GroupKFold cross-validation.
Saves final model to models/ranker_rollout_regimeB.json
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
import xgboost as xgb

from mapf.features import FEATURE_NAMES

FEATURE_FAMILIES = {
    'corridor'          : ['corridor_score_i', 'corridor_score_j'],
    'congestion'        : ['local_density_i', 'local_density_j'],
    'conflict_frequency': ['pair_conflict_count'],
    'goal_distance'     : ['goal_dist_i', 'goal_dist_j'],
    'affected_agents'   : ['num_agents_in_conflict_graph', 'cardinal_degree_i', 'cardinal_degree_j'],
    'downstream'        : ['downstream_conflict_count_i', 'downstream_conflict_count_j'],
}


class RandomRanker:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    def predict(self, X):
        return self.rng.random(len(X))


class LinearRanker:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)


def train_rollout_ranker():
    print("=" * 60)
    print("Phase 2.1: Train XGBRanker on REAL Rollout Labels")
    print("=" * 60)

    data_path = "data/rollout_training_regimeB.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run rollout_labeler.py first.")
        return

    df = pd.read_csv(data_path)
    n_queries = df['query_id'].nunique()
    print(f"Loaded {len(df)} candidate rows across {n_queries} queries.")

    # Drop tied queries (no ranking signal)
    variance = df.groupby('query_id')['relevance'].nunique()
    untied_qids = variance[variance > 1].index
    df_clean = df[df['query_id'].isin(untied_qids)].copy()
    dropped = n_queries - df_clean['query_id'].nunique()
    print(f"Dropped {dropped} tied queries ? {df_clean['query_id'].nunique()} usable queries remain.")
    print(f"Usable rows: {len(df_clean)}")

    features = [f for f in FEATURE_NAMES if f in df_clean.columns]
    print(f"Features   : {len(features)}")

    # Group by MAPF instance (not query) to avoid leakage
    df_clean['instance_group'] = (
        df_clean['map'] + "_" +
        df_clean['k'].astype(str) + "_" +
        df_clean['scenario_id'].astype(str)
    )
    groups = df_clean['instance_group']
    n_instances = groups.nunique()
    print(f"Instances  : {n_instances}")

    n_splits = min(5, n_instances)
    if n_instances < 2:
        print("ERROR: Not enough instances for cross-validation.")
        return
    print(f"CV folds   : {n_splits}")
    print()

    gkf = GroupKFold(n_splits=n_splits)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    xgb_spearmans, lin_spearmans, rnd_spearmans = [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_clean, df_clean['relevance'], groups)):
        train_df = df_clean.iloc[train_idx].sort_values(['query_id', 'candidate_index'])
        val_df   = df_clean.iloc[val_idx].sort_values(['query_id', 'candidate_index'])

        X_tr, y_tr = train_df[features], train_df['relevance']
        X_va, y_va = val_df[features],   val_df['relevance']

        # ---- XGBRanker ----
        ranker = xgb.XGBRanker(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            objective='rank:pairwise',
            tree_method='hist',
            eval_metric='ndcg@3',
            random_state=42 + fold,
            n_jobs=-1
        )
        ranker.fit(X_tr, y_tr, qid=train_df['query_id'])
        p_xgb = ranker.predict(X_va)

        # ---- LinearRanker ----
        lin = LinearRanker()
        lin.fit(X_tr, y_tr)
        p_lin = lin.predict(X_va)

        # ---- RandomRanker ----
        rnd = RandomRanker(seed=42 + fold)
        p_rnd = rnd.predict(X_va)

        # Query-level Spearman
        val_df = val_df.copy()
        val_df['p_xgb'] = p_xgb
        val_df['p_lin'] = p_lin
        val_df['p_rnd'] = p_rnd

        sp_xgb_fold, sp_lin_fold, sp_rnd_fold = [], [], []
        for qid, grp in val_df.groupby('query_id'):
            if len(grp) >= 2 and grp['relevance'].nunique() > 1:
                sx, _ = spearmanr(grp['p_xgb'], grp['relevance'])
                sl, _ = spearmanr(grp['p_lin'], grp['relevance'])
                sr, _ = spearmanr(grp['p_rnd'], grp['relevance'])
                if not np.isnan(sx): sp_xgb_fold.append(sx)
                if not np.isnan(sl): sp_lin_fold.append(sl)
                if not np.isnan(sr): sp_rnd_fold.append(sr)

        f_xgb = np.mean(sp_xgb_fold) if sp_xgb_fold else 0.0
        f_lin = np.mean(sp_lin_fold) if sp_lin_fold else 0.0
        f_rnd = np.mean(sp_rnd_fold) if sp_rnd_fold else 0.0
        xgb_spearmans.append(f_xgb)
        lin_spearmans.append(f_lin)
        rnd_spearmans.append(f_rnd)
        print(f"  Fold {fold+1}: XGB={f_xgb:+.4f}  Lin={f_lin:+.4f}  Rnd={f_rnd:+.4f}  "
              f"(val queries={len(sp_xgb_fold)})")

    mean_xgb = float(np.mean(xgb_spearmans))
    mean_lin = float(np.mean(lin_spearmans))
    mean_rnd = float(np.mean(rnd_spearmans))

    print()
    print("--- 5-Fold Cross-Validation Summary (Rollout Labels) ---")
    print(f"  XGBRanker    Mean Spearman: {mean_xgb:+.4f}  << main metric")
    print(f"  LinearRanker Mean Spearman: {mean_lin:+.4f}")
    print(f"  RandomRanker Mean Spearman: {mean_rnd:+.4f}")
    print(f"  XGB lift over Random      : {mean_xgb - mean_rnd:+.4f}")

    # Retrain on full clean dataset
    print()
    print("Retraining final model on all usable data...")
    full_sorted = df_clean.sort_values(['query_id', 'candidate_index'])
    final_ranker = xgb.XGBRanker(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        objective='rank:pairwise',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    final_ranker.fit(full_sorted[features], full_sorted['relevance'], qid=full_sorted['query_id'])
    model_path = "models/ranker_rollout_regimeB.json"
    final_ranker.save_model(model_path)
    print(f"Model saved -> {model_path}")

    # Feature importances
    importances = dict(zip(features, [float(x) for x in final_ranker.feature_importances_]))
    print()
    print("--- Feature Importances (Rollout-trained) ---")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        bar = '#' * int(imp * 30)
        print(f"  {feat:35s}: {imp:.4f}  {bar}")

    # Feature-family ablations
    print()
    print("--- Feature-Family Ablations ---")
    ablation_results = {}
    for fam_name, fam_cols in FEATURE_FAMILIES.items():
        abl_feats = [f for f in features if f not in fam_cols]
        fold_sps = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df_clean, df_clean['relevance'], groups)):
            tr = df_clean.iloc[train_idx].sort_values(['query_id', 'candidate_index'])
            va = df_clean.iloc[val_idx].sort_values(['query_id', 'candidate_index'])
            ar = xgb.XGBRanker(
                n_estimators=100, learning_rate=0.05, max_depth=4,
                objective='rank:pairwise', tree_method='hist',
                random_state=42 + fold, n_jobs=-1
            )
            ar.fit(tr[abl_feats], tr['relevance'], qid=tr['query_id'])
            preds = ar.predict(va[abl_feats])
            va = va.copy()
            va['pred'] = preds
            sps = []
            for qid, grp in va.groupby('query_id'):
                if len(grp) >= 2 and grp['relevance'].nunique() > 1:
                    sp, _ = spearmanr(grp['pred'], grp['relevance'])
                    if not np.isnan(sp): sps.append(sp)
            fold_sps.append(np.mean(sps) if sps else 0.0)

        abl_mean = float(np.mean(fold_sps))
        delta = abl_mean - mean_xgb
        ablation_results[fam_name] = {'spearman': abl_mean, 'delta': delta}
        print(f"  No {fam_name:20s}: Spearman={abl_mean:+.4f}  D={delta:+.4f}")
        ar.save_model(f"models/rollout_ablation_no_{fam_name}.json")

    # Save summary
    summary = {
        'data_source'          : 'rollout_training_regimeB.csv',
        'b_rollout'            : 1500,
        't_rollout'            : 2.0,
        'total_rows'           : len(df),
        'usable_queries'       : int(df_clean['query_id'].nunique()),
        'dropped_tied_queries' : int(dropped),
        'mean_xgb_spearman'    : mean_xgb,
        'mean_lin_spearman'    : mean_lin,
        'mean_rnd_spearman'    : mean_rnd,
        'xgb_lift_over_random' : mean_xgb - mean_rnd,
        'feature_importances'  : importances,
        'ablations'            : ablation_results,
    }
    summary_path = "results/phase2_training_summary_rollout.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Full summary -> {summary_path}")
    print("=" * 60)
    print("Phase 2.1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    train_rollout_ranker()
