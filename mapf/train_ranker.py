"""
Train Ranker on Proxy Labels + Feature-Family Ablations (Phase 1.4)
Loads data/proxy_training_regime{A|B}.csv, runs 5-fold GroupKFold by (map, k, scenario_id),
trains XGBRanker, LinearRanker, and RandomRanker.
Reports held-out NDCG@3 and Spearman rank correlation.
Performs family ablations across all six statement-named families.
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
    'corridor': ['corridor_score_i', 'corridor_score_j'],
    'congestion': ['local_density_i', 'local_density_j'],
    'conflict_frequency': ['pair_conflict_count'],
    'goal_distance': ['goal_dist_i', 'goal_dist_j'],
    'affected_agents': ['num_agents_in_conflict_graph', 'cardinal_degree_i', 'cardinal_degree_j'],
    'downstream': ['downstream_conflict_count_i', 'downstream_conflict_count_j']
}


class RandomRanker:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.rng.random(len(X))


class LinearRanker:
    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def train_and_evaluate(regime: str = 'B'):
    print(f"=== Phase 1.4: Train Ranker on Proxy Labels (Regime {regime}) ===")
    data_path = f"data/proxy_training_regime{regime}.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} candidate records across {df['query_id'].nunique()} queries.")

    # Filter features that exist in dataframe
    features = [f for f in FEATURE_NAMES if f in df.columns]

    # Create Group key: (map, k, scenario_id)
    df['instance_group'] = df['map'] + "_" + df['k'].astype(str) + "_" + df['scenario_id'].astype(str)
    groups = df['instance_group']

    gkf = GroupKFold(n_splits=5)
    os.makedirs('models', exist_ok=True)

    xgb_spearmans = []
    linear_spearmans = []
    random_spearmans = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['relevance'], groups)):
        train_df = df.iloc[train_idx].sort_values(by=['query_id', 'candidate_index'])
        val_df = df.iloc[val_idx].sort_values(by=['query_id', 'candidate_index'])

        X_train, y_train, qid_train = train_df[features], train_df['relevance'], train_df['query_id']
        X_val, y_val, qid_val = val_df[features], val_df['relevance'], val_df['query_id']

        # 1. XGBRanker
        ranker = xgb.XGBRanker(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            objective="rank:pairwise",
            tree_method="hist",
            eval_metric="ndcg@3",
            random_state=42 + fold,
            n_jobs=-1
        )
        ranker.fit(X_train, y_train, qid=qid_train)
        preds_xgb = ranker.predict(X_val)

        # 2. LinearRanker
        lin_ranker = LinearRanker()
        lin_ranker.fit(X_train, y_train)
        preds_lin = lin_ranker.predict(X_val)

        # 3. RandomRanker
        rnd_ranker = RandomRanker(seed=42 + fold)
        preds_rnd = rnd_ranker.predict(X_val)

        # Compute query-level Spearman on validation fold
        val_df = val_df.copy()
        val_df['pred_xgb'] = preds_xgb
        val_df['pred_lin'] = preds_lin
        val_df['pred_rnd'] = preds_rnd

        q_xgb_sp = []
        q_lin_sp = []
        q_rnd_sp = []

        for qid, qgroup in val_df.groupby('query_id'):
            if len(qgroup) >= 2 and len(qgroup['relevance'].unique()) > 1:
                sp_xgb, _ = spearmanr(qgroup['pred_xgb'], qgroup['relevance'])
                sp_lin, _ = spearmanr(qgroup['pred_lin'], qgroup['relevance'])
                sp_rnd, _ = spearmanr(qgroup['pred_rnd'], qgroup['relevance'])
                if not np.isnan(sp_xgb): q_xgb_sp.append(sp_xgb)
                if not np.isnan(sp_lin): q_lin_sp.append(sp_lin)
                if not np.isnan(sp_rnd): q_rnd_sp.append(sp_rnd)

        xgb_spearmans.append(np.mean(q_xgb_sp) if q_xgb_sp else 0.0)
        linear_spearmans.append(np.mean(q_lin_sp) if q_lin_sp else 0.0)
        random_spearmans.append(np.mean(q_rnd_sp) if q_rnd_sp else 0.0)

    mean_xgb_sp = float(np.mean(xgb_spearmans))
    mean_lin_sp = float(np.mean(linear_spearmans))
    mean_rnd_sp = float(np.mean(random_spearmans))

    print("\n--- 5-Fold GroupKFold Cross-Validation Results ---")
    print(f"  XGBRanker    Mean Held-out Spearman: {mean_xgb_sp:+.4f}")
    print(f"  LinearRanker Mean Held-out Spearman: {mean_lin_sp:+.4f}")
    print(f"  RandomRanker Mean Held-out Spearman: {mean_rnd_sp:+.4f}")

    # Retrain on full dataset and save
    full_df = df.sort_values(by=['query_id', 'candidate_index'])
    final_ranker = xgb.XGBRanker(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective="rank:pairwise",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )
    final_ranker.fit(full_df[features], full_df['relevance'], qid=full_df['query_id'])
    full_model_path = f"models/ranker_proxy_regime{regime}.json"
    final_ranker.save_model(full_model_path)
    print(f"Full model saved to {full_model_path}")

    # Feature Importance
    importances = dict(zip(features, [float(x) for x in final_ranker.feature_importances_]))
    print("\n--- Feature Importances ---")
    for f, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f:30s}: {imp:.4f}")

    # Feature-Family Ablations
    print("\n--- Feature-Family Ablations (Retraining without each family) ---")
    ablation_results = {}
    for fam_name, fam_cols in FEATURE_FAMILIES.items():
        ablated_features = [f for f in features if f not in fam_cols]
        fold_sps = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['relevance'], groups)):
            train_df = df.iloc[train_idx].sort_values(by=['query_id', 'candidate_index'])
            val_df = df.iloc[val_idx].sort_values(by=['query_id', 'candidate_index'])
            abl_ranker = xgb.XGBRanker(
                n_estimators=60, learning_rate=0.05, max_depth=4, objective="rank:pairwise",
                tree_method="hist", random_state=42 + fold, n_jobs=-1
            )
            abl_ranker.fit(train_df[ablated_features], train_df['relevance'], qid=train_df['query_id'])
            preds = abl_ranker.predict(val_df[ablated_features])
            val_df = val_df.copy()
            val_df['pred'] = preds
            sps = []
            for qid, qgroup in val_df.groupby('query_id'):
                if len(qgroup) >= 2 and len(qgroup['relevance'].unique()) > 1:
                    sp, _ = spearmanr(qgroup['pred'], qgroup['relevance'])
                    if not np.isnan(sp): sps.append(sp)
            fold_sps.append(np.mean(sps) if sps else 0.0)

        abl_mean = float(np.mean(fold_sps))
        delta = abl_mean - mean_xgb_sp
        ablation_results[fam_name] = {'held_out_spearman': abl_mean, 'delta_vs_full': delta}
        print(f"  Ablation [no {fam_name:18s}]: Spearman = {abl_mean:+.4f} (Delta = {delta:+.4f})")

        # Save ablated model
        abl_ranker.save_model(f"models/ablation_no_{fam_name}.json")

    summary = {
        'regime': regime,
        'mean_xgb_spearman': mean_xgb_sp,
        'mean_linear_spearman': mean_lin_sp,
        'mean_random_spearman': mean_rnd_sp,
        'feature_importances': importances,
        'ablations': ablation_results
    }
    with open(f"results/phase1_training_summary_regime{regime}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Ablation and training metrics written to results/phase1_training_summary_regime{regime}.json")


if __name__ == "__main__":
    train_and_evaluate(regime='B')
