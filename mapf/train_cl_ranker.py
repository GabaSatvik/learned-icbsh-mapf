# mapf/train_cl_ranker.py
"""
Trains and evaluates the CL-ICBS-h Surrogate Ranker:
1. 5-Fold GroupKFold Cross-Validation (XGBRanker vs Linear vs Random)
2. Cross-Map Transfer Validation (Train on Random -> Test on Room, and vice versa)
3. 7-Family Feature Ablation Study (including Cached-Lookahead family)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
try:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBRanker
except ImportError:
    from mapf.baselines import GroupKFold, Ridge, StandardScaler, make_pipeline, XGBRanker
from mapf.features import FEATURE_NAMES


FEATURE_FAMILIES = {
    "corridor": ["corridor_score_i", "corridor_score_j"],
    "congestion": ["local_density_i", "local_density_j"],
    "conflict_frequency": ["pair_conflict_count"],
    "goal_distance": ["goal_dist_i", "goal_dist_j"],
    "affected_agents": [
        "num_agents_in_conflict_graph",
        "cardinal_degree_i",
        "cardinal_degree_j",
        "mvc_vertex_cover_membership",
    ],
    "downstream": [
        "downstream_conflict_count_i",
        "downstream_conflict_count_j",
        "marginal_conflict_delta_min",
    ],
    "cached_lookahead": [
        "delta_g_i",
        "delta_g_j",
        "delta_g_min",
        "delta_g_max",
        "delta_f_res",
    ],
}


def compute_mean_query_spearman(df_test: pd.DataFrame, preds: np.ndarray) -> float:
    from scipy.stats import rankdata
    qids = df_test["query_id"].values
    y_all = df_test["oracle_score"].values
    p_all = np.asarray(preds, dtype=np.float64)
    _, idx_start, counts = np.unique(qids, return_index=True, return_counts=True)
    order = np.argsort(idx_start)
    idx_start = idx_start[order]
    counts = counts[order]
    corrs = []
    for st, cnt in zip(idx_start, counts):
        if cnt < 2:
            continue
        yt = y_all[st : st + cnt]
        yp = p_all[st : st + cnt]
        if np.std(yt) > 1e-7 and np.std(yp) > 1e-7:
            rt = rankdata(yt)
            rp = rankdata(yp)
            rt_c = rt - np.mean(rt)
            rp_c = rp - np.mean(rp)
            denom = np.sqrt(np.sum(rt_c ** 2) * np.sum(rp_c ** 2))
            if denom > 1e-9:
                corrs.append(float(np.sum(rt_c * rp_c) / denom))
    return float(np.mean(corrs)) if corrs else 0.0


def build_xgb_ranker() -> XGBRanker:
    return XGBRanker(
        objective="rank:pairwise",
        n_estimators=80,
        max_depth=3,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=4,
    )


def train_and_evaluate(
    csv_path: str = "data/cl_oracle_training.csv",
    model_out: str = "models/ranker_cl_oracle.json",
    linear_out: str = "models/linear_cl_oracle.pkl",
    summary_out: str = "results/phase3_training_summary.json",
):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    print(f"[Train CL-Ranker] Loaded {len(df)} rows ({df['query_id'].nunique()} queries) with {len(feature_cols)} features.")

    df = df.sort_values("query_id").reset_index(drop=True)
    X = df[feature_cols].values
    y = df["relevance"].values
    groups_inst = (df["map_name"].astype(str) + "_k" + df["k"].astype(str) + "_s" + df["scenario_id"].astype(str)).values

    # 1. 5-Fold GroupKFold Cross-Validation
    gkf = GroupKFold(n_splits=5)
    xgb_folds, lin_folds, rand_folds = [], [], []
    rng = np.random.default_rng(42)

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups_inst), 1):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        X_tr, y_tr = df_tr[feature_cols].values, df_tr["relevance"].values
        X_te = df_te[feature_cols].values
        q_sizes_tr = df_tr.groupby("query_id", sort=False).size().values

        ranker = build_xgb_ranker()
        ranker.fit(X_tr, y_tr, group=q_sizes_tr)
        xgb_preds = ranker.predict(X_te)
        xgb_rho = compute_mean_query_spearman(df_te, xgb_preds)
        xgb_folds.append(xgb_rho)

        lin_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        lin_model.fit(X_tr, y_tr)
        lin_preds = lin_model.predict(X_te)
        lin_rho = compute_mean_query_spearman(df_te, lin_preds)
        lin_folds.append(lin_rho)

        rand_preds = rng.uniform(0, 1, size=len(df_te))
        rand_rho = compute_mean_query_spearman(df_te, rand_preds)
        rand_folds.append(rand_rho)

        print(f"  Fold {fold_idx}: XGBRanker={xgb_rho:+.4f} | Linear={lin_rho:+.4f} | Random={rand_rho:+.4f}")

    print(
        f"\n[5-Fold CV Mean] XGBRanker: {np.mean(xgb_folds):+.4f} (±{np.std(xgb_folds):.4f}) | "
        f"Linear: {np.mean(lin_folds):+.4f} | Random: {np.mean(rand_folds):+.4f}"
    )

    # 2. Cross-Map Generalization Test
    cross_map_results = {}
    maps = df["map_name"].unique()
    if len(maps) >= 2:
        for train_map in maps:
            df_tr = df[df["map_name"] == train_map]
            df_te = df[df["map_name"] != train_map]
            if len(df_tr) == 0 or len(df_te) == 0:
                continue
            q_sizes_tr = df_tr.groupby("query_id", sort=False).size().values
            cm_ranker = build_xgb_ranker()
            cm_ranker.fit(df_tr[feature_cols].values, df_tr["relevance"].values, group=q_sizes_tr)
            cm_preds = cm_ranker.predict(df_te[feature_cols].values)
            cm_rho = compute_mean_query_spearman(df_te, cm_preds)
            cross_map_results[f"train_{train_map}_test_other"] = cm_rho
            print(f"[Cross-Map Transfer] Train on {train_map} -> Test on unseen map(s): Spearman = {cm_rho:+.4f}")

    # 3. Feature-Family Ablation Study
    ablation_results = {"full_model": float(np.mean(xgb_folds))}
    for fam_name, fam_cols in FEATURE_FAMILIES.items():
        keep_cols = [c for c in feature_cols if c not in fam_cols]
        if len(keep_cols) == len(feature_cols):
            continue
        fam_rhos = []
        for tr_idx, te_idx in gkf.split(X, y, groups=groups_inst):
            df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
            q_sizes_tr = df_tr.groupby("query_id", sort=False).size().values
            ab_ranker = build_xgb_ranker()
            ab_ranker.fit(df_tr[keep_cols].values, df_tr["relevance"].values, group=q_sizes_tr)
            ab_preds = ab_ranker.predict(df_te[keep_cols].values)
            fam_rhos.append(compute_mean_query_spearman(df_te, ab_preds))
        mean_ab = float(np.mean(fam_rhos))
        delta = mean_ab - ablation_results["full_model"]
        ablation_results[f"no_{fam_name}"] = {"spearman": mean_ab, "delta": delta}
        print(f"  [Ablation] No {fam_name:<20}: Spearman={mean_ab:+.4f} (Delta={delta:+.4f})")

    # 4. Train Final Production Models on Full Dataset
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    full_q_sizes = df.groupby("query_id", sort=False).size().values
    final_xgb = build_xgb_ranker()
    final_xgb.fit(X, y, group=full_q_sizes)
    final_xgb.save_model(model_out)

    final_lin = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    final_lin.fit(X, y)
    with open(linear_out, "wb") as f:
        pickle.dump(final_lin, f)

    importances = {
        col: float(imp) for col, imp in zip(feature_cols, final_xgb.feature_importances_)
    }
    sorted_imp = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    summary = {
        "n_rows": int(len(df)),
        "n_queries": int(df["query_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "cv_xgb_folds": xgb_folds,
        "cv_xgb_mean": float(np.mean(xgb_folds)),
        "cv_linear_mean": float(np.mean(lin_folds)),
        "cv_random_mean": float(np.mean(rand_folds)),
        "cross_map_transfer": cross_map_results,
        "feature_importances": sorted_imp,
        "ablations": ablation_results,
    }
    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Train CL-Ranker] Saved final model to {model_out} and summary to {summary_out}.")


if __name__ == "__main__":
    train_and_evaluate()
