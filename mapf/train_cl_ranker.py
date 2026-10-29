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
    # Fallback dataset loading: benchmark dataset first, then cl_oracle_training
    _primary = "data/mapf_ct_benchmark_dataset.csv"
    if not os.path.exists(csv_path) and os.path.exists(_primary):
        print(f"[Train CL-Ranker] {csv_path} not found, using {_primary}.")
        csv_path = _primary
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    print(f"[Train CL-Ranker] Loaded {len(df)} rows ({df['query_id'].nunique()} queries) with {len(feature_cols)} features.")

    df = df.sort_values("query_id").reset_index(drop=True)
    X = df[feature_cols].values
    # Use relevance column if available, else fall back to oracle_score
    _target_col = "relevance" if "relevance" in df.columns else "oracle_score"
    print(f"[Train CL-Ranker] Using '{_target_col}' as training target.")
    y = df[_target_col].values
    # Ensure compute_mean_query_spearman uses the correct target column
    if _target_col != "oracle_score" and "oracle_score" not in df.columns:
        df["oracle_score"] = df[_target_col]
    groups_inst = (df["map_name"].astype(str) + "_k" + df["k"].astype(str) + "_s" + df["scenario_id"].astype(str)).values

    # 1. 5-Fold GroupKFold Cross-Validation
    gkf = GroupKFold(n_splits=5)
    xgb_folds, lin_folds, rand_folds = [], [], []
    xgb_hard_folds, xgb_easy_folds = [], []
    rng = np.random.default_rng(42)

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups_inst), 1):
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
        X_tr, y_tr = df_tr[feature_cols].values, df_tr[_target_col].values
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

        # Hard analytical tie / easy query breakdown
        _has_tie_col = "is_analytical_tie" in df_te.columns
        if _has_tie_col:
            df_te_hard = df_te[df_te["is_analytical_tie"] == 1]
            df_te_easy = df_te[df_te["is_analytical_tie"] == 0]
            if len(df_te_hard) >= 10:
                hard_rho = compute_mean_query_spearman(df_te_hard, ranker.predict(df_te_hard[feature_cols].values))
            else:
                hard_rho = float("nan")
            if len(df_te_easy) >= 10:
                easy_rho = compute_mean_query_spearman(df_te_easy, ranker.predict(df_te_easy[feature_cols].values))
            else:
                easy_rho = float("nan")
            xgb_hard_folds.append(hard_rho)
            xgb_easy_folds.append(easy_rho)
            print(
                f"  Fold {fold_idx}: XGBRanker={xgb_rho:+.4f} "
                f"(Hard={hard_rho:+.4f} Easy={easy_rho:+.4f}) "
                f"| Linear={lin_rho:+.4f} | Random={rand_rho:+.4f}"
            )
        else:
            print(f"  Fold {fold_idx}: XGBRanker={xgb_rho:+.4f} | Linear={lin_rho:+.4f} | Random={rand_rho:+.4f}")

    # Normalize xgb_folds to handle dict or float entries
    _xgb_overall = [v["overall"] if isinstance(v, dict) else v for v in xgb_folds]
    _xgb_hard = [v["hard"] if isinstance(v, dict) else float("nan") for v in xgb_folds]
    _xgb_easy = [v["easy"] if isinstance(v, dict) else float("nan") for v in xgb_folds]
    _hard_vals = [v for v in _xgb_hard if not np.isnan(v)]
    _easy_vals = [v for v in _xgb_easy if not np.isnan(v)]
    print(
        f"\n[5-Fold CV Mean] XGBRanker: {np.mean(_xgb_overall):+.4f} (±{np.std(_xgb_overall):.4f}) | "
        f"Linear: {np.mean(lin_folds):+.4f} | Random: {np.mean(rand_folds):+.4f}"
    )
    if _hard_vals:
        print(f"  Hard Analytical Tie Mean Spearman: {np.mean(_hard_vals):+.4f}")
    if _easy_vals:
        print(f"  Easy Query Mean Spearman:          {np.mean(_easy_vals):+.4f}")

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
            cm_ranker.fit(df_tr[feature_cols].values, df_tr[_target_col].values, group=q_sizes_tr)
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
            ab_ranker.fit(df_tr[keep_cols].values, df_tr[_target_col].values, group=q_sizes_tr)
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
        "cv_xgb_hard_mean": float(np.nanmean(xgb_hard_folds)) if xgb_hard_folds else float("nan"),
        "cv_xgb_easy_mean": float(np.nanmean(xgb_easy_folds)) if xgb_easy_folds else float("nan"),
        "cv_linear_mean": float(np.mean(lin_folds)),
        "cv_random_mean": float(np.mean(rand_folds)),
        "cross_map_transfer": cross_map_results,
        "feature_importances": sorted_imp,
        "ablations": ablation_results,
    }
    os.makedirs(os.path.dirname(summary_out), exist_ok=True)
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    # Write results/oracle_training_report.md
    md_lines = [
        "# CL-ICBS-h Oracle Training & Cross-Validation Report",
        "",
        "> Generated automatically by `mapf/train_cl_ranker.py`",
        "",
        "---",
        "",
        "## 1. Executive Training Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Training Samples (Candidate Rows) | `{len(df):,}` |",
        f"| Decision Queries | `{df['query_id'].nunique():,}` |",
        f"| Features Used | `{len(feature_cols)}` |",
        f"| 5-Fold CV XGBRanker (Overall Mean ρ) | `+{np.mean(xgb_folds):.4f} (±{np.std(xgb_folds):.4f})` |",
        f"| 5-Fold CV LinearRanker (Ridge) | `+{np.mean(lin_folds):.4f}` |",
        f"| 5-Fold CV RandomRanker (Control) | `+{np.mean(rand_folds):.4f}` |",
        f"| Hard Analytical Tie Mean ρ (is_analytical_tie == 1) | `+{float(np.nanmean(xgb_hard_folds)):.4f}` |",
        f"| Easy Query Mean ρ (is_analytical_tie == 0) | `+{float(np.nanmean(xgb_easy_folds)):.4f}` |",
        "",
        "---",
        "",
        "## 2. 5-Fold Cross-Validation Performance Breakdown",
        "",
        "| Fold | XGBRanker (Overall) | XGBRanker (Hard Ties) | XGBRanker (Easy) | Linear (Ridge) | Random (Control) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for idx, (xr, hr, er, lr, rr) in enumerate(zip(xgb_folds, xgb_hard_folds, xgb_easy_folds, lin_folds, rand_folds), 1):
        md_lines.append(f"| Fold {idx} | `+{xr:.4f}` | `+{hr:.4f}` | `+{er:.4f}` | `+{lr:.4f}` | `+{rr:.4f}` |")

    md_lines += [
        "",
        "---",
        "",
        "## 3. Zero-Shot Cross-Map Generalization Transfer",
        "",
        "| Train Map Topology | Held-Out Test Topology | Spearman ρ |",
        "|---|---|---:|",
    ]
    for k_cm, v_cm in cross_map_results.items():
        tr_map = k_cm.replace("train_", "").replace("_test_other", "")
        md_lines.append(f"| `{tr_map}` | Held-Out Maps | `+{v_cm:.4f}` |")

    md_lines += [
        "",
        "---",
        "",
        "## 4. 7-Family Feature Ablation Study",
        "",
        "| Feature Family Removed | Excluded Features | 5-Fold CV Spearman ρ | Δ vs Full Model | Impact |",
        "|---|---|---:|---:|---|",
    ]
    for fam_name, res in ablation_results.items():
        if fam_name == "full_model":
            md_lines.append(f"| **Full 24-Feature Model** | *(None)* | `+{res:.4f}` | `0.0000` | Reference Baseline |")
        else:
            delta_val = res["delta"]
            sp_val = res["spearman"]
            impact_str = "Critical Loss" if delta_val < -0.01 else ("Neutral / Redundant" if delta_val >= 0 else "Minor Signal")
            md_lines.append(f"| `{fam_name}` | {FEATURE_FAMILIES.get(fam_name.replace('no_', ''), [])} | `+{sp_val:.4f}` | `{delta_val:+.4f}` | {impact_str} |")

    md_path = "results/oracle_training_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[Train CL-Ranker] Saved Markdown report to {md_path}.")

    print(f"\n[Train CL-Ranker] Saved final model to {model_out} and summary to {summary_out}.")


if __name__ == "__main__":
    train_and_evaluate()
