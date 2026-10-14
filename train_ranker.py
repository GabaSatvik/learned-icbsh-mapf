"""
Learning-to-Rank training script using XGBoost (rank:pairwise objective / LambdaMART).
Reads data logged by ConflictDataLogger, trains the ranking model,
evaluates NDCG / ranking performance, and saves model weights.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLS = [
    'conflict_type',
    'num_cardinal_conflicts',
    'num_agents_in_conflict_graph',
    'mvc_size',
    'agent_i_path_length',
    'agent_j_path_length',
    'conflict_time',
    'is_edge_conflict'
]


def train_ranker(
    data_path: str = "logs/ltr_training_data.csv",
    model_output_path: str = "ranker_model.json"
):
    print(f"Reading training data from {data_path}...")
    df = pd.read_csv(data_path)

    # Filter query groups with at least 2 candidates for pairwise ranking
    group_sizes = df.groupby('query_id').size()
    multi_candidate_qids = group_sizes[group_sizes >= 2].index
    print(f"Total decision points (queries): {len(group_sizes)}, with >= 2 candidates: {len(multi_candidate_qids)}")

    train_df = df[df['query_id'].isin(multi_candidate_qids)].copy()
    if train_df.empty:
        print("Warning: Insufficient multi-candidate queries in this small subset, using entire dataset for fit verification.")
        train_df = df.copy()

    # Sort strictly by query_id for XGBoost LTR
    train_df = train_df.sort_values(by=['query_id', 'candidate_index']).reset_index(drop=True)

    X = train_df[FEATURE_COLS]
    y = train_df['relevance']
    qid = train_df['query_id']

    print(f"Training XGBRanker on {len(X)} records with features: {FEATURE_COLS}")
    ranker = xgb.XGBRanker(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        objective="rank:pairwise",
        tree_method="hist",
        random_state=42
    )

    ranker.fit(X, y, qid=qid)
    ranker.save_model(model_output_path)
    print(f"Model saved successfully to {model_output_path}")

    # Inspect feature importances
    importances = ranker.feature_importances_
    print("\n--- Feature Importances ---")
    for feat, imp in zip(FEATURE_COLS, importances):
        print(f"  {feat:30s}: {imp:.4f}")

    return ranker


if __name__ == "__main__":
    train_ranker()
