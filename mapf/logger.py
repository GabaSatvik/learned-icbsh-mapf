"""
Data Logger for Learning-to-Rank (LambdaMART/XGBoost rank:pairwise).
Logs feature sets for all candidate conflicts at each Constraint Tree branching decision,
along with outcomes/relevance labels.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional


class ConflictDataLogger:
    """
    Instruments ICBS-h to log decision points where conflict choices are made.
    For Learning-to-Rank:
      - query_id / qid: Unique identifier for each CT node branching point where a choice among candidate conflicts is made.
      - Each candidate conflict is an item under this query_id with its 6 features.
      - relevance_label: Score representing the quality of choosing that conflict.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._current_query_id = 0

    def get_next_query_id(self) -> int:
        self._current_query_id += 1
        return self._current_query_id

    def log_branch_decision(
        self,
        query_id: int,
        candidate_features: List[Dict[str, float]],
        chosen_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        outcomes: Optional[List[float]] = None
    ):
        """
        Logs a set of candidate conflicts evaluated at a single CT node.
        outcomes: optional list of relevance scores (e.g., higher = better choice).
        If outcomes is not given, chosen conflict gets higher relevance (e.g. 2 vs 0).
        """
        num_candidates = len(candidate_features)
        for idx, feat in enumerate(candidate_features):
            is_chosen = (idx == chosen_index)
            # Relevance label for LTR:
            if outcomes is not None and len(outcomes) == num_candidates:
                relevance = outcomes[idx]
            else:
                # Cardinal choice or heuristic chosen conflict gets graded relevance:
                relevance = 2.0 if is_chosen else (1.0 if feat.get('conflict_type', 0.0) == 2.0 else 0.0)

            record = {
                'query_id': query_id,
                'candidate_index': idx,
                'is_chosen': int(is_chosen),
                'relevance': relevance,
                **feat
            }
            if metadata:
                record.update({f"meta_{k}": v for k, v in metadata.items()})

            self.records.append(record)

    def save_to_csv(self, filename: str = "ltr_training_data.csv") -> str:
        filepath = os.path.join(self.log_dir, filename)
        df = pd.DataFrame(self.records)
        if not df.empty:
            # Sort by query_id as strictly required by XGBoost LTR
            df = df.sort_values(by=['query_id', 'candidate_index'])
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records across {df['query_id'].nunique() if not df.empty else 0} decision points to {filepath}")
        return filepath

    def save_to_json(self, filename: str = "ltr_training_data.json") -> str:
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=2)
        return filepath

    def clear(self):
        self.records.clear()
        self._current_query_id = 0
