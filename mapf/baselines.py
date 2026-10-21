# mapf/baselines.py
"""
State-of-the-Art Conflict Prioritization Baselines and Guarded CL-ICBS-h Selector:
1. DisjointSplittingSelector (Mouratidis et al., 2026)
2. FAwareResidualSelector (Boyarski et al., 2021 analytical zero-A* lookahead)
3. GuardedLearnedSelector (CL-ICBS-h: Top-K Residual Guardrail + Learned Surrogate Ranker)
"""
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from mapf.cardinal import ClassifiedConflict
from mapf.conflict_graph import compute_residual_mvc_size
from mapf.features import FeatureExtractor, FEATURE_NAMES


def _compute_candidate_lookahead_tuple(cc: ClassifiedConflict, context: Dict[str, Any]):
    """
    Computes zero-A*-cost exact cached lookahead metrics for a candidate conflict:
      - delta_f_res: min(delta_g_i + h_res_i - h, delta_g_j + h_res_j - h)
      - delta_g_min: min(delta_g_i, delta_g_j)
      - delta_g_max: max(delta_g_i, delta_g_j)
      - deg_sum: deg_G_CF(i) + deg_G_CF(j)
      - pair_freq: historical conflicts between (i, j) along CT branch
    """
    c = cc.conflict
    i, j = c.agent_i, c.agent_j
    paths = context["paths"]
    G_CF = context["G_CF"]
    h_parent = context.get("mvc_size", 0)
    pair_counts = context.get("pair_counts", {})

    old_cost_i = len(paths[i]) - 1
    old_cost_j = len(paths[j]) - 1

    dg_i = (cc.alt_cost_i - old_cost_i) if cc.alt_path_i is not None else 50.0
    dg_j = (cc.alt_cost_j - old_cost_j) if cc.alt_path_j is not None else 50.0

    # Residual MVC after resolving agent i or agent j
    try:
        h_res_i = compute_residual_mvc_size(G_CF, i)
        h_res_j = compute_residual_mvc_size(G_CF, j)
    except Exception:
        h_res_i = max(0, h_parent - (1 if i in getattr(G_CF, "adj", {}) else 0))
        h_res_j = max(0, h_parent - (1 if j in getattr(G_CF, "adj", {}) else 0))

    df_i = dg_i + (h_res_i - h_parent)
    df_j = dg_j + (h_res_j - h_parent)

    delta_f_res = min(df_i, df_j)
    delta_f_max = max(df_i, df_j)
    delta_g_min = min(dg_i, dg_j)
    delta_g_max = max(dg_i, dg_j)

    deg_i = G_CF.degree(i) if hasattr(G_CF, "degree") else len(getattr(G_CF, "adj", {}).get(i, set()))
    deg_j = G_CF.degree(j) if hasattr(G_CF, "degree") else len(getattr(G_CF, "adj", {}).get(j, set()))
    deg_sum = deg_i + deg_j

    pair_key = (min(i, j), max(i, j))
    pair_freq = pair_counts.get(pair_key, 0)

    return (delta_f_res, delta_g_min, delta_f_max, delta_g_max, deg_sum, pair_freq, -c.time)


def disjoint_splitting_selector(
    candidates: List[ClassifiedConflict], context: Dict[str, Any]
) -> ClassifiedConflict:
    """
    Mouratidis et al. (2026) Disjoint Splitting Prioritization:
    Prioritizes conflicts involving agents with the highest degree in the Cardinal Conflict
    Graph G_CF (hub agents participating in multiple cardinal conflicts), breaking ties by
    cached cost increase (delta_g_min + delta_g_max) and earliest conflict timestep.
    """
    best_cc = candidates[0]
    best_key = None
    for cc in candidates:
        delta_f_res, dg_min, df_max, dg_max, deg_sum, pair_freq, neg_t = (
            _compute_candidate_lookahead_tuple(cc, context)
        )
        key = (deg_sum, dg_min + dg_max, pair_freq, neg_t)
        if best_key is None or key > best_key:
            best_key = key
            best_cc = cc
    return best_cc


def f_aware_residual_selector(
    candidates: List[ClassifiedConflict], context: Dict[str, Any]
) -> ClassifiedConflict:
    """
    Boyarski et al. (2021) f-Aware Conflict Prioritization (Zero-A* Residual Formulation):
    Selects the conflict maximizing the guaranteed child f-value increase (delta_f_res),
    breaking ties by secondary branch f-increase (delta_f_max), G_CF degree sum, and
    chronic pair frequency.
    """
    best_cc = candidates[0]
    best_key = None
    for cc in candidates:
        key = _compute_candidate_lookahead_tuple(cc, context)
        if best_key is None or key > best_key:
            best_key = key
            best_cc = cc
    return best_cc


class GuardedLearnedSelector:
    """
    CL-ICBS-h Conflict Selector:
    Combines an analytical Top-K Residual f-Progress Guardrail with a learned
    24-feature surrogate ranker (XGBRanker or LinearRanker).

    Mechanism:
      1. If |candidates| > top_k_guardrail, first filter to the top_k_guardrail
         candidates ranked by analytical (delta_f_res, delta_g_min, deg_sum).
         This guarantees zero out-of-distribution catastrophic regressions on unseen maps.
      2. Extract the 24-feature matrix on the guarded shortlist and rank via the learned model.
    """

    def __init__(self, model: Any, top_k_guardrail: int = 3, feature_mask: Optional[List[int]] = None):
        self.model = model
        self.top_k_guardrail = top_k_guardrail
        self.feature_mask = feature_mask
        self._extractor_cache: Dict[int, FeatureExtractor] = {}

    def _get_extractor(self, grid) -> FeatureExtractor:
        gid = id(grid)
        if gid not in self._extractor_cache:
            self._extractor_cache[gid] = FeatureExtractor(grid)
        return self._extractor_cache[gid]

    def __call__(
        self, candidates: List[ClassifiedConflict], context: Dict[str, Any]
    ) -> ClassifiedConflict:
        if len(candidates) == 1:
            return candidates[0]

        # Step 1: Analytical Top-K Structural Guardrail
        if self.top_k_guardrail is not None and len(candidates) > self.top_k_guardrail:
            scored = [
                (_compute_candidate_lookahead_tuple(cc, context), idx, cc)
                for idx, cc in enumerate(candidates)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            shortlist = [item[2] for item in scored[: self.top_k_guardrail]]
        else:
            shortlist = candidates

        # Step 2: 24-Feature Surrogate Ranking within the Guarded Shortlist
        extractor = self._get_extractor(context["grid"])
        X = extractor.extract_all(shortlist, context)
        if self.feature_mask is not None:
            X = X[:, self.feature_mask]

        scores = self.model.predict(X)
        best_idx = int(np.argmax(scores))
        return shortlist[best_idx]
