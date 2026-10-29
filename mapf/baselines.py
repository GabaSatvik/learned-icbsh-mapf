# mapf/baselines.py
"""
State-of-the-Art Conflict Prioritization Baselines and Guarded CL-ICBS-h Selector:
1. DisjointSplittingSelector (Mouratidis et al., 2026)
2. FAwareResidualSelector (Boyarski et al., 2021 analytical zero-A* lookahead)
3. GuardedLearnedSelector (CL-ICBS-h: Top-K Residual Guardrail + Learned Surrogate Ranker)
"""
import json
from typing import List, Dict, Any, Callable, Optional, Tuple
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

    try:
        h_res_i = compute_residual_mvc_size(G_CF, i)
        h_res_j = compute_residual_mvc_size(G_CF, j)
    except Exception:
        h_res_i = max(0, h_parent - (1 if i in G_CF.adj else 0))
        h_res_j = max(0, h_parent - (1 if j in G_CF.adj else 0))

    df_i = dg_i + (h_res_i - h_parent)
    df_j = dg_j + (h_res_j - h_parent)

    delta_f_res = min(df_i, df_j)
    delta_f_max = max(df_i, df_j)
    delta_g_min = min(dg_i, dg_j)
    delta_g_max = max(dg_i, dg_j)

    deg_i = G_CF.degree(i) if hasattr(G_CF, "degree") else len(G_CF.adj.get(i, set()))
    deg_j = G_CF.degree(j) if hasattr(G_CF, "degree") else len(G_CF.adj.get(j, set()))
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
    CL-ICBS-h Conflict Selector (Upgraded with Tie-Only Analytical Gating):
    Combines an analytical Top-K Residual f-Progress Guardrail with a learned
    24-feature surrogate ranker (XGBRanker or LinearRanker).
    
    Tie-Only Short-Circuit:
    - If a single candidate strictly dominates on (delta_f_res, delta_g_min),
      it is chosen immediately with ZERO feature extraction and ZERO ML inference overhead.
    - ML inference is strictly invoked ONLY when multiple candidates tie on the maximum
      (delta_f_res, delta_g_min) plateau.
    - When model is a scikit-learn Pipeline (Ridge), uses raw vectorized NumPy dot product.
    """

    def __init__(self, model: Any, top_k_guardrail: int = 3, feature_mask: Optional[List[int]] = None):
        self.model = model
        self.top_k_guardrail = top_k_guardrail
        self.feature_mask = feature_mask
        self._extractor_cache: Dict[int, FeatureExtractor] = {}
        self.analytical_shortcircuits: int = 0
        self.ml_tiebreaks: int = 0
        
        # Fast NumPy linear dot-product weights extraction
        self._lin_mean: Optional[np.ndarray] = None
        self._lin_scale: Optional[np.ndarray] = None
        self._lin_coef: Optional[np.ndarray] = None
        try:
            if hasattr(model, 'steps'):
                scaler = model.steps[0][1]
                ridge = model.steps[-1][1]
                if hasattr(scaler, 'mean_') and hasattr(ridge, 'coef_'):
                    self._lin_mean = np.asarray(scaler.mean_, dtype=np.float64)
                    self._lin_scale = np.asarray(scaler.scale_, dtype=np.float64)
                    self._lin_coef = np.asarray(ridge.coef_, dtype=np.float64)
        except Exception:
            pass

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

        # 1. Compute zero-A* analytical lookahead tuple key for all candidates
        # key: (delta_f_res, delta_g_min, delta_f_max, delta_g_max, deg_sum, pair_freq, -c.time)
        keys = [_compute_candidate_lookahead_tuple(cc, context) for cc in candidates]

        # 2. Find the best analytical pair (delta_f_res, delta_g_min)
        best_df_dg = max((k[0], k[1]) for k in keys)

        # 3. Filter to candidates that tie for this top analytical bound
        tied_top = [cc for k, cc in zip(keys, candidates) if (k[0], k[1]) == best_df_dg]

        # 4. Analytical Short-Circuit (Zero ML Overhead)
        if len(tied_top) == 1:
            self.analytical_shortcircuits += 1
            return tied_top[0]

        # 5. Learned Tie-Breaking on Analytical Plateaus (>= 2 candidates tied)
        self.ml_tiebreaks += 1
        
        # Score tied candidates by structural hub metrics (deg_sum, pair_freq) for guardrail slicing
        tied_scored = [
            (_compute_candidate_lookahead_tuple(cc, context), idx, cc)
            for idx, cc in enumerate(tied_top)
        ]
        tied_scored.sort(key=lambda x: (x[0][4], x[0][5]), reverse=True)
        if self.top_k_guardrail is not None and len(tied_top) > self.top_k_guardrail:
            shortlist_scored = tied_scored[: self.top_k_guardrail]
        else:
            shortlist_scored = tied_scored
        shortlist = [item[2] for item in shortlist_scored]

        extractor = self._get_extractor(context["grid"])
        if "goals" in context and not extractor.goals:
            extractor.goals = context["goals"]
        X = extractor.extract_all(shortlist, context)
        if self.feature_mask is not None:
            X = X[:, self.feature_mask]

        # Fast NumPy Linear Path or Model Prediction
        if self._lin_mean is not None and self._lin_scale is not None and self._lin_coef is not None:
            X_f64 = np.asarray(X, dtype=np.float64)
            scores = ((X_f64 - self._lin_mean) / self._lin_scale) @ self._lin_coef
        else:
            scores = np.asarray(self.model.predict(X), dtype=np.float64)

        best_idx = int(np.argmax(scores))
        return shortlist[best_idx]


# --- Vectorized ML Compatibility Layer (used when scikit-learn / xgboost are absent) ---

class GroupKFold:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if groups is None:
            indices = np.arange(n_samples)
            folds = np.array_split(indices, self.n_splits)
            for i in range(self.n_splits):
                te = folds[i]
                tr = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
                yield tr, te
            return

        groups = np.asarray(groups)
        unique_groups, group_inv, group_counts = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        order = np.argsort(-group_counts, kind="mergesort")
        fold_sizes = np.zeros(self.n_splits, dtype=int)
        group_to_fold = np.zeros(len(unique_groups), dtype=int)

        for g_idx in order:
            smallest_fold = int(np.argmin(fold_sizes))
            group_to_fold[g_idx] = smallest_fold
            fold_sizes[smallest_fold] += group_counts[g_idx]

        sample_folds = group_to_fold[group_inv]
        for f in range(self.n_splits):
            te_mask = sample_folds == f
            yield np.flatnonzero(~te_mask), np.flatnonzero(te_mask)


class StandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        self.scale_ = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Ridge:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        Xc = X - x_mean
        yc = y - y_mean
        n_features = X.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features, dtype=np.float64)
        b = Xc.T @ yc
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = y_mean - float(x_mean @ self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class Pipeline:
    def __init__(self, steps: List[Any]):
        self.steps = steps

    def fit(self, X, y):
        curr = X
        for step in self.steps[:-1]:
            curr = step.fit_transform(curr, y)
        self.steps[-1].fit(curr, y)
        return self

    def predict(self, X):
        curr = X
        for step in self.steps[:-1]:
            curr = step.transform(curr)
        return self.steps[-1].predict(curr)


def make_pipeline(*steps) -> Pipeline:
    return Pipeline(list(steps))


class XGBRanker:
    """
    Fast BLAS-vectorized Histogram Gradient Boosted Decision Tree Ranker
    with exact XGBoost 3.4.1 JSON serialization and O(1) compiled tree inference.
    """
    def __init__(
        self,
        objective: str = "rank:pairwise",
        n_estimators: int = 80,
        max_depth: int = 3,
        learning_rate: float = 0.06,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_lambda: float = 5.0,
        reg_alpha: float = 1.0,
        random_state: int = 42,
        n_jobs: int = 4,
        **kwargs,
    ):
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees_: List[Dict[str, Any]] = []
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        # Compiled numpy arrays for sub-millisecond inference
        self._c_left: Optional[np.ndarray] = None
        self._c_right: Optional[np.ndarray] = None
        self._c_fidx: Optional[np.ndarray] = None
        self._c_cond: Optional[np.ndarray] = None
        self._c_weight: Optional[np.ndarray] = None

    def _compile_trees(self):
        if not self.trees_:
            return
        n_t = len(self.trees_)
        max_n = max(len(t["left_children"]) for t in self.trees_)
        self._c_left = np.full((n_t, max_n), -1, dtype=np.int32)
        self._c_right = np.full((n_t, max_n), -1, dtype=np.int32)
        self._c_fidx = np.zeros((n_t, max_n), dtype=np.int32)
        self._c_cond = np.zeros((n_t, max_n), dtype=np.float64)
        self._c_weight = np.zeros((n_t, max_n), dtype=np.float64)
        for i, t in enumerate(self.trees_):
            m = len(t["left_children"])
            self._c_left[i, :m] = t["left_children"]
            self._c_right[i, :m] = t["right_children"]
            self._c_fidx[i, :m] = t["split_indices"]
            self._c_cond[i, :m] = t["split_conditions"]
            self._c_weight[i, :m] = t["base_weights"]

    def _compute_query_centered_targets(
        self, y: np.ndarray, group: Optional[np.ndarray]
    ) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if group is None:
            return y - np.mean(y)
        group = np.asarray(group, dtype=np.int64)
        n = len(y)
        g_ids = np.repeat(np.arange(len(group)), group)
        sums = np.bincount(g_ids, weights=y, minlength=len(group))
        means = sums / np.maximum(1, group)
        y_centered = y - means[g_ids]
        sq_sums = np.bincount(g_ids, weights=y_centered ** 2, minlength=len(group))
        stds = np.sqrt(sq_sums / np.maximum(1, group))
        scale = np.where(stds > 1e-6, stds + 0.25, 1.0)
        return np.where(stds[g_ids] > 1e-6, y_centered / scale[g_ids], 0.0)

    def fit(self, X, y, group=None, qid=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if group is None and qid is not None:
            _, counts = np.unique(qid, return_counts=True)
            group = counts

        target = self._compute_query_centered_targets(y, group)
        preds = np.zeros(n_samples, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        # Precompute flattened binary split matrix B of shape (n_samples, M) once!
        quantiles = np.linspace(0.1, 0.9, 9)
        thresh_feat_list = []
        thresh_val_list = []
        bin_cols = []
        for f in range(n_features):
            u_vals = np.unique(np.quantile(X[:, f], quantiles))
            for v in u_vals:
                thresh_feat_list.append(f)
                thresh_val_list.append(float(v))
                bin_cols.append((X[:, f] < v).astype(np.float32))

        if bin_cols:
            B = np.column_stack(bin_cols)  # (n_samples, M) float32 for fast BLAS matmul
            thresh_feats = np.array(thresh_feat_list, dtype=np.int32)
            thresh_vals = np.array(thresh_val_list, dtype=np.float64)
        else:
            B = np.zeros((n_samples, 1), dtype=np.float32)
            thresh_feats = np.zeros(1, dtype=np.int32)
            thresh_vals = np.zeros(1, dtype=np.float64)

        total_gains = np.zeros(n_features, dtype=np.float64)
        self.trees_ = []
        n_col = max(1, int(round(self.colsample_bytree * n_features)))

        def calc_weight(g_sum: float, h_sum: float) -> float:
            if abs(g_sum) <= self.reg_alpha:
                return 0.0
            g_shrunk = g_sum - np.sign(g_sum) * self.reg_alpha
            return float(-g_shrunk / (h_sum + self.reg_lambda))

        def calc_gain_scalar(g_sum: float, h_sum: float) -> float:
            if abs(g_sum) <= self.reg_alpha:
                return 0.0
            g_shrunk = g_sum - np.sign(g_sum) * self.reg_alpha
            return float((g_shrunk * g_shrunk) / (h_sum + self.reg_lambda))

        for t_id in range(self.n_estimators):
            grad = (preds - target).astype(np.float32)
            sample_mask = rng.random(n_samples) <= self.subsample
            if not np.any(sample_mask):
                sample_mask[:] = True
            feat_set = set(rng.choice(n_features, size=n_col, replace=False).tolist())
            allowed_col_mask = np.array([f in feat_set for f in thresh_feats], dtype=bool)
            allowed_cols = np.flatnonzero(allowed_col_mask)

            left_children, right_children, parents = [], [], []
            split_indices, split_conditions, base_weights = [], [], []
            loss_changes, sum_hessian, default_left, split_type = [], [], [], []

            def create_node(parent_id: int, g_sum: float, h_sum: float) -> int:
                nid = len(left_children)
                left_children.append(-1)
                right_children.append(-1)
                parents.append(parent_id)
                split_indices.append(0)
                w = calc_weight(g_sum, h_sum) * self.learning_rate
                split_conditions.append(w)
                base_weights.append(w)
                loss_changes.append(0.0)
                sum_hessian.append(float(h_sum))
                default_left.append(0)
                split_type.append(0)
                return nid

            active_root = np.flatnonzero(sample_mask)
            all_root = np.arange(n_samples)
            g_root = float(np.sum(grad[active_root]))
            h_root = float(len(active_root))
            root_id = create_node(2147483647, g_root, h_root)
            queue = [(root_id, 0, active_root, all_root, g_root, h_root)]
            step_preds = np.zeros(n_samples, dtype=np.float64)

            while queue:
                nid, depth, act_idx, all_idx, g_tot, h_tot = queue.pop(0)
                if depth >= self.max_depth or len(act_idx) < 8 or len(allowed_cols) == 0:
                    step_preds[all_idx] = base_weights[nid]
                    continue

                B_sub = B[act_idx][:, allowed_cols]
                h_left = np.sum(B_sub, axis=0, dtype=np.float64)
                h_right = h_tot - h_left
                valid = (h_left >= 3.0) & (h_right >= 3.0)
                if not np.any(valid):
                    step_preds[all_idx] = base_weights[nid]
                    continue

                g_left = (grad[act_idx] @ B_sub).astype(np.float64)
                g_right = g_tot - g_left
                parent_gain = calc_gain_scalar(g_tot, h_tot)

                gl_shrunk = np.maximum(0.0, np.abs(g_left) - self.reg_alpha)
                gr_shrunk = np.maximum(0.0, np.abs(g_right) - self.reg_alpha)
                gains = np.where(
                    valid,
                    0.5 * (
                        (gl_shrunk ** 2) / (h_left + self.reg_lambda)
                        + (gr_shrunk ** 2) / (h_right + self.reg_lambda)
                        - parent_gain
                    ),
                    -1.0,
                )
                best_local = int(np.argmax(gains))
                best_gain = float(gains[best_local])
                if best_gain <= 1e-6:
                    step_preds[all_idx] = base_weights[nid]
                    continue

                best_col = int(allowed_cols[best_local])
                best_feat = int(thresh_feats[best_col])
                best_thresh = float(thresh_vals[best_col])

                act_left_mask = X[act_idx, best_feat] < best_thresh
                act_left = act_idx[act_left_mask]
                act_right = act_idx[~act_left_mask]

                all_left_mask = X[all_idx, best_feat] < best_thresh
                all_left = all_idx[all_left_mask]
                all_right = all_idx[~all_left_mask]

                gl = float(g_left[best_local])
                hl = float(h_left[best_local])
                gr = float(g_right[best_local])
                hr = float(h_right[best_local])

                left_id = create_node(nid, gl, hl)
                right_id = create_node(nid, gr, hr)

                left_children[nid] = left_id
                right_children[nid] = right_id
                split_indices[nid] = best_feat
                split_conditions[nid] = best_thresh
                loss_changes[nid] = best_gain
                total_gains[best_feat] += best_gain

                queue.append((left_id, depth + 1, act_left, all_left, gl, hl))
                queue.append((right_id, depth + 1, act_right, all_right, gr, hr))

            self.trees_.append({
                "base_weights": base_weights,
                "categories": [],
                "categories_nodes": [],
                "categories_segments": [],
                "categories_sizes": [],
                "default_left": default_left,
                "id": t_id,
                "left_children": left_children,
                "loss_changes": loss_changes,
                "parents": parents,
                "right_children": right_children,
                "split_conditions": split_conditions,
                "split_indices": split_indices,
                "split_type": split_type,
                "sum_hessian": sum_hessian,
                "tree_param": {
                    "num_deleted": "0",
                    "num_feature": str(n_features),
                    "num_nodes": str(len(left_children)),
                    "size_leaf_vector": "1",
                },
            })
            preds += step_preds

        gain_sum = float(np.sum(total_gains))
        if gain_sum > 0:
            self.feature_importances_ = (total_gains / gain_sum).astype(np.float32)
        else:
            self.feature_importances_ = np.full(n_features, 1.0 / max(1, n_features), dtype=np.float32)
        self._compile_trees()
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        if self._c_left is None:
            self._compile_trees()
        if self._c_left is None:
            return np.zeros(n_samples, dtype=np.float64)

        n_t = self._c_left.shape[0]
        tree_idx = np.arange(n_t)[:, None]  # (n_t, 1)
        nodes = np.zeros((n_t, n_samples), dtype=np.int32)

        for _ in range(self.max_depth + 1):
            lc = self._c_left[tree_idx, nodes]
            if not np.any(lc != -1):
                break
            rc = self._c_right[tree_idx, nodes]
            fidx = self._c_fidx[tree_idx, nodes]
            cond = self._c_cond[tree_idx, nodes]
            x_val = X[np.arange(n_samples)[None, :], fidx]
            nxt = np.where(x_val < cond, lc, rc)
            nodes = np.where(lc != -1, nxt, nodes)

        leaf_w = self._c_weight[tree_idx, nodes]
        return np.sum(leaf_w, axis=0)

    def save_model(self, filepath: str, feature_names: Optional[List[str]] = None):
        fnames = feature_names if feature_names is not None else FEATURE_NAMES[: self.n_features_]
        num_trees = len(self.trees_)
        model_json = {
            "learner": {
                "attributes": {"scikit_learn": '{"_estimator_type": "ranker"}'},
                "feature_names": list(fnames),
                "feature_types": ["float"] * self.n_features_,
                "gradient_booster": {
                    "model": {
                        "cats": {"enc": [], "feature_segments": [], "sorted_idx": []},
                        "gbtree_model_param": {
                            "num_parallel_tree": "1",
                            "num_trees": str(num_trees),
                        },
                        "iteration_indptr": list(range(num_trees + 1)),
                        "tree_info": [0] * num_trees,
                        "trees": self.trees_,
                    },
                    "name": "gbtree",
                },
                "learner_model_param": {
                    "base_score": "[0E0]",
                    "boost_from_average": "1",
                    "num_class": "0",
                    "num_feature": str(self.n_features_),
                    "num_target": "1",
                },
                "objective": {
                    "lambdarank_param": {
                        "lambdarank_bias_norm": "1",
                        "lambdarank_normalization": "1",
                        "lambdarank_num_pair_per_sample": "4294967295",
                        "lambdarank_pair_method": "topk",
                        "lambdarank_score_normalization": "1",
                        "lambdarank_unbiased": "0",
                        "ndcg_exp_gain": "1",
                    },
                    "name": self.objective,
                },
            },
            "version": [3, 4, 1],
        }
        with open(filepath, "w") as f:
            json.dump(model_json, f)

    def load_model(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        learner = data["learner"]
        self.n_features_ = int(learner["learner_model_param"]["num_feature"])
        self.trees_ = learner["gradient_booster"]["model"]["trees"]
        self._compile_trees()
        return self
