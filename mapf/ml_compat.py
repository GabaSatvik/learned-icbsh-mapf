"""
Pure NumPy/SciPy compatibility layer for GroupKFold, StandardScaler, Ridge, and XGBRanker.
Used automatically when scikit-learn / xgboost are not installed in the runtime environment,
and serializes models to the exact XGBoost 3.4.1 JSON schema.
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


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
        # Sort groups by size descending (exact sklearn GroupKFold algorithm)
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
            tr_idx = np.flatnonzero(~te_mask)
            te_idx = np.flatnonzero(te_mask)
            yield tr_idx, te_idx


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
    Histogram Gradient Boosted Decision Tree Ranker with exact XGBoost 3.4.1 JSON serialization.
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

    def _compute_query_centered_targets(
        self, y: np.ndarray, group: Optional[np.ndarray]
    ) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if group is None:
            return y - np.mean(y)
        out = np.empty_like(y)
        idx = 0
        for g_size in group:
            g_size = int(g_size)
            sl = y[idx : idx + g_size]
            mean_g = np.mean(sl)
            std_g = np.std(sl)
            if std_g > 1e-6:
                out[idx : idx + g_size] = (sl - mean_g) / (std_g + 0.25)
            else:
                out[idx : idx + g_size] = 0.0
            idx += g_size
        return out

    def _fit_single_tree(
        self,
        X: np.ndarray,
        grad: np.ndarray,
        sample_mask: np.ndarray,
        feat_indices: np.ndarray,
        tree_id: int,
        candidate_thresholds: List[np.ndarray],
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape
        # Build a binary tree up to max_depth
        left_children = []
        right_children = []
        parents = []
        split_indices = []
        split_conditions = []
        base_weights = []
        loss_changes = []
        sum_hessian = []
        default_left = []
        split_type = []

        feat_gains = np.zeros(n_features, dtype=np.float64)
        leaf_preds = np.zeros(n_samples, dtype=np.float64)

        def calc_weight(g_sum: float, h_sum: float) -> float:
            if abs(g_sum) <= self.reg_alpha:
                return 0.0
            g_shrunk = g_sum - np.sign(g_sum) * self.reg_alpha
            return float(-g_shrunk / (h_sum + self.reg_lambda))

        def calc_gain(g_sum: float, h_sum: float) -> float:
            if abs(g_sum) <= self.reg_alpha:
                return 0.0
            g_shrunk = g_sum - np.sign(g_sum) * self.reg_alpha
            return float((g_shrunk * g_shrunk) / (h_sum + self.reg_lambda))

        # BFS node queue for building XGBoost node array
        # Each queue item: (node_idx, depth, active_sample_indices_for_fitting, all_sample_indices_for_pred)
        active_root = np.flatnonzero(sample_mask)
        all_root = np.arange(n_samples)

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

        g_root = float(np.sum(grad[active_root]))
        h_root = float(len(active_root))
        root_id = create_node(2147483647, g_root, h_root)
        queue = [(root_id, 0, active_root, all_root, g_root, h_root)]

        while queue:
            nid, depth, act_idx, all_idx, g_tot, h_tot = queue.pop(0)
            if depth >= self.max_depth or len(act_idx) < 8:
                leaf_preds[all_idx] = base_weights[nid]
                continue

            parent_gain = calc_gain(g_tot, h_tot)
            best_gain = 0.0
            best_feat = -1
            best_thresh = 0.0

            for f in feat_indices:
                vals = X[act_idx, f]
                thresh_list = candidate_thresholds[f]
                if len(thresh_list) == 0:
                    continue
                # Vectorized threshold evaluation: (n_active, n_thresh)
                mask_left = vals[:, None] < thresh_list[None, :]
                h_left = np.sum(mask_left, axis=0).astype(np.float64)
                h_right = h_tot - h_left
                valid = (h_left >= 3) & (h_right >= 3)
                if not np.any(valid):
                    continue
                g_left = grad[act_idx] @ mask_left
                g_right = g_tot - g_left

                for t_i in np.flatnonzero(valid):
                    gl = float(g_left[t_i])
                    hl = float(h_left[t_i])
                    gr = float(g_right[t_i])
                    hr = float(h_right[t_i])
                    gain = 0.5 * (calc_gain(gl, hl) + calc_gain(gr, hr) - parent_gain)
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = int(f)
                        best_thresh = float(thresh_list[t_i])

            if best_feat < 0 or best_gain <= 1e-6:
                leaf_preds[all_idx] = base_weights[nid]
                continue

            # Split node nid
            act_left_mask = X[act_idx, best_feat] < best_thresh
            act_left = act_idx[act_left_mask]
            act_right = act_idx[~act_left_mask]

            all_left_mask = X[all_idx, best_feat] < best_thresh
            all_left = all_idx[all_left_mask]
            all_right = all_idx[~all_left_mask]

            gl = float(np.sum(grad[act_left]))
            hl = float(len(act_left))
            gr = float(np.sum(grad[act_right]))
            hr = float(len(act_right))

            left_id = create_node(nid, gl, hl)
            right_id = create_node(nid, gr, hr)

            left_children[nid] = left_id
            right_children[nid] = right_id
            split_indices[nid] = best_feat
            split_conditions[nid] = best_thresh
            loss_changes[nid] = float(best_gain)
            feat_gains[best_feat] += best_gain

            queue.append((left_id, depth + 1, act_left, all_left, gl, hl))
            queue.append((right_id, depth + 1, act_right, all_right, gr, hr))

        tree_dict = {
            "base_weights": base_weights,
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": [],
            "default_left": default_left,
            "id": tree_id,
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
        }
        return tree_dict, leaf_preds, feat_gains

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

        # Precompute 16 quantile candidate thresholds per feature
        candidate_thresholds = []
        quantiles = np.linspace(0.06, 0.94, 15)
        for f in range(n_features):
            u_vals = np.unique(np.quantile(X[:, f], quantiles))
            candidate_thresholds.append(u_vals)

        total_gains = np.zeros(n_features, dtype=np.float64)
        self.trees_ = []

        n_col = max(1, int(round(self.colsample_bytree * n_features)))

        for t_id in range(self.n_estimators):
            # Gradient of pairwise/query-centered squared rank loss: grad = preds - target
            if group is not None:
                pred_centered = self._compute_query_centered_targets(preds, group)
                grad = pred_centered - target
            else:
                grad = preds - target

            sample_mask = rng.random(n_samples) <= self.subsample
            if not np.any(sample_mask):
                sample_mask[:] = True
            feat_indices = np.sort(rng.choice(n_features, size=n_col, replace=False))

            tree_dict, step_preds, step_gains = self._fit_single_tree(
                X, grad, sample_mask, feat_indices, t_id, candidate_thresholds
            )
            self.trees_.append(tree_dict)
            preds += step_preds
            total_gains += step_gains

        gain_sum = float(np.sum(total_gains))
        if gain_sum > 0:
            self.feature_importances_ = (total_gains / gain_sum).astype(np.float32)
        else:
            self.feature_importances_ = np.full(n_features, 1.0 / max(1, n_features), dtype=np.float32)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        preds = np.zeros(n_samples, dtype=np.float64)
        for tree in self.trees_:
            left = tree["left_children"]
            right = tree["right_children"]
            s_idx = tree["split_indices"]
            s_cond = tree["split_conditions"]
            weights = tree["base_weights"]

            nodes = np.zeros(n_samples, dtype=int)
            # Depth is at most max_depth (e.g. 3-6)
            while True:
                lc = np.array([left[n] for n in nodes], dtype=int)
                active = lc != -1
                if not np.any(active):
                    break
                act_rows = np.flatnonzero(active)
                act_nodes = nodes[act_rows]
                f_cols = np.array([s_idx[n] for n in act_nodes], dtype=int)
                conds = np.array([s_cond[n] for n in act_nodes], dtype=np.float64)
                x_vals = X[act_rows, f_cols]
                go_left = x_vals < conds
                next_nodes = np.where(
                    go_left,
                    [left[n] for n in act_nodes],
                    [right[n] for n in act_nodes],
                )
                nodes[act_rows] = next_nodes
            preds += np.array([weights[n] for n in nodes], dtype=np.float64)
        return preds

    def save_model(self, filepath: str, feature_names: Optional[List[str]] = None):
        from mapf.features import FEATURE_NAMES
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
        return self
