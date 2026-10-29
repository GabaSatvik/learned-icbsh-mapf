"""
24-Feature extraction module for Learning-to-Rank conflict prioritization in CL-ICBS-h.
Covers all 6 statement-named feature families + baseline state context (18 features)
plus 6 zero-A*-cost Cached-Lookahead features (24 total features).
"""

from collections import deque
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from mapf.grid import GridMap
from mapf.low_level import get_location, get_path_cost, compute_bfs_h_table
from mapf.conflicts import detect_all_conflicts, count_single_agent_conflicts, count_single_agent_conflicts
from mapf.cardinal import ClassifiedConflict
from mapf.conflict_graph import CardinalConflictGraph, compute_residual_mvc_size

BASE_FEATURE_NAMES = [
    # Family 1: Corridor structure (Option c: Distance-Transform Value to nearest obstacle, clipped [0, 5])
    "corridor_score_i",
    "corridor_score_j",
    # Family 2: Congestion (Number of other agents within Manhattan radius <= 2 at timestep t)
    "local_density_i",
    "local_density_j",
    # Family 3: Conflict frequency (Historical collision count for pair (i, j) along CT branch)
    "pair_conflict_count",
    # Family 4: Goal distance (Exact shortest-path distance from conflict cell to each agent's goal)
    "goal_dist_i",
    "goal_dist_j",
    # Family 5: Affected agents (Vertices in G_CF and degrees of agent_i, agent_j in G_CF)
    "num_agents_in_conflict_graph",
    "cardinal_degree_i",
    "cardinal_degree_j",
    # Family 6: Downstream conflicts (Conflicts in 1-step lookahead)
    "downstream_conflict_count_i",
    "downstream_conflict_count_j",
    # Baseline context features
    "conflict_type",          # 2 = cardinal, 1 = semi-cardinal, 0 = non-cardinal
    "mvc_size",               # Current node h(N) = MVC(G_CF)
    "conflict_time",          # Timestep t of conflict
    "is_edge_conflict",       # 1 if EDGE conflict, 0 if VERTEX conflict
    "agent_i_path_length",    # Cost of agent_i current path
    "agent_j_path_length",    # Cost of agent_j current path
]

# Full 24-feature suite for CL-ICBS-h
FEATURE_NAMES = BASE_FEATURE_NAMES + [
    "delta_g_i",                  # Exact 1-step path cost increase for agent i
    "delta_g_j",                  # Exact 1-step path cost increase for agent j
    "delta_g_min",                # min(delta_g_i, delta_g_j)
    "delta_g_max",                # max(delta_g_i, delta_g_j)
    "delta_f_res",                # Guaranteed 1-step residual f-progress min(dg_i + dh_res_i, dg_j + dh_res_j)
    "marginal_conflict_delta_min",# min(marginal_conflict_delta_i, marginal_conflict_delta_j)
]

EXTENDED_FEATURE_NAMES = FEATURE_NAMES


class FeatureExtractor:
    def __init__(self, grid: GridMap, goals: Optional[List[Tuple[int, int]]] = None):
        self.grid = grid
        self.goals = goals or []
        self.dist_transform = self._compute_obstacle_distance_transform()
        self.goal_dist_tables: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        for g in self.goals:
            if g not in self.goal_dist_tables:
                self.goal_dist_tables[g] = compute_bfs_h_table(self.grid, g)

    def _compute_obstacle_distance_transform(self) -> np.ndarray:
        """
        Option (c) Corridor Feature: Multi-source BFS from all obstacle cells and map borders.
        Clipped to [0, 5]. Narrow 1-cell corridors get value 1; open areas get up to 5.
        """
        H, W = self.grid.height, self.grid.width
        dist = np.full((H, W), 5, dtype=np.int32)
        queue = deque()

        for r in range(H):
            for c in range(W):
                if not self.grid.is_passable(r, c):
                    dist[r, c] = 0
                    queue.append((r, c))
                elif r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    dist[r, c] = 1
                    queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            if d >= 5:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))

        return np.clip(dist, 0, 5)

    def _get_goal_dist(self, goal: Tuple[int, int], pos: Tuple[int, int]) -> int:
        if goal not in self.goal_dist_tables:
            self.goal_dist_tables[goal] = compute_bfs_h_table(self.grid, goal)
        table = self.goal_dist_tables[goal]
        return table.get(pos, abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    def extract_all(
        self, classified: List[ClassifiedConflict], context: Dict[str, Any]
    ) -> np.ndarray:
        paths = context["paths"]
        goals = context.get("goals", self.goals)
        G_CF = context["G_CF"]
        mvc_size = context.get("mvc_size", 0)
        pair_counts = context.get("pair_counts", {})
        return self.extract_conflict_features(
            classified, paths, G_CF, mvc_size, pair_counts, goals=goals, include_extended=True
        )

    def extract_conflict_features(
        self,
        classified: List[ClassifiedConflict],
        paths: List[List[Tuple[int, int]]],
        G_CF: CardinalConflictGraph,
        mvc_size: int,
        pair_counts: Dict[Tuple[int, int], int],
        goals: Optional[List[Tuple[int, int]]] = None,
        include_extended: bool = True,
    ) -> np.ndarray:
        active_goals = goals if goals is not None else self.goals
        n_cols = len(FEATURE_NAMES) if include_extended else len(BASE_FEATURE_NAMES)
        if not classified:
            return np.zeros((0, n_cols), dtype=np.float32)

        k = len(paths)
        num_cf_agents = G_CF.num_vertices()
        rows = []
        res_mvc_cache: Dict[int, int] = {}

        for cc in classified:
            c = cc.conflict
            i, j = c.agent_i, c.agent_j
            t = c.time
            pos_i = c.loc1
            pos_j = c.loc2 if c.conflict_type.upper() == "EDGE" and c.loc2 is not None else c.loc1

            # 1. Corridor structure
            corridor_i = float(self.dist_transform[pos_i[0], pos_i[1]])
            corridor_j = float(self.dist_transform[pos_j[0], pos_j[1]])

            # 2. Congestion: other agents within Manhattan radius <= 2 at timestep t
            density_i = 0
            density_j = 0
            for a in range(k):
                if a == i or a == j:
                    continue
                pos_a = get_location(paths[a], t)
                if abs(pos_a[0] - pos_i[0]) + abs(pos_a[1] - pos_i[1]) <= 2:
                    density_i += 1
                if abs(pos_a[0] - pos_j[0]) + abs(pos_a[1] - pos_j[1]) <= 2:
                    density_j += 1

            # 3. Conflict frequency
            pair_key = (min(i, j), max(i, j))
            pair_freq = float(pair_counts.get(pair_key, 0))

            # 4. Goal distance
            g_dist_i = float(self._get_goal_dist(active_goals[i], pos_i)) if i < len(active_goals) else 0.0
            g_dist_j = float(self._get_goal_dist(active_goals[j], pos_j)) if j < len(active_goals) else 0.0

            # 5. Affected agents
            deg_i = float(G_CF.get_degree(i))
            deg_j = float(G_CF.get_degree(j))

            # 6. Downstream conflicts
            if cc.alt_total_conflicts_i < 999999:
                downstream_i = float(cc.alt_total_conflicts_i)
            elif cc.alt_path_i is not None:
                # O(k*T) single-agent scan instead of O(k^2*T) full detect_all_conflicts
                _max_len = max(len(p) for p in paths)
                downstream_i = float(count_single_agent_conflicts(paths, i, cc.alt_path_i, _max_len))
            else:
                downstream_i = 50.0

            if cc.alt_total_conflicts_j < 999999:
                downstream_j = float(cc.alt_total_conflicts_j)
            elif cc.alt_path_j is not None:
                # O(k*T) single-agent scan instead of O(k^2*T) full detect_all_conflicts
                _max_len = max(len(p) for p in paths)
                downstream_j = float(count_single_agent_conflicts(paths, j, cc.alt_path_j, _max_len))
            else:
                downstream_j = 50.0

            if cc.classification == "CARDINAL":
                ctype_val = 2.0
            elif cc.classification == "SEMI_CARDINAL":
                ctype_val = 1.0
            else:
                ctype_val = 0.0

            is_edge = 1.0 if c.conflict_type.upper() == "EDGE" else 0.0
            len_i = float(get_path_cost(paths[i]))
            len_j = float(get_path_cost(paths[j]))

            row = [
                corridor_i,
                corridor_j,
                float(density_i),
                float(density_j),
                pair_freq,
                g_dist_i,
                g_dist_j,
                float(num_cf_agents),
                deg_i,
                deg_j,
                downstream_i,
                downstream_j,
                ctype_val,
                float(mvc_size),
                float(t),
                is_edge,
                len_i,
                len_j,
            ]

            if include_extended:
                old_cost_i = len(paths[i]) - 1
                old_cost_j = len(paths[j]) - 1
                dg_i = float(cc.alt_cost_i - old_cost_i) if cc.alt_path_i is not None else 50.0
                dg_j = float(cc.alt_cost_j - old_cost_j) if cc.alt_path_j is not None else 50.0
                dg_i = min(dg_i, 50.0)
                dg_j = min(dg_j, 50.0)
                dg_min = min(dg_i, dg_j)
                dg_max = max(dg_i, dg_j)

                if i not in res_mvc_cache:
                    res_mvc_cache[i] = compute_residual_mvc_size(G_CF, i) if deg_i > 0 else mvc_size
                if j not in res_mvc_cache:
                    res_mvc_cache[j] = compute_residual_mvc_size(G_CF, j) if deg_j > 0 else mvc_size

                df_i = dg_i + float(res_mvc_cache[i] - mvc_size)
                df_j = dg_j + float(res_mvc_cache[j] - mvc_size)
                delta_f_res = min(df_i, df_j)

                mc_i = min(float(cc.marginal_conflict_delta_i), 25.0) if cc.marginal_conflict_delta_i < 999999 else 25.0
                mc_j = min(float(cc.marginal_conflict_delta_j), 25.0) if cc.marginal_conflict_delta_j < 999999 else 25.0
                mc_min = min(mc_i, mc_j)

                row.extend([dg_i, dg_j, dg_min, dg_max, delta_f_res, mc_min])

            rows.append(row)

        return np.array(rows, dtype=np.float32)
