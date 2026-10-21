"""
Feature extraction module for Learning-to-Rank conflict prioritization in CBS/ICBS-h.
Supports all 24 features across the 7 feature families:
1. Corridor Structure
2. Congestion
3. Conflict Frequency
4. Goal Distance
5. Affected Agents in Cardinal Conflict Graph
6. Predicted Downstream Conflicts
7. Cached Lookahead & Residual f-Progress
"""
from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np

from mapf.conflicts import Conflict, get_agent_pos_at_t, detect_all_conflicts
from mapf.grid import Grid
from mapf.low_level import manhattan_distance
from mapf.cardinal import ClassifiedConflict
from mapf.conflict_graph import compute_residual_mvc_size


FEATURE_NAMES = [
    # Family 1: Corridor structure
    "corridor_score_i",
    "corridor_score_j",
    # Family 2: Congestion
    "local_density_i",
    "local_density_j",
    # Family 3: Conflict frequency
    "pair_conflict_count",
    # Family 4: Goal distance
    "goal_dist_i",
    "goal_dist_j",
    # Family 5: Number of affected agents
    "num_agents_in_conflict_graph",
    "cardinal_degree_i",
    "cardinal_degree_j",
    "mvc_vertex_cover_membership",
    # Family 6: Predicted downstream conflicts
    "downstream_conflict_count_i",
    "downstream_conflict_count_j",
    "marginal_conflict_delta_min",
    # Family 7: Cached lookahead & residual f-progress
    "delta_g_i",
    "delta_g_j",
    "delta_g_min",
    "delta_g_max",
    "delta_f_res",
    # Baseline features
    "conflict_type",
    "mvc_size",
    "conflict_time",
    "is_edge_conflict",
    "agent_i_path_length",
]


def compute_grid_distance_transform(grid: Grid) -> Dict[Tuple[int, int], int]:
    """Computes Manhattan distance to nearest obstacle for each passable cell."""
    dt: Dict[Tuple[int, int], int] = {}
    queue = []

    for y in range(grid.height):
        for x in range(grid.width):
            if not grid.is_valid(x, y):
                continue
            is_border = (x == 0 or x == grid.width - 1 or y == 0 or y == grid.height - 1)
            has_adj_obs = any(
                not grid.is_valid(nx, ny)
                for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            )
            if is_border or has_adj_obs:
                dt[(x, y)] = 1
                queue.append((x, y))

    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dt[curr]
        for nx, ny in grid.get_neighbors(curr[0], curr[1]):
            if (nx, ny) not in dt:
                dt[(nx, ny)] = d + 1
                queue.append((nx, ny))

    return dt


class FeatureExtractor:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.dt = compute_grid_distance_transform(grid)

    def extract_all(self, classified: List[ClassifiedConflict], context: Dict[str, Any]) -> np.ndarray:
        """
        Extracts 24 features for each candidate conflict.
        Returns np.ndarray of shape (len(classified), 24).
        """
        paths = context["paths"]
        goals = context["goals"]
        G_CF = context.get("G_CF")
        mvc_size = context.get("mvc_size", 0)
        pair_counts = context.get("pair_counts", {})

        num_agents = len(paths)
        num_cg = float(len(G_CF.vertices)) if G_CF and hasattr(G_CF, "vertices") else 0.0

        rows = []
        for cc in classified:
            c = cc.conflict
            i, j = c.agent_i, c.agent_j
            t = c.time
            loc = c.loc_i

            # Family 1: Corridor structure
            corridor_val = float(min(5.0, float(self.dt.get(loc, 1))))
            corridor_i = corridor_val
            corridor_j = corridor_val

            # Family 2: Congestion
            density = 0.0
            for a_idx in range(num_agents):
                if a_idx != i and a_idx != j:
                    pos = get_agent_pos_at_t(paths[a_idx], t)
                    if manhattan_distance(pos, loc) <= 2:
                        density += 1.0
            density_i = density
            density_j = density

            # Family 3: Conflict frequency
            pair_key = (min(i, j), max(i, j))
            pair_freq = float(pair_counts.get(pair_key, 0))

            # Family 4: Goal distance
            g_dist_i = float(manhattan_distance(loc, goals[i])) if i < len(goals) else 0.0
            g_dist_j = float(manhattan_distance(loc, goals[j])) if j < len(goals) else 0.0

            # Family 5: Affected agents
            deg_i = float(G_CF.degree(i)) if G_CF and hasattr(G_CF, "degree") else 0.0
            deg_j = float(G_CF.degree(j)) if G_CF and hasattr(G_CF, "degree") else 0.0
            in_vc = 1.0 if G_CF and hasattr(G_CF, "vertices") and (i in G_CF.vertices or j in G_CF.vertices) else 0.0

            # Family 6: Downstream conflicts
            down_i = 0.0
            if cc.alt_path_i is not None:
                # Fast conflict check against remaining paths
                for other_id in range(num_agents):
                    if other_id != i:
                        p_other = paths[other_id]
                        max_len = max(len(cc.alt_path_i), len(p_other))
                        for step_t in range(1, max_len):
                            pos_i = get_agent_pos_at_t(cc.alt_path_i, step_t)
                            pos_o = get_agent_pos_at_t(p_other, step_t)
                            if pos_i == pos_o:
                                down_i += 1.0
                            elif step_t > 0:
                                prev_i = get_agent_pos_at_t(cc.alt_path_i, step_t - 1)
                                prev_o = get_agent_pos_at_t(p_other, step_t - 1)
                                if prev_i == pos_o and pos_i == prev_o and prev_i != pos_i:
                                    down_i += 1.0
            else:
                down_i = 10.0

            down_j = 0.0
            if cc.alt_path_j is not None:
                for other_id in range(num_agents):
                    if other_id != j:
                        p_other = paths[other_id]
                        max_len = max(len(cc.alt_path_j), len(p_other))
                        for step_t in range(1, max_len):
                            pos_j = get_agent_pos_at_t(cc.alt_path_j, step_t)
                            pos_o = get_agent_pos_at_t(p_other, step_t)
                            if pos_j == pos_o:
                                down_j += 1.0
                            elif step_t > 0:
                                prev_j = get_agent_pos_at_t(cc.alt_path_j, step_t - 1)
                                prev_o = get_agent_pos_at_t(p_other, step_t - 1)
                                if prev_j == pos_o and pos_j == prev_o and prev_j != pos_j:
                                    down_j += 1.0
            else:
                down_j = 10.0

            marginal_delta_min = min(down_i, down_j)

            # Family 7: Cached lookahead & residual f-progress
            old_cost_i = len(paths[i]) - 1
            old_cost_j = len(paths[j]) - 1

            dg_i = (cc.alt_cost_i - old_cost_i) if cc.alt_path_i is not None else 50.0
            dg_j = (cc.alt_cost_j - old_cost_j) if cc.alt_path_j is not None else 50.0

            if str(cc.conflict_type).upper() == "CARDINAL":
                dg_i = max(1.0, dg_i)
                dg_j = max(1.0, dg_j)

            dg_min = min(dg_i, dg_j)
            dg_max = max(dg_i, dg_j)

            try:
                h_res_i = compute_residual_mvc_size(G_CF, i)
                h_res_j = compute_residual_mvc_size(G_CF, j)
            except Exception:
                h_res_i = max(0, mvc_size - (1 if G_CF and i in G_CF.vertices else 0))
                h_res_j = max(0, mvc_size - (1 if G_CF and j in G_CF.vertices else 0))

            df_i = dg_i + (h_res_i - mvc_size)
            df_j = dg_j + (h_res_j - mvc_size)
            delta_f_res = min(df_i, df_j)

            # Baseline features
            ctype_upper = str(cc.conflict_type).upper()
            c_val = 2.0 if ctype_upper == "CARDINAL" else (1.0 if ctype_upper == "SEMI_CARDINAL" else 0.0)
            is_edge = 1.0 if str(c.conflict_type).upper() == "EDGE" else 0.0

            row = [
                corridor_i,
                corridor_j,
                density_i,
                density_j,
                pair_freq,
                g_dist_i,
                g_dist_j,
                num_cg,
                deg_i,
                deg_j,
                in_vc,
                down_i,
                down_j,
                marginal_delta_min,
                dg_i,
                dg_j,
                dg_min,
                dg_max,
                delta_f_res,
                c_val,
                float(mvc_size),
                float(t),
                is_edge,
                float(len(paths[i])),
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def extract_features(
        self,
        conflict: Conflict,
        conflict_class: str,
        all_classified_conflicts: List[Tuple[Conflict, str]],
        paths: List[List[Tuple[int, int]]],
        goals: List[Tuple[int, int]],
        mvc_size: int,
        pair_history: Optional[Dict[Tuple[int, int], int]] = None,
        downstream_counts: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """Legacy compatibility wrapper."""
        a1, a2 = conflict.agent_1, conflict.agent_2
        t = conflict.time
        loc = conflict.loc_1
        corridor = float(min(5, self.dt.get(loc, 1)))
        return {
            'corridor_score_i': corridor,
            'corridor_score_j': corridor,
            'local_density_i': 0.0,
            'local_density_j': 0.0,
            'pair_conflict_count': float(pair_history.get((min(a1, a2), max(a1, a2)), 0) if pair_history else 0),
            'goal_dist_i': float(manhattan_distance(loc, goals[a1])) if a1 < len(goals) else 0.0,
            'goal_dist_j': float(manhattan_distance(loc, goals[a2])) if a2 < len(goals) else 0.0,
            'num_agents_in_conflict_graph': 0.0,
            'cardinal_degree_i': 0.0,
            'cardinal_degree_j': 0.0,
            'downstream_conflict_count_i': float(downstream_counts[0]) if downstream_counts else 0.0,
            'downstream_conflict_count_j': float(downstream_counts[1]) if downstream_counts else 0.0,
            'conflict_type': 2.0 if conflict_class.lower() == 'cardinal' else (1.0 if conflict_class.lower() == 'semi-cardinal' else 0.0),
            'mvc_size': float(mvc_size),
            'conflict_time': float(t),
            'is_edge_conflict': 1.0 if str(conflict.conflict_type).lower() == 'edge' else 0.0,
            'agent_i_path_length': float(len(paths[a1])),
            'agent_j_path_length': float(len(paths[a2]))
        }
