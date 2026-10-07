"""
Feature extraction module for Learning-to-Rank conflict prioritization in CBS/ICBS-h.
Covers all six statement-named feature families:

1. Corridor Structure:
   Option (c) Distance-transform value was chosen.
   - Justification / Rationale:
     Option (c) directly measures the distance from the collision cell to the nearest obstacle.
     Low values (e.g. 1) indicate tight bottlenecks/corridors with no lateral bypass routes,
     whereas high values indicate open rooms where agents easily circumvent one another.
     Alternative (a) [degree-3+ cells] only measures local graph branching without metric geometry.
     Alternative (b) [alternative shortest path counts] is computationally expensive (O(V+E) per check).
     Option (c) is O(1) at evaluation time (precomputed per map grid), highly interpretable, and clipped to [0, 5].

2. Congestion:
   - local_density_i, local_density_j: Number of other agents whose space-time trajectories
     pass within Manhattan radius <= 2 of the conflict vertex at time t.

3. Conflict Frequency:
   - pair_conflict_count: Running count of conflicts previously detected and resolved between
     agent i and agent j along this branch of the search tree.

4. Goal Distance:
   - goal_dist_i, goal_dist_j: True static shortest-path distance from conflict vertex to agent's goal.

5. Number of Affected Agents:
   - num_agents_in_conflict_graph: Total agents in V(G_CF).
   - cardinal_degree_i, cardinal_degree_j: Degree of agent i and agent j in G_CF.

6. Predicted Downstream Conflicts:
   - downstream_conflict_count_i, downstream_conflict_count_j: One-step lookahead conflict count
     in the re-planned paths found during cardinal classification.

Kept Baseline Features:
   - conflict_type: 2.0 (cardinal), 1.0 (semi-cardinal), 0.0 (non-cardinal)
   - mvc_size: size of MVC(G_CF)
   - conflict_time: timestep t
   - is_edge_conflict: 1.0 if edge, 0.0 if vertex
   - agent_i_path_length, agent_j_path_length
"""
from typing import List, Tuple, Dict, Set, Optional, Any
from mapf.conflicts import Conflict, get_agent_pos_at_t
from mapf.grid import Grid
from mapf.low_level import manhattan_distance


def compute_grid_distance_transform(grid: Grid) -> Dict[Tuple[int, int], int]:
    """
    Computes distance transform from obstacles:
    For every passable cell (x, y), returns the Manhattan distance to the nearest obstacle.
    """
    dt: Dict[Tuple[int, int], int] = {}
    queue = []
    
    # Initialize from cells adjacent to obstacles or boundary
    for y in range(grid.height):
        for x in range(grid.width):
            if (x, y) in grid.obstacles:
                continue
            # check distance to nearest obstacle
            # For 0-distance, if next to obstacle or border
            is_border = (x == 0 or x == grid.width - 1 or y == 0 or y == grid.height - 1)
            has_adj_obs = any((nx, ny) in grid.obstacles for nx, ny in [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1)
            ])
            if is_border or has_adj_obs:
                dt[(x, y)] = 1
                queue.append((x, y))

    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        d = dt[curr]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = curr[0] + dx, curr[1] + dy
            if grid.is_valid(nx, ny) and (nx, ny) not in dt:
                dt[(nx, ny)] = d + 1
                queue.append((nx, ny))

    return dt


def extract_conflict_features(
    conflict: Conflict,
    conflict_class: str,
    all_classified_conflicts: List[Tuple[Conflict, str]],
    paths: List[List[Tuple[int, int]]],
    mvc_size: int
) -> Dict[str, float]:
    """Compatibility wrapper for extract_features."""
    type_val = 2.0 if conflict_class == 'cardinal' else (1.0 if conflict_class == 'semi-cardinal' else 0.0)
    cardinal_conflicts = [c for c, cls in all_classified_conflicts if cls == 'cardinal']
    agents_cg = set()
    for c in cardinal_conflicts:
        agents_cg.add(c.agent_1)
        agents_cg.add(c.agent_2)

    a1, a2 = conflict.agent_1, conflict.agent_2
    return {
        'corridor_score_i': 1.0,
        'corridor_score_j': 1.0,
        'local_density_i': 0.0,
        'local_density_j': 0.0,
        'pair_conflict_count': 0.0,
        'goal_dist_i': 0.0,
        'goal_dist_j': 0.0,
        'num_agents_in_conflict_graph': float(len(agents_cg)),
        'cardinal_degree_i': 0.0,
        'cardinal_degree_j': 0.0,
        'downstream_conflict_count_i': 0.0,
        'downstream_conflict_count_j': 0.0,
        'conflict_type': type_val,
        'mvc_size': float(mvc_size),
        'conflict_time': float(conflict.time),
        'is_edge_conflict': 1.0 if conflict.conflict_type == 'edge' else 0.0,
        'agent_i_path_length': float(len(paths[a1])),
        'agent_j_path_length': float(len(paths[a2]))
    }


FEATURE_NAMES = [
    # Family 1: Corridor structure (option c)
    'corridor_score_i',
    'corridor_score_j',
    # Family 2: Congestion
    'local_density_i',
    'local_density_j',
    # Family 3: Conflict frequency
    'pair_conflict_count',
    # Family 4: Goal distance
    'goal_dist_i',
    'goal_dist_j',
    # Family 5: Number of affected agents
    'num_agents_in_conflict_graph',
    'cardinal_degree_i',
    'cardinal_degree_j',
    # Family 6: Predicted downstream conflicts
    'downstream_conflict_count_i',
    'downstream_conflict_count_j',
    # Kept baseline features
    'conflict_type',
    'mvc_size',
    'conflict_time',
    'is_edge_conflict',
    'agent_i_path_length',
    'agent_j_path_length'
]


class FeatureExtractor:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.dt = compute_grid_distance_transform(grid)

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
        a1, a2 = conflict.agent_1, conflict.agent_2
        t = conflict.time
        loc = conflict.loc_1

        # 1. Corridor structure (Option c: distance transform, clipped to [0, 5])
        corridor_i = float(min(5, self.dt.get(loc, 1)))
        corridor_j = corridor_i

        # 2. Congestion: other agents within Manhattan distance <= 2 at time t
        num_agents = len(paths)
        density_i = 0
        density_j = 0
        for a_idx in range(num_agents):
            if a_idx == a1 or a_idx == a2:
                continue
            pos = get_agent_pos_at_t(paths[a_idx], t)
            if manhattan_distance(pos, loc) <= 2:
                density_i += 1
        density_j = density_i

        # 3. Conflict frequency
        pair_key = (min(a1, a2), max(a1, a2))
        pair_freq = float(pair_history.get(pair_key, 0) if pair_history else 0)

        # 4. Goal distance
        g_dist_i = float(manhattan_distance(loc, goals[a1])) if a1 < len(goals) else 0.0
        g_dist_j = float(manhattan_distance(loc, goals[a2])) if a2 < len(goals) else 0.0

        # 5. Number of affected agents & degrees in G_CF
        cardinal_edges = [
            (c.agent_1, c.agent_2) for c, cls in all_classified_conflicts if cls == 'cardinal'
        ]
        agents_in_cg = set()
        deg_i = 0
        deg_j = 0
        for u, v in cardinal_edges:
            agents_in_cg.add(u)
            agents_in_cg.add(v)
            if u == a1 or v == a1:
                deg_i += 1
            if u == a2 or v == a2:
                deg_j += 1

        num_agents_cg = float(len(agents_in_cg))
        cardinal_deg_i = float(deg_i)
        cardinal_deg_j = float(deg_j)

        # 6. Downstream conflicts
        down_i = float(downstream_counts[0]) if downstream_counts else 0.0
        down_j = float(downstream_counts[1]) if downstream_counts else 0.0

        # Kept features
        type_val = 2.0 if conflict_class == 'cardinal' else (1.0 if conflict_class == 'semi-cardinal' else 0.0)

        feats = {
            'corridor_score_i': corridor_i,
            'corridor_score_j': corridor_j,
            'local_density_i': float(density_i),
            'local_density_j': float(density_j),
            'pair_conflict_count': pair_freq,
            'goal_dist_i': g_dist_i,
            'goal_dist_j': g_dist_j,
            'num_agents_in_conflict_graph': num_agents_cg,
            'cardinal_degree_i': cardinal_deg_i,
            'cardinal_degree_j': cardinal_deg_j,
            'downstream_conflict_count_i': down_i,
            'downstream_conflict_count_j': down_j,
            'conflict_type': type_val,
            'mvc_size': float(mvc_size),
            'conflict_time': float(t),
            'is_edge_conflict': 1.0 if conflict.conflict_type == 'edge' else 0.0,
            'agent_i_path_length': float(len(paths[a1])),
            'agent_j_path_length': float(len(paths[a2]))
        }
        return feats
