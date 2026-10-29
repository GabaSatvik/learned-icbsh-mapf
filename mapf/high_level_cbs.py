"""
Standard Conflict-Based Search (Sharon et al., 2012).
Uses exact 2D Reverse-BFS heuristics in low-level Space-Time A* and branches on first-found conflict.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from mapf.grid import GridMap
from mapf.low_level import (
    ConstraintSet,
    space_time_astar,
    get_path_cost,
    compute_bfs_h_table,
    reset_astar_call_count,
    get_astar_call_count,
)
from mapf.conflicts import Conflict, detect_first_conflict
from mapf.cardinal import build_constraint_for_agent


@dataclass(order=True)
class CBSNode:
    cost: int
    node_id: int
    paths: List[List[Tuple[int, int]]] = field(compare=False)
    constraints: List[ConstraintSet] = field(compare=False)


class PlainCBS:
    def __init__(self, grid: GridMap):
        self.grid = grid

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        timeout: float = 60.0,
        max_expansions: Optional[int] = None,
        shared_h_tables: Optional[List[Dict[Tuple[int, int], int]]] = None,
        shared_path_cache: Optional[Dict[Tuple[int, Tuple[frozenset, frozenset]], Optional[List[Tuple[int, int]]]]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        reset_astar_call_count()
        k = len(starts)

        h_tables = shared_h_tables if shared_h_tables is not None else [compute_bfs_h_table(self.grid, goals[i]) for i in range(k)]
        for i in range(k):
            if starts[i] not in h_tables[i]:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "astar_calls": get_astar_call_count(),
                "runtime": time.perf_counter() - start_time,
                }

        max_base_dist = max(h_tables[i][starts[i]] for i in range(k)) if k > 0 else 0
        max_t = max(max_base_dist + 80, self.grid.width + self.grid.height + 50)

        path_cache = shared_path_cache if shared_path_cache is not None else {}
        root_constraints = [ConstraintSet() for _ in range(k)]
        root_paths = []

        for i in range(k):
            p = space_time_astar(
                self.grid, starts[i], goals[i], root_constraints[i], max_timesteps=max_t, h_table=h_tables[i]
            )
            if p is None:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "astar_calls": get_astar_call_count(),
                "runtime": time.perf_counter() - start_time,
                }
            path_cache[(i, root_constraints[i].freeze_key())] = p
            root_paths.append(p)

        root_cost = sum(get_path_cost(p) for p in root_paths)
        node_counter = 0
        open_list = [CBSNode(cost=root_cost, node_id=node_counter, paths=root_paths, constraints=root_constraints)]
        nodes_expanded = 0

        while open_list:
            if time.perf_counter() - start_time > timeout:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "astar_calls": get_astar_call_count(),
                "runtime": time.perf_counter() - start_time,
                }
            if max_expansions is not None and nodes_expanded >= max_expansions:
                return {
                    "status": "MAX_EXPANSIONS",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "astar_calls": get_astar_call_count(),
                "runtime": time.perf_counter() - start_time,
                }

            curr = heapq.heappop(open_list)
            nodes_expanded += 1

            conflict = detect_first_conflict(curr.paths)
            if conflict is None:
                return {
                    "status": "SOLVED",
                    "cost": curr.cost,
                    "paths": curr.paths,
                    "nodes_expanded": nodes_expanded,
                    "astar_calls": get_astar_call_count(),
                "runtime": time.perf_counter() - start_time,
                }

            for agent_id in (conflict.agent_i, conflict.agent_j):
                new_cs = build_constraint_for_agent(conflict, agent_id, curr.constraints[agent_id])
                ckey = (agent_id, new_cs.freeze_key())
                if ckey not in path_cache:
                    path_cache[ckey] = space_time_astar(
                        self.grid,
                        starts[agent_id],
                        goals[agent_id],
                        new_cs,
                        max_timesteps=max_t,
                        h_table=h_tables[agent_id],
                    )
                new_path = path_cache[ckey]
                if new_path is not None:
                    child_paths = list(curr.paths)
                    child_paths[agent_id] = new_path
                    child_constraints = list(curr.constraints)
                    child_constraints[agent_id] = new_cs
                    child_cost = sum(get_path_cost(p) for p in child_paths)
                    node_counter += 1
                    heapq.heappush(
                        open_list,
                        CBSNode(
                            cost=child_cost,
                            node_id=node_counter,
                            paths=child_paths,
                            constraints=child_constraints,
                        ),
                    )

        return {
            "status": "UNSOLVABLE",
            "cost": None,
            "nodes_expanded": nodes_expanded,
            "astar_calls": get_astar_call_count(),
            "runtime": time.perf_counter() - start_time,
        }


StandardCBS = PlainCBS
