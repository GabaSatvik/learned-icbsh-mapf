"""
Standard Conflict-Based Search (Sharon et al., 2012).
Implements PlainCBS solver with standard dict return format.
"""
import heapq
import time
from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass, field

from mapf.grid import Grid
from mapf.low_level import space_time_astar, compute_bfs_h_table, get_path_cost, SpaceTimeAStar
from mapf.conflicts import Conflict, find_first_conflict


class PlainCBS:
    """
    Standard Plain CBS solver (Sharon et al., 2012).
    Branches on the first detected conflict without heuristics.
    """
    def __init__(self, grid: Grid):
        self.grid = grid

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        timeout: float = 30.0,
        max_expansions: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        k = len(starts)
        max_t = self.grid.width + self.grid.height + 50

        # Compute static heuristic tables for all agents
        h_tables = [compute_bfs_h_table(self.grid, goals[i]) for i in range(k)]
        if any(starts[i] not in h_tables[i] for i in range(k)):
            return {
                "status": "UNSOLVABLE",
                "cost": None,
                "nodes_expanded": 0,
                "paths": None,
                "decision_nodes": 0,
            }

        # Root paths
        root_paths = []
        for i in range(k):
            p = space_time_astar(self.grid, starts[i], goals[i], [], max_t, h_tables[i])
            if p is None:
                return {
                    "status": "UNSOLVABLE",
                    "cost": None,
                    "nodes_expanded": 0,
                    "paths": None,
                    "decision_nodes": 0,
                }
            root_paths.append(p)

        root_cost = sum(get_path_cost(p) for p in root_paths)
        # Heap items: (cost, node_id, paths, constraints)
        node_id = 0
        open_list = [(root_cost, node_id, root_paths, [])]
        nodes_expanded = 0

        while open_list:
            if time.perf_counter() - start_time > timeout:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "paths": None,
                    "decision_nodes": 0,
                }

            if max_expansions is not None and nodes_expanded >= max_expansions:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "paths": None,
                    "decision_nodes": 0,
                }

            cost, _, curr_paths, curr_cons = heapq.heappop(open_list)
            nodes_expanded += 1

            conflict = find_first_conflict(curr_paths)
            if conflict is None:
                return {
                    "status": "SOLVED",
                    "cost": cost,
                    "nodes_expanded": nodes_expanded,
                    "paths": curr_paths,
                    "decision_nodes": 0,
                }

            # Branch on first conflict
            for agent in (conflict.agent_1, conflict.agent_2):
                other = conflict.agent_2 if agent == conflict.agent_1 else conflict.agent_1
                new_c = conflict.get_constraint_for_agent(agent, other)
                child_cons = curr_cons + [new_c]

                agent_cons = [c for c in child_cons if getattr(c, "agent", -1) == agent]
                new_p = space_time_astar(
                    self.grid, starts[agent], goals[agent], agent_cons, max_t, h_tables[agent]
                )

                if new_p is not None:
                    child_paths = list(curr_paths)
                    child_paths[agent] = new_p
                    child_cost = sum(get_path_cost(p) for p in child_paths)
                    node_id += 1
                    heapq.heappush(open_list, (child_cost, node_id, child_paths, child_cons))

        return {
            "status": "UNSOLVABLE",
            "cost": None,
            "nodes_expanded": nodes_expanded,
            "paths": None,
            "decision_nodes": 0,
        }


class StandardCBS:
    """Legacy wrapper for StandardCBS."""
    def __init__(self, grid: Grid):
        self._cbs = PlainCBS(grid)

    def solve(self, starts, goals, time_limit: float = 30.0):
        res = self._cbs.solve(starts, goals, timeout=time_limit)
        stats = {
            "expanded": res["nodes_expanded"],
            "time": 0.0,
            "cost": res["cost"],
            "timeout": (res["status"] == "TIMEOUT"),
        }
        return res["paths"], stats
