"""
ICBS-h implementation (Improved Conflict-Based Search with MVC Heuristic).
Supports pluggable conflict selectors, analytical lookahead, and max expansions.
"""
import heapq
import time
from typing import List, Tuple, Set, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from mapf.grid import Grid
from mapf.low_level import space_time_astar, compute_bfs_h_table, get_path_cost, SpaceTimeAStar
from mapf.conflicts import Conflict, detect_all_conflicts
from mapf.cardinal import classify_conflicts, ClassifiedConflict
from mapf.conflict_graph import compute_h4_and_graph, ConflictGraph


class ICBShSolver:
    """
    ICBS-h Solver with pluggable conflict prioritization policy.
    Maintains 100% cost-optimality and supports:
      - Regime A: full ICBS-h with MVC heuristic
      - Pluggable conflict selectors
      - Reusable cached lookahead paths from conflict classification
    """
    def __init__(
        self,
        grid: Grid,
        conflict_selector: Optional[Callable[[List[ClassifiedConflict], Dict[str, Any]], ClassifiedConflict]] = None,
        regime: str = "A",
    ):
        self.grid = grid
        self.conflict_selector = conflict_selector
        self.regime = regime

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        timeout: float = 30.0,
        max_expansions: Optional[int] = None,
        initial_constraints: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        k = len(starts)
        max_t = self.grid.width + self.grid.height + 50

        # Precompute BFS heuristic distance tables for all agents
        h_tables = [compute_bfs_h_table(self.grid, goals[i]) for i in range(k)]
        if any(starts[i] not in h_tables[i] for i in range(k)):
            return {
                "status": "UNSOLVABLE",
                "cost": None,
                "nodes_expanded": 0,
                "paths": None,
                "decision_nodes": 0,
            }

        init_cons = list(initial_constraints) if initial_constraints else []

        # Find initial paths for each agent
        root_paths = []
        for i in range(k):
            cons_i = [c for c in init_cons if getattr(c, "agent", -1) == i]
            p = space_time_astar(self.grid, starts[i], goals[i], cons_i, max_t, h_tables[i])
            if p is None:
                return {
                    "status": "UNSOLVABLE",
                    "cost": None,
                    "nodes_expanded": 0,
                    "paths": None,
                    "decision_nodes": 0,
                }
            root_paths.append(p)

        root_g = sum(get_path_cost(p) for p in root_paths)
        initial_conflicts = detect_all_conflicts(root_paths)
        if not initial_conflicts:
            return {
                "status": "SOLVED",
                "cost": root_g,
                "nodes_expanded": 1,
                "paths": root_paths,
                "decision_nodes": 0,
            }

        classified_init = classify_conflicts(
            self.grid, starts, goals, root_paths, init_cons, initial_conflicts, max_t, False, h_tables
        )
        root_h, _ = compute_h4_and_graph(classified_init)
        root_f = root_g + root_h

        node_id = 0
        # Heap items: (f_cost, h_cost, node_id, g_cost, paths, constraints, pair_counts)
        open_list = [(root_f, root_h, node_id, root_g, root_paths, init_cons, {})]
        nodes_expanded = 0
        decision_nodes = 0

        while open_list:
            if time.perf_counter() - start_time > timeout:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "paths": None,
                    "decision_nodes": decision_nodes,
                }

            if max_expansions is not None and nodes_expanded >= max_expansions:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "paths": None,
                    "decision_nodes": decision_nodes,
                }

            f_val, h_val, _, curr_g, curr_paths, curr_cons, curr_pairs = heapq.heappop(open_list)
            nodes_expanded += 1

            conflicts = detect_all_conflicts(curr_paths)
            if not conflicts:
                return {
                    "status": "SOLVED",
                    "cost": curr_g,
                    "nodes_expanded": nodes_expanded,
                    "paths": curr_paths,
                    "decision_nodes": decision_nodes,
                }

            classified = classify_conflicts(
                self.grid, starts, goals, curr_paths, curr_cons, conflicts, max_t, False, h_tables
            )
            mvc_size, G_CF = compute_h4_and_graph(classified)

            context = {
                "grid": self.grid,
                "paths": curr_paths,
                "goals": goals,
                "G_CF": G_CF,
                "mvc_size": mvc_size,
                "pair_counts": curr_pairs,
                "classified": classified,
                "constraints": curr_cons,
            }

            if len(classified) == 1:
                chosen_cc = classified[0]
            else:
                decision_nodes += 1
                if self.conflict_selector is not None:
                    chosen_cc = self.conflict_selector(classified, context)
                else:
                    # Default ICBS-h: Cardinal > Semi-Cardinal > Non-Cardinal
                    cards = [cc for cc in classified if cc.conflict_type == "CARDINAL"]
                    if cards:
                        cards.sort(key=lambda x: x.conflict.time)
                        chosen_cc = cards[0]
                    else:
                        semis = [cc for cc in classified if cc.conflict_type == "SEMI_CARDINAL"]
                        if semis:
                            semis.sort(key=lambda x: x.conflict.time)
                            chosen_cc = semis[0]
                        else:
                            classified.sort(key=lambda x: x.conflict.time)
                            chosen_cc = classified[0]

            c = chosen_cc.conflict
            i, j = c.agent_i, c.agent_j
            next_pairs = dict(curr_pairs)
            pk = (min(i, j), max(i, j))
            next_pairs[pk] = next_pairs.get(pk, 0) + 1

            for branch_idx, (agent_id, other_id) in enumerate([(i, j), (j, i)]):
                new_c = c.get_constraint_for_agent(agent_id, other_id)
                child_cons = curr_cons + [new_c]

                # Reuse cached lookahead path if valid
                if branch_idx == 0:
                    child_path = chosen_cc.alt_path_i
                else:
                    child_path = chosen_cc.alt_path_j

                if child_path is None:
                    # Branch is infeasible
                    continue

                child_paths = list(curr_paths)
                child_paths[agent_id] = child_path
                child_g = sum(get_path_cost(p) for p in child_paths)

                c_child = detect_all_conflicts(child_paths)
                if not c_child:
                    child_h = 0
                else:
                    cc_child = classify_conflicts(
                        self.grid, starts, goals, child_paths, child_cons, c_child, max_t, False, h_tables
                    )
                    child_h, _ = compute_h4_and_graph(cc_child)

                child_f = child_g + child_h
                node_id += 1
                heapq.heappush(
                    open_list, (child_f, child_h, node_id, child_g, child_paths, child_cons, next_pairs)
                )

        return {
            "status": "UNSOLVABLE",
            "cost": None,
            "nodes_expanded": nodes_expanded,
            "paths": None,
            "decision_nodes": decision_nodes,
        }


class ICBS_H:
    """Legacy wrapper for backward compatibility."""
    def __init__(self, grid: Grid, logger: Any = None, enable_bypass: bool = True):
        self._solver = ICBShSolver(grid)

    def solve(self, starts, goals, time_limit: float = 30.0, metadata: Any = None):
        res = self._solver.solve(starts, goals, timeout=time_limit)
        stats = {
            "expanded": res["nodes_expanded"],
            "time": 0.0,
            "cost": res["cost"],
            "timeout": (res["status"] == "TIMEOUT"),
        }
        return res["paths"], stats
