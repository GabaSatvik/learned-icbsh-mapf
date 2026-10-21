"""
High-Level ICBS-h Solver (Boyarski et al., 2015 + Felner et al., 2018) with pluggable
conflict-prioritization hooks, node classification caching, and strict priority-tier invariants.
"""

import heapq
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from mapf.grid import GridMap
from mapf.low_level import (
    ConstraintSet,
    space_time_astar,
    get_path_cost,
    compute_bfs_h_table,
)
from mapf.conflicts import Conflict, Constraint, detect_all_conflicts
from mapf.cardinal import (
    ClassifiedConflict,
    classify_conflicts,
    build_constraint_for_agent,
    _extract_per_agent_constraint_sets,
)
from mapf.conflict_graph import CardinalConflictGraph, compute_h4_and_graph


class ICBSNode:
    def __init__(
        self,
        constraints: List[ConstraintSet],
        paths: List[List[Tuple[int, int]]],
        g: int,
        h: int,
        depth: int = 0,
        pair_counts: Optional[Dict[Tuple[int, int], int]] = None,
        node_id: int = 0,
        conflicts_cache: Optional[List[Conflict]] = None,
        classified_cache: Optional[List[ClassifiedConflict]] = None,
        g_cf_cache: Optional[CardinalConflictGraph] = None,
        flat_constraints: Optional[List[Constraint]] = None,
        bypass_steps: int = 0,
    ):
        self.constraints = constraints
        self.paths = paths
        self.g = g
        self.h = h
        self.f = g + h
        self.depth = depth
        self.pair_counts = dict(pair_counts) if pair_counts else {}
        self.node_id = node_id
        self.conflicts_cache = conflicts_cache
        self.classified_cache = classified_cache
        self.g_cf_cache = g_cf_cache
        self.flat_constraints = list(flat_constraints) if flat_constraints else []
        self.bypass_steps = bypass_steps

    def __lt__(self, other: "ICBSNode"):
        if self.f != other.f:
            return self.f < other.f
        if self.h != other.h:
            return self.h < other.h
        c_self = len(self.conflicts_cache) if self.conflicts_cache is not None else 999999
        c_other = len(other.conflicts_cache) if other.conflicts_cache is not None else 999999
        if c_self != c_other:
            return c_self < c_other
        return self.node_id < other.node_id


class ICBShSolver:
    def __init__(
        self,
        grid: GridMap,
        conflict_selector: Optional[
            Callable[[List[ClassifiedConflict], Dict[str, Any]], ClassifiedConflict]
        ] = None,
        enable_bypass: bool = True,
        regime: str = "A",
        strict_tiering: bool = True,
    ):
        self.grid = grid
        self.conflict_selector = conflict_selector
        self.enable_bypass = enable_bypass
        self.regime = regime
        self.strict_tiering = strict_tiering

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        timeout: float = 60.0,
        max_expansions: Optional[int] = None,
        initial_constraints: Optional[List[Any]] = None,
        on_node_callback: Optional[
            Callable[[ICBSNode, List[ClassifiedConflict], CardinalConflictGraph], None]
        ] = None,
        shared_h_tables: Optional[List[Dict[Tuple[int, int], int]]] = None,
        shared_path_cache: Optional[Dict[Tuple[int, Tuple[frozenset, frozenset]], Optional[List[Tuple[int, int]]]]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        k = len(starts)

        h_tables = shared_h_tables if shared_h_tables is not None else [compute_bfs_h_table(self.grid, goals[i]) for i in range(k)]
        for i in range(k):
            if starts[i] not in h_tables[i]:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "decision_nodes": 0,
                    "runtime": time.perf_counter() - start_time,
                }

        max_base_dist = max(h_tables[i][starts[i]] for i in range(k)) if k > 0 else 0
        max_t = max(max_base_dist + 80, self.grid.width + self.grid.height + 50)

        path_cache = shared_path_cache if shared_path_cache is not None else {}

        root_constraints = _extract_per_agent_constraint_sets(k, initial_constraints)
        root_flat_constraints: List[Constraint] = []
        if initial_constraints:
            for item in initial_constraints:
                if isinstance(item, Constraint):
                    root_flat_constraints.append(item)

        root_paths = []
        for i in range(k):
            p = space_time_astar(
                self.grid,
                starts[i],
                goals[i],
                root_constraints[i],
                max_timesteps=max_t,
                h_table=h_tables[i],
            )
            if p is None:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "decision_nodes": 0,
                    "runtime": time.perf_counter() - start_time,
                }
            path_cache[(i, root_constraints[i].freeze_key())] = p
            root_paths.append(p)

        root_g = sum(get_path_cost(p) for p in root_paths)
        root_conflicts = detect_all_conflicts(root_paths)
        if not root_conflicts:
            return {
                "status": "SOLVED",
                "cost": root_g,
                "paths": root_paths,
                "nodes_expanded": 1,
                "bypass_count": 0,
                "decision_nodes": 0,
                "runtime": time.perf_counter() - start_time,
            }

        classified_root = classify_conflicts(
            self.grid,
            starts,
            goals,
            root_paths,
            root_constraints,
            root_conflicts,
            enable_bypass=False,
            max_timesteps=max_t,
            h_tables=h_tables,
            path_cache=path_cache,
            return_bypass=False,
        )
        root_h, root_g_cf = compute_h4_and_graph(classified_root)

        node_counter = 0
        root = ICBSNode(
            root_constraints,
            root_paths,
            root_g,
            root_h,
            depth=0,
            node_id=node_counter,
            conflicts_cache=root_conflicts,
            classified_cache=classified_root,
            g_cf_cache=root_g_cf,
            flat_constraints=root_flat_constraints,
        )
        open_list = [root]

        nodes_expanded = 0
        bypass_count = 0
        decision_nodes = 0

        while open_list:
            if time.perf_counter() - start_time > timeout:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "runtime": time.perf_counter() - start_time,
                }

            if max_expansions is not None and nodes_expanded >= max_expansions:
                return {
                    "status": "MAX_EXPANSIONS",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "runtime": time.perf_counter() - start_time,
                }

            node = heapq.heappop(open_list)
            nodes_expanded += 1

            conflicts = (
                node.conflicts_cache
                if node.conflicts_cache is not None
                else detect_all_conflicts(node.paths)
            )
            if not conflicts:
                return {
                    "status": "SOLVED",
                    "cost": node.g,
                    "paths": node.paths,
                    "nodes_expanded": nodes_expanded,
                    "bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "runtime": time.perf_counter() - start_time,
                }

            can_bypass = self.enable_bypass and (node.bypass_steps < k * 3)
            if node.classified_cache is not None and not can_bypass:
                classified = node.classified_cache
                bypass_info = None
            elif node.classified_cache is not None and can_bypass:
                # Check if cached classified conflicts already offer a valid bypass!
                bypass_info = None
                tot_before = len(conflicts)
                for cc in node.classified_cache:
                    if cc.classification != "CARDINAL":
                        if (
                            cc.cost_increase_i == 0
                            and cc.alt_path_i is not None
                            and cc.alt_total_conflicts_i < tot_before
                        ):
                            bypass_info = (cc.conflict.agent_i, cc.alt_path_i, cc.conflict)
                            break
                        if (
                            cc.cost_increase_j == 0
                            and cc.alt_path_j is not None
                            and cc.alt_total_conflicts_j < tot_before
                        ):
                            bypass_info = (cc.conflict.agent_j, cc.alt_path_j, cc.conflict)
                            break
                classified = node.classified_cache
            else:
                classified, bypass_info = classify_conflicts(
                    self.grid,
                    starts,
                    goals,
                    node.paths,
                    node.constraints,
                    conflicts,
                    enable_bypass=can_bypass,
                    max_timesteps=max_t,
                    h_tables=h_tables,
                    path_cache=path_cache,
                    return_bypass=True,
                )

            if bypass_info is not None:
                bp_agent, bp_path, _ = bypass_info
                bypass_count += 1
                new_paths = list(node.paths)
                new_paths[bp_agent] = bp_path
                new_conflicts = detect_all_conflicts(new_paths)
                if not new_conflicts:
                    return {
                        "status": "SOLVED",
                        "cost": node.g,
                        "paths": new_paths,
                        "nodes_expanded": nodes_expanded,
                        "bypass_count": bypass_count,
                        "decision_nodes": decision_nodes,
                        "runtime": time.perf_counter() - start_time,
                    }
                new_classified = classify_conflicts(
                    self.grid,
                    starts,
                    goals,
                    new_paths,
                    node.constraints,
                    new_conflicts,
                    enable_bypass=False,
                    max_timesteps=max_t,
                    h_tables=h_tables,
                    path_cache=path_cache,
                    return_bypass=False,
                )
                new_h, new_g_cf = compute_h4_and_graph(new_classified)
                node_counter += 1
                bp_node = ICBSNode(
                    node.constraints,
                    new_paths,
                    node.g,
                    max(node.h, new_h),
                    depth=node.depth,
                    pair_counts=node.pair_counts,
                    node_id=node_counter,
                    conflicts_cache=new_conflicts,
                    classified_cache=new_classified,
                    g_cf_cache=new_g_cf,
                    flat_constraints=node.flat_constraints,
                    bypass_steps=node.bypass_steps + 1,
                )
                heapq.heappush(open_list, bp_node)
                continue

            g_cf = (
                node.g_cf_cache
                if node.g_cf_cache is not None
                else CardinalConflictGraph(classified)
            )

            if on_node_callback is not None:
                on_node_callback(node, classified, g_cf)

            # Priority tiering: CARDINAL > SEMI_CARDINAL > NON_CARDINAL
            cardinals = [cc for cc in classified if cc.classification == "CARDINAL"]
            semis = [cc for cc in classified if cc.classification == "SEMI_CARDINAL"]
            nons = [cc for cc in classified if cc.classification == "NON_CARDINAL"]

            if self.strict_tiering or self.regime == "A":
                if cardinals:
                    candidates = cardinals
                elif semis:
                    candidates = semis
                else:
                    candidates = nons
            else:
                candidates = classified

            if len(candidates) >= 2:
                decision_nodes += 1
                if self.conflict_selector is not None:
                    context = {
                        "grid": self.grid,
                        "paths": node.paths,
                        "starts": starts,
                        "goals": goals,
                        "G_CF": g_cf,
                        "mvc_size": node.h,
                        "pair_counts": node.pair_counts,
                        "classified": classified,
                        "constraints": node.flat_constraints,
                    }
                    chosen_cc = self.conflict_selector(candidates, context)
                else:
                    chosen_cc = candidates[0]
            else:
                chosen_cc = candidates[0]

            c = chosen_cc.conflict
            ai, aj = c.agent_i, c.agent_j
            pair_key = (min(ai, aj), max(ai, aj))
            next_pair_counts = dict(node.pair_counts)
            next_pair_counts[pair_key] = next_pair_counts.get(pair_key, 0) + 1

            for branch_idx, agent_id in enumerate((ai, aj)):
                other_agent = aj if agent_id == ai else ai
                new_cs = build_constraint_for_agent(c, agent_id, node.constraints[agent_id])
                new_con_obj = c.get_constraint_for_agent(agent_id, other_agent)

                if branch_idx == 0 and chosen_cc.alt_path_i is not None:
                    new_path = chosen_cc.alt_path_i
                elif branch_idx == 1 and chosen_cc.alt_path_j is not None:
                    new_path = chosen_cc.alt_path_j
                else:
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

                if new_path is None:
                    continue

                child_paths = list(node.paths)
                child_paths[agent_id] = new_path
                child_constraints = list(node.constraints)
                child_constraints[agent_id] = new_cs
                child_flat = list(node.flat_constraints) + [new_con_obj]
                child_g = sum(get_path_cost(p) for p in child_paths)

                child_conflicts = detect_all_conflicts(child_paths)
                if not child_conflicts:
                    child_classified = []
                    child_h = 0
                    child_g_cf = CardinalConflictGraph([])
                else:
                    # Incremental classification: reuse parent ClassifiedConflict for unaffected pairs
                    parent_cc_map = {cc.conflict: cc for cc in classified}
                    reused_cc = []
                    new_confs = []
                    n_tot_child = len(child_conflicts)
                    for ch_c in child_conflicts:
                        if ch_c.agent_i != agent_id and ch_c.agent_j != agent_id and ch_c in parent_cc_map:
                            p_cc = parent_cc_map[ch_c]
                            reused_cc.append(
                                ClassifiedConflict(
                                    conflict=p_cc.conflict,
                                    classification=p_cc.classification,
                                    cost_increase_i=p_cc.cost_increase_i,
                                    cost_increase_j=p_cc.cost_increase_j,
                                    alt_path_i=p_cc.alt_path_i,
                                    alt_path_j=p_cc.alt_path_j,
                                    alt_cost_i=p_cc.alt_cost_i,
                                    alt_cost_j=p_cc.alt_cost_j,
                                    alt_total_conflicts_i=(n_tot_child + p_cc.marginal_conflict_delta_i) if p_cc.marginal_conflict_delta_i < 999999 else 999999,
                                    alt_total_conflicts_j=(n_tot_child + p_cc.marginal_conflict_delta_j) if p_cc.marginal_conflict_delta_j < 999999 else 999999,
                                    marginal_conflict_delta_i=p_cc.marginal_conflict_delta_i,
                                    marginal_conflict_delta_j=p_cc.marginal_conflict_delta_j,
                                )
                            )
                        else:
                            new_confs.append(ch_c)
                    if new_confs:
                        newly_classified = classify_conflicts(
                            self.grid,
                            starts,
                            goals,
                            child_paths,
                            child_constraints,
                            new_confs,
                            enable_bypass=False,
                            max_timesteps=max_t,
                            h_tables=h_tables,
                            path_cache=path_cache,
                            return_bypass=False,
                        )
                        # Fix total conflict count baseline for newly_classified
                        for ncc in newly_classified:
                            if ncc.marginal_conflict_delta_i < 999999:
                                ncc.alt_total_conflicts_i = n_tot_child + ncc.marginal_conflict_delta_i
                            if ncc.marginal_conflict_delta_j < 999999:
                                ncc.alt_total_conflicts_j = n_tot_child + ncc.marginal_conflict_delta_j
                    else:
                        newly_classified = []
                    child_classified = reused_cc + newly_classified
                    child_h, child_g_cf = compute_h4_and_graph(child_classified)

                # Pathmax: f(child) >= f(parent)
                effective_h = max(child_h, node.f - child_g)
                node_counter += 1
                child_node = ICBSNode(
                    child_constraints,
                    child_paths,
                    child_g,
                    effective_h,
                    depth=node.depth + 1,
                    pair_counts=next_pair_counts,
                    node_id=node_counter,
                    conflicts_cache=child_conflicts,
                    classified_cache=child_classified,
                    g_cf_cache=child_g_cf,
                    flat_constraints=child_flat,
                    bypass_steps=0,
                )
                heapq.heappush(open_list, child_node)

        return {
            "status": "UNSOLVABLE",
            "cost": None,
            "nodes_expanded": nodes_expanded,
            "bypass_count": bypass_count,
            "decision_nodes": decision_nodes,
            "runtime": time.perf_counter() - start_time,
        }


ICBSHSolver = ICBShSolver
ICBS_H = ICBShSolver
