"""
High-Level ICBS-h Solver (Boyarski et al., 2015 + Felner et al., 2018) with pluggable
conflict-prioritization hooks, node classification caching, and strict priority-tier invariants.
"""

from mapf.search_limits import bounded_solve
from mapf.validation import validate_instance, validate_constrained_paths
import heapq
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from mapf.grid import GridMap
from mapf.low_level import (
    ConstraintSet,
    space_time_astar,
    get_path_cost,
    compute_bfs_h_table,
    reset_astar_call_count,
    get_astar_call_count,
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

    @bounded_solve
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
        on_decision_callback: Optional[Callable[..., None]] = None,
        shared_h_tables: Optional[List[Dict[Tuple[int, int], int]]] = None,
        shared_path_cache: Optional[Dict[Tuple[int, Tuple[frozenset, frozenset]], Optional[List[Tuple[int, int]]]]] = None,
        initial_node: Optional[ICBSNode] = None,
        forced_first_conflict: Optional[Conflict] = None,
        progress: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve, optionally continuing a saved descendant problem.

        Continuation validates saved paths and their individual optimal costs,
        rebuilds caches, and recomputes root MVC rather than trusting a supplied
        inherited h. The original global OPEN frontier is not restored.
        A forced first split suppresses bypass only at that root. Progress
        exposes a lower bound on expansions even on cooperative interruption.
        """
        start_time = time.perf_counter()
        if progress is not None:
            progress.clear()
            progress.update(nodes_expanded=0, forced_split_applied=False)
        if initial_node is not None and initial_constraints is not None:
            raise ValueError('Use saved-node constraints or initial_constraints, not both')
        if forced_first_conflict is not None and initial_node is None:
            raise ValueError('Forced continuation requires a saved node')
        reset_astar_call_count()
        if hasattr(self.conflict_selector, 'ml_tiebreaks'):
            self.conflict_selector.ml_tiebreaks = 0
        if hasattr(self.conflict_selector, 'analytical_shortcircuits'):
            self.conflict_selector.analytical_shortcircuits = 0
        k = len(starts)
        if not validate_instance(self.grid, starts, goals):
            return dict(status="UNSOLVABLE", cost=None, nodes_expanded=0,
                        runtime=time.perf_counter()-start_time)

        h_tables = shared_h_tables if shared_h_tables is not None else [compute_bfs_h_table(self.grid, goals[i]) for i in range(k)]
        for i in range(k):
            if starts[i] not in h_tables[i]:
                return {
                    "status": "NO_ROOT_PATH",
                    "cost": None,
                    "nodes_expanded": 0,
                    "decision_nodes": 0,
                    "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

                "runtime": time.perf_counter() - start_time,
                }

        max_t = None  # Exact low-level search uses a constraint-derived complete bound.

        path_cache = shared_path_cache if shared_path_cache is not None else {}

        root_constraints = ([cs.copy() for cs in initial_node.constraints] if initial_node is not None
                            else _extract_per_agent_constraint_sets(k, initial_constraints))
        if initial_node is not None:
            validate_constrained_paths(self.grid, starts, goals, initial_node.paths, root_constraints)
            if initial_node.g != sum(map(get_path_cost, initial_node.paths)):
                raise ValueError('Saved node cost differs from path lengths')
        root_flat_constraints: List[Constraint] = []
        if initial_node is not None:
            for agent, cs in enumerate(root_constraints):
                root_flat_constraints.extend(Constraint(agent,t,(r,c)) for r,c,t in sorted(cs.vertex_constraints))
                root_flat_constraints.extend(Constraint(agent,t,(r,c),(rr,cc),'EDGE')
                                             for r,c,rr,cc,t in sorted(cs.edge_constraints))
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
                    "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

                "runtime": time.perf_counter() - start_time,
                }
            if initial_node is not None:
                saved = initial_node.paths[i]
                if len(saved) != len(p):
                    raise ValueError('Saved individual path is not optimal under its constraints')
                p = list(saved)
            path_cache[(i, root_constraints[i].freeze_key())] = p
            root_paths.append(p)

        root_g = sum(get_path_cost(p) for p in root_paths)
        root_conflicts = detect_all_conflicts(root_paths)
        if not root_conflicts:
            if forced_first_conflict is not None:
                raise ValueError('No conflict to force in saved state')
            if progress is not None:
                progress['nodes_expanded'] = 1
            return {
                "status": "SOLVED",
                "cost": root_g,
                "paths": root_paths,
                "nodes_expanded": 1,
                "bypass_count": 0,
                "decision_nodes": 0,
                "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

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
            depth=initial_node.depth if initial_node is not None else 0,
            pair_counts=initial_node.pair_counts if initial_node is not None else None,
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
        forced_pending = forced_first_conflict

        while open_list:
            if time.perf_counter() - start_time > timeout:
                return {
                    "status": "TIMEOUT",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

                "runtime": time.perf_counter() - start_time,
                }

            if max_expansions is not None and nodes_expanded >= max_expansions:
                return {
                    "status": "MAX_EXPANSIONS",
                    "cost": None,
                    "nodes_expanded": nodes_expanded,
                    "bypass_count": bypass_count,
                    "decision_nodes": decision_nodes,
                    "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

                "runtime": time.perf_counter() - start_time,
                }

            node = heapq.heappop(open_list)
            nodes_expanded += 1
            if progress is not None:
                progress['nodes_expanded'] = nodes_expanded

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
                    "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

                "runtime": time.perf_counter() - start_time,
                }

            can_bypass = self.enable_bypass and (node.bypass_steps < k * 3) and forced_pending is None
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
                        "astar_calls": get_astar_call_count(),
                "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

                "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

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

            if on_decision_callback is not None and len(candidates) >= 2:
                # The solver is paused before choosing or mutating this decision.
                # Consumers must clone mutable members before running counterfactuals.
                on_decision_callback(
                    node,
                    classified,
                    g_cf,
                    candidates,
                    {
                        "open_list": open_list,
                        "node_counter": node_counter,
                        "nodes_expanded": nodes_expanded,
                        "bypass_count": bypass_count,
                        "decision_nodes": decision_nodes,
                        "h_tables": h_tables,
                        "path_cache": path_cache,
                        "starts": starts,
                        "goals": goals,
                    },
                )

            if forced_pending is not None:
                matches = [cc for cc in candidates if cc.conflict == forced_pending]
                if len(matches) != 1:
                    raise ValueError('Forced conflict is absent from the eligible priority tier')
                chosen_cc = matches[0]
                forced_pending = None
                if progress is not None:
                    progress['forced_split_applied'] = True
                if len(candidates) >= 2:
                    decision_nodes += 1
            elif len(candidates) >= 2:
                decision_nodes += 1
                if self.conflict_selector is not None:
                    context = {
                        "grid": self.grid,
                        "paths": node.paths,
                        "starts": starts,
                        "goals": goals,
                        "G_CF": g_cf,
                        "mvc_size": g_cf.exact_mvc_size(),
                        "effective_h": node.h,
                        "pair_counts": node.pair_counts,
                        "classified": classified,
                        "constraints": node.flat_constraints,
                        "per_agent_constraints": node.constraints,
                        "h_tables": h_tables,
                    }
                    chosen_cc = self.conflict_selector(candidates, context)
                else:
                    chosen_cc = candidates[0]
            else:
                chosen_cc = candidates[0]

            if not any(chosen_cc is cc for cc in candidates):
                raise ValueError("Selector returned a conflict outside the eligible tier")
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
                    # Paths are memoized, but contextual conflict counts must be
                    # recomputed for this child, including untouched agent pairs.
                    child_classified = classify_conflicts(
                        self.grid, starts, goals, child_paths, child_constraints,
                        child_conflicts, enable_bypass=False, h_tables=h_tables,
                        path_cache=path_cache, return_bypass=False)
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
            "astar_calls": get_astar_call_count(),
            "ml_tiebreaks": getattr(self.conflict_selector, "ml_tiebreaks", 0),

            "analytical_shortcircuits": getattr(self.conflict_selector, "analytical_shortcircuits", 0),

            "runtime": time.perf_counter() - start_time,
        }


ICBSHSolver = ICBShSolver
ICBS_H = ICBShSolver
