"""Counterfactual ICBS-h rollouts from a complete live OPEN frontier."""
from __future__ import annotations

import copy
import heapq
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mapf.baselines import f_aware_residual_selector
from mapf.cardinal import ClassifiedConflict, build_constraint_for_agent, classify_conflicts
from mapf.conflict_graph import CardinalConflictGraph, compute_h4_and_graph
from mapf.conflicts import Conflict, detect_all_conflicts
from mapf.icbs_h import ICBSNode
from mapf.low_level import get_astar_call_count, get_path_cost, reset_astar_call_count, space_time_astar
from mapf.validation import validate_solution


def conflict_id(conflict: Conflict) -> str:
    loc2 = "" if conflict.loc2 is None else f"-{conflict.loc2[0]}-{conflict.loc2[1]}"
    return (f"{conflict.agent_i}-{conflict.agent_j}-{conflict.conflict_type.lower()}-"
            f"{conflict.time}-{conflict.loc1[0]}-{conflict.loc1[1]}{loc2}")


@dataclass
class FullSearchSnapshot:
    current: ICBSNode
    open_list: List[ICBSNode]
    classified: List[ClassifiedConflict]
    candidates: List[ClassifiedConflict]
    starts: List[Tuple[int, int]]
    goals: List[Tuple[int, int]]
    h_tables: List[Dict[Tuple[int, int], int]]
    path_cache: Dict[Any, Any]
    node_counter: int
    parent_nodes_expanded: int
    parent_bypass_count: int
    parent_decision_nodes: int
    mode: str = "full_search_state"
    grid: Any = None


def capture_full_search_snapshot(node, classified, graph, candidates, state) -> FullSearchSnapshot:
    """Clone all ordering-relevant live state before the original choice."""
    del graph
    bundle = copy.deepcopy(dict(current=node, open_list=state["open_list"],
                                classified=classified, candidates=candidates,
                                path_cache=state["path_cache"]))
    return FullSearchSnapshot(
        current=bundle["current"], open_list=bundle["open_list"],
        classified=bundle["classified"], candidates=bundle["candidates"],
        starts=list(state["starts"]), goals=list(state["goals"]),
        h_tables=copy.deepcopy(state["h_tables"]), path_cache=bundle["path_cache"],
        node_counter=int(state["node_counter"]),
        parent_nodes_expanded=int(state["nodes_expanded"]),
        parent_bypass_count=int(state["bypass_count"]),
        parent_decision_nodes=int(state["decision_nodes"]),
    )


def _eligible(classified: Sequence[ClassifiedConflict]) -> List[ClassifiedConflict]:
    tiers = {"CARDINAL": 0, "SEMI_CARDINAL": 1, "NON_CARDINAL": 2}
    best = min(tiers[c.classification] for c in classified)
    return [c for c in classified if tiers[c.classification] == best]


def _context(snapshot, node, classified, graph):
    return dict(grid=snapshot.grid, paths=node.paths, starts=snapshot.starts,
                goals=snapshot.goals, G_CF=graph, mvc_size=graph.exact_mvc_size(),
                effective_h=node.h, pair_counts=node.pair_counts,
                classified=classified, constraints=node.flat_constraints,
                per_agent_constraints=node.constraints, h_tables=snapshot.h_tables)


def _conflict_tuple(c):
    return (c.agent_i, c.agent_j, c.time, c.conflict_type, c.loc1, c.loc2)


def _checkpoint(budget, solved, solution_cost, expansions, generated, peak_open,
                open_list, lb0, plateau_escape, bypass_count, infeasible_children,
                start_wall, start_cpu, status):
    final_lb = min((n.f for n in open_list), default=None)
    return dict(
        budget=budget, solved=solved, solution_cost=solution_cost,
        ct_expansions=expansions, ct_generated=generated,
        low_level_searches=get_astar_call_count(), replanning_count=get_astar_call_count(),
        wall_seconds=time.perf_counter()-start_wall,
        cpu_seconds=time.process_time()-start_cpu, peak_open_size=peak_open,
        final_open_size=len(open_list), initial_global_lower_bound=lb0,
        final_global_lower_bound=final_lb,
        lower_bound_increase=None if final_lb is None else final_lb-lb0,
        plateau_escape_expansions=plateau_escape,
        plateau_escape_censored=plateau_escape is None and not solved,
        solved_before_plateau_escape=solved and plateau_escape is None,
        bypass_count=bypass_count, infeasible_child_count=infeasible_children,
        status=status, censored=not solved and status in {"EXPANSION_CAP", "TIME_CAP"})


def evaluate_conflict_with_rollout(
    grid: Any,
    solver_snapshot: FullSearchSnapshot,
    candidate_conflict: ClassifiedConflict,
    expansion_budgets: Iterable[int],
    continuation_policy: str = "residual",
    timeout: Optional[float] = None,
    trace: bool = False,
    force_first: bool = True,
) -> Dict[str, Any]:
    """Force one current conflict, then continue ordinary residual ICBS-h."""
    if solver_snapshot.mode != "full_search_state":
        raise ValueError("This evaluator requires a full_search_state snapshot")
    if continuation_policy != "residual":
        raise ValueError("Only residual continuation is supported")
    budgets = sorted(set(int(b) for b in expansion_budgets))
    if not budgets or budgets[0] <= 0:
        raise ValueError("Expansion budgets must be positive")
    snapshot = copy.deepcopy(solver_snapshot)
    snapshot.grid = grid
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    reset_astar_call_count()
    open_list = snapshot.open_list + [snapshot.current]
    heapq.heapify(open_list)
    lb0 = min(n.f for n in open_list)
    peak_open, expansions, generated = len(open_list), 0, 0
    bypass_count = infeasible_children = 0
    plateau_escape = None
    forced_pending = copy.deepcopy(candidate_conflict.conflict) if force_first else None
    forced_node_id = snapshot.current.node_id
    node_counter, path_cache = snapshot.node_counter, snapshot.path_cache
    results, trajectory, chosen_trace = {}, [], []
    solved, solution_cost, solution_paths, status = False, None, None, "EXPANSION_CAP"

    while open_list and expansions < budgets[-1]:
        if timeout is not None and time.perf_counter()-start_wall >= timeout:
            status = "TIME_CAP"
            break
        node = heapq.heappop(open_list)
        expansions += 1
        if forced_pending is not None and node.node_id != forced_node_id:
            raise RuntimeError("OPEN replay did not select the captured node first")
        conflicts = node.conflicts_cache if node.conflicts_cache is not None else detect_all_conflicts(node.paths)
        if not conflicts:
            solved, solution_cost, solution_paths, status = True, node.g, node.paths, "SOLVED"
            validate_solution(grid, snapshot.starts, snapshot.goals, solution_paths, solution_cost)
        else:
            can_bypass = node.bypass_steps < len(snapshot.starts)*3 and forced_pending is None
            if node.classified_cache is not None and not can_bypass:
                classified, bypass_info = node.classified_cache, None
            elif node.classified_cache is not None:
                classified, bypass_info = node.classified_cache, None
                before = len(conflicts)
                for cc in classified:
                    if cc.classification != "CARDINAL":
                        if cc.cost_increase_i == 0 and cc.alt_path_i is not None and cc.alt_total_conflicts_i < before:
                            bypass_info = (cc.conflict.agent_i, cc.alt_path_i)
                            break
                        if cc.cost_increase_j == 0 and cc.alt_path_j is not None and cc.alt_total_conflicts_j < before:
                            bypass_info = (cc.conflict.agent_j, cc.alt_path_j)
                            break
            else:
                classified, raw = classify_conflicts(
                    grid, snapshot.starts, snapshot.goals, node.paths, node.constraints,
                    conflicts, enable_bypass=can_bypass, h_tables=snapshot.h_tables,
                    path_cache=path_cache, return_bypass=True)
                bypass_info = None if raw is None else raw[:2]
            if bypass_info is not None:
                bp_agent, bp_path = bypass_info
                bypass_count += 1
                new_paths = list(node.paths); new_paths[bp_agent] = bp_path
                new_conflicts = detect_all_conflicts(new_paths)
                if not new_conflicts:
                    solved, solution_cost, solution_paths, status = True, node.g, new_paths, "SOLVED"
                    validate_solution(grid, snapshot.starts, snapshot.goals, solution_paths, solution_cost)
                else:
                    new_classified = classify_conflicts(
                        grid, snapshot.starts, snapshot.goals, new_paths, node.constraints,
                        new_conflicts, enable_bypass=False, h_tables=snapshot.h_tables,
                        path_cache=path_cache, return_bypass=False)
                    new_h, new_graph = compute_h4_and_graph(new_classified)
                    node_counter += 1
                    heapq.heappush(open_list, ICBSNode(
                        node.constraints, new_paths, node.g, max(node.h, new_h),
                        depth=node.depth, pair_counts=node.pair_counts, node_id=node_counter,
                        conflicts_cache=new_conflicts, classified_cache=new_classified,
                        g_cf_cache=new_graph, flat_constraints=node.flat_constraints,
                        bypass_steps=node.bypass_steps+1))
                    generated += 1
            else:
                graph = node.g_cf_cache if node.g_cf_cache is not None else CardinalConflictGraph(classified)
                candidates = _eligible(classified)
                forced = forced_pending is not None
                if forced:
                    matches = [cc for cc in candidates if cc.conflict == forced_pending]
                    if len(matches) != 1:
                        raise ValueError("Forced conflict is absent from eligible tier")
                    chosen, forced_pending = matches[0], None
                elif len(candidates) > 1:
                    chosen = f_aware_residual_selector(candidates, _context(snapshot, node, classified, graph))
                else:
                    chosen = candidates[0]
                chosen_trace.append(dict(expansion=expansions, node_id=node.node_id,
                                         conflict=_conflict_tuple(chosen.conflict), forced=forced))
                c, ai, aj = chosen.conflict, chosen.conflict.agent_i, chosen.conflict.agent_j
                pair_key = (min(ai, aj), max(ai, aj))
                pair_counts = dict(node.pair_counts)
                pair_counts[pair_key] = pair_counts.get(pair_key, 0)+1
                for branch_idx, agent in enumerate((ai, aj)):
                    other = aj if agent == ai else ai
                    new_cs = build_constraint_for_agent(c, agent, node.constraints[agent])
                    con_obj = c.get_constraint_for_agent(agent, other)
                    if branch_idx == 0 and chosen.alt_path_i is not None:
                        new_path = chosen.alt_path_i
                    elif branch_idx == 1 and chosen.alt_path_j is not None:
                        new_path = chosen.alt_path_j
                    else:
                        key = (agent, new_cs.freeze_key())
                        if key not in path_cache:
                            path_cache[key] = space_time_astar(
                                grid, snapshot.starts[agent], snapshot.goals[agent],
                                new_cs, h_table=snapshot.h_tables[agent])
                        new_path = path_cache[key]
                    if new_path is None:
                        infeasible_children += 1
                        continue
                    paths = list(node.paths); paths[agent] = new_path
                    constraints = list(node.constraints); constraints[agent] = new_cs
                    flat = list(node.flat_constraints)+[con_obj]
                    child_g = sum(get_path_cost(p) for p in paths)
                    child_conflicts = detect_all_conflicts(paths)
                    if child_conflicts:
                        child_classified = classify_conflicts(
                            grid, snapshot.starts, snapshot.goals, paths, constraints,
                            child_conflicts, enable_bypass=False, h_tables=snapshot.h_tables,
                            path_cache=path_cache, return_bypass=False)
                        child_h, child_graph = compute_h4_and_graph(child_classified)
                    else:
                        child_classified, child_h, child_graph = [], 0, CardinalConflictGraph([])
                    node_counter += 1
                    heapq.heappush(open_list, ICBSNode(
                        constraints, paths, child_g, max(child_h, node.f-child_g),
                        depth=node.depth+1, pair_counts=pair_counts, node_id=node_counter,
                        conflicts_cache=child_conflicts, classified_cache=child_classified,
                        g_cf_cache=child_graph, flat_constraints=flat, bypass_steps=0))
                    generated += 1
        peak_open = max(peak_open, len(open_list))
        current_lb = min((n.f for n in open_list), default=None)
        if plateau_escape is None and current_lb is not None and current_lb > lb0:
            plateau_escape = expansions
        if trace:
            trajectory.append(dict(expansion=expansions, global_min_open_f=current_lb))
        if expansions in budgets:
            results[expansions] = _checkpoint(
                expansions, solved, solution_cost, expansions, generated, peak_open,
                open_list, lb0, plateau_escape, bypass_count, infeasible_children,
                start_wall, start_cpu, status if solved else "EXPANSION_CAP")
        if solved:
            break
    if not open_list and not solved:
        status = "UNSOLVABLE"
    for budget in budgets:
        if budget not in results:
            results[budget] = _checkpoint(
                budget, solved, solution_cost, expansions, generated, peak_open,
                open_list, lb0, plateau_escape, bypass_count, infeasible_children,
                start_wall, start_cpu, status)
    return dict(
        mode=snapshot.mode, continuation_policy=continuation_policy,
        candidate_conflict_id=conflict_id(candidate_conflict.conflict),
        forced_split_applied=(not force_first) or forced_pending is None,
        checkpoints=[results[b] for b in budgets], chosen_conflicts=chosen_trace,
        trajectory=trajectory, solution_paths=solution_paths if solved else None)
