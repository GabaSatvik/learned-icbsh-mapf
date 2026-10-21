"""
Conflict classification (Cardinal, Semi-Cardinal, Non-Cardinal) and Bypass logic (Boyarski et al., 2015).
Includes memoized path caching and exact O(T * k) marginal conflict tracking for CL-ICBS-h.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union, Any
from mapf.grid import GridMap
from mapf.low_level import ConstraintSet, coerce_constraint_set, space_time_astar, get_path_cost
from mapf.conflicts import Conflict, Constraint, detect_all_conflicts, count_single_agent_conflicts


@dataclass
class ClassifiedConflict:
    conflict: Conflict
    classification: str  # "CARDINAL", "SEMI_CARDINAL", "NON_CARDINAL"
    cost_increase_i: int
    cost_increase_j: int
    alt_path_i: Optional[List[Tuple[int, int]]]
    alt_path_j: Optional[List[Tuple[int, int]]]
    alt_cost_i: float
    alt_cost_j: float
    alt_total_conflicts_i: int = 999999
    alt_total_conflicts_j: int = 999999
    marginal_conflict_delta_i: int = 999999
    marginal_conflict_delta_j: int = 999999

    @property
    def conflict_type(self) -> str:
        return self.classification


def build_constraint_for_agent(
    conflict: Conflict, agent_id: int, base_constraints: ConstraintSet
) -> ConstraintSet:
    """Returns a new ConstraintSet with the branch constraint added for agent_id."""
    new_c = base_constraints.copy()
    t = conflict.time
    if conflict.conflict_type.upper() == "VERTEX":
        r, c = conflict.loc1
        new_c.add_vertex_constraint(r, c, t)
    else:
        r1, c1 = conflict.loc1
        r2, c2 = conflict.loc2 if conflict.loc2 is not None else conflict.loc1
        if agent_id == conflict.agent_i:
            new_c.add_edge_constraint(r1, c1, r2, c2, t)
        else:
            new_c.add_edge_constraint(r2, c2, r1, c1, t)
    return new_c


def _extract_per_agent_constraint_sets(
    k: int, constraints: Any
) -> List[ConstraintSet]:
    if not constraints:
        return [ConstraintSet() for _ in range(k)]
    if isinstance(constraints, list) and len(constraints) == k and all(
        isinstance(x, ConstraintSet) for x in constraints
    ):
        return constraints
    # Flat list of Constraint objects or legacy tuples
    res = [ConstraintSet() for _ in range(k)]
    for item in constraints:
        if isinstance(item, Constraint):
            if 0 <= item.agent < k:
                res[item.agent].add_constraint(item)
        elif isinstance(item, tuple) and len(item) == 3:
            ag, ctype, data = item
            if 0 <= ag < k:
                if ctype == "vertex":
                    res[ag].add_vertex_constraint(data[0], data[1], data[2])
                else:
                    res[ag].add_edge_constraint(data[0], data[1], data[2], data[3], data[4])
    return res


def classify_conflicts(
    grid: GridMap,
    arg2: Any,
    arg3: Any,
    arg4: Any,
    constraints: Any,
    conflicts: Optional[List[Conflict]] = None,
    enable_bypass: bool = False,
    max_timesteps: Optional[int] = None,
    h_tables: Optional[List[Dict[Tuple[int, int], int]]] = None,
    path_cache: Optional[
        Dict[Tuple[int, Tuple[frozenset, frozenset]], Optional[List[Tuple[int, int]]]]
    ] = None,
    return_bypass: bool = False,
) -> Union[
    List[ClassifiedConflict],
    Tuple[List[ClassifiedConflict], Optional[Tuple[int, List[Tuple[int, int]], Conflict]]],
]:
    """
    Classifies all conflicts into CARDINAL, SEMI_CARDINAL, or NON_CARDINAL.
    Supports both calling conventions:
      - classify_conflicts(grid, starts, goals, paths, constraints, conflicts, ...)
      - classify_conflicts(grid, paths, starts, goals, constraints, conflicts, ...)
    If return_bypass is False (default), returns List[ClassifiedConflict].
    If return_bypass is True, returns (List[ClassifiedConflict], bypass_info).
    """
    # Disambiguate (starts, goals, paths) vs (paths, starts, goals)
    if arg2 and isinstance(arg2[0], list):
        paths, starts, goals = arg2, arg3, arg4
    else:
        starts, goals, paths = arg2, arg3, arg4

    k = len(paths)
    per_agent_cs = _extract_per_agent_constraint_sets(k, constraints)

    if conflicts is None:
        conflicts = detect_all_conflicts(paths)

    total_conflicts_before = len(conflicts)
    max_path_len = max((len(p) for p in paths), default=0)

    agent_base_conflicts: Dict[int, int] = {}
    for c in conflicts:
        agent_base_conflicts[c.agent_i] = agent_base_conflicts.get(c.agent_i, 0) + 1
        agent_base_conflicts[c.agent_j] = agent_base_conflicts.get(c.agent_j, 0) + 1

    classified_list: List[ClassifiedConflict] = []
    # Cache (agent_id, freeze_key) -> (alt_path, agent_conflict_count) within this node
    alt_conflict_cache: Dict[Tuple[int, Tuple[frozenset, frozenset]], int] = {}

    for c in conflicts:
        ai = c.agent_i
        aj = c.agent_j
        cost_i = get_path_cost(paths[ai])
        cost_j = get_path_cost(paths[aj])

        # Replan agent_i
        ci = build_constraint_for_agent(c, ai, per_agent_cs[ai])
        hi = h_tables[ai] if h_tables is not None else None
        fkey_i = ci.freeze_key()
        cache_key_i = (ai, fkey_i)

        if path_cache is not None:
            if cache_key_i not in path_cache:
                path_cache[cache_key_i] = space_time_astar(
                    grid, starts[ai], goals[ai], ci, max_timesteps=max_timesteps, h_table=hi
                )
            alt_path_i = path_cache[cache_key_i]
        else:
            alt_path_i = space_time_astar(
                grid, starts[ai], goals[ai], ci, max_timesteps=max_timesteps, h_table=hi
            )
        alt_cost_i = get_path_cost(alt_path_i) if alt_path_i is not None else float("inf")

        # Replan agent_j
        cj = build_constraint_for_agent(c, aj, per_agent_cs[aj])
        hj = h_tables[aj] if h_tables is not None else None
        fkey_j = cj.freeze_key()
        cache_key_j = (aj, fkey_j)

        if path_cache is not None:
            if cache_key_j not in path_cache:
                path_cache[cache_key_j] = space_time_astar(
                    grid, starts[aj], goals[aj], cj, max_timesteps=max_timesteps, h_table=hj
                )
            alt_path_j = path_cache[cache_key_j]
        else:
            alt_path_j = space_time_astar(
                grid, starts[aj], goals[aj], cj, max_timesteps=max_timesteps, h_table=hj
            )
        alt_cost_j = get_path_cost(alt_path_j) if alt_path_j is not None else float("inf")

        inc_i = (alt_cost_i - cost_i) if alt_path_i is not None else float("inf")
        inc_j = (alt_cost_j - cost_j) if alt_path_j is not None else float("inf")

        if inc_i > 0 and inc_j > 0:
            ctype = "CARDINAL"
        elif inc_i > 0 or inc_j > 0:
            ctype = "SEMI_CARDINAL"
        else:
            ctype = "NON_CARDINAL"

        base_c_i = agent_base_conflicts.get(ai, 1)
        base_c_j = agent_base_conflicts.get(aj, 1)

        if alt_path_i is not None:
            if cache_key_i not in alt_conflict_cache:
                alt_conflict_cache[cache_key_i] = count_single_agent_conflicts(
                    paths, ai, alt_path_i, max_path_len
                )
            agent_c_i = alt_conflict_cache[cache_key_i]
            marg_delta_i = agent_c_i - base_c_i
            alt_tot_i = total_conflicts_before + marg_delta_i
        else:
            alt_tot_i = 999999
            marg_delta_i = 999999

        if alt_path_j is not None:
            if cache_key_j not in alt_conflict_cache:
                alt_conflict_cache[cache_key_j] = count_single_agent_conflicts(
                    paths, aj, alt_path_j, max_path_len
                )
            agent_c_j = alt_conflict_cache[cache_key_j]
            marg_delta_j = agent_c_j - base_c_j
            alt_tot_j = total_conflicts_before + marg_delta_j
        else:
            alt_tot_j = 999999
            marg_delta_j = 999999

        if enable_bypass and return_bypass and ctype != "CARDINAL":
            if inc_i == 0 and alt_path_i is not None and alt_tot_i < total_conflicts_before:
                return classified_list, (ai, alt_path_i, c)
            if inc_j == 0 and alt_path_j is not None and alt_tot_j < total_conflicts_before:
                return classified_list, (aj, alt_path_j, c)

        classified_list.append(
            ClassifiedConflict(
                conflict=c,
                classification=ctype,
                cost_increase_i=int(inc_i) if inc_i != float("inf") else 999999,
                cost_increase_j=int(inc_j) if inc_j != float("inf") else 999999,
                alt_path_i=alt_path_i,
                alt_path_j=alt_path_j,
                alt_cost_i=alt_cost_i,
                alt_cost_j=alt_cost_j,
                alt_total_conflicts_i=alt_tot_i,
                alt_total_conflicts_j=alt_tot_j,
                marginal_conflict_delta_i=marg_delta_i,
                marginal_conflict_delta_j=marg_delta_j,
            )
        )

    if return_bypass:
        return classified_list, None
    return classified_list
