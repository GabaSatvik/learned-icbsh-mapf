"""
Conflict classification and bypassing as defined in ICBS (Boyarski et al., 2015).
"""
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass

from mapf.conflicts import Conflict, Constraint
from mapf.low_level import space_time_astar, compute_bfs_h_table, SpaceTimeAStar


@dataclass
class ClassifiedConflict:
    conflict: Conflict
    conflict_type: str  # "CARDINAL", "SEMI_CARDINAL", "NON_CARDINAL"
    alt_path_i: Optional[List[Tuple[int, int]]] = None
    alt_path_j: Optional[List[Tuple[int, int]]] = None
    alt_cost_i: float = 0.0
    alt_cost_j: float = 0.0


def get_constraints_for_agent(
    all_constraints: List[Any],
    agent_id: int
) -> Tuple[Set[Tuple[int, int, int]], Set[Tuple[int, int, int, int, int]]]:
    """
    Returns (vertex_constraints, edge_constraints) for a specific agent.
    """
    v_cons = set()
    e_cons = set()
    for c in all_constraints:
        if hasattr(c, "agent"):
            if c.agent == agent_id:
                if str(c.constraint_type).upper() == "VERTEX":
                    v_cons.add((c.loc_1[0], c.loc_1[1], c.time))
                else:
                    v_cons_loc2 = c.loc_2 if c.loc_2 is not None else c.loc_1
                    e_cons.add((c.loc_1[0], c.loc_1[1], v_cons_loc2[0], v_cons_loc2[1], c.time))
        elif isinstance(c, tuple) and len(c) == 3 and isinstance(c[1], str):
            aid, ctype, data = c
            if aid == agent_id:
                if str(ctype).lower() == "vertex":
                    v_cons.add(data)
                elif str(ctype).lower() == "edge":
                    e_cons.add(data)
    return v_cons, e_cons


def classify_conflict(
    conflict: Conflict,
    paths: List[List[Tuple[int, int]]],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    existing_constraints: List[Any],
    low_level_solver: SpaceTimeAStar
) -> Tuple[str, Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """Legacy classify_conflict returning (str, bypass_1, bypass_2)."""
    grid = low_level_solver.grid
    a1, a2 = conflict.agent_1, conflict.agent_2
    orig_cost_1 = len(paths[a1]) - 1
    orig_cost_2 = len(paths[a2]) - 1

    c1 = conflict.get_constraint_for_agent(a1, a2)
    c2 = conflict.get_constraint_for_agent(a2, a1)

    v1, e1 = get_constraints_for_agent(list(existing_constraints) + [c1], a1)
    new_path_1 = low_level_solver.find_path(starts[a1], goals[a1], v1, e1)

    v2, e2 = get_constraints_for_agent(list(existing_constraints) + [c2], a2)
    new_path_2 = low_level_solver.find_path(starts[a2], goals[a2], v2, e2)

    cost_inc_1 = (new_path_1 is None) or ((len(new_path_1) - 1) > orig_cost_1)
    cost_inc_2 = (new_path_2 is None) or ((len(new_path_2) - 1) > orig_cost_2)

    bp1 = new_path_1 if (new_path_1 is not None and len(new_path_1) - 1 == orig_cost_1) else None
    bp2 = new_path_2 if (new_path_2 is not None and len(new_path_2) - 1 == orig_cost_2) else None

    if cost_inc_1 and cost_inc_2:
        return "cardinal", None, None
    elif cost_inc_1 or cost_inc_2:
        return "semi-cardinal", bp1, bp2
    else:
        return "non-cardinal", bp1, bp2


def classify_conflicts(
    grid,
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    paths: List[List[Tuple[int, int]]],
    constraints: List[Any],
    conflicts: List[Conflict],
    max_timesteps: int = 1024,
    enable_bypass: bool = False,
    h_tables: Optional[List[Dict[Tuple[int, int], int]]] = None,
) -> List[ClassifiedConflict]:
    """
    Classifies all conflicts into CARDINAL, SEMI_CARDINAL, or NON_CARDINAL.
    Caches alt_path_i, alt_cost_i, alt_path_j, alt_cost_j for exact zero-A* lookahead.
    """
    if h_tables is None:
        h_tables = [compute_bfs_h_table(grid, goals[i]) for i in range(len(starts))]

    classified = []
    for c in conflicts:
        i, j = c.agent_i, c.agent_j
        cost_i = len(paths[i]) - 1
        cost_j = len(paths[j]) - 1

        con_i = c.get_constraint_for_agent(i, j)
        cons_i = [cn for cn in constraints if getattr(cn, "agent", -1) == i] + [con_i]
        alt_p_i = space_time_astar(grid, starts[i], goals[i], cons_i, max_timesteps, h_tables[i])

        if alt_p_i is None:
            inc_i = True
            alt_c_i = float(cost_i + 50.0)
        else:
            alt_c_i = float(len(alt_p_i) - 1)
            inc_i = (alt_c_i > cost_i)

        con_j = c.get_constraint_for_agent(j, i)
        cons_j = [cn for cn in constraints if getattr(cn, "agent", -1) == j] + [con_j]
        alt_p_j = space_time_astar(grid, starts[j], goals[j], cons_j, max_timesteps, h_tables[j])

        if alt_p_j is None:
            inc_j = True
            alt_c_j = float(cost_j + 50.0)
        else:
            alt_c_j = float(len(alt_p_j) - 1)
            inc_j = (alt_c_j > cost_j)

        if inc_i and inc_j:
            ctype = "CARDINAL"
        elif inc_i or inc_j:
            ctype = "SEMI_CARDINAL"
        else:
            ctype = "NON_CARDINAL"

        classified.append(
            ClassifiedConflict(
                conflict=c,
                conflict_type=ctype,
                alt_path_i=alt_p_i,
                alt_path_j=alt_p_j,
                alt_cost_i=alt_c_i,
                alt_cost_j=alt_c_j,
            )
        )
    return classified
