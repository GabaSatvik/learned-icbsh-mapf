"""
Conflict classification and bypassing as defined in ICBS (Boyarski et al., 2015).
"""
from typing import List, Tuple, Set, Dict, Optional
from mapf.conflicts import Conflict
from mapf.low_level import SpaceTimeAStar


def get_constraints_for_agent(
    all_constraints: List[Tuple[int, str, Tuple]],
    agent_id: int
) -> Tuple[Set[Tuple[int, int, int]], Set[Tuple[int, int, int, int, int]]]:
    """
    Returns (vertex_constraints, edge_constraints) for a specific agent.
    all_constraints elements are: (agent_id, type, data)
      vertex: data is (x, y, t)
      edge: data is (from_x, from_y, to_x, to_y, t)
    """
    v_cons = set()
    e_cons = set()
    for aid, ctype, data in all_constraints:
        if aid == agent_id:
            if ctype == 'vertex':
                v_cons.add(data)
            elif ctype == 'edge':
                e_cons.add(data)
    return v_cons, e_cons


def classify_conflict(
    conflict: Conflict,
    paths: List[List[Tuple[int, int]]],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    existing_constraints: List[Tuple[int, str, Tuple]],
    low_level_solver: SpaceTimeAStar
) -> Tuple[str, Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """
    Classifies a conflict as:
      - 'cardinal': resolving it strictly increases cost for both child branches
      - 'semi-cardinal': increases cost for exactly one child branch
      - 'non-cardinal': does not increase cost for either child branch

    Also returns alternate paths found (if any) with the same cost, which can be used for bypassing!
    """
    a1, a2 = conflict.agent_1, conflict.agent_2
    orig_cost_1 = len(paths[a1]) - 1
    orig_cost_2 = len(paths[a2]) - 1

    # Format constraints for child 1 (constraining a1) and child 2 (constraining a2)
    if conflict.conflict_type == 'vertex':
        c1 = (a1, 'vertex', (conflict.loc_1[0], conflict.loc_1[1], conflict.time))
        c2 = (a2, 'vertex', (conflict.loc_1[0], conflict.loc_1[1], conflict.time))
    else:
        # Edge conflict: a1 cannot go loc_1 -> loc_2 at time t
        # a2 cannot go loc_2 -> loc_1 at time t
        c1 = (a1, 'edge', (conflict.loc_1[0], conflict.loc_1[1], conflict.loc_2[0], conflict.loc_2[1], conflict.time))
        c2 = (a2, 'edge', (conflict.loc_2[0], conflict.loc_2[1], conflict.loc_1[0], conflict.loc_1[1], conflict.time))

    v1, e1 = get_constraints_for_agent(existing_constraints + [c1], a1)
    new_path_1 = low_level_solver.find_path(starts[a1], goals[a1], v1, e1)

    v2, e2 = get_constraints_for_agent(existing_constraints + [c2], a2)
    new_path_2 = low_level_solver.find_path(starts[a2], goals[a2], v2, e2)

    cost_increase_1 = (new_path_1 is None) or ((len(new_path_1) - 1) > orig_cost_1)
    cost_increase_2 = (new_path_2 is None) or ((len(new_path_2) - 1) > orig_cost_2)

    bypass_path_1 = new_path_1 if (new_path_1 is not None and len(new_path_1) - 1 == orig_cost_1) else None
    bypass_path_2 = new_path_2 if (new_path_2 is not None and len(new_path_2) - 1 == orig_cost_2) else None

    if cost_increase_1 and cost_increase_2:
        return 'cardinal', None, None
    elif cost_increase_1 or cost_increase_2:
        return 'semi-cardinal', bypass_path_1, bypass_path_2
    else:
        return 'non-cardinal', bypass_path_1, bypass_path_2
