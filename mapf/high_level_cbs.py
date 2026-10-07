"""
Standard Conflict-Based Search (Sharon et al., 2012).
"""
import heapq
import time
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass, field
from mapf.grid import Grid
from mapf.low_level import SpaceTimeAStar
from mapf.conflicts import Conflict, find_first_conflict
from mapf.cardinal import get_constraints_for_agent


@dataclass(order=True)
class CTNode:
    cost: int
    id: int
    paths: List[List[Tuple[int, int]]] = field(compare=False)
    constraints: List[Tuple[int, str, Tuple]] = field(compare=False)  # (agent_id, type, data)
    conflicts: List[Conflict] = field(default_factory=list, compare=False)


class StandardCBS:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.low_level = SpaceTimeAStar(grid)

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        time_limit: float = 30.0
    ) -> Tuple[Optional[List[List[Tuple[int, int]]]], Dict]:
        start_time = time.time()
        num_agents = len(starts)
        num_expanded = 0
        node_id_counter = 0

        # Root node
        root_paths = []
        for i in range(num_agents):
            p = self.low_level.find_path(starts[i], goals[i], set(), set())
            if p is None:
                return None, {'expanded': 0, 'time': time.time() - start_time, 'cost': None}
            root_paths.append(p)

        root_cost = sum(len(p) - 1 for p in root_paths)
        root = CTNode(
            cost=root_cost,
            id=node_id_counter,
            paths=root_paths,
            constraints=[]
        )
        node_id_counter += 1

        open_list = [root]

        while open_list:
            if time.time() - start_time > time_limit:
                return None, {'expanded': num_expanded, 'time': time.time() - start_time, 'timeout': True}

            curr = heapq.heappop(open_list)
            num_expanded += 1

            conflict = find_first_conflict(curr.paths)
            if conflict is None:
                return curr.paths, {
                    'expanded': num_expanded,
                    'time': time.time() - start_time,
                    'cost': curr.cost
                }

            # Branch on conflict
            a1, a2 = conflict.agent_1, conflict.agent_2
            if conflict.conflict_type == 'vertex':
                c1 = (a1, 'vertex', (conflict.loc_1[0], conflict.loc_1[1], conflict.time))
                c2 = (a2, 'vertex', (conflict.loc_1[0], conflict.loc_1[1], conflict.time))
            else:
                c1 = (a1, 'edge', (conflict.loc_1[0], conflict.loc_1[1], conflict.loc_2[0], conflict.loc_2[1], conflict.time))
                c2 = (a2, 'edge', (conflict.loc_2[0], conflict.loc_2[1], conflict.loc_1[0], conflict.loc_1[1], conflict.time))

            for agent_constrained, new_constraint in [(a1, c1), (a2, c2)]:
                child_constraints = list(curr.constraints) + [new_constraint]
                v_cons, e_cons = get_constraints_for_agent(child_constraints, agent_constrained)

                new_path = self.low_level.find_path(
                    starts[agent_constrained],
                    goals[agent_constrained],
                    v_cons,
                    e_cons
                )

                if new_path is not None:
                    child_paths = list(curr.paths)
                    child_paths[agent_constrained] = new_path
                    child_cost = sum(len(p) - 1 for p in child_paths)
                    child_node = CTNode(
                        cost=child_cost,
                        id=node_id_counter,
                        paths=child_paths,
                        constraints=child_constraints
                    )
                    node_id_counter += 1
                    heapq.heappush(open_list, child_node)

        return None, {'expanded': num_expanded, 'time': time.time() - start_time, 'cost': None}
