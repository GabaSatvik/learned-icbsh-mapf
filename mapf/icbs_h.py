"""
ICBS-h implementation (Improved Conflict-Based Search with MVC Heuristic).
Supports:
  1. Conflict classification (cardinal, semi-cardinal, non-cardinal)
  2. Bypassing conflicts
  3. Conflict graph and Minimum Vertex Cover (MVC) admissible heuristic
  4. Instrumenting conflict choices to the ConflictDataLogger for Learning-to-Rank.
"""
import heapq
import time
from typing import List, Tuple, Set, Optional, Dict, Any
from dataclasses import dataclass, field

from mapf.grid import Grid
from mapf.low_level import SpaceTimeAStar
from mapf.conflicts import Conflict, find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.conflict_graph import ConflictGraph
from mapf.features import extract_conflict_features
from mapf.logger import ConflictDataLogger


@dataclass(order=True)
class ICBSNode:
    f_cost: int
    h_cost: int
    g_cost: int
    id: int
    paths: List[List[Tuple[int, int]]] = field(compare=False)
    constraints: List[Tuple[int, str, Tuple]] = field(compare=False)


class ICBS_H:
    def __init__(self, grid: Grid, logger: Optional[ConflictDataLogger] = None, enable_bypass: bool = True):
        self.grid = grid
        self.low_level = SpaceTimeAStar(grid)
        self.logger = logger
        self.enable_bypass = enable_bypass

    def _compute_heuristic(self, classified_conflicts: List[Tuple[Conflict, str]]) -> int:
        """Admissible heuristic = size of Minimum Vertex Cover of the cardinal conflict graph."""
        cardinal_edges = [
            (c.agent_1, c.agent_2)
            for c, cls in classified_conflicts
            if cls == 'cardinal'
        ]
        if not cardinal_edges:
            return 0
        cg = ConflictGraph(cardinal_edges)
        return cg.compute_mvc()

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        time_limit: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None
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

        root_g = sum(len(p) - 1 for p in root_paths)

        # Classify conflicts for root
        initial_conflicts = find_all_conflicts(root_paths)
        classified_initial = [
            (c, classify_conflict(c, root_paths, starts, goals, [], self.low_level)[0])
            for c in initial_conflicts
        ]
        root_h = self._compute_heuristic(classified_initial)
        root = ICBSNode(
            f_cost=root_g + root_h,
            h_cost=root_h,
            g_cost=root_g,
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

            all_conflicts = find_all_conflicts(curr.paths)
            if not all_conflicts:
                return curr.paths, {
                    'expanded': num_expanded,
                    'time': time.time() - start_time,
                    'cost': curr.g_cost
                }

            # Classify all conflicts and check for bypass opportunities
            classified_conflicts = []
            bypass_found = False

            for c in all_conflicts:
                c_class, bp1, bp2 = classify_conflict(
                    c, curr.paths, starts, goals, curr.constraints, self.low_level
                )
                classified_conflicts.append((c, c_class))

                # Check bypass (if enabled, adopt alternative path without expanding CT)
                if self.enable_bypass and not bypass_found:
                    if bp1 is not None:
                        new_paths = list(curr.paths)
                        new_paths[c.agent_1] = bp1
                        new_conflicts = find_all_conflicts(new_paths)
                        if len(new_conflicts) < len(all_conflicts):
                            curr.paths = new_paths
                            bypass_found = True
                            break
                    elif bp2 is not None:
                        new_paths = list(curr.paths)
                        new_paths[c.agent_2] = bp2
                        new_conflicts = find_all_conflicts(new_paths)
                        if len(new_conflicts) < len(all_conflicts):
                            curr.paths = new_paths
                            bypass_found = True
                            break

            if bypass_found:
                # Re-insert curr node into open list with updated paths
                curr_h = self._compute_heuristic(
                    [(c, classify_conflict(c, curr.paths, starts, goals, curr.constraints, self.low_level)[0])
                     for c in find_all_conflicts(curr.paths)]
                )
                curr.h_cost = curr_h
                curr.f_cost = curr.g_cost + curr_h
                heapq.heappush(open_list, curr)
                continue

            # Prioritize: Cardinal > Semi-Cardinal > Non-Cardinal
            # Extract features for all candidate conflicts
            current_mvc = self._compute_heuristic(classified_conflicts)
            features_list = []
            for c, c_cls in classified_conflicts:
                feat = extract_conflict_features(
                    c, c_cls, classified_conflicts, curr.paths, current_mvc
                )
                features_list.append(feat)

            # Standard ICBS preference: pick first cardinal, else semi, else non
            priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
            best_idx = 0
            best_rank = 999
            for idx, (c, c_cls) in enumerate(classified_conflicts):
                rank = priority_rank.get(c_cls, 3)
                if rank < best_rank:
                    best_rank = rank
                    best_idx = idx

            # Log this decision point if logger is active
            if self.logger is not None and len(classified_conflicts) >= 1:
                qid = self.logger.get_next_query_id()
                self.logger.log_branch_decision(
                    query_id=qid,
                    candidate_features=features_list,
                    chosen_index=best_idx,
                    metadata=metadata
                )

            chosen_conflict = classified_conflicts[best_idx][0]

            # Branch on chosen conflict
            a1, a2 = chosen_conflict.agent_1, chosen_conflict.agent_2
            if chosen_conflict.conflict_type == 'vertex':
                c1 = (a1, 'vertex', (chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.time))
                c2 = (a2, 'vertex', (chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.time))
            else:
                c1 = (a1, 'edge', (chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.loc_2[0], chosen_conflict.loc_2[1], chosen_conflict.time))
                c2 = (a2, 'edge', (chosen_conflict.loc_2[0], chosen_conflict.loc_2[1], chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.time))

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
                    child_g = sum(len(p) - 1 for p in child_paths)

                    # Compute ICBS-h heuristic for child node
                    child_conflicts = find_all_conflicts(child_paths)
                    classified_child = [
                        (cc, classify_conflict(cc, child_paths, starts, goals, child_constraints, self.low_level)[0])
                        for cc in child_conflicts
                    ]
                    child_h = self._compute_heuristic(classified_child)

                    child_node = ICBSNode(
                        f_cost=child_g + child_h,
                        h_cost=child_h,
                        g_cost=child_g,
                        id=node_id_counter,
                        paths=child_paths,
                        constraints=child_constraints
                    )
                    node_id_counter += 1
                    heapq.heappush(open_list, child_node)

        return None, {'expanded': num_expanded, 'time': time.time() - start_time, 'cost': None}
