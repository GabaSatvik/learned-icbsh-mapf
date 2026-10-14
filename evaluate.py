"""
Evaluation and inference script:
Compares Standard CBS, ICBS-h (arbitrary tie-break), and ICBS-h with the trained XGBRanker model
on unseen benchmark scenarios on random-32-32-20 map.
"""
import os
import glob
import xgboost as xgb
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import heapq
import time

from mapf.grid import Grid, parse_scenario
from mapf.conflicts import Conflict, find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.conflict_graph import ConflictGraph
from mapf.features import extract_conflict_features
from mapf.high_level_cbs import StandardCBS
from mapf.icbs_h import ICBS_H, ICBSNode
from train_ranker import FEATURE_COLS


class LearnedICBS_H(ICBS_H):
    """
    ICBS-h solver where conflict selection among candidates is ranked by the trained XGBRanker model.
    """
    def __init__(self, grid: Grid, model_path: str = "ranker_model.json", enable_bypass: bool = True):
        super().__init__(grid, logger=None, enable_bypass=enable_bypass)
        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(model_path)

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

        root_paths = []
        for i in range(num_agents):
            p = self.low_level.find_path(starts[i], goals[i], set(), set())
            if p is None:
                return None, {'expanded': 0, 'time': time.time() - start_time, 'cost': None}
            root_paths.append(p)

        root_g = sum(len(p) - 1 for p in root_paths)
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

            # Classify all conflicts and check bypass
            classified_conflicts = []
            bypass_found = False

            for c in all_conflicts:
                c_class, bp1, bp2 = classify_conflict(
                    c, curr.paths, starts, goals, curr.constraints, self.low_level
                )
                classified_conflicts.append((c, c_class))

                if self.enable_bypass and not bypass_found:
                    if bp1 is not None:
                        new_paths = list(curr.paths)
                        new_paths[c.agent_1] = bp1
                        if len(find_all_conflicts(new_paths)) < len(all_conflicts):
                            curr.paths = new_paths
                            bypass_found = True
                            break
                    elif bp2 is not None:
                        new_paths = list(curr.paths)
                        new_paths[c.agent_2] = bp2
                        if len(find_all_conflicts(new_paths)) < len(all_conflicts):
                            curr.paths = new_paths
                            bypass_found = True
                            break

            if bypass_found:
                curr_h = self._compute_heuristic(
                    [(c, classify_conflict(c, curr.paths, starts, goals, curr.constraints, self.low_level)[0])
                     for c in find_all_conflicts(curr.paths)]
                )
                curr.h_cost = curr_h
                curr.f_cost = curr.g_cost + curr_h
                heapq.heappush(open_list, curr)
                continue

            current_mvc = self._compute_heuristic(classified_conflicts)
            features_list = [
                extract_conflict_features(c, c_cls, classified_conflicts, curr.paths, current_mvc)
                for c, c_cls in classified_conflicts
            ]

            # Use XGBRanker to predict the best conflict to branch on
            if len(features_list) == 1:
                best_idx = 0
            else:
                import pandas as pd
                X_cand = pd.DataFrame(features_list)[FEATURE_COLS]
                scores = self.ranker.predict(X_cand)
                best_idx = int(np.argmax(scores))

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


def run_benchmark_verification():
    map_file = "data/random-32-32-20.map"
    grid = Grid.from_file(map_file)

    # Use scenario 23, 24, 25 (test scenarios not seen in training)
    test_scens = [
        "data/scen-random/random-32-32-20-random-23.scen",
        "data/scen-random/random-32-32-20-random-24.scen",
        "data/scen-random/random-32-32-20-random-25.scen"
    ]

    plain_icbs = ICBS_H(grid)
    learned_icbs = LearnedICBS_H(grid, model_path="ranker_model.json")

    print("\n=================================================================")
    print(" VERIFICATION BENCHMARK: Plain ICBS-h vs Learned LTR ICBS-h")
    print(" Map: random-32-32-20 (Unseen Test Scenarios: 23, 24, 25)")
    print("=================================================================")
    print(f"{'Scenario':<25} | {'Agents':<6} | {'Plain Nodes':<12} | {'Plain Time':<10} | {'LTR Nodes':<10} | {'LTR Time':<10} | {'Cost Match':<10}")
    print("-" * 95)

    for scen_file in test_scens:
        scen_name = os.path.basename(scen_file)
        for k in [4, 6, 8, 10]:
            instances = parse_scenario(scen_file, max_agents=k)
            starts = [inst[0] for inst in instances]
            goals = [inst[1] for inst in instances]

            p_sol, p_stats = plain_icbs.solve(starts, goals, time_limit=10.0)
            l_sol, l_stats = learned_icbs.solve(starts, goals, time_limit=10.0)

            cost_match = (p_stats.get('cost') == l_stats.get('cost')) if p_sol and l_sol else False
            print(f"{scen_name:<25} | {k:<6} | {p_stats.get('expanded', -1):<12} | {p_stats.get('time', -1):.4f}s    | {l_stats.get('expanded', -1):<10} | {l_stats.get('time', -1):.4f}s  | {str(cost_match):<10}")


if __name__ == "__main__":
    run_benchmark_verification()
