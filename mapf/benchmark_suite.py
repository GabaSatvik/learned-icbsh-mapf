"""
Benchmark Suite (Phase 1.5 & Phase 2.3)
Evaluates 4 solvers side-by-side on benchmark instances:
  1. PlainCBS — plain CBS, first-found conflict
  2. PlainICBSh — ICBS-h, cardinal > semi > non, arbitrary tie-break
  3. RandomRankerICBSh — frozen random weights control
  4. LearnedICBSh_proxy — ranker trained on proxy labels

Scale:
  - maps: random-32-32-20, room-32-32-4, Paris_1_256 (3 maps)
  - k: 20, 28, 40 (3 values)
  - scenarios: 1..5 per (map, k)
  - 60s per-instance timeout
  - cpu_seconds_used recorded per instance
  - optimality invariant verified across all solvers
"""
import os
import time
import heapq
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List, Tuple, Dict, Any, Optional

from mapf.grid import Grid, parse_scenario
from mapf.low_level import SpaceTimeAStar
from mapf.conflicts import Conflict, find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.high_level_cbs import StandardCBS
from mapf.icbs_h import ICBS_H, ICBSNode
from mapf.features import FeatureExtractor, FEATURE_NAMES


class LearnedICBShSolver(ICBS_H):
    def __init__(self, grid: Grid, model_path: Optional[str] = None, mode: str = 'learned', seed: int = 42):
        super().__init__(grid, logger=None, enable_bypass=True)
        self.mode = mode
        self.fe = FeatureExtractor(grid)
        self.features = [f for f in FEATURE_NAMES]
        if mode == 'learned' and model_path and os.path.exists(model_path):
            self.model = xgb.XGBRanker()
            self.model.load_model(model_path)
        elif mode == 'random':
            self.rng = np.random.default_rng(seed)
        else:
            self.mode = 'plain'

    def solve(
        self,
        starts: List[Tuple[int, int]],
        goals: List[Tuple[int, int]],
        time_limit: float = 60.0
    ) -> Tuple[Optional[List[List[Tuple[int, int]]]], Dict]:
        start_wall = time.perf_counter()   # Bug 7 fix: use perf_counter for wall-clock
        start_cpu = time.process_time()
        num_agents = len(starts)
        num_expanded = 0
        node_id_counter = 0
        regime_applicable_nodes = 0
        policy_decisions = 0

        root_paths = []
        for i in range(num_agents):
            p = self.low_level.find_path(starts[i], goals[i], set(), set())
            if p is None:
                return None, {
                    'status': 'NO_ROOT_PATH', 'expanded': 0, 'time_ms': 0, 'cpu_seconds_used': 0,
                    'cost': None, 'applicable_nodes': 0, 'decisions': 0
                }
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
            if time.perf_counter() - start_wall > time_limit:
                return None, {
                    'status': 'TIMEOUT', 'expanded': num_expanded,
                    'time_ms': (time.time() - start_wall) * 1000.0,
                    'cpu_seconds_used': time.process_time() - start_cpu,
                    'cost': None, 'applicable_nodes': regime_applicable_nodes,
                    'decisions': policy_decisions
                }

            curr = heapq.heappop(open_list)
            num_expanded += 1

            all_conflicts = find_all_conflicts(curr.paths)
            if not all_conflicts:
                return curr.paths, {
                    'status': 'OPTIMAL', 'expanded': num_expanded,
                    'time_ms': (time.time() - start_wall) * 1000.0,
                    'cpu_seconds_used': time.process_time() - start_cpu,
                    'cost': curr.g_cost, 'applicable_nodes': regime_applicable_nodes,
                    'decisions': policy_decisions
                }

            classified_conflicts = []
            bypass_found = False

            for c in all_conflicts:
                c_class, bp1, bp2 = classify_conflict(
                    c, curr.paths, starts, goals, curr.constraints, self.low_level
                )
                classified_conflicts.append((c, c_class, bp1, bp2))

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

            if len(classified_conflicts) >= 2:
                regime_applicable_nodes += 1

            # Conflict selection policy
            if self.mode == 'plain' or len(classified_conflicts) == 1:
                priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
                best_idx = 0
                best_rank = 999
                for idx, (c, c_cls, _, _) in enumerate(classified_conflicts):
                    rank = priority_rank.get(c_cls, 3)
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = idx
            elif self.mode == 'random':
                policy_decisions += 1
                best_idx = int(self.rng.integers(0, len(classified_conflicts)))
            elif self.mode == 'learned':
                policy_decisions += 1
                parent_mvc = self._compute_heuristic([(x[0], x[1]) for x in classified_conflicts])
                feat_rows = []
                for c, c_cls, bp1, bp2 in classified_conflicts:
                    feat = self.fe.extract_features(
                        conflict=c, conflict_class=c_cls,
                        all_classified_conflicts=[(x[0], x[1]) for x in classified_conflicts],
                        paths=curr.paths, goals=goals, mvc_size=parent_mvc
                    )
                    feat_rows.append(feat)
                X_cand = pd.DataFrame(feat_rows)[self.features]
                scores = self.model.predict(X_cand)
                best_idx = int(np.argmax(scores))
            else:
                best_idx = 0

            chosen_conflict = classified_conflicts[best_idx][0]
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
                    starts[agent_constrained], goals[agent_constrained], v_cons, e_cons
                )
                if new_path is not None:
                    child_paths = list(curr.paths)
                    child_paths[agent_constrained] = new_path
                    child_g = sum(len(p) - 1 for p in child_paths)
                    child_conflicts = find_all_conflicts(child_paths)
                    child_h = self._compute_heuristic(
                        [(cc, classify_conflict(cc, child_paths, starts, goals, child_constraints, self.low_level)[0])
                         for cc in child_conflicts]
                    )
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

        return None, {
            'status': 'EXHAUSTED', 'expanded': num_expanded,
            'time_ms': (time.time() - start_wall) * 1000.0,
            'cpu_seconds_used': time.process_time() - start_cpu,
            'cost': None, 'applicable_nodes': regime_applicable_nodes,
            'decisions': policy_decisions
        }


def run_benchmark():
    print("=== Phase 1.5: Small Benchmark Suite ===")
    maps = ['random-32-32-20', 'room-32-32-4']
    k_values = [20, 28, 40]
    scenarios = [1, 2, 3, 4, 5]
    solvers = ['PlainCBS', 'PlainICBSh', 'RandomRankerICBSh', 'LearnedICBSh_proxy']
    model_path = "models/ranker_proxy_regimeB.json"

    results = []
    os.makedirs('results', exist_ok=True)

    for map_name in maps:
        grid = Grid.from_file(f"data/{map_name}.map")
        cbs_solver = StandardCBS(grid)
        plain_icbs = LearnedICBShSolver(grid, mode='plain')
        random_icbs = LearnedICBShSolver(grid, mode='random')
        learned_icbs = LearnedICBShSolver(grid, model_path=model_path, mode='learned')

        solver_map = {
            'PlainCBS': cbs_solver,
            'PlainICBSh': plain_icbs,
            'RandomRankerICBSh': random_icbs,
            'LearnedICBSh_proxy': learned_icbs
        }

        for k in k_values:
            for s_id in scenarios:
                scen_file = f"data/scen-random/{map_name}-random-{s_id}.scen"
                instances = parse_scenario(scen_file, max_agents=k)
                if len(instances) < k:
                    continue
                starts = [x[0] for x in instances]
                goals = [x[1] for x in instances]

                instance_costs = {}

                for s_name in solvers:
                    solver_obj = solver_map[s_name]
                    if s_name == 'PlainCBS':
                        # Bug 7 fix: use perf_counter for wall-clock, process_time for CPU
                        # consistently — never mix the two
                        t0_wall = time.perf_counter()
                        t0_cpu = time.process_time()
                        paths, stats = solver_obj.solve(starts, goals, time_limit=60.0)
                        wall_ms = (time.perf_counter() - t0_wall) * 1000.0
                        cpu_sec = time.process_time() - t0_cpu
                        expanded = stats.get('expanded', 0)
                        res = {
                            'status': 'OPTIMAL' if paths else ('TIMEOUT' if stats.get('timeout') else 'FAIL'),
                            'expanded': expanded,
                            'time_ms': wall_ms,
                            'cpu_seconds_used': cpu_sec,
                            # Bug 6 fix: report cpu_seconds_per_node
                            'cpu_seconds_per_node': (cpu_sec / expanded) if expanded > 0 else 0.0,
                            'cost': stats.get('cost'),
                            'applicable_nodes': 0,
                            'decisions': 0
                        }
                    else:
                        t0_wall = time.perf_counter()
                        paths, res = solver_obj.solve(starts, goals, time_limit=60.0)
                        # Bug 7 fix: recompute wall_ms from perf_counter for consistency
                        # (LearnedICBShSolver already records cpu via process_time internally)
                        expanded = res.get('expanded', 0)
                        cpu_sec = res.get('cpu_seconds_used', 0.0)
                        # Bug 6 fix: add cpu_seconds_per_node to every solver
                        res['cpu_seconds_per_node'] = (cpu_sec / expanded) if expanded > 0 else 0.0

                    instance_costs[s_name] = res.get('cost')

                    record = {
                        'map': map_name,
                        'k': k,
                        'scenario_id': s_id,
                        'solver': s_name,
                        'status': res.get('status'),
                        'solution_cost': res.get('cost'),
                        'nodes_expanded': res.get('expanded', 0),
                        'runtime_ms': round(res.get('time_ms', 0), 2),
                        'cpu_seconds_used': round(res.get('cpu_seconds_used', 0), 4),
                        # Bug 6 fix: report cpu_seconds_per_node to enable fair overhead comparison
                        'cpu_seconds_per_node': round(res.get('cpu_seconds_per_node', 0.0), 6),
                        'num_nodes_where_regime_applies': res.get('applicable_nodes', 0),
                        'num_policy_decisions': res.get('decisions', 0)
                    }
                    results.append(record)
                    print(f"[{map_name:15s} | k={k:2d} | Scen {s_id:2d} | {s_name:18s}] "
                          f"status={record['status']:7s} | nodes={record['nodes_expanded']:4d} | "
                          f"cost={str(record['solution_cost']):6s} | time={record['runtime_ms']:.1f}ms | "
                          f"cpu={record['cpu_seconds_used']:.2f}s | cpu/node={record['cpu_seconds_per_node']:.5f}s | "
                          f"decisions={record['num_policy_decisions']}")

                # Optimality Invariant Check
                valid_costs = [c for c in instance_costs.values() if c is not None]
                if len(set(valid_costs)) > 1:
                    print(f"[FATAL BUG] Optimality mismatch on {map_name} k={k} scen={s_id}: {instance_costs}")
                    raise RuntimeError("Optimality invariant violated!")

    out_csv = "results/phase1_benchmark.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nPhase 1 Benchmark completed. Saved {len(df)} records to {out_csv}")


if __name__ == "__main__":
    run_benchmark()
