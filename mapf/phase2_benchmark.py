"""
Phase 2 Benchmark Suite
=======================
Compares 4 solvers using the ROLLOUT-trained model:
  1. PlainCBS            -- plain CBS baseline
  2. PlainICBSh          -- ICBS-h, cardinal > semi > non (no learned ranking)
  3. RandomRankerICBSh   -- random conflict selection (control)
  4. LearnedICBSh_rollout-- XGBRanker trained on real rollout labels (Phase 2)

Maps: random-32-32-20, room-32-32-4
k   : 20, 28
Scen: 1..5
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List, Tuple, Optional, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mapf.grid import Grid, parse_scenario
from mapf.benchmark_suite import LearnedICBShSolver
from mapf.high_level_cbs import StandardCBS

ROLLOUT_MODEL = "models/ranker_rollout_regimeB.json"


def run_phase2_benchmark(
    maps=('random-32-32-20', 'room-32-32-4'),
    k_values=(20, 28),
    scenarios=range(1, 6),
    time_limit=60.0,
    output_csv='results/phase2_benchmark.csv'
):
    print("=" * 70)
    print("Phase 2 Benchmark — Rollout-Trained Learned Ranker vs Baselines")
    print("=" * 70)
    print(f"Model : {ROLLOUT_MODEL}")
    print(f"Maps  : {list(maps)}")
    print(f"k     : {list(k_values)}")
    print(f"Scens : {list(scenarios)}")
    print()

    os.makedirs('results', exist_ok=True)
    results = []
    solver_names = ['PlainCBS', 'PlainICBSh', 'RandomRankerICBSh', 'LearnedICBSh_rollout']

    for map_name in maps:
        map_path = f"data/{map_name}.map"
        if not os.path.exists(map_path):
            print(f"SKIP: {map_path} not found")
            continue

        grid = Grid.from_file(map_path)

        cbs_solver     = StandardCBS(grid)
        plain_icbs     = LearnedICBShSolver(grid, mode='plain')
        random_icbs    = LearnedICBShSolver(grid, mode='random', seed=99)
        learned_icbs   = LearnedICBShSolver(grid, model_path=ROLLOUT_MODEL, mode='learned')

        solver_map = {
            'PlainCBS'             : cbs_solver,
            'PlainICBSh'           : plain_icbs,
            'RandomRankerICBSh'    : random_icbs,
            'LearnedICBSh_rollout' : learned_icbs,
        }

        for k in k_values:
            for s_id in scenarios:
                scen_file = f"data/scen-random/{map_name}-random-{s_id}.scen"
                if not os.path.exists(scen_file):
                    continue
                instances = parse_scenario(scen_file, max_agents=k)
                if len(instances) < k:
                    continue
                starts = [x[0] for x in instances[:k]]
                goals  = [x[1] for x in instances[:k]]

                instance_costs = {}

                for s_name in solver_names:
                    solver_obj = solver_map[s_name]
                    t0w = time.time()
                    t0c = time.process_time()

                    if s_name == 'PlainCBS':
                        paths, stats = solver_obj.solve(starts, goals, time_limit=time_limit)
                        res = {
                            'status'           : 'OPTIMAL' if paths else ('TIMEOUT' if stats.get('timeout') else 'FAIL'),
                            'expanded'         : stats.get('expanded', 0),
                            'time_ms'          : (time.time() - t0w) * 1000.0,
                            'cpu_seconds_used' : time.process_time() - t0c,
                            'cost'             : stats.get('cost'),
                            'applicable_nodes' : 0,
                            'decisions'        : 0,
                        }
                    else:
                        paths, res = solver_obj.solve(starts, goals, time_limit=time_limit)

                    instance_costs[s_name] = res.get('cost')

                    record = {
                        'map'            : map_name,
                        'k'              : k,
                        'scenario_id'    : s_id,
                        'solver'         : s_name,
                        'status'         : res.get('status'),
                        'solution_cost'  : res.get('cost'),
                        'nodes_expanded' : res.get('expanded', 0),
                        'runtime_ms'     : round(res.get('time_ms', 0), 2),
                        'cpu_sec'        : round(res.get('cpu_seconds_used', 0), 4),
                        'policy_decisions': res.get('decisions', 0),
                    }
                    results.append(record)

                    tag = f"[{map_name:15s}|k={k:2d}|s={s_id}|{s_name:22s}]"
                    print(f"{tag} {record['status']:7s} nodes={record['nodes_expanded']:5d} "
                          f"cost={str(record['solution_cost']):6s} "
                          f"time={record['runtime_ms']:7.1f}ms decisions={record['policy_decisions']}")

                # Optimality check
                valid = [c for c in instance_costs.values() if c is not None]
                if len(set(valid)) > 1:
                    print(f"  [FATAL] Optimality mismatch! {instance_costs}")
                    raise RuntimeError("Optimality invariant violated!")

                print()

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Summary table
    print("=" * 70)
    print("SUMMARY: Median Nodes Expanded per (map, k, solver)")
    print("=" * 70)
    optimal_df = df[df['status'] == 'OPTIMAL']
    if not optimal_df.empty:
        pivot = optimal_df.pivot_table(
            values='nodes_expanded',
            index=['map', 'k'],
            columns='solver',
            aggfunc='median'
        )
        print(pivot.to_string())
        print()

        # Compute reduction vs PlainICBSh
        print("--- Node Reduction: LearnedICBSh_rollout vs PlainICBSh (median) ---")
        for (map_name, k), row in pivot.iterrows():
            plain = row.get('PlainICBSh', None)
            learned = row.get('LearnedICBSh_rollout', None)
            if plain is not None and learned is not None and plain > 0:
                pct = 100.0 * (plain - learned) / plain
                sign = "REDUCTION" if learned < plain else "INCREASE"
                print(f"  {map_name} k={k}: Plain={plain:.0f}  Learned={learned:.0f}  "
                      f"Delta={plain-learned:+.0f} ({pct:+.1f}%)  [{sign}]")

        # Optimality check summary
        print()
        total = len(df['map'].unique()) * len(df['k'].unique()) * len(list(scenarios)) * len(solver_names)
        opt_count = len(df[df['status'] == 'OPTIMAL'])
        print(f"Optimality: {opt_count}/{len(df)} runs solved optimally")
        cost_match = df.groupby(['map', 'k', 'scenario_id'])['solution_cost'].nunique()
        mismatches = (cost_match[cost_match > 1]).sum()
        print(f"Cost mismatches across solvers: {mismatches} (should be 0)")

    print(f"\nResults saved -> {output_csv}")
    print("=" * 70)
    return output_csv


if __name__ == '__main__':
    run_phase2_benchmark()
