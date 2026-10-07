"""
Phase 0 — Baseline Compute Calibration
Measures compute scaling, checks optimality invariant on hand-built instances,
evaluates k=20 instances on random-32-32-20, extrapolates compute budgets,
and applies hard compute-reduction rules per instructions.
"""
import time
import os
import sys
from typing import List, Tuple, Dict

from mapf.grid import Grid, parse_scenario
from mapf.high_level_cbs import StandardCBS
from mapf.icbs_h import ICBS_H


def run_handbuilt_calibration() -> bool:
    print("=== Step 1: Hand-built 2-agent optimality check (3 instances) ===")
    test_cases = [
        # Instance 1: 5x5 head-on
        (Grid(5, 5, set()), [(0, 2), (4, 2)], [(4, 2), (0, 2)]),
        # Instance 2: 5x5 diagonal cross
        (Grid(5, 5, set()), [(0, 0), (4, 4)], [(4, 4), (0, 0)]),
        # Instance 3: 5x5 with a central obstacle
        (Grid(5, 5, {(2, 2)}), [(1, 2), (3, 2)], [(3, 2), (1, 2)])
    ]
    all_matched = True
    for idx, (grid, starts, goals) in enumerate(test_cases, 1):
        cbs = StandardCBS(grid)
        icbs = ICBS_H(grid)
        _, cbs_stats = cbs.solve(starts, goals, time_limit=5.0)
        _, icbs_stats = icbs.solve(starts, goals, time_limit=5.0)

        cbs_cost = cbs_stats.get('cost')
        icbs_cost = icbs_stats.get('cost')
        match = (cbs_cost == icbs_cost and cbs_cost is not None)
        print(f"  Test {idx}: CBS cost={cbs_cost} (nodes={cbs_stats.get('expanded')}) | "
              f"ICBS-h cost={icbs_cost} (nodes={icbs_stats.get('expanded')}) -> Match={match}")
        if not match:
            all_matched = False
    return all_matched


def run_benchmark_calibration() -> Tuple[float, float, int]:
    print("\n=== Step 2: random-32-32-20 at k=20, scenarios 1..3 ===")
    map_file = "data/random-32-32-20.map"
    grid = Grid.from_file(map_file)
    cbs = StandardCBS(grid)
    icbs = ICBS_H(grid)

    total_cbs_cpu = 0.0
    total_icbs_cpu = 0.0
    total_nodes = 0
    scenarios = [
        "data/scen-random/random-32-32-20-random-1.scen",
        "data/scen-random/random-32-32-20-random-2.scen",
        "data/scen-random/random-32-32-20-random-3.scen"
    ]

    for scen_file in scenarios:
        inst = parse_scenario(scen_file, max_agents=20)
        starts = [x[0] for x in inst]
        goals = [x[1] for x in inst]

        t0_cpu = time.process_time()
        t0_wall = time.time()
        _, s_cbs = cbs.solve(starts, goals, time_limit=15.0)
        cbs_wall = time.time() - t0_wall
        cbs_cpu = time.process_time() - t0_cpu
        total_cbs_cpu += cbs_cpu

        t1_cpu = time.process_time()
        t1_wall = time.time()
        _, s_icbs = icbs.solve(starts, goals, time_limit=15.0)
        icbs_wall = time.time() - t1_wall
        icbs_cpu = time.process_time() - t1_cpu
        total_icbs_cpu += icbs_cpu
        total_nodes += s_icbs.get('expanded', 0)

        print(f"  {os.path.basename(scen_file)} (k=20):")
        print(f"    Plain CBS:  nodes={s_cbs.get('expanded')}, cost={s_cbs.get('cost')}, wall={cbs_wall:.3f}s, cpu={cbs_cpu:.3f}s")
        print(f"    ICBS-h:     nodes={s_icbs.get('expanded')}, cost={s_icbs.get('cost')}, wall={icbs_wall:.3f}s, cpu={icbs_cpu:.3f}s")

    avg_icbs_cpu_per_instance = total_icbs_cpu / len(scenarios)
    avg_nodes_per_instance = max(total_nodes // len(scenarios), 1)
    return avg_icbs_cpu_per_instance, total_icbs_cpu, avg_nodes_per_instance


def evaluate_budget_and_reductions(avg_cpu_per_inst: float, avg_nodes_per_inst: int) -> Dict[str, str]:
    print("\n=== Step 3 & 4: Extrapolate Phase 1 and Phase 2 compute cost ===")
    # Phase 1: Proxy labeling: ~0.05s per query (no rollout)
    # Phase 2: Rollout labeling: B_rollout=300, T_rollout=0.5s, top-3 candidates
    # Estimated queries per instance at training (3 maps, 3 k-values, 5 scenarios = 45 instances)
    total_training_instances = 45
    est_queries_per_inst = min(100, max(15, avg_nodes_per_inst))
    
    # In Phase 1, proxy extraction is fast (pure feature calculation):
    phase1_est_cpu_hours = (total_training_instances * est_queries_per_inst * 0.005) / 3600.0
    
    # In Phase 2 rollout labeling: each query evaluates 3 candidates with up to 0.5s rollout
    # Average rollout duration: ~0.15s per candidate * 3 candidates = 0.45s per query
    phase2_est_cpu_hours = (total_training_instances * est_queries_per_inst * 3 * 0.15) / 3600.0
    # Add full benchmark sweep (e.g. 3 maps * 3 k * 10 scenarios * 6 solvers * ~0.5s)
    phase2_bench_cpu_hours = (3 * 3 * 10 * 6 * avg_cpu_per_inst) / 3600.0
    phase2_total_cpu_hours = phase2_est_cpu_hours + phase2_bench_cpu_hours

    print(f"  Estimated Phase 1 proxy labeling compute: {phase1_est_cpu_hours:.3f} CPU-hours")
    print(f"  Estimated Phase 2 rollout labeling compute: {phase2_est_cpu_hours:.3f} CPU-hours")
    print(f"  Estimated Phase 2 full benchmark compute: {phase2_bench_cpu_hours:.3f} CPU-hours")
    print(f"  Estimated Phase 2 total compute: {phase2_total_cpu_hours:.3f} CPU-hours")

    config = {
        'max_queries_per_instance': '100',
        'drop_paris_from_training': 'False',
        'training_scenarios_count': '5',
        'status': 'full scale affordable'
    }

    reductions_applied = []
    if phase1_est_cpu_hours > 4.0:
        config['max_queries_per_instance'] = '50'
        reductions_applied.append("Reduced max_queries_per_instance to 50 (Phase 1 rollout/proxy threshold exceeded)")

    if phase2_total_cpu_hours > 30.0:
        config['max_queries_per_instance'] = '50'
        config['drop_paris_from_training'] = 'True'
        reductions_applied.append("Reduced max_queries_per_instance to 50 and dropped Paris_1_256 from training (Phase 2 > 30 CPU-hours)")

    if phase2_total_cpu_hours > 50.0:
        config['training_scenarios_count'] = '3'
        reductions_applied.append("Reduced training scenarios to 3 per (map, k) (Phase 2 > 50 CPU-hours)")

    if reductions_applied:
        print("\n[AUTO-REDUCTION APPLIED]")
        for r in reductions_applied:
            print("  *", r)
        config['status'] = "; ".join(reductions_applied)
    else:
        print("\n[RESULT] full scale affordable")

    return config


def main():
    opt_ok = run_handbuilt_calibration()
    if not opt_ok:
        print("[ERROR] Hand-built optimality check failed! Halting.")
        sys.exit(1)

    avg_cpu, total_cpu, avg_nodes = run_benchmark_calibration()
    config = evaluate_budget_and_reductions(avg_cpu, avg_nodes)

    os.makedirs('results', exist_ok=True)
    with open('results/phase0_summary.txt', 'w') as f:
        f.write(f"Handbuilt optimality passed: {opt_ok}\n")
        f.write(f"Benchmark average ICBS-h CPU seconds: {avg_cpu:.4f}\n")
        f.write(f"Status: {config['status']}\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    print("Calibration summary saved to results/phase0_summary.txt")


if __name__ == "__main__":
    main()
