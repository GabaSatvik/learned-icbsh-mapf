# mapf/phase3_benchmark.py
"""
Definitive 6-Solver, 25-Scenario Statistical Benchmark Suite for CL-ICBS-h.
Parallelized across CPU cores with Wilcoxon signed-rank significance tests and 95% CIs.
"""
import os
import time
import random
import multiprocessing as mp
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
try:
    from xgboost import XGBRanker
except ImportError:
    from mapf.baselines import XGBRanker
from mapf.grid import load_map, load_scenarios
from mapf.low_level import compute_bfs_h_table
from mapf.high_level_cbs import PlainCBS
from mapf.icbs_h import ICBShSolver
from mapf.baselines import (
    disjoint_splitting_selector,
    f_aware_residual_selector,
    GuardedLearnedSelector,
)

_WORKER_CACHE: Dict[str, Any] = {}


def _run_single_instance_all_solvers(args) -> List[Dict[str, Any]]:
    map_name, map_path, scen_path, k, scen_id, timeout, model_path = args
    if map_path not in _WORKER_CACHE:
        _WORKER_CACHE[map_path] = load_map(map_path)
    grid = _WORKER_CACHE[map_path]

    if model_path not in _WORKER_CACHE:
        xgb_model = XGBRanker()
        xgb_model.load_model(model_path)
        _WORKER_CACHE[model_path] = xgb_model
    xgb_model = _WORKER_CACHE[model_path]

    scenarios = load_scenarios(scen_path, k, max_scenarios=scen_id)
    if len(scenarios) < scen_id:
        return []
    starts, goals = scenarios[scen_id - 1]

    # Precompute BFS heuristic tables and shared low-level A* cache once for this instance
    shared_h_tables = [compute_bfs_h_table(grid, goals[i]) for i in range(k)]
    shared_path_cache = {}

    rng = random.Random(42 + scen_id * 100 + k)

    def random_selector(cands, ctx):
        return rng.choice(cands)

    cl_selector = GuardedLearnedSelector(xgb_model, top_k_guardrail=3)

    solvers = [
        ("PlainCBS", PlainCBS(grid)),
        ("PlainICBSh", ICBShSolver(grid, conflict_selector=None, regime="A")),
        ("RandomRankerICBSh", ICBShSolver(grid, conflict_selector=random_selector, regime="A")),
        ("DisjointSplittingICBSh", ICBShSolver(grid, conflict_selector=disjoint_splitting_selector, regime="A")),
        ("FAwareResidualICBSh", ICBShSolver(grid, conflict_selector=f_aware_residual_selector, regime="A")),
        ("Learned_CL_ICBSh", ICBShSolver(grid, conflict_selector=cl_selector, regime="A")),
    ]

    rows = []
    solved_costs = []
    for s_name, solver in solvers:
        t0 = time.perf_counter()
        if s_name == "PlainCBS":
            res = solver.solve(
                starts,
                goals,
                timeout=min(timeout, 1.5),
                max_expansions=350,
                shared_h_tables=shared_h_tables,
                shared_path_cache=shared_path_cache,
            )
        else:
            res = solver.solve(
                starts,
                goals,
                timeout=min(timeout, 2.5),
                max_expansions=150,
                shared_h_tables=shared_h_tables,
                shared_path_cache=shared_path_cache,
            )
        wall_t = time.perf_counter() - t0
        if res["status"] == "SOLVED" and res["cost"] is not None:
            solved_costs.append((s_name, res["cost"]))
        rows.append({
            "map_name": map_name,
            "k": k,
            "scenario_id": scen_id,
            "solver": s_name,
            "status": res["status"],
            "cost": res["cost"],
            "nodes_expanded": res["nodes_expanded"],
            "runtime": round(wall_t, 4),
            "decision_nodes": res.get("decision_nodes", 0),
        })

    # Verify 100% Optimality Invariant across all solvers that solved this instance
    if len(solved_costs) >= 2:
        ref_solver, ref_cost = solved_costs[0]
        for s_name, c_val in solved_costs[1:]:
            if c_val != ref_cost:
                raise AssertionError(
                    f"OPTIMALITY VIOLATION on {map_name} k={k} scen={scen_id}: "
                    f"{ref_solver}={ref_cost} vs {s_name}={c_val}"
                )

    return rows


def run_phase3_benchmark(
    model_path: str = "models/ranker_cl_oracle.json",
    output_csv: str = "results/phase3_benchmark.csv",
    report_out: str = "results/phase3_statistical_summary.md",
    n_scenarios: int = 25,
    timeout: float = 60.0,
    num_workers: int = 12,
):
    configs = [
        ("random-32-32-20", "data/random-32-32-20.map", "data/random-32-32-20-random-1.scen", [20, 28]),
        ("room-32-32-4", "data/room-32-32-4.map", "data/room-32-32-4-random-1.scen", [16, 20]),
    ]

    tasks = []
    for map_name, map_path, scen_path, k_list in configs:
        if not os.path.exists(map_path) or not os.path.exists(scen_path):
            continue
        for k in k_list:
            for scen_id in range(1, n_scenarios + 1):
                tasks.append((map_name, map_path, scen_path, k, scen_id, timeout, model_path))

    n_procs = min(num_workers, mp.cpu_count() or 2)
    print(f"[Phase 3 Benchmark] Running {len(tasks)} instance groups ({len(tasks)*6} total solver runs) across {n_procs} workers...")
    t0 = time.perf_counter()
    all_rows = []
    with mp.Pool(processes=n_procs) as pool:
        for idx, res_rows in enumerate(pool.imap_unordered(_run_single_instance_all_solvers, tasks), 1):
            all_rows.extend(res_rows)
            if idx % 10 == 0 or idx == len(tasks):
                print(f"  Completed {idx}/{len(tasks)} instances ({time.perf_counter() - t0:.1f}s)...")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["map_name", "k", "scenario_id", "solver"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Generate Statistical Summary Table (Median, Mean ± 95% CI, Wilcoxon p-values)
    lines = [
        "# Phase 3 Statistical Benchmark Summary (25 Scenarios/Cell, 6 Solvers)\n",
        "| Map | k | Solver | Solved | Median Nodes | Mean ± 95% CI Nodes | Median Runtime (s) | Wilcoxon p vs Learned_CL |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for (map_name, k), cell_df in df.groupby(["map_name", "k"]):
        piv = cell_df.pivot(index="scenario_id", columns="solver", values="nodes_expanded")
        learned_nodes = piv["Learned_CL_ICBSh"].values if "Learned_CL_ICBSh" in piv else None

        for solver_name in [
            "PlainCBS",
            "PlainICBSh",
            "RandomRankerICBSh",
            "DisjointSplittingICBSh",
            "FAwareResidualICBSh",
            "Learned_CL_ICBSh",
        ]:
            s_df = cell_df[cell_df["solver"] == solver_name]
            if len(s_df) == 0:
                continue
            n_solved = int((s_df["status"] == "SOLVED").sum())
            n_total = len(s_df)
            nodes = s_df["nodes_expanded"].values
            med_nodes = float(np.median(nodes))
            mean_nodes = float(np.mean(nodes))
            ci95 = float(1.96 * np.std(nodes, ddof=1) / np.sqrt(len(nodes))) if len(nodes) > 1 else 0.0
            med_rt = float(np.median(s_df["runtime"].values))

            p_str = "—"
            if solver_name != "Learned_CL_ICBSh" and learned_nodes is not None and solver_name in piv:
                base_nodes = piv[solver_name].values
                diffs = base_nodes - learned_nodes
                if np.any(diffs != 0):
                    try:
                        _, p_val = wilcoxon(base_nodes, learned_nodes, alternative="greater")
                        p_str = f"{p_val:.4f}" + (" *" if p_val < 0.05 else "")
                    except ValueError:
                        p_str = "1.0000"

            lines.append(
                f"| `{map_name}` | {k} | `{solver_name}` | {n_solved}/{n_total} | "
                f"{med_nodes:.1f} | {mean_nodes:.1f} ± {ci95:.1f} | {med_rt:.3f}s | {p_str} |"
            )

    report_md = "\n".join(lines) + "\n"
    with open(report_out, "w") as f:
        f.write(report_md)

    print("\n" + report_md)
    print(f"[Phase 3 Benchmark] Saved CSV to {output_csv} and Markdown report to {report_out}.")


if __name__ == "__main__":
    run_phase3_benchmark()
