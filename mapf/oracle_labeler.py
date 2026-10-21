# mapf/oracle_labeler.py
"""
Hybrid 1-Step Exact Child Lookahead + Micro-Rollout Oracle Labeler for CL-ICBS-h.
Parallelized across CPU cores to generate ~8,000-12,000 high-signal rows.
"""
import os
import time
import random
import multiprocessing as mp
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from mapf.grid import load_map, load_scenarios
from mapf.low_level import compute_bfs_h_table, space_time_astar, get_path_cost
from mapf.conflicts import detect_all_conflicts
from mapf.cardinal import classify_conflicts, ClassifiedConflict
from mapf.conflict_graph import compute_h4_and_graph, compute_residual_mvc_size
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.icbs_h import ICBShSolver
from mapf.baselines import _compute_candidate_lookahead_tuple


def evaluate_candidate_hybrid_oracle(
    grid,
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    h_tables: List[dict],
    node_paths: List[List[Tuple[int, int]]],
    node_constraints: List[Any],
    node_f: float,
    cc: ClassifiedConflict,
    context: Dict[str, Any] = None,
    B_micro: int = 20,
    T_micro: float = 0.025,
) -> Dict[str, float]:
    """
    Computes exact 1-step child f-values (f_1, f_2) using cached alt_path_i / alt_path_j
    plus a bounded micro-rollout to break any f-ties.
    """
    c = cc.conflict
    max_t = grid.width + grid.height + 50
    child_f_values = []
    child_conflict_counts = []
    child_dg_values = []

    G_CF = context.get("G_CF") if context is not None else None
    h_parent = context.get("mvc_size", 0) if context is not None else 0
    node_g = sum(get_path_cost(p) for p in node_paths)

    for branch_idx, agent_id in enumerate([c.agent_i, c.agent_j]):
        other_agent = c.agent_j if agent_id == c.agent_i else c.agent_i
        old_cost = get_path_cost(node_paths[agent_id])

        if branch_idx == 0 and cc.alt_path_i is not None:
            new_path = cc.alt_path_i
            c_cnt = cc.alt_total_conflicts_i if cc.alt_total_conflicts_i < 999999 else 0
        elif branch_idx == 1 and cc.alt_path_j is not None:
            new_path = cc.alt_path_j
            c_cnt = cc.alt_total_conflicts_j if cc.alt_total_conflicts_j < 999999 else 0
        else:
            new_c = c.get_constraint_for_agent(agent_id, other_agent)
            new_constraints = list(node_constraints) + [new_c]
            agent_cons = [con for con in new_constraints if getattr(con, "agent", -1) == agent_id]
            new_path = space_time_astar(
                grid,
                starts[agent_id],
                goals[agent_id],
                agent_cons,
                max_timesteps=max_t,
                h_table=h_tables[agent_id],
            )
            c_cnt = -1

        if new_path is None:
            child_f_values.append(node_f + 50.0)
            child_conflict_counts.append(0)
            child_dg_values.append(50.0)
            continue

        new_cost = get_path_cost(new_path)
        dg = float(new_cost - old_cost)
        child_dg_values.append(dg)
        g_child = node_g + dg

        if c_cnt < 0:
            new_paths = list(node_paths)
            new_paths[agent_id] = new_path
            c_child = detect_all_conflicts(new_paths)
            c_cnt = len(c_child)
        child_conflict_counts.append(c_cnt)

        if c_cnt == 0:
            child_f_values.append(float(g_child))
        elif G_CF is not None:
            h_res = compute_residual_mvc_size(G_CF, agent_id)
            child_f_values.append(float(g_child + h_res))
        else:
            child_f_values.append(float(g_child + max(0, h_parent - 1)))

    f_min = min(child_f_values)
    f_max = max(child_f_values)
    df_min = f_min - node_f
    df_max = f_max - node_f
    c_min = min(child_conflict_counts)
    dg_min = min(child_dg_values)
    dg_max = max(child_dg_values)

    deg_sum = 0.0
    if G_CF is not None:
        deg_sum = float(G_CF.degree(c.agent_i) + G_CF.degree(c.agent_j))

    # Fast micro-rollout when child does not immediately solve
    if c_min == 0:
        micro_nodes = 1.0
    elif B_micro > 0 and T_micro > 0:
        def forced_root_selector(cands, ctx):
            for cand in cands:
                if (
                    cand.conflict.agent_i == c.agent_i
                    and cand.conflict.agent_j == c.agent_j
                    and cand.conflict.time == c.time
                ):
                    return cand
            return cands[0]

        sub_solver = ICBShSolver(grid, conflict_selector=forced_root_selector, regime="A")
        res = sub_solver.solve(
            starts,
            goals,
            timeout=T_micro,
            max_expansions=B_micro,
            initial_constraints=list(node_constraints),
        )
        if res["status"] == "SOLVED":
            micro_nodes = float(res["nodes_expanded"])
        else:
            micro_nodes = float(B_micro) * 1.25 + 0.5 * c_min - 8.0 * df_min - 2.0 * df_max
    else:
        micro_nodes = float(B_micro) * 1.25 + 0.5 * c_min - 8.0 * df_min - 2.0 * df_max

    # Unified Oracle Score (higher is better for ranking):
    # Combines exact 1-step lower-bound jump (Huang et al. 2021) + inverse micro-subtree size
    oracle_score = (
        20.0 * df_min
        + 5.0 * df_max
        - 0.25 * micro_nodes
        - 0.1 * c_min
        + 0.25 * dg_min
        + 0.08 * dg_max
        + 0.15 * deg_sum
        - 0.005 * float(c.time)
    )

    return {
        "df_min": df_min,
        "df_max": df_max,
        "c_min": float(c_min),
        "micro_nodes": micro_nodes,
        "oracle_score": oracle_score,
    }


def _worker_process_instance(args) -> List[Dict[str, Any]]:
    map_name, map_path, scen_path, k, scen_idx, max_queries_per_inst, max_cands = args
    grid = load_map(map_path)
    scenarios = load_scenarios(scen_path, k, max_scenarios=scen_idx)
    if len(scenarios) < scen_idx:
        return []

    starts, goals = scenarios[scen_idx - 1]
    h_tables = [compute_bfs_h_table(grid, goals[i]) for i in range(k)]
    if any(starts[i] not in h_tables[i] for i in range(k)):
        return []

    extractor = FeatureExtractor(grid)
    records = []
    queries_collected = [0]
    rng = random.Random(1000 + scen_idx * 37 + k)

    def record_query(pool_cands: List[ClassifiedConflict], context: Dict[str, Any]):
        if len(pool_cands) < 2 or queries_collected[0] >= max_queries_per_inst:
            return None
        if len(pool_cands) > max_cands:
            scored = [
                (_compute_candidate_lookahead_tuple(cc, context), idx, cc)
                for idx, cc in enumerate(pool_cands)
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_half = [x[2] for x in scored[: max_cands - 1]]
            rem = [x[2] for x in scored[max_cands - 1 :]]
            sampled_cands = top_half + [rng.choice(rem)]
        else:
            sampled_cands = list(pool_cands)

        feats_matrix = extractor.extract_all(sampled_cands, context)
        node_paths = context["paths"]
        node_constraints = context.get("constraints", [])
        node_g = sum(get_path_cost(p) for p in node_paths)
        node_f = node_g + context.get("mvc_size", 0)

        # Run micro-rollout on first 6 queries per instance, fast exact 1-step child lookahead on rest
        b_mic = 16 if queries_collected[0] < 6 else 0
        t_mic = 0.02 if queries_collected[0] < 6 else 0.0

        oracle_results = []
        for cc in sampled_cands:
            ores = evaluate_candidate_hybrid_oracle(
                grid,
                starts,
                goals,
                h_tables,
                node_paths,
                node_constraints,
                node_f,
                cc,
                context=context,
                B_micro=b_mic,
                T_micro=t_mic,
            )
            oracle_results.append(ores)

        scores = np.array([r["oracle_score"] for r in oracle_results], dtype=float)
        s_min, s_max = float(np.min(scores)), float(np.max(scores))

        if s_max - s_min > 1e-6:
            rel_cont = 2.0 * (scores - s_min) / (s_max - s_min)
            thresh_80 = np.percentile(scores, 80.0)
            rel_bin = (scores >= thresh_80).astype(float)
        else:
            rel_cont = np.ones_like(scores)
            rel_bin = np.ones_like(scores)

        qid = f"{map_name}_k{k}_s{scen_idx}_q{queries_collected[0]}"
        queries_collected[0] += 1

        for idx, cc in enumerate(sampled_cands):
            row = {
                "query_id": qid,
                "map_name": map_name,
                "k": k,
                "scenario_id": scen_idx,
                "relevance": float(rel_cont[idx]),
                "binary_label": int(rel_bin[idx]),
                "oracle_score": float(scores[idx]),
                "df_min": float(oracle_results[idx]["df_min"]),
                "micro_nodes": float(oracle_results[idx]["micro_nodes"]),
            }
            for f_idx, fname in enumerate(FEATURE_NAMES):
                row[fname] = float(feats_matrix[idx, f_idx])
            records.append(row)

        best_idx = int(np.argmax(scores))
        return sampled_cands[best_idx]

    def harvesting_selector(candidates: List[ClassifiedConflict], context: Dict[str, Any]):
        if queries_collected[0] < max_queries_per_inst and len(candidates) >= 2:
            chosen = record_query(candidates, context)
            if chosen is not None:
                return chosen
        return f_aware_Follow(candidates, context)

    def on_node_cb(node, classified, g_cf):
        # Also harvest multi-conflict nodes when cardinal tier had only 1 conflict
        if queries_collected[0] < max_queries_per_inst and len(classified) >= 2:
            cardinals = [cc for cc in classified if cc.classification == "CARDINAL"]
            if len(cardinals) < 2:
                ctx = {
                    "grid": grid,
                    "paths": node.paths,
                    "starts": starts,
                    "goals": goals,
                    "G_CF": g_cf,
                    "mvc_size": node.h,
                    "pair_counts": node.pair_counts,
                    "classified": classified,
                    "constraints": node.flat_constraints,
                }
                record_query(classified, ctx)

    def f_aware_Follow(cands, ctx):
        best_cc = cands[0]
        best_k = None
        for cc in cands:
            k_tup = _compute_candidate_lookahead_tuple(cc, ctx)
            if best_k is None or k_tup > best_k:
                best_k = k_tup
                best_cc = cc
        return best_cc

    solver = ICBShSolver(grid, conflict_selector=harvesting_selector, regime="A")
    solver.solve(starts, goals, timeout=8.0, max_expansions=90, on_node_callback=on_node_cb)

    # If the optimal trajectory solved quickly, run a randomized exploration pass to collect deeper CT decision queries
    if queries_collected[0] < max_queries_per_inst // 2:
        def exploratory_selector(cands, ctx):
            if queries_collected[0] < max_queries_per_inst and len(cands) >= 2:
                record_query(cands, ctx)
            return rng.choice(cands)

        exp_solver = ICBShSolver(grid, conflict_selector=exploratory_selector, regime="A")
        exp_solver.solve(starts, goals, timeout=4.0, max_expansions=60, on_node_callback=on_node_cb)

    return records


def generate_cl_oracle_dataset(
    output_csv: str = "data/cl_oracle_training.csv",
    max_queries_per_inst: int = 35,
    max_cands_per_query: int = 5,
    num_workers: int = 12,
):
    t0 = time.perf_counter()
    configs = [
        ("random-32-32-20", "data/random-32-32-20.map", "data/random-32-32-20-random-1.scen", [16, 20, 24, 28], range(1, 21)),
        ("room-32-32-4", "data/room-32-32-4.map", "data/room-32-32-4-random-1.scen", [12, 16, 20, 24], range(1, 21)),
    ]

    tasks = []
    for map_name, map_path, scen_path, k_list, scen_range in configs:
        if not os.path.exists(map_path) or not os.path.exists(scen_path):
            print(f"[WARN] Skipping {map_name}: missing {map_path} or {scen_path}")
            continue
        for k in k_list:
            for s_idx in scen_range:
                tasks.append((map_name, map_path, scen_path, k, s_idx, max_queries_per_inst, max_cands_per_query))

    print(f"[Oracle Labeler] Launching {len(tasks)} instance tasks across {num_workers} CPU workers...")
    all_rows = []
    with mp.Pool(processes=min(num_workers, mp.cpu_count() or 4)) as pool:
        for inst_rows in pool.imap_unordered(_worker_process_instance, tasks):
            all_rows.extend(inst_rows)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    elapsed = time.perf_counter() - t0
    n_queries = df["query_id"].nunique() if len(df) > 0 else 0
    tied_queries = 0
    if n_queries > 0:
        for _, grp in df.groupby("query_id"):
            if grp["oracle_score"].nunique() <= 1:
                tied_queries += 1

    tie_pct = (100.0 * tied_queries / n_queries) if n_queries > 0 else 100.0
    print(f"[Oracle Labeler] Completed in {elapsed:.1f}s ({elapsed/3600:.2f} wall-clock hours).")
    print(f"[Oracle Labeler] Saved {len(df)} candidate rows across {n_queries} queries to {output_csv}.")
    print(f"[Oracle Labeler] Tied Queries: {tied_queries}/{n_queries} ({tie_pct:.2f}%).")


if __name__ == "__main__":
    generate_cl_oracle_dataset()
