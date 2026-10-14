"""
Rollout Labeler (Phase 2)
=========================
Generates REAL training labels by running short ICBS-h rollouts for each candidate conflict.

Key differences from proxy_labeler.py:
  - Labels come from actual subtree expansions (B_rollout=1500, T_rollout=2.0s)
    NOT from a hand-crafted proxy formula.
  - Fully parallelised across CPU cores using multiprocessing.Pool
  - Checkpointing: saves progress every CHECKPOINT_EVERY queries so crashes don't lose work
  - tqdm progress bar shows real-time progress

How rollout labeling works:
  At each CT branching node, we have N candidate conflicts to choose from.
  For each candidate c_i:
    1. Tentatively add the constraint from branching on c_i
    2. Replan the affected agent
    3. Run a short ICBS-h search on the resulting child node (budget: B_rollout expansions or T_rollout seconds)
    4. Record rollout_cost = expansions used (or penalty B_rollout*1.5 if unsolved)
  Lower rollout_cost => that conflict was a better choice => higher relevance.

The resulting CSV is a drop-in replacement for proxy_training_regimeB.csv.
"""

import os
import sys
import time
import heapq
import argparse
import logging
import signal
import pickle
import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

import pandas as pd

# ── Add project root to path so we can import mapf ──────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mapf.grid import Grid, parse_scenario
from mapf.low_level import SpaceTimeAStar
from mapf.conflicts import Conflict, find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.conflict_graph import ConflictGraph
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.icbs_h import ICBSNode

# ── Configuration ────────────────────────────────────────────────────────────
B_ROLLOUT      = 1500     # Max CT expansions per rollout (was 300 — that was the bug)
T_ROLLOUT      = 2.0      # Max wall-clock seconds per rollout
PENALTY_MULT   = 1.5      # Penalty factor when rollout hits budget: cost = B_ROLLOUT * PENALTY_MULT
INSTANCE_LIMIT = 60.0     # Max seconds per MAPF instance (parent search)
MAX_QUERIES_PER_INSTANCE = 100
CHECKPOINT_EVERY = 50     # Save CSV every N completed queries

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Short rollout: runs a mini ICBS-h search from a given child node
# Returns number of CT nodes expanded (lower = better conflict choice)
# ─────────────────────────────────────────────────────────────────────────────
def run_short_rollout(
    child_paths: List[List[Tuple[int, int]]],
    child_constraints: List,
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    low_level: SpaceTimeAStar,
    b_rollout: int = B_ROLLOUT,
    t_rollout: float = T_ROLLOUT
) -> int:
    """
    Run a short ICBS-h search from a child CT node.
    Returns expansion count. If unsolved within budget, returns penalty.
    """
    penalty = int(b_rollout * PENALTY_MULT)
    t_start = time.time()
    expansions = 0

    # Compute heuristic for starting child node
    initial_conflicts = find_all_conflicts(child_paths)
    if not initial_conflicts:
        return 0  # Already solved — perfect choice

    classified = []
    for c in initial_conflicts:
        c_cls, _, _ = classify_conflict(c, child_paths, starts, goals, child_constraints, low_level)
        classified.append((c, c_cls))

    # MVC heuristic
    cardinal_edges = [(c.agent_1, c.agent_2) for c, cls in classified if cls == 'cardinal']
    h = 0
    if cardinal_edges:
        from mapf.conflict_graph import ConflictGraph
        h = ConflictGraph(cardinal_edges).compute_mvc()

    child_g = sum(len(p) - 1 for p in child_paths)
    root = ICBSNode(
        f_cost=child_g + h,
        h_cost=h,
        g_cost=child_g,
        id=0,
        paths=list(child_paths),
        constraints=list(child_constraints)
    )

    open_list = [root]
    node_id = 1

    while open_list:
        if expansions >= b_rollout or (time.time() - t_start) >= t_rollout:
            # Return ACTUAL expansions (not fixed penalty) — different candidates
            # exhaust budget at different depths, giving genuine ranking signal
            # even for rollouts that don't reach a solved state.
            return expansions + b_rollout  # offset above solved-case range

        curr = heapq.heappop(open_list)
        expansions += 1

        all_conflicts = find_all_conflicts(curr.paths)
        if not all_conflicts:
            return expansions  # Solved!

        # Classify conflicts
        classified_curr = []
        for c in all_conflicts:
            c_cls, _, _ = classify_conflict(c, curr.paths, starts, goals, curr.constraints, low_level)
            classified_curr.append((c, c_cls))

        # Pick best conflict: cardinal > semi > non
        priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
        best_idx = min(range(len(classified_curr)),
                       key=lambda i: priority_rank.get(classified_curr[i][1], 3))
        chosen_c = classified_curr[best_idx][0]

        # Branch
        a1, a2 = chosen_c.agent_1, chosen_c.agent_2
        if chosen_c.conflict_type == 'vertex':
            c1 = (a1, 'vertex', (chosen_c.loc_1[0], chosen_c.loc_1[1], chosen_c.time))
            c2 = (a2, 'vertex', (chosen_c.loc_1[0], chosen_c.loc_1[1], chosen_c.time))
        else:
            c1 = (a1, 'edge', (chosen_c.loc_1[0], chosen_c.loc_1[1], chosen_c.loc_2[0], chosen_c.loc_2[1], chosen_c.time))
            c2 = (a2, 'edge', (chosen_c.loc_2[0], chosen_c.loc_2[1], chosen_c.loc_1[0], chosen_c.loc_1[1], chosen_c.time))

        for agent_idx, new_c in [(a1, c1), (a2, c2)]:
            child_constr = list(curr.constraints) + [new_c]
            v_cons, e_cons = get_constraints_for_agent(child_constr, agent_idx)
            new_path = low_level.find_path(starts[agent_idx], goals[agent_idx], v_cons, e_cons)
            if new_path is None:
                continue
            cp = list(curr.paths)
            cp[agent_idx] = new_path
            cg = sum(len(p) - 1 for p in cp)

            # Simplified h for rollout (skip MVC to save time — just 0)
            child_node = ICBSNode(
                f_cost=cg,
                h_cost=0,
                g_cost=cg,
                id=node_id,
                paths=cp,
                constraints=child_constr
            )
            node_id += 1
            heapq.heappush(open_list, child_node)

    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# Worker function — runs ONE instance (map + k + scenario_id)
# Returns list of record dicts
# ─────────────────────────────────────────────────────────────────────────────
def _worker(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker: processes one (map, k, scenario) instance and returns records."""
    map_name     = args['map_name']
    map_path     = args['map_path']
    scen_file    = args['scen_file']
    k            = args['k']
    s_id         = args['s_id']
    start_query  = args['start_query_id']   # offset so query IDs are globally unique

    records = []

    try:
        grid    = Grid.from_file(map_path)
        fe      = FeatureExtractor(grid)
        ll      = SpaceTimeAStar(grid)

        instances = parse_scenario(scen_file, max_agents=k)
        if len(instances) < k:
            return records

        starts = [x[0] for x in instances[:k]]
        goals  = [x[1] for x in instances[:k]]

        # Build unconstrained root paths
        root_paths = []
        for i in range(k):
            p = ll.find_path(starts[i], goals[i], set(), set())
            if p is None:
                return records
            root_paths.append(p)

        # Run a simplified CBS-like expansion loop to collect decision points
        from mapf.icbs_h import ICBSNode
        from mapf.conflict_graph import ConflictGraph

        def compute_h(classified):
            edges = [(c.agent_1, c.agent_2) for c, cls in classified if cls == 'cardinal']
            if not edges:
                return 0
            return ConflictGraph(edges).compute_mvc()

        root_g = sum(len(p) - 1 for p in root_paths)
        initial_conflicts = find_all_conflicts(root_paths)
        classified_init = []
        for c in initial_conflicts:
            c_cls, _, _ = classify_conflict(c, root_paths, starts, goals, [], ll)
            classified_init.append((c, c_cls))
        root_h = compute_h(classified_init)

        root = ICBSNode(
            f_cost=root_g + root_h,
            h_cost=root_h,
            g_cost=root_g,
            id=0,
            paths=root_paths,
            constraints=[]
        )

        open_list = [root]
        node_id_ctr = 1
        queries_this_instance = 0
        local_query_id = start_query
        pair_history: Dict[Tuple[int, int], int] = {}
        t_inst = time.time()

        while open_list and queries_this_instance < MAX_QUERIES_PER_INSTANCE:
            if time.time() - t_inst > INSTANCE_LIMIT:
                break

            curr = heapq.heappop(open_list)
            all_conflicts = find_all_conflicts(curr.paths)
            if not all_conflicts:
                break  # solved

            # Classify
            classified_conflicts = []
            bypass_found = False
            for c in all_conflicts:
                c_cls, bp1, bp2 = classify_conflict(c, curr.paths, starts, goals, curr.constraints, ll)
                classified_conflicts.append((c, c_cls, bp1, bp2))

                # Bypass check
                if not bypass_found:
                    for bp, agent_idx in [(bp1, c.agent_1), (bp2, c.agent_2)]:
                        if bp is not None:
                            test = list(curr.paths)
                            test[agent_idx] = bp
                            if len(find_all_conflicts(test)) < len(all_conflicts):
                                curr.paths = test
                                bypass_found = True
                                break
                if bypass_found:
                    break

            if bypass_found:
                curr_cl = [(c, classify_conflict(c, curr.paths, starts, goals, curr.constraints, ll)[0])
                           for c in find_all_conflicts(curr.paths)]
                curr.h_cost = compute_h(curr_cl)
                curr.f_cost = curr.g_cost + curr.h_cost
                heapq.heappush(open_list, curr)
                continue

            parent_mvc = compute_h([(c, cls) for c, cls, _, _ in classified_conflicts])
            candidates = classified_conflicts  # Regime B: all conflicts

            # Only log if >= 2 candidates (need ranking signal)
            if len(candidates) >= 2:
                local_query_id += 1
                queries_this_instance += 1

                cand_feats   = []
                rollout_costs = []

                for c_idx, (c, c_cls, bp1, bp2) in enumerate(candidates):
                    # --- Feature extraction ---
                    a1, a2 = c.agent_1, c.agent_2

                    # Downstream estimate via bp paths
                    down_i, down_j = 0, 0
                    if bp1 is not None:
                        tp = list(curr.paths); tp[a1] = bp1
                        down_i = len(find_all_conflicts(tp))
                    if bp2 is not None:
                        tp = list(curr.paths); tp[a2] = bp2
                        down_j = len(find_all_conflicts(tp))

                    feat = fe.extract_features(
                        conflict=c,
                        conflict_class=c_cls,
                        all_classified_conflicts=[(x[0], x[1]) for x in classified_conflicts],
                        paths=curr.paths,
                        goals=goals,
                        mvc_size=parent_mvc,
                        pair_history=pair_history,
                        downstream_counts=(down_i, down_j)
                    )
                    cand_feats.append((c_idx, feat))

                    # --- ROLLOUT: branch on this candidate, run short search ---
                    if c.conflict_type == 'vertex':
                        rc1 = (a1, 'vertex', (c.loc_1[0], c.loc_1[1], c.time))
                        rc2 = (a2, 'vertex', (c.loc_1[0], c.loc_1[1], c.time))
                    else:
                        rc1 = (a1, 'edge', (c.loc_1[0], c.loc_1[1], c.loc_2[0], c.loc_2[1], c.time))
                        rc2 = (a2, 'edge', (c.loc_2[0], c.loc_2[1], c.loc_1[0], c.loc_1[1], c.time))

                    # Branch on agent 1 side (left branch)
                    child_constr = list(curr.constraints) + [rc1]
                    v_cons, e_cons = get_constraints_for_agent(child_constr, a1)
                    new_path = ll.find_path(starts[a1], goals[a1], v_cons, e_cons)

                    if new_path is not None:
                        child_paths = list(curr.paths)
                        child_paths[a1] = new_path
                        cost = run_short_rollout(child_paths, child_constr, starts, goals, ll)
                    else:
                        cost = B_ROLLOUT * 2  # infeasible branch: worst possible cost

                    rollout_costs.append(float(cost))

                # --- Compute relevance: lower cost = better = higher relevance ---
                min_c = min(rollout_costs)
                max_c = max(rollout_costs)

                for (c_idx, feat), r_cost in zip(cand_feats, rollout_costs):
                    if max_c > min_c:
                        norm_rel = 2.0 * (max_c - r_cost) / (max_c - min_c)
                    else:
                        norm_rel = 1.0  # all tied — no signal this query

                    row = {
                        'query_id'       : local_query_id,
                        'candidate_index': c_idx,
                        'relevance'      : round(norm_rel, 4),
                        'rollout_cost'   : round(r_cost, 1),
                        'map'            : map_name,
                        'k'              : k,
                        'scenario_id'    : s_id,
                        **feat
                    }
                    records.append(row)

            # Advance main search — pick best conflict by standard ICBS priority
            priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
            best_idx = min(range(len(classified_conflicts)),
                           key=lambda i: priority_rank.get(classified_conflicts[i][1], 3))
            chosen = classified_conflicts[best_idx][0]
            pk = (min(chosen.agent_1, chosen.agent_2), max(chosen.agent_1, chosen.agent_2))
            pair_history[pk] = pair_history.get(pk, 0) + 1

            # Branch main search
            a1, a2 = chosen.agent_1, chosen.agent_2
            if chosen.conflict_type == 'vertex':
                mc1 = (a1, 'vertex', (chosen.loc_1[0], chosen.loc_1[1], chosen.time))
                mc2 = (a2, 'vertex', (chosen.loc_1[0], chosen.loc_1[1], chosen.time))
            else:
                mc1 = (a1, 'edge', (chosen.loc_1[0], chosen.loc_1[1], chosen.loc_2[0], chosen.loc_2[1], chosen.time))
                mc2 = (a2, 'edge', (chosen.loc_2[0], chosen.loc_2[1], chosen.loc_1[0], chosen.loc_1[1], chosen.time))

            for ag, nc in [(a1, mc1), (a2, mc2)]:
                cc = list(curr.constraints) + [nc]
                vc, ec = get_constraints_for_agent(cc, ag)
                np_ = ll.find_path(starts[ag], goals[ag], vc, ec)
                if np_ is not None:
                    cp = list(curr.paths); cp[ag] = np_
                    cg = sum(len(p) - 1 for p in cp)
                    cf = find_all_conflicts(cp)
                    cl = [(c_, classify_conflict(c_, cp, starts, goals, cc, ll)[0]) for c_ in cf]
                    ch = compute_h(cl)
                    heapq.heappush(open_list, ICBSNode(
                        f_cost=cg + ch, h_cost=ch, g_cost=cg,
                        id=node_id_ctr, paths=cp, constraints=cc
                    ))
                    node_id_ctr += 1

    except Exception as e:
        log.warning(f"Worker failed [{map_name} k={k} scen={s_id}]: {e}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────
def run_rollout_labeling(
    maps: List[str] = ['random-32-32-20', 'room-32-32-4'],
    k_values: List[int] = [20, 28],
    scenarios_per_group: int = 5,
    num_workers: int = 12,
    output_csv: str = 'data/rollout_training_regimeB.csv',
    checkpoint_dir: str = 'data/rollout_checkpoints'
) -> str:
    os.makedirs('data', exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 2: Rollout Labeling")
    log.info(f"  Maps      : {maps}")
    log.info(f"  k values  : {k_values}")
    log.info(f"  Scenarios : {scenarios_per_group} per group")
    log.info(f"  CPU cores : {num_workers}")
    log.info(f"  B_rollout : {B_ROLLOUT}  T_rollout: {T_ROLLOUT}s")
    log.info(f"  Output    : {output_csv}")
    log.info("=" * 60)

    # Build job list
    jobs = []
    query_offset = 0
    for map_name in maps:
        map_path = f"data/{map_name}.map"
        if not os.path.exists(map_path):
            log.warning(f"Map not found: {map_path}, skipping.")
            continue
        for k in k_values:
            for s_id in range(1, scenarios_per_group + 1):
                scen_file = f"data/scen-random/{map_name}-random-{s_id}.scen"
                if not os.path.exists(scen_file):
                    log.warning(f"Scenario not found: {scen_file}, skipping.")
                    continue
                jobs.append({
                    'map_name'      : map_name,
                    'map_path'      : map_path,
                    'scen_file'     : scen_file,
                    'k'             : k,
                    's_id'          : s_id,
                    'start_query_id': query_offset,
                })
                # Offset so IDs don't collide across workers
                query_offset += MAX_QUERIES_PER_INSTANCE

    log.info(f"Total jobs to process: {len(jobs)}")

    # Try-load existing checkpoint
    ckpt_file = os.path.join(checkpoint_dir, 'progress.pkl')
    completed_keys = set()
    all_records = []

    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, 'rb') as f:
                saved = pickle.load(f)
                completed_keys = saved.get('completed_keys', set())
                all_records    = saved.get('records', [])
            log.info(f"Checkpoint loaded: {len(completed_keys)} jobs already done, {len(all_records)} records.")
        except Exception:
            log.warning("Could not load checkpoint — starting fresh.")

    # Filter out already-done jobs
    pending = [j for j in jobs if (j['map_name'], j['k'], j['s_id']) not in completed_keys]
    log.info(f"Jobs remaining: {len(pending)}")

    if not pending:
        log.info("All jobs already complete. Writing final CSV.")
    else:
        t0 = time.time()
        done_count = 0
        batch_size = num_workers * 2  # Process in batches to checkpoint

        try:
            # Try to import tqdm for nice progress bars
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            log.info("tqdm not installed — showing plain progress. Install with: pip install tqdm")

        with Pool(processes=num_workers) as pool:
            if use_tqdm:
                it = tqdm(pool.imap_unordered(_worker, pending),
                          total=len(pending),
                          desc="Rollout jobs",
                          unit="job",
                          dynamic_ncols=True)
            else:
                it = pool.imap_unordered(_worker, pending)

            for job_idx, (job, result_records) in enumerate(zip(pending, it)):
                all_records.extend(result_records)
                completed_keys.add((job['map_name'], job['k'], job['s_id']))
                done_count += 1

                # Print plain progress if no tqdm
                if not use_tqdm and done_count % 5 == 0:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (len(pending) - done_count) / rate if rate > 0 else 0
                    log.info(f"Progress: {done_count}/{len(pending)} jobs | "
                             f"Records: {len(all_records)} | "
                             f"ETA: {eta/60:.1f} min")

                # Checkpoint periodically
                if done_count % CHECKPOINT_EVERY == 0:
                    with open(ckpt_file, 'wb') as f:
                        pickle.dump({'completed_keys': completed_keys, 'records': all_records}, f)
                    if not use_tqdm:
                        log.info(f"Checkpoint saved at job {done_count}.")

        elapsed_total = time.time() - t0
        log.info(f"All jobs done in {elapsed_total/60:.1f} min. Total records: {len(all_records)}")

    # Write final CSV
    if not all_records:
        log.error("No records generated! Check map/scenario paths.")
        return output_csv

    df = pd.DataFrame(all_records)
    df = df.sort_values(by=['query_id', 'candidate_index']).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    n_queries = df['query_id'].nunique()
    tied = df.groupby('query_id')['rollout_cost'].nunique().eq(1).sum()
    tied_pct = 100.0 * tied / n_queries if n_queries > 0 else 0.0

    log.info("=" * 60)
    log.info(f"Rollout labeling COMPLETE")
    log.info(f"  Output CSV : {output_csv}")
    log.info(f"  Total rows : {len(df)}")
    log.info(f"  Queries    : {n_queries}")
    log.info(f"  Tied queries (no signal): {tied} / {n_queries} ({tied_pct:.1f}%)")
    log.info(f"  Columns    : {list(df.columns)}")
    log.info("=" * 60)

    # Save final checkpoint
    with open(ckpt_file, 'wb') as f:
        pickle.dump({'completed_keys': completed_keys, 'records': all_records}, f)

    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2 Rollout Labeler')
    parser.add_argument('--maps', nargs='+', default=['random-32-32-20', 'room-32-32-4'],
                        help='Map names (without .map extension)')
    parser.add_argument('--k', nargs='+', type=int, default=[20, 28],
                        help='Agent counts to evaluate')
    parser.add_argument('--scenarios', type=int, default=5,
                        help='Number of scenarios per (map, k) combination')
    parser.add_argument('--workers', type=int, default=12,
                        help='CPU cores to use (default: 12, max on your machine: 16)')
    parser.add_argument('--output', type=str, default='data/rollout_training_regimeB.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    run_rollout_labeling(
        maps=args.maps,
        k_values=args.k,
        scenarios_per_group=args.scenarios,
        num_workers=args.workers,
        output_csv=args.output
    )
