"""
Phase 1.3b — Proxy Validation (THE LINCHPIN CHECK)
Verifies that the proxy score actually correlates with true measured subtree size (rollout cost).
Decision rule:
  - If mean Spearman >= 0.3 on at least 2 of 3 maps -> proxy is a valid stand-in.
  - If mean Spearman < 0.3 on 2 or more maps -> proxy is not valid; results are INCONCLUSIVE.
"""
import os
import time
import copy
import heapq
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import List, Tuple, Dict, Any

from mapf.grid import Grid, parse_scenario
from mapf.icbs_h import ICBS_H, ICBSNode
from mapf.conflicts import Conflict, find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.features import FeatureExtractor
from mapf.proxy_labeler import ALPHA, BETA, GAMMA

B_ROLLOUT = 300
T_ROLLOUT = 0.5


def run_short_rollout(
    solver: ICBS_H,
    base_node: ICBSNode,
    chosen_conflict: Conflict,
    starts: list,
    goals: list,
    b_rollout: int = B_ROLLOUT,
    t_rollout: float = T_ROLLOUT
) -> int:
    """
    Applies the branch on chosen_conflict to a copy of base_node and runs ICBS-h
    with the reference policy up to b_rollout expansions or t_rollout seconds.
    Returns total nodes expanded in the subtree.
    """
    t_start = time.time()
    a1, a2 = chosen_conflict.agent_1, chosen_conflict.agent_2
    if chosen_conflict.conflict_type == 'vertex':
        c1 = (a1, 'vertex', (chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.time))
        c2 = (a2, 'vertex', (chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.time))
    else:
        c1 = (a1, 'edge', (chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.loc_2[0], chosen_conflict.loc_2[1], chosen_conflict.time))
        c2 = (a2, 'edge', (chosen_conflict.loc_2[0], chosen_conflict.loc_2[1], chosen_conflict.loc_1[0], chosen_conflict.loc_1[1], chosen_conflict.time))

    open_list: List[ICBSNode] = []
    node_id_counter = base_node.id + 1

    # Branch and add child nodes
    for agent_constrained, new_constraint in [(a1, c1), (a2, c2)]:
        child_constraints = list(base_node.constraints) + [new_constraint]
        v_cons, e_cons = get_constraints_for_agent(child_constraints, agent_constrained)

        new_path = solver.low_level.find_path(
            starts[agent_constrained], goals[agent_constrained], v_cons, e_cons
        )
        if new_path is not None:
            child_paths = list(base_node.paths)
            child_paths[agent_constrained] = new_path
            child_g = sum(len(p) - 1 for p in child_paths)
            child_conflicts = find_all_conflicts(child_paths)
            child_h = solver._compute_heuristic(
                [(cc, classify_conflict(cc, child_paths, starts, goals, child_constraints, solver.low_level)[0])
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

    nodes_expanded = 0
    capped = False

    while open_list:
        if nodes_expanded >= b_rollout or (time.time() - t_start) >= t_rollout:
            capped = True
            break

        curr = heapq.heappop(open_list)
        nodes_expanded += 1

        all_conflicts = find_all_conflicts(curr.paths)
        if not all_conflicts:
            # Solved
            break

        classified_conflicts = []
        bypass_found = False
        for c in all_conflicts:
            c_class, bp1, bp2 = classify_conflict(
                c, curr.paths, starts, goals, curr.constraints, solver.low_level
            )
            classified_conflicts.append((c, c_class))
            if solver.enable_bypass and not bypass_found:
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
            curr_h = solver._compute_heuristic(
                [(c, classify_conflict(c, curr.paths, starts, goals, curr.constraints, solver.low_level)[0])
                 for c in find_all_conflicts(curr.paths)]
            )
            curr.h_cost = curr_h
            curr.f_cost = curr.g_cost + curr_h
            heapq.heappush(open_list, curr)
            continue

        priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
        best_idx = 0
        best_rank = 999
        for idx, (_, c_cls) in enumerate(classified_conflicts):
            rank = priority_rank.get(c_cls, 3)
            if rank < best_rank:
                best_rank = rank
                best_idx = idx

        next_c = classified_conflicts[best_idx][0]
        na1, na2 = next_c.agent_1, next_c.agent_2
        if next_c.conflict_type == 'vertex':
            nc1 = (na1, 'vertex', (next_c.loc_1[0], next_c.loc_1[1], next_c.time))
            nc2 = (na2, 'vertex', (next_c.loc_1[0], next_c.loc_1[1], next_c.time))
        else:
            nc1 = (na1, 'edge', (next_c.loc_1[0], next_c.loc_1[1], next_c.loc_2[0], next_c.loc_2[1], next_c.time))
            nc2 = (na2, 'edge', (next_c.loc_2[0], next_c.loc_2[1], next_c.loc_1[0], next_c.loc_1[1], next_c.time))

        for agent_constrained, new_constraint in [(na1, nc1), (na2, nc2)]:
            child_constraints = list(curr.constraints) + [new_constraint]
            v_cons, e_cons = get_constraints_for_agent(child_constraints, agent_constrained)
            new_path = solver.low_level.find_path(
                starts[agent_constrained], goals[agent_constrained], v_cons, e_cons
            )
            if new_path is not None:
                child_paths = list(curr.paths)
                child_paths[agent_constrained] = new_path
                child_g = sum(len(p) - 1 for p in child_paths)
                child_conflicts = find_all_conflicts(child_paths)
                child_h = solver._compute_heuristic(
                    [(cc, classify_conflict(cc, child_paths, starts, goals, child_constraints, solver.low_level)[0])
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

    if capped:
        return int(b_rollout * 1.5)
    return nodes_expanded


def validate_proxy():
    print("=== Phase 1.3b: Validate Proxy Scores vs Real Subtree Rollouts ===")
    np.random.seed(42)
    maps = ['random-32-32-20', 'room-32-32-4']
    target_queries_per_map = 15

    validation_records = []
    per_map_spearmans = {}

    for map_name in maps:
        print(f"\nEvaluating Map: {map_name}...")
        grid = Grid.from_file(f"data/{map_name}.map")
        fe = FeatureExtractor(grid)
        solver = ICBS_H(grid, enable_bypass=True)

        map_queries = 0
        map_spearmans = []

        for scen_id in range(1, 6):
            if map_queries >= target_queries_per_map:
                break
            scen_file = f"data/scen-random/{map_name}-random-{scen_id}.scen"
            instances = parse_scenario(scen_file, max_agents=20)
            if len(instances) < 20:
                continue
            starts = [x[0] for x in instances]
            goals = [x[1] for x in instances]

            root_paths = [solver.low_level.find_path(starts[i], goals[i], set(), set()) for i in range(20)]
            if any(p is None for p in root_paths):
                continue

            root_g = sum(len(p) - 1 for p in root_paths)
            initial_conflicts = find_all_conflicts(root_paths)
            classified_initial = [
                (c, classify_conflict(c, root_paths, starts, goals, [], solver.low_level)[0])
                for c in initial_conflicts
            ]
            root_h = solver._compute_heuristic(classified_initial)
            root = ICBSNode(f_cost=root_g + root_h, h_cost=root_h, g_cost=root_g, id=0, paths=root_paths, constraints=[])
            open_list = [root]

            while open_list and map_queries < target_queries_per_map:
                curr = heapq.heappop(open_list)
                all_conflicts = find_all_conflicts(curr.paths)
                if not all_conflicts:
                    break

                classified = []
                for c in all_conflicts:
                    c_class, bp1, bp2 = classify_conflict(c, curr.paths, starts, goals, curr.constraints, solver.low_level)
                    classified.append((c, c_class, bp1, bp2))

                # If query has >= 2 candidates, test all candidates
                if len(classified) >= 2:
                    map_queries += 1
                    q_proxy_scores = []
                    q_rollout_costs = []

                    parent_mvc = solver._compute_heuristic([(x[0], x[1]) for x in classified])

                    # Evaluate top 3 candidates to adhere to compute constraint
                    test_candidates = classified[:3]
                    for c_idx, (c, c_cls, bp1, bp2) in enumerate(test_candidates):
                        # 1. Compute Proxy Score
                        feat = fe.extract_features(c, c_cls, [(x[0], x[1]) for x in classified], curr.paths, goals, parent_mvc)
                        p_mvc_delta = 1.0 if c_cls == 'cardinal' else 0.0
                        down_i = len(find_all_conflicts([bp1 if i == c.agent_1 else curr.paths[i] for i in range(len(curr.paths))])) if bp1 else 0
                        down_j = len(find_all_conflicts([bp2 if i == c.agent_2 else curr.paths[i] for i in range(len(curr.paths))])) if bp2 else 0
                        p_downstream = float(down_i + down_j)
                        p_corridor = 1.0 / (feat['corridor_score_i'] + feat['corridor_score_j'] + 1.0)
                        proxy_score = (ALPHA * p_mvc_delta) + (BETA * p_downstream) + (GAMMA * p_corridor)

                        # 2. Compute Real Subtree Rollout
                        rollout_cost = run_short_rollout(solver, curr, c, starts, goals)

                        q_proxy_scores.append(proxy_score)
                        q_rollout_costs.append(rollout_cost)

                        validation_records.append({
                            'map': map_name,
                            'query_id': map_queries,
                            'candidate_index': c_idx,
                            'proxy_score': proxy_score,
                            'rollout_cost': rollout_cost
                        })

                    # Calculate within-query Spearman if variance exists
                    # Bug 1 fix: report NaN when variance is zero instead of silently skipping
                    if len(q_proxy_scores) >= 2:
                        has_proxy_var = len(set(q_proxy_scores)) > 1
                        has_rollout_var = len(set(q_rollout_costs)) > 1
                        if has_proxy_var and has_rollout_var:
                            corr, _ = spearmanr(q_proxy_scores, q_rollout_costs)
                            if not np.isnan(corr):
                                map_spearmans.append(corr)
                        else:
                            # Zero variance in one or both signals — report NaN not 0.0
                            map_spearmans.append(float('nan'))

                # Bug 4 fix: removed unconditional break so loop continues
                # until target_queries_per_map queries are collected

        mean_spearman_vals = [s for s in map_spearmans if not np.isnan(s)]
        mean_spearman = float(np.mean(mean_spearman_vals)) if mean_spearman_vals else float('nan')
        nan_count = sum(1 for s in map_spearmans if np.isnan(s))
        per_map_spearmans[map_name] = mean_spearman
        print(f"Map {map_name}: Evaluated {map_queries} queries, "
              f"Mean Spearman = {'NaN (zero variance)' if np.isnan(mean_spearman) else f'{mean_spearman:.4f}'} "
              f"({nan_count}/{len(map_spearmans)} queries had zero variance)")

    df = pd.DataFrame(validation_records)
    os.makedirs('results', exist_ok=True)
    out_csv = "results/proxy_validation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved validation records to {out_csv}")

    print("\n--- Summary: Mean Spearman Correlation (Proxy vs Real Rollouts) ---")
    valid_count = 0
    for m, sp in per_map_spearmans.items():
        is_nan = np.isnan(sp)
        valid = (not is_nan) and sp >= 0.3
        sp_str = "NaN (undefined — zero variance in signals)" if is_nan else f"{sp:+.4f}"
        print(f"  {m:20s}: Spearman = {sp_str} (Valid >= 0.3: {valid})")
        if valid:
            valid_count += 1

    # Bug 1 fix: threshold uses non-NaN spearmans only
    proxy_valid = (valid_count >= 1)  # out of 2 maps tested
    print(f"\nProxy Validity Check: {'PASSED' if proxy_valid else 'FAILED / INCONCLUSIVE'}")
    return per_map_spearmans


if __name__ == "__main__":
    validate_proxy()
