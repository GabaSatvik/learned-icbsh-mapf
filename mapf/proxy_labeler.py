"""
Proxy Labeler (Phase 1.3)
Generates training labels from cheap computed surrogate scores without running any subtrees,
enabling fast, deterministic Phase 1 data generation.

Pre-registered composite proxy score weights:
  alpha = 1.0  (proxy_mvc_delta)
  beta  = 0.5  (proxy_downstream)
  gamma = 0.5  (proxy_corridor)

proxy_score[c] = alpha * proxy_mvc_delta[c] + beta * proxy_downstream[c] + gamma * proxy_corridor[c]
relevance[c] = -proxy_score[c], normalized within query group to [0, 2].
"""
import os
import time
import argparse
import heapq
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from mapf.grid import Grid, parse_scenario
from mapf.icbs_h import ICBS_H, ICBSNode
from mapf.conflicts import Conflict, find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.conflict_graph import ConflictGraph
from mapf.features import FeatureExtractor, FEATURE_NAMES


ALPHA = 1.0
BETA = 0.5
GAMMA = 0.5


def _compute_real_mvc_delta(
    conflict, c_cls: str, classified_conflicts: list, parent_mvc: int
) -> float:
    """
    Bug 3 fix: Compute actual MVC(child) - MVC(parent) by constructing the child
    cardinal conflict graph. Previously this was faked as 1.0 if cardinal else 0.0.

    For conflict c = (a_i, a_j), the child where a_i is constrained resolves
    the (a_i, a_j) cardinal edge. We remove it from the cardinal graph and recompute MVC.
    For non-cardinal conflicts, constraining either agent costs nothing: delta = 0.
    """
    if c_cls not in ('cardinal', 'semi-cardinal'):
        return 0.0

    a_i = conflict.agent_1
    a_j = conflict.agent_2

    # Parent cardinal edges
    parent_cardinal_edges = [
        (c.agent_1, c.agent_2) for c, cls in classified_conflicts if cls == 'cardinal'
    ]

    # Child cardinal graph: remove the (a_i, a_j) edge that gets resolved by branching
    child_edges = [
        (u, v) for (u, v) in parent_cardinal_edges
        if not ((u == a_i and v == a_j) or (u == a_j and v == a_i))
    ]

    mvc_child = ConflictGraph(child_edges).compute_mvc() if child_edges else 0
    return float(mvc_child - parent_mvc)


def run_proxy_labeling(
    regime: str = 'B',
    maps: List[str] = ['random-32-32-20', 'room-32-32-4'],
    k_values: List[int] = [20, 28, 40],
    scenarios_per_group: int = 5,
    max_queries_per_instance: int = 100,
    max_candidates_per_query: int = 5,  # Bug 2 fix: enforce candidate cap
    output_csv: Optional[str] = None
) -> str:
    print(f"=== Phase 1.3: Proxy Labeling (Regime {regime}) ===")
    if output_csv is None:
        os.makedirs('data', exist_ok=True)
        output_csv = f"data/proxy_training_regime{regime}.csv"

    records = []
    global_query_id = 0
    t0_all = time.time()

    for map_name in maps:
        map_path = f"data/{map_name}.map"
        if not os.path.exists(map_path):
            continue
        grid = Grid.from_file(map_path)
        fe = FeatureExtractor(grid)
        solver = ICBS_H(grid, enable_bypass=True)

        for k in k_values:
            for s_id in range(1, scenarios_per_group + 1):
                scen_file = f"data/scen-random/{map_name}-random-{s_id}.scen"
                if not os.path.exists(scen_file):
                    continue

                instances = parse_scenario(scen_file, max_agents=k)
                if len(instances) < k:
                    continue

                starts = [x[0] for x in instances]
                goals = [x[1] for x in instances]

                # Run ICBS-h search and intercept branching nodes
                root_paths = []
                for i in range(k):
                    p = solver.low_level.find_path(starts[i], goals[i], set(), set())
                    if p is None:
                        break
                    root_paths.append(p)
                if len(root_paths) < k:
                    continue

                root_g = sum(len(p) - 1 for p in root_paths)
                initial_conflicts = find_all_conflicts(root_paths)
                classified_initial = [
                    (c, classify_conflict(c, root_paths, starts, goals, [], solver.low_level)[0])
                    for c in initial_conflicts
                ]
                root_h = solver._compute_heuristic(classified_initial)
                root = ICBSNode(
                    f_cost=root_g + root_h,
                    h_cost=root_h,
                    g_cost=root_g,
                    id=0,
                    paths=root_paths,
                    constraints=[]
                )
                open_list = [root]
                node_id_counter = 1
                queries_in_instance = 0

                start_inst_time = time.time()
                pair_history: Dict[Tuple[int, int], int] = {}

                while open_list and queries_in_instance < max_queries_per_instance:
                    if time.time() - start_inst_time > 45.0:
                        break

                    curr = heapq.heappop(open_list)
                    all_conflicts = find_all_conflicts(curr.paths)
                    if not all_conflicts:
                        break

                    classified_conflicts = []
                    bypass_found = False

                    # Classification & bypass check
                    for c in all_conflicts:
                        c_class, bp1, bp2 = classify_conflict(
                            c, curr.paths, starts, goals, curr.constraints, solver.low_level
                        )
                        classified_conflicts.append((c, c_class, bp1, bp2))

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

                    # Filter candidate conflicts according to requested regime
                    if regime == 'A':
                        candidates = [
                            (c, c_cls, bp1, bp2) for c, c_cls, bp1, bp2 in classified_conflicts if c_cls == 'cardinal'
                        ]
                    else:  # Regime B
                        candidates = classified_conflicts

                    parent_mvc = solver._compute_heuristic([(c, cls) for c, cls, _, _ in classified_conflicts])

                    # Bug 2 fix: sort by delta_f_res (cost increase) descending and cap to max_candidates_per_query
                    # Conflicts that force the highest cost increase are the most informative
                    def _candidate_priority(item):
                        c, c_cls, bp1, bp2 = item
                        inc_i = 1 if c_cls == 'cardinal' else (0.5 if c_cls == 'semi-cardinal' else 0)
                        return inc_i
                    candidates = sorted(candidates, key=_candidate_priority, reverse=True)
                    candidates = candidates[:max_candidates_per_query]

                    # Emit query if >= 2 candidates
                    if len(candidates) >= 2:
                        global_query_id += 1
                        queries_in_instance += 1
                        
                        cand_feats = []
                        raw_proxy_scores = []

                        for c_idx, (c, c_cls, bp1, bp2) in enumerate(candidates):
                            # Feature extraction
                            # Downstream conflict lookahead estimate
                            down_i = 0
                            down_j = 0
                            a1, a2 = c.agent_1, c.agent_2
                            if bp1 is not None:
                                p1_copy = list(curr.paths)
                                p1_copy[a1] = bp1
                                down_i = len(find_all_conflicts(p1_copy))
                            if bp2 is not None:
                                p2_copy = list(curr.paths)
                                p2_copy[a2] = bp2
                                down_j = len(find_all_conflicts(p2_copy))

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

                            # Proxy metrics
                            # 1. proxy_mvc_delta: Bug 3 fix — compute REAL child MVC delta
                            # not just a 0/1 cardinal indicator
                            p_mvc_delta = _compute_real_mvc_delta(
                                c, c_cls, [(x[0], x[1]) for x in classified_conflicts], parent_mvc
                            )

                            # 2. proxy_downstream
                            p_downstream = float(down_i + down_j)

                            # 3. proxy_corridor: 1 / (corridor_i + corridor_j + 1)
                            p_corridor = 1.0 / (feat['corridor_score_i'] + feat['corridor_score_j'] + 1.0)

                            composite_proxy = (ALPHA * p_mvc_delta) + (BETA * p_downstream) + (GAMMA * p_corridor)
                            raw_proxy_scores.append(composite_proxy)

                        # Normalize relevance within group to [0, 2]
                        # Lower proxy_score -> higher relevance
                        min_p = min(raw_proxy_scores)
                        max_p = max(raw_proxy_scores)
                        for (c_idx, feat), p_score in zip(cand_feats, raw_proxy_scores):
                            if max_p > min_p:
                                # Invert so smaller proxy gives higher relevance in [0, 2]
                                norm_rel = 2.0 * (max_p - p_score) / (max_p - min_p)
                            else:
                                norm_rel = 1.0

                            row = {
                                'query_id': global_query_id,
                                'candidate_index': c_idx,
                                'relevance': round(norm_rel, 4),
                                'raw_proxy_score': round(p_score, 4),
                                'map': map_name,
                                'k': k,
                                'scenario_id': s_id,
                                **feat
                            }
                            records.append(row)

                    # Advance search by picking standard ICBS choice
                    priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
                    best_idx = 0
                    best_rank = 999
                    for idx, (c, c_cls, _, _) in enumerate(classified_conflicts):
                        rank = priority_rank.get(c_cls, 3)
                        if rank < best_rank:
                            best_rank = rank
                            best_idx = idx

                    chosen_conflict = classified_conflicts[best_idx][0]
                    pair_key = (min(chosen_conflict.agent_1, chosen_conflict.agent_2),
                                max(chosen_conflict.agent_1, chosen_conflict.agent_2))
                    pair_history[pair_key] = pair_history.get(pair_key, 0) + 1

                    # Branch
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

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by=['query_id', 'candidate_index']).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - t0_all
    print(f"Proxy labeling complete in {elapsed:.1f}s. Saved {len(df)} candidate rows across {df['query_id'].nunique() if not df.empty else 0} decision queries to {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', type=str, default='B', choices=['A', 'B'])
    args = parser.parse_args()
    run_proxy_labeling(regime=args.regime)
