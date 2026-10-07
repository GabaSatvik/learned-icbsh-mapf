"""
Conflict-Density Auditor (Phase 1.1)
Empirically determines at which (map, k) pairs Regime A (≥2 cardinal conflicts)
and Regime B (≥2 total conflicts) occur.
"""
import os
import time
import pandas as pd
from typing import List, Dict, Any

from mapf.grid import Grid, parse_scenario
from mapf.icbs_h import ICBS_H, ICBSNode
from mapf.conflicts import find_all_conflicts
from mapf.cardinal import classify_conflict, get_constraints_for_agent
from mapf.conflict_graph import ConflictGraph
import heapq


def audit_instance(grid: Grid, starts: list, goals: list, timeout: float = 60.0) -> Dict[str, Any]:
    solver = ICBS_H(grid, enable_bypass=True)
    start_time = time.time()
    start_cpu = time.process_time()

    total_nodes = 0
    nodes_ge2_total = 0
    nodes_ge2_cardinal = 0
    max_total = 0
    max_cardinal = 0

    root_paths = []
    for i in range(len(starts)):
        p = solver.low_level.find_path(starts[i], goals[i], set(), set())
        if p is None:
            return {
                'status': 'NO_ROOT_PATH',
                'total_nodes': 0,
                'cpu_seconds_used': time.process_time() - start_cpu,
                'max_total_conflicts_at_any_node': 0,
                'fraction_of_nodes_with_ge2_total_conflicts': 0.0,
                'max_cardinal_conflicts_at_any_node': 0,
                'fraction_of_nodes_with_ge2_cardinal_conflicts': 0.0,
            }
        root_paths.append(p)

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

    status = 'TIMEOUT'
    while open_list:
        if time.time() - start_time > timeout:
            status = 'TIMEOUT'
            break

        curr = heapq.heappop(open_list)
        total_nodes += 1

        all_conflicts = find_all_conflicts(curr.paths)
        if not all_conflicts:
            status = 'OPTIMAL'
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

        # Audit conflict counts at this branching node
        n_total = len(classified_conflicts)
        n_cardinal = sum(1 for _, cls in classified_conflicts if cls == 'cardinal')

        if n_total > max_total:
            max_total = n_total
        if n_cardinal > max_cardinal:
            max_cardinal = n_cardinal

        if n_total >= 2:
            nodes_ge2_total += 1
        if n_cardinal >= 2:
            nodes_ge2_cardinal += 1

        # Standard ICBS branching
        priority_rank = {'cardinal': 0, 'semi-cardinal': 1, 'non-cardinal': 2}
        best_idx = 0
        best_rank = 999
        for idx, (_, c_cls) in enumerate(classified_conflicts):
            rank = priority_rank.get(c_cls, 3)
            if rank < best_rank:
                best_rank = rank
                best_idx = idx

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

            new_path = solver.low_level.find_path(
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
                    (cc, classify_conflict(cc, child_paths, starts, goals, child_constraints, solver.low_level)[0])
                    for cc in child_conflicts
                ]
                child_h = solver._compute_heuristic(classified_child)

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

    cpu_used = time.process_time() - start_cpu
    frac_total = (nodes_ge2_total / total_nodes) if total_nodes > 0 else 0.0
    frac_cardinal = (nodes_ge2_cardinal / total_nodes) if total_nodes > 0 else 0.0

    return {
        'status': status,
        'total_nodes': total_nodes,
        'cpu_seconds_used': cpu_used,
        'max_total_conflicts_at_any_node': max_total,
        'fraction_of_nodes_with_ge2_total_conflicts': frac_total,
        'max_cardinal_conflicts_at_any_node': max_cardinal,
        'fraction_of_nodes_with_ge2_cardinal_conflicts': frac_cardinal,
    }


def run_audit():
    print("=== Phase 1.1: Conflict-Density Audit ===")
    maps = ['random-32-32-20', 'room-32-32-4', 'Paris_1_256']
    k_values = [12, 20, 28, 40]
    scenarios_per_group = 5

    records = []
    os.makedirs('results', exist_ok=True)

    for map_name in maps:
        map_path = f"data/{map_name}.map"
        if not os.path.exists(map_path):
            print(f"Skipping missing map {map_path}")
            continue
        print(f"\nLoading map: {map_name}...")
        grid = Grid.from_file(map_path)

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

                res = audit_instance(grid, starts, goals, timeout=60.0)
                record = {
                    'map': map_name,
                    'k': k,
                    'scenario': s_id,
                    **res
                }
                records.append(record)
                print(f"[{map_name} | k={k:2d} | Scen {s_id:2d}] status={res['status']:7s} | "
                      f"nodes={res['total_nodes']:3d} | "
                      f"max_tot={res['max_total_conflicts_at_any_node']} (ge2_frac={res['fraction_of_nodes_with_ge2_total_conflicts']:.2f}) | "
                      f"max_card={res['max_cardinal_conflicts_at_any_node']} (ge2_card_frac={res['fraction_of_nodes_with_ge2_cardinal_conflicts']:.2f}) | "
                      f"cpu={res['cpu_seconds_used']:.2f}s")

    df = pd.DataFrame(records)
    out_csv = "results/conflict_density_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nAudit completed. Saved {len(df)} records to {out_csv}")

    # Decision rule evaluation
    # Regime A: frac_ge2_cardinal > 0 or max_cardinal >= 2 in >= 30% of instances somewhere
    # Check if there is any (map, k) cell where instances with >=2 cardinal conflicts appear in >= 30% of instances
    # Or globally across instances
    regime_a_instances = df[df['max_cardinal_conflicts_at_any_node'] >= 2]
    regime_b_instances = df[df['max_total_conflicts_at_any_node'] >= 2]

    frac_a_overall = len(regime_a_instances) / len(df) if len(df) > 0 else 0
    frac_b_overall = len(regime_b_instances) / len(df) if len(df) > 0 else 0

    # Also check per (map, k) maximum instance prevalence
    grouped = df.groupby(['map', 'k'])
    max_rate_a_in_any_cell = grouped.apply(lambda g: (g['max_cardinal_conflicts_at_any_node'] >= 2).mean()).max()
    max_rate_b_in_any_cell = grouped.apply(lambda g: (g['max_total_conflicts_at_any_node'] >= 2).mean()).max()

    print("\n--- Conflict Density Decision Rule ---")
    print(f"Regime A (>=2 cardinal conflicts): Max cell rate = {max_rate_a_in_any_cell*100:.1f}%, Overall rate = {frac_a_overall*100:.1f}%")
    print(f"Regime B (>=2 total conflicts):    Max cell rate = {max_rate_b_in_any_cell*100:.1f}%, Overall rate = {frac_b_overall*100:.1f}%")

    regime_a_applies = (max_rate_a_in_any_cell >= 0.30)
    regime_b_applies = (max_rate_b_in_any_cell >= 0.50)

    if regime_a_applies and regime_b_applies:
        verdict = "BOTH_APPLICABLE"
    elif regime_b_applies:
        verdict = "PATH_B_PRIMARY"
    elif regime_a_applies:
        verdict = "PATH_A_PRIMARY"
    else:
        verdict = "NO_DOMAIN"

    print(f"\nVERDICT: {verdict}")
    with open("results/density_audit_verdict.txt", "w") as f:
        f.write(f"{verdict}\n")
        f.write(f"Regime A max cell rate: {max_rate_a_in_any_cell:.4f}\n")
        f.write(f"Regime B max cell rate: {max_rate_b_in_any_cell:.4f}\n")


if __name__ == "__main__":
    run_audit()
