"""
Phase 1.6 — Gate Decision Evaluator
Reads results/phase1_benchmark.csv and results/proxy_validation.csv.
Evaluates the three pre-registered gates:
  1. Directional consistency: The sign of (LearnedICBSh_proxy - PlainICBSh) median node count
     is negative (fewer nodes) on at least 7 of 9 (map, k) cells.
  2. Proxy validity: Mean Spearman >= 0.3 on at least 2 of 3 maps.
  3. Optimality preserved on every solved instance.

Emits verdict: PASS, Case NULL, or Case INCONCLUSIVE.
"""
import os
import pandas as pd
import numpy as np


def evaluate_phase1_gate():
    print("=== Phase 1.6: Gate Decision Evaluation ===")
    bench_file = "results/phase1_benchmark.csv"
    val_file = "results/proxy_validation.csv"

    if not os.path.exists(bench_file):
        print(f"Error: {bench_file} missing.")
        return
    if not os.path.exists(val_file):
        print(f"Error: {val_file} missing.")
        return

    bench_df = pd.read_csv(bench_file)
    val_df = pd.read_csv(val_file)

    # 1. Optimality Invariant Check
    optimality_passed = True
    for (m, k, s_id), group in bench_df.groupby(['map', 'k', 'scenario_id']):
        costs = group['solution_cost'].dropna().unique()
        if len(costs) > 1:
            print(f"Optimality mismatch at {m} k={k} scen={s_id}: {costs}")
            optimality_passed = False
    print(f"Gate 3 (Optimality Preserved): {'PASSED' if optimality_passed else 'FAILED'}")

    # 2. Proxy Validity Check
    per_map_spearmans = {}
    for map_name, group in val_df.groupby('map'):
        # compute spearman per query then average
        q_sps = []
        for qid, qg in group.groupby('query_id'):
            if len(qg['proxy_score'].unique()) > 1 and len(qg['rollout_cost'].unique()) > 1:
                from scipy.stats import spearmanr
                sp, _ = spearmanr(qg['proxy_score'], qg['rollout_cost'])
                if not np.isnan(sp):
                    q_sps.append(sp)
        per_map_spearmans[map_name] = np.mean(q_sps) if q_sps else 0.0

    valid_maps_count = sum(1 for sp in per_map_spearmans.values() if sp >= 0.3)
    proxy_valid = (valid_maps_count >= 2)
    print("\nGate 2 (Proxy Validity):")
    for m, sp in per_map_spearmans.items():
        print(f"  {m:20s}: Mean Spearman = {sp:+.4f} (>= 0.3: {sp >= 0.3})")
    print(f"  Proxy Validity Status: {'PASSED' if proxy_valid else 'FAILED (Mean Spearman < 0.3 on >= 2 maps)'}")

    # 3. Directional Consistency Check
    # (LearnedICBSh_proxy - PlainICBSh) median node count negative on >= 7 of (map, k) cells
    pivoted = bench_df.pivot_table(
        index=['map', 'k', 'scenario_id'],
        columns='solver',
        values='nodes_expanded'
    ).reset_index()

    pivoted['diff_learned_plain'] = pivoted['LearnedICBSh_proxy'] - pivoted['PlainICBSh']

    cell_medians = pivoted.groupby(['map', 'k'])['diff_learned_plain'].median()
    print("\nGate 1 (Directional Consistency - Median Node Difference Learned vs Plain):")
    negative_cells = 0
    total_cells = len(cell_medians)
    for (m, k), med_diff in cell_medians.items():
        is_neg = (med_diff < 0)
        if is_neg:
            negative_cells += 1
        print(f"  [{m:15s} | k={k:2d}]: Median Node Delta = {med_diff:+6.1f} (Fewer nodes: {is_neg})")

    directional_passed = (negative_cells >= 7)
    print(f"  Directional Consistency Status: {'PASSED' if directional_passed else 'FAILED'} ({negative_cells}/{total_cells} cells negative; required >= 7)")

    # Overall Verdict
    gate_passed = directional_passed and proxy_valid and optimality_passed
    print("\n=======================================================")
    if gate_passed:
        verdict = "GATE_PASSED"
        report_type = "PHASE2_TRIGGER"
    else:
        verdict = "GATE_FAILED"
        if proxy_valid:
            report_type = "CASE_NULL"
        else:
            report_type = "CASE_INCONCLUSIVE"

    print(f"FINAL VERDICT: {verdict} ({report_type})")
    print("=======================================================")

    with open("results/phase1_gate_verdict.txt", "w") as f:
        f.write(f"Verdict: {verdict}\n")
        f.write(f"Report Type: {report_type}\n")
        f.write(f"Optimality: {optimality_passed}\n")
        f.write(f"Proxy Validity: {proxy_valid} (valid maps: {valid_maps_count})\n")
        f.write(f"Directional Consistency: {directional_passed} ({negative_cells}/{total_cells} cells negative)\n")

    return verdict, report_type


if __name__ == "__main__":
    evaluate_phase1_gate()
