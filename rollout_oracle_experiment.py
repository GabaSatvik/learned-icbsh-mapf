"""Harvest live ICBS-h decisions and evaluate the deep rollout oracle."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mapf.baselines import _compute_candidate_lookahead_tuple, f_aware_residual_selector
from mapf.features import FeatureExtractor
from mapf.grid import load_map
from mapf.icbs_h import ICBShSolver
from mapf.ml_revision_policy import SavedLinear, candidate_pool, immediate_children, teacher_key
from mapf.rollout_oracle import capture_full_search_snapshot, conflict_id, evaluate_conflict_with_rollout

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments" / "rollout_oracle_instances.json"


class CapturedEnough(RuntimeError):
    pass


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def context(snapshot, graph=None):
    node = snapshot.current
    graph = graph or node.g_cf_cache
    return dict(
        grid=snapshot.grid, paths=node.paths, starts=snapshot.starts, goals=snapshot.goals,
        G_CF=graph, mvc_size=graph.exact_mvc_size(), effective_h=node.h,
        pair_counts=node.pair_counts, classified=snapshot.classified,
        constraints=node.flat_constraints, per_agent_constraints=node.constraints,
        h_tables=snapshot.h_tables)


def harvest_one(instance):
    grid = load_map(ROOT / instance["map_path"])
    starts = list(map(tuple, instance["starts"]))
    goals = list(map(tuple, instance["goals"]))
    saved = []
    def capture(node, classified, graph, candidates, state):
        snap = capture_full_search_snapshot(node, classified, graph, candidates, state)
        snap.grid = grid
        saved.append(snap)
        raise CapturedEnough()
    try:
        ICBShSolver(grid, conflict_selector=f_aware_residual_selector).solve(
            starts, goals, timeout=20, max_expansions=3000, on_decision_callback=capture)
    except CapturedEnough:
        pass
    return grid, saved[0] if saved else None


def base_record(run_id, query_id, instance, snapshot, cc, index, pool_size, teacher,
                analytical, residual_rank, model_score):
    c, node = cc.conflict, snapshot.current
    graph = node.g_cf_cache
    degrees = [graph.degree(a) for a in graph.adj]
    branches = teacher.get("branches", [])
    fvals = [b.get("effective_f") for b in branches]
    return dict(
        run_id=run_id, query_id=query_id, instance_id=instance["id"],
        family_id=instance["family_id"], split=instance["split"],
        scenario_family=instance["family_id"], map=instance["map_name"],
        agent_count=instance["k"], seed=instance.get("seed"),
        rollout_mode="full_search_state", continuation_policy="residual",
        candidate_conflict_id=conflict_id(c), candidate_index=index,
        candidate_pool_size=pool_size, conflict_tier=cc.classification,
        agent_i=c.agent_i, agent_j=c.agent_j, conflict_timestep=c.time,
        conflict_type=c.conflict_type, current_g=node.g, current_h=node.h,
        current_f=node.f, global_min_open_f=min([node.f]+[n.f for n in snapshot.open_list]),
        open_size=len(snapshot.open_list)+1,
        conflict_count=len(node.conflicts_cache or snapshot.classified),
        conflict_graph_nodes=graph.num_vertices(), conflict_graph_edges=len(graph.edges),
        conflict_graph_max_degree=max(degrees, default=0), mvc=graph.exact_mvc_size(),
        total_path_cost=sum(len(p)-1 for p in node.paths),
        min_path_length=min(map(len, node.paths)), max_path_length=max(map(len, node.paths)),
        one_step_status=teacher["status"],
        F_left=fvals[0] if len(fvals)>0 else None, F_right=fvals[1] if len(fvals)>1 else None,
        one_step_min_f=teacher.get("priority", [None, None])[0] if teacher.get("priority") else None,
        one_step_max_f=teacher.get("priority", [None, None])[1] if teacher.get("priority") else None,
        one_step_teacher_key=json.dumps(teacher.get("priority")),
        delta_g_left=(branches[0].get("g")-node.g) if len(branches)>0 and branches[0].get("g") is not None else None,
        delta_g_right=(branches[1].get("g")-node.g) if len(branches)>1 and branches[1].get("g") is not None else None,
        analytical_key=json.dumps(analytical), residual_rank=residual_rank,
        model_b_score=model_score)


def rollout_job(args):
    grid, snapshot, cc, budgets, timeout, trace = args
    return evaluate_conflict_with_rollout(
        grid, snapshot, cc, budgets, timeout=timeout, trace=trace)


def run(mode, target_states, budgets, rollout_timeout):
    cfg = json.loads(CONFIG.read_text())
    instances = cfg["instances"]
    outdir = ROOT / "results" / "rollout_oracle" / mode
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    model = SavedLinear(ROOT / "models" / "model_b.json")
    raw, states = [], []
    for instance in instances:
        if len(states) >= target_states:
            break
        grid, snapshot = harvest_one(instance)
        if snapshot is None:
            continue
        snapshot.grid = grid
        ctx = context(snapshot)
        pool = candidate_pool(snapshot.candidates, ctx)
        if len(pool) < 2:
            continue
        extractor = FeatureExtractor(grid, snapshot.goals)
        features = extractor.extract_all(pool, ctx)
        scores = model.predict(features)
        analytical = [_compute_candidate_lookahead_tuple(cc, ctx) for cc in pool]
        residual_order = sorted(range(len(pool)), key=lambda i: analytical[i], reverse=True)
        residual_ranks = {idx: rank for rank, idx in enumerate(residual_order)}
        query_id = f"{instance['id']}/deep-q0"
        candidate_rows = []
        prepared = []
        for i, cc in enumerate(pool):
            teacher = immediate_children(cc, ctx, seconds=5)
            base = base_record(run_id, query_id, instance, snapshot, cc, i, len(pool),
                               teacher, analytical[i], residual_ranks[i], float(scores[i]))
            prepared.append((i, cc, base))
        jobs = [(grid, snapshot, cc, budgets, rollout_timeout,
                 len(states) < 2 and i < 2) for i, cc, _ in prepared]
        if mode == "pilot" and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as executor:
                rollout_results = list(executor.map(rollout_job, jobs))
        else:
            rollout_results = [rollout_job(job) for job in jobs]
        for (i, cc, base), result in zip(prepared, rollout_results):
            for checkpoint in result["checkpoints"]:
                row = dict(base)
                row.update(checkpoint)
                raw.append(row)
                candidate_rows.append(row)
            if result["trajectory"]:
                path = outdir / f"trajectory_{len(states):03d}_{i:02d}.csv"
                write_csv(path, result["trajectory"])
        states.append(summarize_state(query_id, instance, candidate_rows, budgets[-1]))
        print(f"{mode}: {len(states)}/{target_states} {query_id} pool={len(pool)}", flush=True)
    write_csv(outdir / "raw_rollouts.csv", raw)
    write_csv(outdir / "state_summary.csv", states)
    metadata = dict(
        run_id=run_id, git_commit=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"), dirty=bool(git("status", "--porcelain")),
        python=sys.version, platform=platform.platform(), seed=71, mode=mode,
        target_states=target_states, collected_states=len(states), budgets=budgets,
        continuation_policy="residual", rollout_mode="full_search_state",
        state_selection="first informative live residual-policy decision per frozen train/development instance")
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return raw, states, metadata


def summarize_state(query_id, instance, rows, final_budget):
    final = [r for r in rows if r["budget"] == final_budget]
    solved = [r for r in final if r["solved"]]
    vals = [r["ct_expansions"] for r in solved]
    return dict(
        query_id=query_id, instance_id=instance["id"], family_id=instance["family_id"],
        split=instance["split"], map=instance["map_name"], agent_count=instance["k"],
        candidate_count=len(final), exact_candidates=len(solved),
        all_candidates_exact=len(solved)==len(final),
        min_eventual_expansions=min(vals) if vals else None,
        max_eventual_expansions=max(vals) if vals else None,
        expansion_gap=(max(vals)-min(vals)) if vals else None,
        expansion_ratio=(max(vals)/min(vals)) if vals and min(vals)>0 else None,
        optimal_costs=json.dumps(sorted({r["solution_cost"] for r in solved})),
        optimal_cost_preserved=len({r["solution_cost"] for r in solved}) <= 1)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def selector_metrics(raw, budgets):
    by_state = defaultdict(list)
    for row in raw:
        if row["budget"] == budgets[-1]:
            by_state[row["query_id"]].append(row)
    complete = {q: rows for q, rows in by_state.items() if rows and all(r["solved"] for r in rows)}
    policies = ["residual", "one_step"] + [f"rollout_{b}" for b in budgets]
    comparisons, regrets = [], []
    for policy in policies:
        pair_ok = pair_total = top_ok = covered = 0
        rho, regret = [], []
        for q, eventual in complete.items():
            truth = {r["candidate_index"]: r["ct_expansions"] for r in eventual}
            allrows = [r for r in raw if r["query_id"] == q]
            if policy == "residual":
                pick = min(eventual, key=lambda r: r["residual_rank"])
                score = {r["candidate_index"]: -r["residual_rank"] for r in eventual}
            elif policy == "one_step":
                valid = [r for r in eventual if r["one_step_status"] == "COMPLETE"]
                if len(valid) != len(eventual): continue
                pick = max(valid, key=lambda r: (float("inf") if r["one_step_min_f"] == "+inf" else r["one_step_min_f"],
                                                  float("inf") if r["one_step_max_f"] == "+inf" else r["one_step_max_f"]))
                score = {r["candidate_index"]: (
                    float("inf") if r["one_step_min_f"] == "+inf" else r["one_step_min_f"],
                    float("inf") if r["one_step_max_f"] == "+inf" else r["one_step_max_f"]) for r in valid}
            else:
                budget = int(policy.split("_")[1])
                horizon = [r for r in raw if r["query_id"] == q and r["budget"] == budget]
                valid = [r for r in horizon if r["solved"]]
                if not valid: continue
                pick = min(valid, key=lambda r: r["ct_expansions"])
                score = {r["candidate_index"]: -r["ct_expansions"] for r in valid}
                if len(valid) != len(eventual):
                    score.update({r["candidate_index"]: -float("inf") for r in eventual if r["candidate_index"] not in score})
            covered += 1
            best = min(truth.values())
            top_ok += truth[pick["candidate_index"]] == best
            regret.append(truth[pick["candidate_index"]]-best)
            ids = sorted(truth)
            numeric = [score[i][0]*1e6+score[i][1] if isinstance(score[i], tuple) else score[i] for i in ids]
            if len(set(numeric)) > 1:
                rho.append(float(spearmanr(numeric, [-truth[i] for i in ids]).statistic))
            for x in range(len(ids)):
                for y in range(x+1, len(ids)):
                    if truth[ids[x]] == truth[ids[y]] or numeric[x] == numeric[y]: continue
                    pair_total += 1
                    pair_ok += (numeric[x] > numeric[y]) == (truth[ids[x]] < truth[ids[y]])
        comparisons.append(dict(
            policy=policy, complete_states=len(complete), covered_states=covered,
            pairwise_accuracy=pair_ok/pair_total if pair_total else None,
            spearman_mean=float(np.nanmean(rho)) if rho else None,
            top1_accuracy=top_ok/covered if covered else None,
            mean_regret=float(np.mean(regret)) if regret else None,
            median_regret=float(np.median(regret)) if regret else None,
            zero_regret_fraction=float(np.mean(np.array(regret)==0)) if regret else None,
            worst_regret=max(regret) if regret else None))
        regrets.extend(dict(policy=policy, query_id=q, regret=v) for q, v in zip(complete, regret))
    return comparisons, regrets


def analyze(raw, budgets):
    comparisons, regrets = selector_metrics(raw, budgets)
    budget_rows = []
    for b in budgets:
        metric = next(x for x in comparisons if x["policy"] == f"rollout_{b}")
        rows = [r for r in raw if r["budget"] == b]
        budget_rows.append(dict(
            budget=b, pairwise_accuracy=metric["pairwise_accuracy"],
            top1_accuracy=metric["top1_accuracy"], mean_regret=metric["mean_regret"],
            median_regret=metric["median_regret"],
            label_coverage=sum(r["solved"] for r in rows)/len(rows) if rows else 0,
            average_oracle_wall_seconds=float(np.mean([r["wall_seconds"] for r in rows])) if rows else None))
    return comparisons, regrets, budget_rows


def report(mode, raw, states, metadata, budgets):
    comparisons, regrets, budget_rows = analyze(raw, budgets)
    if mode == "pilot":
        out = ROOT / "results" / "rollout_oracle" / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "ranking_comparison.csv", comparisons)
        write_csv(out / "regret_summary.csv", regrets)
        write_csv(out / "budget_comparison.csv", budget_rows)
        summary = dict(metadata=metadata, states=states, ranking=comparisons, budgets=budget_rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    exact = [s for s in states if s["all_candidates_exact"]]
    variable = [s for s in exact if (s["expansion_gap"] or 0) > 0]
    one = next((x for x in comparisons if x["policy"] == "one_step"), {})
    best = max(budget_rows, key=lambda x: ((x["pairwise_accuracy"] or -1), -(x["average_oracle_wall_seconds"] or 0)))
    title = "SMOKE_REPORT.md" if mode == "smoke" else "ROLLOUT_ORACLE_REPORT.md"
    outpath = (ROOT / "results" / "rollout_oracle" / mode / title
               if mode == "smoke" else ROOT / "results" / "rollout_oracle" / title)
    decision = ("PROMISING BUT NEED MORE ORACLE DATA" if len(exact) < 20 else
                "PROCEED WITH ROLLOUT-BASED ML TRAINING" if
                (best["pairwise_accuracy"] or 0) > (one.get("pairwise_accuracy") or 0) else
                "MODIFY ORACLE OBJECTIVE BEFORE TRAINING")
    lines = [
        f"# {mode.title()} rollout-oracle report", "",
        f"- States collected: {len(states)}",
        f"- Fully completed candidate sets: {len(exact)}",
        f"- Completed states with nonzero candidate variability: {len(variable)}",
        f"- Optimal-cost preservation: {all(s['optimal_cost_preserved'] for s in states)}",
        f"- Best observed horizon by pairwise accuracy: {best['budget']}",
        f"- Decision gate: **{decision}**", "",
        "## Ranking comparison", "",
        "| policy | pairwise | top-1 | mean regret | coverage states |",
        "|---|---:|---:|---:|---:|"]
    for x in comparisons:
        lines.append(f"| {x['policy']} | {x['pairwise_accuracy']} | {x['top1_accuracy']} | {x['mean_regret']} | {x['covered_states']} |")
    lines += ["", "Censored candidate pairs are excluded; two unsolved candidates are never treated as tied.",
              "Timing fields are diagnostic and are not used as labels."]
    outpath.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["smoke", "pilot"])
    p.add_argument("--target-states", type=int)
    p.add_argument("--rollout-timeout", type=float, default=30)
    args = p.parse_args()
    budgets = [50, 100, 250] if args.mode == "smoke" else [50, 100, 250, 500, 1500]
    target = args.target_states or (5 if args.mode == "smoke" else 40)
    raw, states, metadata = run(args.mode, target, budgets, args.rollout_timeout)
    report(args.mode, raw, states, metadata, budgets)


if __name__ == "__main__":
    main()
