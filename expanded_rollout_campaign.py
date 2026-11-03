"""Collect later, non-adjacent train/development decisions for oracle expansion."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from mapf.baselines import _compute_candidate_lookahead_tuple, f_aware_residual_selector
from mapf.features import FeatureExtractor
from mapf.grid import load_map
from mapf.icbs_h import ICBShSolver
from mapf.ml_revision_policy import SavedLinear, candidate_pool, immediate_children
from mapf.rollout_oracle import capture_full_search_snapshot
from rollout_oracle_experiment import (
    CONFIG, ROOT, CapturedEnough, base_record, context, rollout_job, write_csv,
)

OUT = ROOT / "results" / "rollout_oracle" / "adaptive" / "new_states"
PARTS = OUT / "raw_parts"
BUDGETS = [50, 100, 250, 500, 1500]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def harvest_ordinal(instance, ordinal):
    grid = load_map(ROOT / instance["map_path"])
    starts = list(map(tuple, instance["starts"]))
    goals = list(map(tuple, instance["goals"]))
    visits, saved = 0, []
    def capture(node, classified, graph, candidates, state):
        nonlocal visits
        if visits == ordinal:
            snap = capture_full_search_snapshot(node, classified, graph, candidates, state)
            snap.grid = grid
            saved.append(snap)
            raise CapturedEnough()
        visits += 1
    try:
        ICBShSolver(grid, conflict_selector=f_aware_residual_selector).solve(
            starts, goals, timeout=30, max_expansions=5000,
            on_decision_callback=capture)
    except CapturedEnough:
        pass
    return grid, saved[0] if saved else None


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge():
    rows = []
    for path in sorted(PARTS.glob("*.csv")):
        rows.extend(read_rows(path))
    write_csv(OUT / "raw_rollouts.csv", rows)
    finals = [r for r in rows if int(r["budget"]) == 1500]
    by_query = {}
    for row in finals:
        by_query.setdefault(row["query_id"], []).append(row)
    summary = []
    for q, group in sorted(by_query.items()):
        solved = [r for r in group if str(r["solved"]).lower() == "true"]
        vals = [int(r["ct_expansions"]) for r in solved]
        summary.append(dict(
            query_id=q, instance_id=group[0]["instance_id"],
            family_id=group[0]["family_id"], map=group[0]["map"],
            candidate_count=len(group), exact_candidates=len(solved),
            all_candidates_exact=len(solved) == len(group),
            min_eventual_expansions=min(vals) if vals else None,
            max_eventual_expansions=max(vals) if vals else None,
            expansion_gap=max(vals)-min(vals) if vals else None,
            optimal_cost_preserved=len({r["solution_cost"] for r in solved}) <= 1))
    write_csv(OUT / "state_summary.csv", summary)
    return summary


def targets():
    cfg = json.loads(CONFIG.read_text())["instances"]
    by_id = {x["id"]: x for x in cfg}
    pilot = pd.read_csv(ROOT / "results" / "rollout_oracle" / "pilot" / "state_summary.csv")
    complete = pilot.all_candidates_exact.astype(str).str.lower().eq("true")
    # Previously tractable states first; map/family order remains frozen.
    return [by_id[x] for x in pilot.loc[complete, "instance_id"]]


def run(target_count, ordinal, timeout, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    model = SavedLinear(ROOT / "models" / "model_b.json")
    collected = 0
    for instance in targets():
        if collected >= target_count:
            break
        part = PARTS / (instance["id"].replace("/", "__") + f"__q{ordinal}.csv")
        if part.exists():
            collected += 1
            print(f"skip checkpoint {instance['id']}", flush=True)
            continue
        grid, snapshot = harvest_ordinal(instance, ordinal)
        if snapshot is None:
            print(f"no q{ordinal} decision {instance['id']}", flush=True)
            continue
        ctx = context(snapshot)
        pool = candidate_pool(snapshot.candidates, ctx)
        if len(pool) < 2:
            continue
        features = FeatureExtractor(grid, snapshot.goals).extract_all(pool, ctx)
        scores = model.predict(features)
        analytical = [_compute_candidate_lookahead_tuple(cc, ctx) for cc in pool]
        order = sorted(range(len(pool)), key=lambda i: analytical[i], reverse=True)
        ranks = {idx: rank for rank, idx in enumerate(order)}
        prepared = []
        query_id = f"{instance['id']}/deep-q{ordinal}"
        for i, cc in enumerate(pool):
            teacher = immediate_children(cc, ctx, seconds=5)
            base = base_record(
                "adaptive-new-states-v1", query_id, instance, snapshot, cc, i,
                len(pool), teacher, analytical[i], ranks[i], float(scores[i]))
            base["decision_ordinal"] = ordinal
            base["minimum_informative_gap_from_q0"] = ordinal
            prepared.append((cc, base))
        jobs = [(grid, snapshot, cc, BUDGETS, timeout, False)
                for cc, _ in prepared]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            results = list(executor.map(rollout_job, jobs))
        rows = []
        for (_, base), result in zip(prepared, results):
            for checkpoint in result["checkpoints"]:
                row = dict(base); row.update(checkpoint); rows.append(row)
        write_csv(part, rows)
        collected += 1
        solved = sum(str(r["solved"]).lower() == "true"
                     for r in rows if r["budget"] == 1500)
        print(f"new-state {collected}/{target_count} {query_id} "
              f"solved={solved}/{len(pool)}", flush=True)
    summary = merge()
    raw = pd.read_csv(OUT / "raw_rollouts.csv")
    metadata = dict(
        campaign="adaptive-new-states-v1", updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        dirty=bool(git("status", "--porcelain")), python=sys.version,
        platform=platform.platform(), budgets=BUDGETS, wall_cap_seconds=timeout,
        workers=workers, state_count=len(summary),
        decision_ordinals=sorted({int(str(q).rsplit("q", 1)[1]) for q in raw.query_id}),
        rollout_mode="full_search_state", continuation_policy="residual",
        state_selection="q5/q10/q15/q20 from previously tractable train/development residual trajectories")
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"merged new states={len(summary)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-count", type=int, default=5)
    p.add_argument("--ordinal", type=int, default=5)
    p.add_argument("--timeout", type=float, default=20)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args.target_count, args.ordinal, args.timeout, args.workers)


if __name__ == "__main__":
    main()
