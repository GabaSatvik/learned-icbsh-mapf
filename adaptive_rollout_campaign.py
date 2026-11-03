"""Recover censored pilot states with longer, append-safe adaptive rollouts."""
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

from mapf.baselines import _compute_candidate_lookahead_tuple
from mapf.features import FeatureExtractor
from mapf.ml_revision_policy import SavedLinear, candidate_pool, immediate_children
from mapf.rollout_oracle import conflict_id
from rollout_oracle_experiment import (
    CONFIG, ROOT, base_record, context, harvest_one, rollout_job, write_csv,
)

OUT = ROOT / "results" / "rollout_oracle" / "adaptive"
PARTS = OUT / "raw_parts"
BUDGETS = [1500, 3000, 5000]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def slug(instance_id):
    return instance_id.replace("/", "__")


def unresolved_instances():
    summary = pd.read_csv(ROOT / "results" / "rollout_oracle" / "pilot" / "state_summary.csv")
    complete = summary.all_candidates_exact.astype(str).str.lower().eq("true")
    return summary.loc[~complete, "instance_id"].tolist()


def pilot_candidates(instance_id):
    raw = pd.read_csv(ROOT / "results" / "rollout_oracle" / "pilot" / "raw_rollouts.csv")
    rows = raw[(raw.instance_id == instance_id) & (raw.budget == 1500)]
    return list(rows.sort_values("candidate_index").candidate_conflict_id)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge_parts():
    rows = []
    for path in sorted(PARTS.glob("*.csv")):
        rows.extend(read_rows(path))
    write_csv(OUT / "raw_rollouts.csv", rows)
    return rows


def summarize(rows):
    by_query = {}
    for row in rows:
        if int(row["budget"]) == BUDGETS[-1]:
            by_query.setdefault(row["query_id"], []).append(row)
    output = []
    for query, group in sorted(by_query.items()):
        solved = [r for r in group if str(r["solved"]).lower() == "true"]
        values = [int(r["ct_expansions"]) for r in solved]
        output.append(dict(
            query_id=query, instance_id=group[0]["instance_id"],
            family_id=group[0]["family_id"], map=group[0]["map"],
            agent_count=group[0]["agent_count"], candidate_count=len(group),
            exact_candidates=len(solved), all_candidates_exact=len(solved) == len(group),
            min_eventual_expansions=min(values) if values else None,
            max_eventual_expansions=max(values) if values else None,
            expansion_gap=max(values)-min(values) if values else None,
            optimal_costs=json.dumps(sorted({r["solution_cost"] for r in solved})),
            optimal_cost_preserved=len({r["solution_cost"] for r in solved}) <= 1))
    write_csv(OUT / "state_summary.csv", output)
    return output


def run(limit=None, timeout=60.0, workers=4, dry_run=False):
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    by_id = {x["id"]: x for x in cfg["instances"]}
    targets = unresolved_instances()
    if limit is not None:
        targets = targets[:limit]
    model = SavedLinear(ROOT / "models" / "model_b.json")
    completed = 0
    for instance_id in targets:
        part = PARTS / f"{slug(instance_id)}.csv"
        if part.exists():
            print(f"skip checkpoint {instance_id}", flush=True)
            completed += 1
            continue
        instance = by_id[instance_id]
        grid, snapshot = harvest_one(instance)
        if snapshot is None:
            raise RuntimeError(f"Could not recapture {instance_id}")
        snapshot.grid = grid
        ctx = context(snapshot)
        pool = candidate_pool(snapshot.candidates, ctx)
        ids = [conflict_id(cc.conflict) for cc in pool]
        expected = pilot_candidates(instance_id)
        if ids != expected:
            raise RuntimeError(
                f"Candidate replay mismatch for {instance_id}: {ids} != {expected}")
        print(f"verified {instance_id} candidates={len(pool)}", flush=True)
        if dry_run:
            completed += 1
            continue

        features = FeatureExtractor(grid, snapshot.goals).extract_all(pool, ctx)
        scores = model.predict(features)
        analytical = [_compute_candidate_lookahead_tuple(cc, ctx) for cc in pool]
        order = sorted(range(len(pool)), key=lambda i: analytical[i], reverse=True)
        ranks = {idx: rank for rank, idx in enumerate(order)}
        query_id = f"{instance_id}/adaptive-q0"
        prepared = []
        for i, cc in enumerate(pool):
            teacher = immediate_children(cc, ctx, seconds=5)
            base = base_record(
                "adaptive-recovery-v1", query_id, instance, snapshot, cc, i,
                len(pool), teacher, analytical[i], ranks[i], float(scores[i]))
            base["source_pilot_query_id"] = f"{instance_id}/deep-q0"
            base["adaptive_wall_cap_seconds"] = timeout
            prepared.append((i, cc, base))
        jobs = [(grid, snapshot, cc, BUDGETS, timeout, False)
                for _, cc, _ in prepared]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            results = list(executor.map(rollout_job, jobs))
        rows = []
        for (_, _, base), result in zip(prepared, results):
            for checkpoint in result["checkpoints"]:
                row = dict(base)
                row.update(checkpoint)
                rows.append(row)
        write_csv(part, rows)
        completed += 1
        solved = sum(str(r["solved"]).lower() == "true"
                     for r in rows if r["budget"] == BUDGETS[-1])
        print(f"adaptive {completed}/{len(targets)} {instance_id} "
              f"solved={solved}/{len(pool)}", flush=True)

    if not dry_run:
        rows = merge_parts()
        states = summarize(rows)
        metadata = dict(
            campaign="adaptive-recovery-v1", created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            git_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
            dirty=bool(git("status", "--porcelain")), python=sys.version,
            platform=platform.platform(), budgets=BUDGETS, wall_cap_seconds=timeout,
            workers=workers, target_states=len(targets), completed_part_files=len(states),
            rollout_mode="full_search_state", continuation_policy="residual",
            candidate_replay_required=True)
        (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"merged states={len(states)} rows={len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.limit, args.timeout, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
