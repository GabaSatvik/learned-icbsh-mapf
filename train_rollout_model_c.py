"""Freeze a fresh test, replay rollout features, and fit fixed Ridge Model C."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from expanded_rollout_campaign import harvest_ordinal
from mapf.features import FEATURE_NAMES, FeatureExtractor
from mapf.grid import load_map
from mapf.ml_revision_policy import SavedLinear, candidate_pool
from mapf.rollout_oracle import conflict_id
from rollout_oracle_experiment import CONFIG, ROOT, context, write_csv

OUT = ROOT / "results" / "rollout_oracle" / "model_c"
PARTS = OUT / "feature_parts"
DATASET = OUT / "rollout_feature_dataset.csv"
PAIRS = OUT / "rollout_pairs.csv"
MODEL = ROOT / "models" / "model_c_rollout.json"
SEALED = OUT / "sealed_test_manifest.json"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def digest(obj):
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def largest_component(grid):
    unseen = {(r, c) for r in range(grid.height) for c in range(grid.width)
              if grid.is_passable(r, c)}
    best = []
    while unseen:
        start = next(iter(unseen))
        unseen.remove(start)
        q, component = deque([start]), [start]
        while q:
            r, c = q.popleft()
            for nxt in grid.get_neighbors(r, c):
                if nxt in unseen:
                    unseen.remove(nxt)
                    component.append(nxt)
                    q.append(nxt)
        if len(component) > len(best):
            best = component
    return sorted(best)


def freeze_test():
    """Create but do not solve fresh families 30--35."""
    OUT.mkdir(parents=True, exist_ok=True)
    if SEALED.exists():
        manifest = json.loads(SEALED.read_text())
        stored = manifest.pop("manifest_sha256")
        if digest(manifest) != stored:
            raise ValueError("Sealed test manifest checksum mismatch")
        manifest["manifest_sha256"] = stored
        print(f"sealed test already exists: {stored}")
        return
    maps = ["random-32-32-20", "room-32-32-4", "warehouse-32-32-aisles"]
    ks = [16, 24, 32]
    instances = []
    for family in range(30, 36):
        for map_index, map_name in enumerate(maps):
            map_path = ROOT / "data" / f"{map_name}.map"
            grid = load_map(map_path)
            component = largest_component(grid)
            k = ks[(family - 30 + map_index) % len(ks)]
            if len(component) < 2 * k:
                raise ValueError(f"Map too small for sealed instance {map_name} k={k}")
            seed_text = f"model-c-sealed-v1|{map_name}|family-{family}|k{k}"
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:15], 16)
            chosen = random.Random(seed).sample(component, 2 * k)
            starts, goals = chosen[:k], chosen[k:]
            row = dict(
                id=f"{map_name}/family-{family:02d}/k{k}",
                family_id=f"{map_name}/family-{family:02d}",
                split="sealed_test", map_name=map_name,
                map_path=f"data/{map_name}.map", map_sha256=sha_file(map_path),
                k=k, seed=seed, sampling="2k unique locations from largest component",
                starts=starts, goals=goals)
            row["instance_sha256"] = digest(row)
            instances.append(row)
    manifest = dict(
        schema="model-c-sealed-test-v1", created_before_feature_replay=True,
        generation_only_no_solver_execution=True, families=list(range(30, 36)),
        instances=instances)
    manifest["manifest_sha256"] = digest(manifest)
    with SEALED.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"sealed {len(instances)} untouched instances: {manifest['manifest_sha256']}")


def exact_label_rows():
    paths = [
        ROOT / "results" / "rollout_oracle" / "pilot" / "raw_rollouts.csv",
        ROOT / "results" / "rollout_oracle" / "adaptive" / "new_states" / "raw_rollouts.csv",
    ]
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    data["solved"] = data.solved.astype(str).str.lower().eq("true")
    final = data[data.budget == 1500].copy()
    exact_queries = [q for q, g in final.groupby("query_id") if g.solved.all()]
    return final[final.query_id.isin(exact_queries)].copy()


def ordinal(query_id):
    return int(query_id.rsplit("deep-q", 1)[1])


def replay_features():
    if not SEALED.exists():
        raise ValueError("Freeze sealed test before replay")
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())["instances"]
    instances = {x["id"]: x for x in cfg}
    labels = exact_label_rows()
    old_model = SavedLinear(ROOT / "models" / "model_b.json")
    audit = []
    queries = sorted(labels.query_id.unique())
    for number, query in enumerate(queries, 1):
        part = PARTS / (query.replace("/", "__") + ".csv")
        if part.exists():
            print(f"skip feature checkpoint {number}/{len(queries)} {query}", flush=True)
            continue
        group = labels[labels.query_id == query].sort_values("candidate_index")
        instance_id = str(group.iloc[0].instance_id)
        instance = instances[instance_id]
        grid, snapshot = harvest_ordinal(instance, ordinal(query))
        if snapshot is None:
            raise RuntimeError(f"Could not replay {query}")
        ctx = context(snapshot)
        pool = candidate_pool(snapshot.candidates, ctx)
        ids = [conflict_id(cc.conflict) for cc in pool]
        expected = list(group.candidate_conflict_id)
        if ids != expected:
            raise ValueError(f"Candidate mismatch {query}: {ids} != {expected}")
        features = FeatureExtractor(grid, snapshot.goals).extract_all(pool, ctx)
        if features.shape != (len(pool), len(FEATURE_NAMES)) or not np.isfinite(features).all():
            raise ValueError(f"Invalid features for {query}")
        old_scores = old_model.predict(features)
        np.testing.assert_allclose(
            old_scores, group.model_b_score.astype(float).to_numpy(),
            rtol=1e-6, atol=1e-6)
        rows = []
        for index, (label, vector) in enumerate(zip(group.itertuples(), features)):
            row = dict(
                query_id=query, instance_id=instance_id, family_id=label.family_id,
                split=label.split, map=label.map, agent_count=int(label.agent_count),
                decision_ordinal=ordinal(query), candidate_index=index,
                candidate_conflict_id=label.candidate_conflict_id,
                eventual_ct_expansions=int(label.ct_expansions),
                solution_cost=int(float(label.solution_cost)),
                residual_rank=int(label.residual_rank),
                model_b_score=float(label.model_b_score),
                one_step_min_f=label.one_step_min_f,
                one_step_max_f=label.one_step_max_f)
            row.update({name: float(value) for name, value in zip(FEATURE_NAMES, vector)})
            rows.append(row)
        write_csv(part, rows)
        audit.append(dict(query_id=query, candidates=len(rows), candidate_ids_verified=True,
                          model_b_scores_replayed=True))
        print(f"feature replay {number}/{len(queries)} {query}", flush=True)
    rows = []
    for path in sorted(PARTS.glob("*.csv")):
        rows.extend(pd.read_csv(path).to_dict("records"))
    write_csv(DATASET, rows)
    metadata = dict(
        queries=len({r["query_id"] for r in rows}), candidates=len(rows),
        feature_names=FEATURE_NAMES, dataset_sha256=sha_file(DATASET),
        sealed_manifest_sha256=json.loads(SEALED.read_text())["manifest_sha256"],
        candidate_ids_verified=True, model_b_scores_replayed=True,
        git_commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        dirty=bool(git("status", "--porcelain")), python=sys.version,
        platform=platform.platform())
    (OUT / "feature_replay_audit.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def strict_pairs(frame):
    rows = []
    for query, group in frame.groupby("query_id"):
        group = group.sort_values("candidate_index")
        local = []
        records = list(group.itertuples())
        for left, a in enumerate(records):
            for b in records[left+1:]:
                if a.eventual_ct_expansions == b.eventual_ct_expansions:
                    continue
                order = 1 if a.eventual_ct_expansions < b.eventual_ct_expansions else -1
                local.append(dict(
                    query_id=query, family_id=a.family_id, split=a.split,
                    candidate_i=int(a.candidate_index), candidate_j=int(b.candidate_index),
                    conflict_i=a.candidate_conflict_id, conflict_j=b.candidate_conflict_id,
                    expansions_i=int(a.eventual_ct_expansions),
                    expansions_j=int(b.eventual_ct_expansions), order=order,
                    absolute_gap=abs(int(a.eventual_ct_expansions)-int(b.eventual_ct_expansions))))
        for row in local:
            row["query_pair_weight"] = 1 / len(local)
            rows.append(row)
    return pd.DataFrame(rows)


def policy_metrics(frame, pair_frame, score_fn):
    query_metrics, regrets = [], []
    for query, group in frame.groupby("query_id"):
        group = group.sort_values("candidate_index")
        pairs = pair_frame[pair_frame.query_id == query]
        scores = score_fn(group)
        ok = []
        for pair in pairs.itertuples():
            pred = int(scores[pair.candidate_i] > scores[pair.candidate_j]) - int(
                scores[pair.candidate_i] < scores[pair.candidate_j])
            ok.append(.5 if pred == 0 else float(pred == pair.order))
        truth = group.eventual_ct_expansions.astype(int).to_numpy()
        chosen = int(np.argmax(scores))
        regret = int(truth[chosen] - truth.min())
        query_metrics.append(dict(
            query_id=query, family_id=group.iloc[0].family_id,
            pair_accuracy=float(np.mean(ok)) if ok else None,
            top1=regret == 0, regret=regret))
        regrets.append(regret)
    q = pd.DataFrame(query_metrics)
    informative = q.dropna(subset=["pair_accuracy"])
    return dict(
        queries=len(q), informative_queries=len(informative),
        families=int(q.family_id.nunique()),
        pairwise_accuracy=float(informative.pair_accuracy.mean()) if len(informative) else None,
        family_macro_pairwise=float(
            informative.groupby("family_id").pair_accuracy.mean().mean()) if len(informative) else None,
        top1_accuracy=float(q.top1.mean()), mean_regret=float(q.regret.mean()),
        median_regret=float(q.regret.median()),
        family_macro_regret=float(q.groupby("family_id").regret.mean().mean()),
        zero_regret_fraction=float((q.regret == 0).mean()), worst_regret=int(q.regret.max()))


def fit_model():
    if not DATASET.exists() or not SEALED.exists():
        raise ValueError("Freeze and replay before training")
    frame = pd.read_csv(DATASET)
    if set(frame.split) - {"train", "validation"}:
        raise ValueError("Unexpected split in training table")
    pairs = strict_pairs(frame)
    pairs.to_csv(PAIRS, index=False)
    train = frame[frame.split == "train"].copy()
    train_pairs = pairs[pairs.split == "train"]
    if train.family_id.nunique() < 2 or len(train_pairs) == 0:
        raise ValueError("Insufficient training data")
    scaler = StandardScaler().fit(train[FEATURE_NAMES].to_numpy(float))
    X, y, weights = [], [], []
    by_query = {q: g.sort_values("candidate_index") for q, g in train.groupby("query_id")}
    for pair in train_pairs.itertuples():
        z = scaler.transform(by_query[pair.query_id][FEATURE_NAMES].to_numpy(float))
        X.append(z[pair.candidate_i] - z[pair.candidate_j])
        y.append(pair.order)
        weights.append(pair.query_pair_weight)
    estimator = Ridge(alpha=10, fit_intercept=False).fit(
        np.asarray(X), np.asarray(y), sample_weight=np.asarray(weights))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                   coef=estimator.coef_.tolist(), intercept=0.0)
    MODEL.write_text(json.dumps(payload, indent=2))
    model_hash = sha_file(MODEL)
    metadata = dict(
        feature_names=FEATURE_NAMES, backend="sklearn_ridge",
        model_sha256=model_hash, model_name="model_c_rollout",
        alpha=10, fit_intercept=False, standardization="train candidates only",
        pair_weight="1 / strict pairs in query",
        target="lower exact full-search-state residual-continuation CT expansions",
        censored_pair_rule="no jointly censored pair; current fit uses exact state sets only",
        dataset_sha256=sha_file(DATASET), pairs_sha256=sha_file(PAIRS),
        sealed_manifest_sha256=json.loads(SEALED.read_text())["manifest_sha256"],
        training_family_ids=sorted(train.family_id.unique()),
        validation_family_ids=sorted(frame[frame.split == "validation"].family_id.unique()),
        training_queries=int(train.query_id.nunique()), training_candidates=len(train),
        training_pairs=len(train_pairs), validation_used_for_fit=False,
        sealed_test_used=False)
    metadata_path = Path(str(MODEL) + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    loaded = SavedLinear(MODEL)
    np.testing.assert_allclose(
        loaded.predict(train[FEATURE_NAMES].to_numpy(float)),
        estimator.predict(scaler.transform(train[FEATURE_NAMES].to_numpy(float))),
        rtol=1e-12, atol=1e-12)

    old = SavedLinear(ROOT / "models" / "model_b.json")
    evaluations = {}
    for split in ("train", "validation"):
        part = frame[frame.split == split].copy()
        pair_part = pairs[pairs.split == split]
        def model_c(g):
            return loaded.predict(g[FEATURE_NAMES].to_numpy(float))
        def model_b(g):
            return old.predict(g[FEATURE_NAMES].to_numpy(float))
        def residual(g):
            return -g.residual_rank.astype(float).to_numpy()
        def one_step(g):
            values = []
            for row in g.itertuples():
                a = math.inf if str(row.one_step_min_f) == "+inf" else float(row.one_step_min_f)
                b = math.inf if str(row.one_step_max_f) == "+inf" else float(row.one_step_max_f)
                values.append(a * 1_000_000 + b)
            return np.asarray(values)
        evaluations[split] = {
            "model_c": policy_metrics(part, pair_part, model_c),
            "model_b": policy_metrics(part, pair_part, model_b),
            "residual": policy_metrics(part, pair_part, residual),
            "one_step": policy_metrics(part, pair_part, one_step),
        }
    gate = dict(
        zero_serialization_error=True,
        validation_family_macro_beats_model_b=(
            evaluations["validation"]["model_c"]["family_macro_pairwise"] >
            evaluations["validation"]["model_b"]["family_macro_pairwise"]),
        validation_family_macro_beats_residual=(
            evaluations["validation"]["model_c"]["family_macro_pairwise"] >
            evaluations["validation"]["residual"]["family_macro_pairwise"]),
        validation_mean_regret_beats_model_b=(
            evaluations["validation"]["model_c"]["mean_regret"] <
            evaluations["validation"]["model_b"]["mean_regret"]),
    )
    report = dict(metadata=metadata, evaluations=evaluations, development_gate=gate)
    (OUT / "model_c_metrics.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Model C — fixed Ridge with rollout labels", "",
        f"Training uses {metadata['training_queries']} queries, {metadata['training_candidates']} candidates, "
        f"and {metadata['training_pairs']} strict pairs. Validation never enters the fit.", "",
        "| split | policy | family-macro pairwise | top-1 | mean regret | worst regret |",
        "|---|---|---:|---:|---:|---:|"]
    for split, policies in evaluations.items():
        for name, metrics in policies.items():
            lines.append(f"| {split} | {name} | {metrics['family_macro_pairwise']:.3f} | "
                         f"{metrics['top1_accuracy']:.3f} | {metrics['mean_regret']:.3f} | "
                         f"{metrics['worst_regret']} |")
    lines += ["", "## Development gate", ""]
    lines += [f"- {key}: **{value}**" for key, value in gate.items()]
    lines += ["", "The fresh sealed test manifest was generated before fitting and was not solved or inspected."]
    (OUT / "MODEL_C_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "replay", "train", "all"])
    args = parser.parse_args()
    if args.stage in ("freeze", "all"):
        freeze_test()
    if args.stage in ("replay", "all"):
        replay_features()
    if args.stage in ("train", "all"):
        fit_model()


if __name__ == "__main__":
    main()
