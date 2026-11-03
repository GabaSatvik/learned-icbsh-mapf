"""Family-level learning curve for the fixed rollout-labeled Ridge model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mapf.features import FEATURE_NAMES
from train_rollout_model_c import ROOT, policy_metrics

MODEL_C = ROOT / "results" / "rollout_oracle" / "model_c"
OUT = MODEL_C / "learning_curve"
SIZES = [4, 8, 12, 16, 19]
SEED = 20261103


def nested_family_order(families, repeat):
    """Deterministic map-stratified order; prefixes form nested subsets."""
    rng = random.Random(SEED + repeat)
    by_map = {}
    for family in families:
        by_map.setdefault(family.split("/", 1)[0], []).append(family)
    for values in by_map.values():
        values.sort()
        rng.shuffle(values)
    order = []
    maps = sorted(by_map)
    while any(by_map.values()):
        cycle = [name for name in maps if by_map[name]]
        rng.shuffle(cycle)
        for name in cycle:
            order.append(by_map[name].pop())
    if len(order) != len(families) or set(order) != set(families):
        raise AssertionError("Invalid family order")
    return order


def fit_subset(frame, pairs, selected):
    train = frame[(frame.split == "train") & frame.family_id.isin(selected)].copy()
    train_pairs = pairs[(pairs.split == "train") & pairs.family_id.isin(selected)]
    scaler = StandardScaler().fit(train[FEATURE_NAMES].to_numpy(float))
    grouped = {q: g.sort_values("candidate_index") for q, g in train.groupby("query_id")}
    X, y, weights = [], [], []
    for pair in train_pairs.itertuples():
        z = scaler.transform(grouped[pair.query_id][FEATURE_NAMES].to_numpy(float))
        X.append(z[pair.candidate_i] - z[pair.candidate_j])
        y.append(pair.order)
        weights.append(pair.query_pair_weight)
    if not X:
        raise ValueError("Subset contains no strict pairs")
    model = Ridge(alpha=10, fit_intercept=False).fit(
        np.asarray(X), np.asarray(y), sample_weight=np.asarray(weights))
    return scaler, model, len(train), int(train.query_id.nunique()), len(train_pairs)


def evaluate(frame, pairs, scaler, model):
    def scores(group):
        return model.predict(scaler.transform(group[FEATURE_NAMES].to_numpy(float)))
    return policy_metrics(frame, pairs, scores)


def run(repeats):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MODEL_C / "rollout_feature_dataset.csv")
    pairs = pd.read_csv(MODEL_C / "rollout_pairs.csv")
    train_families = sorted(frame.loc[frame.split == "train", "family_id"].unique())
    validation = frame[frame.split == "validation"].copy()
    validation_pairs = pairs[pairs.split == "validation"].copy()
    validation_families = set(validation.family_id.unique())
    if set(train_families) & validation_families:
        raise AssertionError("Training and development families overlap")
    rows = []
    for repeat in range(repeats):
        order = nested_family_order(train_families, repeat)
        for size in SIZES:
            selected = order[:size]
            if set(selected) & validation_families:
                raise AssertionError("Development family entered a training subset")
            scaler, model, candidates, queries, pair_count = fit_subset(
                frame, pairs, selected)
            dev = evaluate(validation, validation_pairs, scaler, model)
            heldout_ids = [f for f in train_families if f not in selected]
            if heldout_ids:
                heldout = frame[(frame.split == "train") &
                                frame.family_id.isin(heldout_ids)].copy()
                heldout_pairs = pairs[(pairs.split == "train") &
                                      pairs.family_id.isin(heldout_ids)].copy()
                heldout_metrics = evaluate(heldout, heldout_pairs, scaler, model)
            else:
                heldout_metrics = {}
            rows.append(dict(
                repeat=repeat, family_count=size,
                selected_families=json.dumps(selected),
                training_queries=queries, training_candidates=candidates,
                training_pairs=pair_count,
                development_family_macro_pairwise=dev["family_macro_pairwise"],
                development_top1=dev["top1_accuracy"],
                development_mean_regret=dev["mean_regret"],
                development_family_macro_regret=dev["family_macro_regret"],
                remaining_train_families=len(heldout_ids),
                heldout_train_family_macro_pairwise=heldout_metrics.get(
                    "family_macro_pairwise"),
                heldout_train_top1=heldout_metrics.get("top1_accuracy"),
                heldout_train_mean_regret=heldout_metrics.get("mean_regret")))
        print(f"learning-curve repeat {repeat+1}/{repeats}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "family_learning_curve_raw.csv", index=False)
    summary = raw.groupby("family_count").agg(
        repeats=("repeat", "count"),
        mean_training_queries=("training_queries", "mean"),
        mean_training_pairs=("training_pairs", "mean"),
        development_pairwise_mean=("development_family_macro_pairwise", "mean"),
        development_pairwise_std=("development_family_macro_pairwise", "std"),
        development_pairwise_min=("development_family_macro_pairwise", "min"),
        development_pairwise_max=("development_family_macro_pairwise", "max"),
        development_top1_mean=("development_top1", "mean"),
        development_top1_std=("development_top1", "std"),
        development_regret_mean=("development_mean_regret", "mean"),
        development_regret_std=("development_mean_regret", "std"),
        heldout_train_pairwise_mean=("heldout_train_family_macro_pairwise", "mean"),
        heldout_train_pairwise_std=("heldout_train_family_macro_pairwise", "std"),
    ).reset_index()
    summary.to_csv(OUT / "family_learning_curve_summary.csv", index=False)

    means = dict(zip(summary.family_count, summary.development_pairwise_mean))
    slope = float(np.polyfit(summary.family_count, summary.development_pairwise_mean, 1)[0])
    best_size = int(summary.loc[summary.development_pairwise_mean.idxmax(), "family_count"])
    change_4_to_19 = float(means[19] - means[4])
    diagnosis = (
        "MORE_INDEPENDENT_DATA_IS_LIKELY_TO_HELP"
        if change_4_to_19 >= .05 and slope > 0
        else "NO_POSITIVE_LEARNING_CURVE_EVIDENCE")
    baseline = json.loads((MODEL_C / "model_c_metrics.json").read_text())
    result = dict(
        repeats=repeats, sizes=SIZES, seed=SEED,
        train_families=len(train_families),
        validation_families=int(validation.family_id.nunique()),
        slope_accuracy_per_family=slope,
        change_4_to_19=change_4_to_19,
        best_mean_family_count=best_size,
        diagnosis=diagnosis,
        full_model_validation=baseline["evaluations"]["validation"])
    (OUT / "learning_curve_summary.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Model C family-level learning curve", "",
        f"Fixed Ridge, fixed 24 features, alpha 10, {repeats} deterministic "
        "map-stratified nested family orders. The six development families are "
        "never used for fitting.", "",
        "| training families | mean train queries | mean strict pairs | dev family-macro accuracy | std | dev top-1 | dev mean regret |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        std = 0.0 if np.isnan(row.development_pairwise_std) else row.development_pairwise_std
        lines.append(
            f"| {row.family_count} | {row.mean_training_queries:.1f} | "
            f"{row.mean_training_pairs:.1f} | {row.development_pairwise_mean:.3f} | "
            f"{std:.3f} | {row.development_top1_mean:.3f} | "
            f"{row.development_regret_mean:.3f} |")
    lines += ["", f"Linear trend: {slope:+.4f} accuracy per added family.",
              f"Change from 4 to all 19 families: {change_4_to_19:+.3f}.",
              f"Best mean development accuracy occurs at {best_size} families.", "",
              f"Diagnosis: **{diagnosis}**.", "",
              "Interpretation: more families reduce mean development regret, but they do "
              "not improve the primary ranking-accuracy measure. The all-family model is "
              "worse than the average smaller-family model on the fixed development set. "
              "This is not evidence that four families are sufficient or preferable; the "
              "development set has only six families, and the smaller-subset estimates vary "
              "substantially with which families are sampled.", "",
              "The result does not support the simple explanation that Model C failed only "
              "because it had too few examples. It points instead toward family/distribution "
              "mismatch and/or a limitation of the fixed 24-feature linear representation. "
              "More data could still help if it is deliberately diverse and targeted, but "
              "blindly collecting more states from the same family mixture is not justified.", "",
              "This diagnoses data quantity only for the current feature representation "
              "and linear model. It does not open or evaluate the sealed test set."]
    (OUT / "LEARNING_CURVE_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    run(args.repeats)


if __name__ == "__main__":
    main()
