"""Diagnose how rollout-ranking signal transfers between scenario families.

This script never reads or evaluates the sealed test instances.  It keeps the
Model C feature set and Ridge configuration fixed and uses only the existing
training/development feature table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from analyze_model_c_learning_curve import fit_subset
from mapf.features import FEATURE_NAMES
from mapf.ml_revision_policy import SavedLinear
from train_rollout_model_c import ROOT, policy_metrics


MODEL_C = ROOT / "results" / "rollout_oracle" / "model_c"
OUT = MODEL_C / "family_transfer"


def policy_scores(name, group, model_c):
    if name == "model_c":
        scaler, model = model_c
        return model.predict(scaler.transform(group[FEATURE_NAMES].to_numpy(float)))
    if name == "model_b":
        return SavedLinear(ROOT / "models" / "model_b.json").predict(
            group[FEATURE_NAMES].to_numpy(float))
    if name == "residual":
        return -group.residual_rank.astype(float).to_numpy()
    if name == "one_step":
        values = []
        for row in group.itertuples():
            left = math.inf if str(row.one_step_min_f) == "+inf" else float(row.one_step_min_f)
            right = math.inf if str(row.one_step_max_f) == "+inf" else float(row.one_step_max_f)
            values.append(left * 1_000_000 + right)
        return np.asarray(values)
    raise ValueError(name)


def evaluate_family(frame, pairs, family, score_fn):
    part = frame[frame.family_id == family].copy()
    pair_part = pairs[pairs.family_id == family].copy()
    return policy_metrics(part, pair_part, score_fn)


def family_policy_table(frame, pairs, full_model):
    validation = frame[frame.split == "validation"].copy()
    validation_pairs = pairs[pairs.split == "validation"].copy()
    rows = []
    for family in sorted(validation.family_id.unique()):
        for policy in ("model_c", "model_b", "residual", "one_step"):
            metrics = evaluate_family(
                validation, validation_pairs, family,
                lambda group, p=policy: policy_scores(p, group, full_model))
            rows.append(dict(family_id=family, policy=policy, **metrics))
    return pd.DataFrame(rows)


def transfer_tables(frame, pairs, train_families, development_families, baseline):
    validation = frame[frame.split == "validation"].copy()
    validation_pairs = pairs[pairs.split == "validation"].copy()
    leave_rows, single_rows = [], []
    for number, omitted in enumerate(train_families, 1):
        selected = [family for family in train_families if family != omitted]
        scaler, model, candidates, queries, pair_count = fit_subset(frame, pairs, selected)
        for development in development_families:
            metrics = evaluate_family(
                validation, validation_pairs, development,
                lambda group, s=scaler, m=model: m.predict(
                    s.transform(group[FEATURE_NAMES].to_numpy(float))))
            base = baseline[development]
            accuracy = metrics["family_macro_pairwise"]
            base_accuracy = base["family_macro_pairwise"]
            leave_rows.append(dict(
                omitted_training_family=omitted,
                development_family=development,
                training_families=len(selected), training_queries=queries,
                training_candidates=candidates, training_pairs=pair_count,
                family_macro_pairwise=accuracy,
                top1_accuracy=metrics["top1_accuracy"],
                mean_regret=metrics["mean_regret"],
                accuracy_delta_vs_full=(accuracy - base_accuracy)
                if accuracy is not None and base_accuracy is not None else np.nan,
                regret_delta_vs_full=metrics["mean_regret"] - base["mean_regret"]))
        print(f"leave-one-family-out {number}/{len(train_families)}", flush=True)

    for number, source in enumerate(train_families, 1):
        try:
            scaler, model, candidates, queries, pair_count = fit_subset(
                frame, pairs, [source])
        except ValueError:
            continue
        for development in development_families:
            metrics = evaluate_family(
                validation, validation_pairs, development,
                lambda group, s=scaler, m=model: m.predict(
                    s.transform(group[FEATURE_NAMES].to_numpy(float))))
            single_rows.append(dict(
                source_training_family=source, development_family=development,
                training_queries=queries, training_candidates=candidates,
                training_pairs=pair_count,
                family_macro_pairwise=metrics["family_macro_pairwise"],
                top1_accuracy=metrics["top1_accuracy"],
                mean_regret=metrics["mean_regret"]))
        print(f"single-family transfer {number}/{len(train_families)}", flush=True)
    return pd.DataFrame(leave_rows), pd.DataFrame(single_rows)


def pair_vectors(frame, pairs, scaler):
    grouped = {
        query: group.sort_values("candidate_index")
        for query, group in frame.groupby("query_id")
    }
    vectors, metadata = [], []
    for pair in pairs.itertuples():
        group = grouped[pair.query_id]
        z = scaler.transform(group[FEATURE_NAMES].to_numpy(float))
        vectors.append(z[pair.candidate_i] - z[pair.candidate_j])
        metadata.append(dict(
            query_id=pair.query_id, family_id=pair.family_id,
            conflict_i=pair.conflict_i, conflict_j=pair.conflict_j,
            order=int(pair.order), absolute_gap=int(pair.absolute_gap),
            query_pair_weight=float(pair.query_pair_weight)))
    return np.asarray(vectors), pd.DataFrame(metadata)


def nearest_pair_consistency(frame, pairs):
    train = frame[frame.split == "train"].copy()
    scaler = StandardScaler().fit(train[FEATURE_NAMES].to_numpy(float))
    train_pairs = pairs[pairs.split == "train"].copy()
    validation_pairs = pairs[pairs.split == "validation"].copy()
    train_x, train_meta = pair_vectors(frame, train_pairs, scaler)
    validation_x, validation_meta = pair_vectors(frame, validation_pairs, scaler)

    train_predictions, train_distances = [], []
    for index, vector in enumerate(train_x):
        eligible = np.flatnonzero(
            train_meta.family_id.to_numpy() != train_meta.iloc[index].family_id)
        candidates = train_x[eligible]
        direct = np.linalg.norm(candidates - vector, axis=1)
        reversed_distance = np.linalg.norm(-candidates - vector, axis=1)
        use_reversed = reversed_distance < direct
        distances = np.minimum(direct, reversed_distance)
        local = int(np.argmin(distances))
        source_index = eligible[local]
        prediction = int(train_meta.iloc[source_index].order)
        if use_reversed[local]:
            prediction *= -1
        train_predictions.append(prediction)
        train_distances.append(float(distances[local]))
    train_meta = train_meta.copy()
    train_meta["nearest_other_family_prediction"] = train_predictions
    train_meta["nearest_other_family_distance"] = train_distances
    train_meta["correct"] = train_meta.order == train_meta.nearest_other_family_prediction

    validation_rows = []
    augmented_x = np.vstack([train_x, -train_x])
    augmented_order = np.concatenate([
        train_meta.order.to_numpy(), -train_meta.order.to_numpy()])
    augmented_source = np.concatenate([
        np.arange(len(train_meta)), np.arange(len(train_meta))])
    for index, vector in enumerate(validation_x):
        distances = np.linalg.norm(augmented_x - vector, axis=1)
        nearest = int(np.argmin(distances))
        source = int(augmented_source[nearest])
        target = validation_meta.iloc[index]
        reference = train_meta.iloc[source]
        validation_rows.append(dict(
            development_query=target.query_id,
            development_family=target.family_id,
            development_conflict_i=target.conflict_i,
            development_conflict_j=target.conflict_j,
            true_order=int(target.order),
            nearest_prediction=int(augmented_order[nearest]),
            correct=int(target.order) == int(augmented_order[nearest]),
            distance=float(distances[nearest]),
            source_training_query=reference.query_id,
            source_training_family=reference.family_id,
            source_conflict_i=reference.conflict_i,
            source_conflict_j=reference.conflict_j,
            source_absolute_gap=int(reference.absolute_gap),
            development_absolute_gap=int(target.absolute_gap),
            query_pair_weight=float(target.query_pair_weight)))
    validation_nearest = pd.DataFrame(validation_rows)

    train_family_accuracy = train_meta.groupby("family_id").apply(
        lambda group: np.average(group.correct, weights=group.query_pair_weight),
        include_groups=False)
    development_family_accuracy = validation_nearest.groupby("development_family").apply(
        lambda group: np.average(group.correct, weights=group.query_pair_weight),
        include_groups=False)
    summary = dict(
        train_cross_family_nearest_pair_accuracy=float(train_family_accuracy.mean()),
        validation_nearest_train_pair_accuracy=float(development_family_accuracy.mean()),
        validation_nearest_train_median_distance=float(validation_nearest.distance.median()),
        train_families=int(train_meta.family_id.nunique()),
        development_families=int(validation_meta.family_id.nunique()))
    return train_meta, validation_nearest, summary


def feature_direction_table(frame, pairs):
    train_pairs = pairs[pairs.split == "train"].copy()
    train = frame[frame.split == "train"].copy()
    grouped = {
        query: group.sort_values("candidate_index")
        for query, group in train.groupby("query_id")
    }
    rows = []
    for family in sorted(train.family_id.unique()):
        family_pairs = train_pairs[train_pairs.family_id == family]
        if family_pairs.empty:
            continue
        for feature in FEATURE_NAMES:
            weighted_correct, total_weight = 0.0, 0.0
            for pair in family_pairs.itertuples():
                values = grouped[pair.query_id][feature].to_numpy(float)
                difference = values[pair.candidate_i] - values[pair.candidate_j]
                prediction = int(difference > 0) - int(difference < 0)
                correct = .5 if prediction == 0 else float(prediction == pair.order)
                weighted_correct += pair.query_pair_weight * correct
                total_weight += pair.query_pair_weight
            rows.append(dict(
                family_id=family, feature=feature,
                direction_accuracy=weighted_correct / total_weight,
                strict_pairs=len(family_pairs)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(MODEL_C / "rollout_feature_dataset.csv")
    pairs = pd.read_csv(MODEL_C / "rollout_pairs.csv")
    if set(frame.split) != {"train", "validation"}:
        raise AssertionError("Unexpected data split; refusing to continue")
    train_families = sorted(frame.loc[frame.split == "train", "family_id"].unique())
    development_families = sorted(
        frame.loc[frame.split == "validation", "family_id"].unique())
    if set(train_families) & set(development_families):
        raise AssertionError("Training and development families overlap")

    coverage = frame.groupby(["split", "family_id"]).agg(
        queries=("query_id", "nunique"),
        candidates=("candidate_index", "size")).reset_index()
    pair_coverage = pairs.groupby(["split", "family_id"]).size().rename(
        "strict_pairs").reset_index()
    coverage = coverage.merge(pair_coverage, how="left", on=["split", "family_id"])
    coverage["strict_pairs"] = coverage.strict_pairs.fillna(0).astype(int)
    coverage["informative"] = coverage.strict_pairs > 0
    coverage.to_csv(OUT / "family_data_coverage.csv", index=False)

    full_scaler, full_model, _, _, _ = fit_subset(frame, pairs, train_families)
    full = (full_scaler, full_model)
    policy_table = family_policy_table(frame, pairs, full)
    policy_table.to_csv(OUT / "development_family_policy_metrics.csv", index=False)
    baseline = {
        family: evaluate_family(
            frame[frame.split == "validation"], pairs[pairs.split == "validation"],
            family, lambda group: full_model.predict(
                full_scaler.transform(group[FEATURE_NAMES].to_numpy(float))))
        for family in development_families
    }

    leave, single = transfer_tables(
        frame, pairs, train_families, development_families, baseline)
    leave.to_csv(OUT / "leave_one_training_family_out.csv", index=False)
    single.to_csv(OUT / "single_family_transfer.csv", index=False)
    influence = leave.groupby("omitted_training_family").agg(
        mean_accuracy_delta_when_removed=("accuracy_delta_vs_full", "mean"),
        min_accuracy_delta_when_removed=("accuracy_delta_vs_full", "min"),
        max_accuracy_delta_when_removed=("accuracy_delta_vs_full", "max"),
        mean_regret_delta_when_removed=("regret_delta_vs_full", "mean"),
        development_families_helped_by_removal=(
            "accuracy_delta_vs_full", lambda values: int((values > 0).sum())),
        development_families_hurt_by_removal=(
            "accuracy_delta_vs_full", lambda values: int((values < 0).sum())),
    ).reset_index().sort_values("mean_accuracy_delta_when_removed", ascending=False)
    influence.to_csv(OUT / "training_family_influence_summary.csv", index=False)

    train_nearest, validation_nearest, nearest_summary = nearest_pair_consistency(
        frame, pairs)
    train_nearest.to_csv(OUT / "train_cross_family_nearest_pairs.csv", index=False)
    validation_nearest.to_csv(OUT / "development_nearest_training_pairs.csv", index=False)
    contradictions = validation_nearest[~validation_nearest.correct].sort_values(
        "distance").head(12)
    contradictions.to_csv(OUT / "closest_cross_family_contradictions.csv", index=False)

    directions = feature_direction_table(frame, pairs)
    directions.to_csv(OUT / "feature_direction_by_family.csv", index=False)
    direction_summary = directions.groupby("feature").agg(
        families=("family_id", "nunique"),
        mean_direction_accuracy=("direction_accuracy", "mean"),
        std_across_families=("direction_accuracy", "std"),
        min_direction_accuracy=("direction_accuracy", "min"),
        max_direction_accuracy=("direction_accuracy", "max"),
        families_positive=("direction_accuracy", lambda values: int((values > .55).sum())),
        families_negative=("direction_accuracy", lambda values: int((values < .45).sum())),
    ).reset_index().sort_values("std_across_families", ascending=False)
    direction_summary.to_csv(OUT / "feature_direction_summary.csv", index=False)

    model_c_rows = policy_table[policy_table.policy == "model_c"]
    residual_rows = policy_table[policy_table.policy == "residual"]
    comparison = model_c_rows.merge(
        residual_rows, on="family_id", suffixes=("_model_c", "_residual"))
    informative_comparison = comparison.dropna(
        subset=["family_macro_pairwise_model_c", "family_macro_pairwise_residual"])
    correct_distances = validation_nearest[validation_nearest.correct].distance
    wrong_distances = validation_nearest[~validation_nearest.correct].distance
    informative_train_families = int(
        coverage.query("split == 'train'").informative.sum())
    informative_development_families = int(
        coverage.query("split == 'validation'").informative.sum())
    harmful = influence.head(5)
    helpful = influence.tail(5).sort_values("mean_accuracy_delta_when_removed")
    result = dict(
        training_families=len(train_families),
        development_families=len(development_families),
        informative_training_families=informative_train_families,
        informative_development_families=informative_development_families,
        development_families_where_model_c_beats_residual=int((
            informative_comparison.family_macro_pairwise_model_c >
            informative_comparison.family_macro_pairwise_residual).sum()),
        model_c_development_family_accuracy={
            row.family_id: (None if pd.isna(row.family_macro_pairwise)
                            else row.family_macro_pairwise)
            for row in model_c_rows.itertuples()},
        residual_development_family_accuracy={
            row.family_id: (None if pd.isna(row.family_macro_pairwise)
                            else row.family_macro_pairwise)
            for row in residual_rows.itertuples()},
        most_harmful_training_families=[
            dict(family=row.omitted_training_family,
                 accuracy_gain_when_removed=row.mean_accuracy_delta_when_removed)
            for row in harmful.itertuples()],
        most_helpful_training_families=[
            dict(family=row.omitted_training_family,
                 accuracy_loss_when_removed=-row.mean_accuracy_delta_when_removed)
            for row in helpful.itertuples()],
        nearest_pair_consistency=nearest_summary,
        median_nearest_distance_when_label_agrees=float(correct_distances.median()),
        median_nearest_distance_when_label_disagrees=float(wrong_distances.median()),
        most_family_dependent_features=direction_summary.head(8).feature.tolist(),
        diagnosis="FROZEN_FEATURE_CONTEXT_MISMATCH_LIKELY_SMALL_SAMPLE_LIMITS_CERTAINTY",
        sealed_test_used=False)
    (OUT / "family_transfer_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False))

    lines = [
        "# Model C family-transfer diagnosis", "",
        "This analysis holds the 24 features and Ridge setup fixed. It uses only the "
        "existing training and development tables; the sealed test is untouched.", "",
        f"There are {len(train_families)} nominal training families, but only "
        f"{informative_train_families} contain a strict "
        "conflict comparison. There are six development families, but only five are "
        "informative for pairwise accuracy. Most families contain only one to five states.", "",
        "## Development families", "",
        "| development family | Model C pairwise | Model B | residual | Model C regret |",
        "|---|---:|---:|---:|---:|"]
    lookup = policy_table.set_index(["family_id", "policy"])
    def metric(value, digits=3):
        return "n/a" if pd.isna(value) else f"{value:.{digits}f}"
    for family in development_families:
        c = lookup.loc[(family, "model_c")]
        b = lookup.loc[(family, "model_b")]
        r = lookup.loc[(family, "residual")]
        lines.append(
            f"| {family} | {metric(c.family_macro_pairwise)} | "
            f"{metric(b.family_macro_pairwise)} | "
            f"{metric(r.family_macro_pairwise)} | "
            f"{metric(c.mean_regret, 2)} |")
    lines += ["", "## Marginal training-family influence", "",
              "Positive delta means Model C improved when that training family was removed.", "",
              "| family removed | mean accuracy delta | dev families improved | dev families worsened |",
              "|---|---:|---:|---:|"]
    for row in influence.itertuples():
        lines.append(
            f"| {row.omitted_training_family} | "
            f"{row.mean_accuracy_delta_when_removed:+.3f} | "
            f"{row.development_families_helped_by_removal} | "
            f"{row.development_families_hurt_by_removal} |")
    lines += ["", "## Feature/label consistency", "",
              f"Nearest-pair label agreement across training families: "
              f"{nearest_summary['train_cross_family_nearest_pair_accuracy']:.3f}.",
              f"Development pair agreement with its nearest training pair: "
              f"{nearest_summary['validation_nearest_train_pair_accuracy']:.3f}.", "",
              f"Median nearest-pair distance is {correct_distances.median():.3f} when the "
              f"label agrees and {wrong_distances.median():.3f} when it disagrees. Distance "
              "therefore does not identify trustworthy transfers in this sample.", "",
              "Low agreement means conflict pairs that look similar in the frozen feature "
              "space can still have opposite deep-rollout preferences. Nearest-neighbor "
              "distance is only diagnostic in this 24-dimensional standardized space.", "",
              "Most family-dependent feature directions: " +
              ", ".join(direction_summary.head(8).feature.tolist()) + ".", "",
              "## Interpretation", "",
              f"Model C beats residual on 0 of {len(informative_comparison)} informative "
              "development families, so the failure is not confined to one held-out family. "
              "Removing some training families helps some development families but hurts "
              "others; no universally harmful family was found. A notable interaction is "
              "warehouse family 13: it transfers best when trained alone, yet the full model "
              "improves when it is removed. That is evidence of unstable mixing, not evidence "
              "that its data should be discarded.", "",
              "The chance-level cross-family consistency and feature-direction reversals "
              "support a frozen-feature/context mismatch: deep rollout quality depends on "
              "search context that the 24 local features do not encode consistently. The "
              "very small number of independent states per family limits certainty, so the "
              "correct response is a targeted feature-and-coverage experiment, not blind data "
              "collection and not deletion of inconvenient families.", "",
              "No production selector was changed and no new model was accepted."]
    (OUT / "FAMILY_TRANSFER_REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
