# mapf/test_features.py
"""
Unit tests for the 24-feature CL-ICBS-h extractor and solver optimality invariants.
"""
import time
from mapf.grid import GridMap
from mapf.low_level import compute_bfs_h_table, space_time_astar
from mapf.conflicts import detect_all_conflicts
from mapf.cardinal import classify_conflicts
from mapf.conflict_graph import compute_h4_and_graph
from mapf.features import FeatureExtractor, FEATURE_NAMES
from mapf.high_level_cbs import PlainCBS
from mapf.icbs_h import ICBShSolver


def build_corridor_instance():
    # 5x5 grid with a central horizontal corridor at row 2
    passable = [
        [True,  True,  True,  True,  True],
        [True,  False, False, False, True],
        [True,  True,  True,  True,  True],
        [True,  False, False, False, True],
        [True,  True,  True,  True,  True],
    ]
    grid = GridMap(5, 5, passable)
    starts = [(2, 0), (2, 4), (0, 2)]
    goals = [(2, 4), (2, 0), (4, 2)]
    return grid, starts, goals


def test_feature_dimensions_and_lookahead():
    grid, starts, goals = build_corridor_instance()
    k = len(starts)
    h_tables = [compute_bfs_h_table(grid, goals[i]) for i in range(k)]

    paths = [
        space_time_astar(grid, starts[i], goals[i], [], max_timesteps=30, h_table=h_tables[i])
        for i in range(k)
    ]
    assert all(p is not None for p in paths), "All agents must have valid root paths"

    conflicts = detect_all_conflicts(paths)
    assert len(conflicts) >= 1, "Expected at least one conflict in corridor instance"

    # Classify conflicts with h_tables
    try:
        classified = classify_conflicts(grid, starts, goals, paths, [], conflicts, max_timesteps=30, h_tables=h_tables)
    except TypeError:
        classified = classify_conflicts(grid, starts, goals, paths, [], conflicts, max_timesteps=30)

    h_val, G_CF = compute_h4_and_graph(classified)

    extractor = FeatureExtractor(grid)
    context = {
        "grid": grid,
        "paths": paths,
        "goals": goals,
        "G_CF": G_CF,
        "mvc_size": h_val,
        "pair_counts": {(0, 1): 2},
        "classified": classified,
    }

    feats = extractor.extract_all(classified, context)
    assert feats.shape == (len(classified), len(FEATURE_NAMES)), (
        f"Expected shape {(len(classified), len(FEATURE_NAMES))}, got {feats.shape}"
    )

    # Verify cached lookahead costs exist on cardinal conflicts
    for idx, cc in enumerate(classified):
        row = dict(zip(FEATURE_NAMES, feats[idx]))
        if cc.conflict_type == "CARDINAL":
            if "delta_g_min" in row:
                assert row["delta_g_min"] >= 1.0, f"Cardinal conflict must have delta_g_min >= 1, got {row['delta_g_min']}"
        assert 0.0 <= row["corridor_score_i"] <= 5.0
        assert 0.0 <= row["corridor_score_j"] <= 5.0

    print(f"[PASS] Extracted {feats.shape[1]} features across {feats.shape[0]} conflicts successfully.")


def test_optimality_invariant():
    grid, starts, goals = build_corridor_instance()
    cbs = PlainCBS(grid)
    icbs = ICBShSolver(grid)

    res_cbs = cbs.solve(starts, goals, timeout=10.0)
    res_icbs = icbs.solve(starts, goals, timeout=10.0)

    assert res_cbs["status"] == "SOLVED" and res_icbs["status"] == "SOLVED"
    assert res_cbs["cost"] == res_icbs["cost"], (
        f"Cost mismatch! PlainCBS={res_cbs['cost']} vs PlainICBSh={res_icbs['cost']}"
    )
    print(
        f"[PASS] Optimality invariant verified: Cost={res_icbs['cost']} "
        f"(PlainCBS nodes={res_cbs['nodes_expanded']}, PlainICBSh nodes={res_icbs['nodes_expanded']})"
    )


if __name__ == "__main__":
    test_feature_dimensions_and_lookahead()
    test_optimality_invariant()
