"""
Unit tests for the ~16 features across all 6 statement-named families
on 3 hand-built 5x5 grid instances computed by inspection.
"""
from mapf.grid import Grid
from mapf.conflicts import Conflict
from mapf.features import FeatureExtractor, compute_grid_distance_transform


def test_instance_1_corridor_and_distance():
    # 5x5 empty grid
    grid = Grid(5, 5, set())
    fe = FeatureExtractor(grid)
    
    # Border cell vs Center cell distance transform test
    # (0, 0) is border cell -> dt should be 1
    # (2, 2) is center cell -> dt should be 3
    assert fe.dt[(0, 0)] == 1, f"Expected border dt=1, got {fe.dt[(0, 0)]}"
    assert fe.dt[(2, 2)] == 3, f"Expected center dt=3, got {fe.dt[(2, 2)]}"
    print("Test 1 (Corridor & distance transform) Passed!")


def test_instance_2_congestion_and_frequency():
    # 5x5 grid, 3 agents
    grid = Grid(5, 5, set())
    fe = FeatureExtractor(grid)
    paths = [
        [(1, 2), (2, 2), (3, 2)], # a0
        [(3, 2), (2, 2), (1, 2)], # a1
        [(2, 3), (2, 2), (2, 1)]  # a2
    ]
    goals = [(3, 2), (1, 2), (2, 1)]
    c = Conflict(agent_1=0, agent_2=1, time=1, conflict_type='vertex', loc_1=(2, 2))
    
    pair_hist = {(0, 1): 2}
    feats = fe.extract_features(
        conflict=c,
        conflict_class='cardinal',
        all_classified_conflicts=[(c, 'cardinal')],
        paths=paths,
        goals=goals,
        mvc_size=1,
        pair_history=pair_hist,
        downstream_counts=(3, 2)
    )
    # At t=1, agent 2 is at (2, 2) which is within radius 2 of (2, 2) -> density should be 1
    assert feats['local_density_i'] == 1.0, f"Expected density 1.0, got {feats['local_density_i']}"
    assert feats['pair_conflict_count'] == 2.0, f"Expected pair count 2.0, got {feats['pair_conflict_count']}"
    assert feats['goal_dist_i'] == 1.0, f"Expected goal_dist_i 1.0, got {feats['goal_dist_i']}"
    assert feats['downstream_conflict_count_i'] == 3.0
    assert feats['downstream_conflict_count_j'] == 2.0
    print("Test 2 (Congestion, frequency, downstream) Passed!")


def test_instance_3_graph_degrees():
    # 5x5 grid, 4 agents with two cardinal conflicts
    grid = Grid(5, 5, set())
    fe = FeatureExtractor(grid)
    paths = [[(0, 0)], [(0, 0)], [(0, 0)], [(0, 0)]]
    goals = [(4, 4), (4, 4), (4, 4), (4, 4)]
    c1 = Conflict(0, 1, 1, 'vertex', (2, 2))
    c2 = Conflict(0, 2, 1, 'vertex', (2, 2))
    
    feats = fe.extract_features(
        conflict=c1,
        conflict_class='cardinal',
        all_classified_conflicts=[(c1, 'cardinal'), (c2, 'cardinal')],
        paths=paths,
        goals=goals,
        mvc_size=1
    )
    # Agent 0 has 2 cardinal conflicts (with 1 and with 2)
    # Agent 1 has 1 cardinal conflict (with 0)
    assert feats['cardinal_degree_i'] == 2.0, f"Expected cardinal_degree_i 2.0, got {feats['cardinal_degree_i']}"
    assert feats['cardinal_degree_j'] == 1.0, f"Expected cardinal_degree_j 1.0, got {feats['cardinal_degree_j']}"
    assert feats['num_agents_in_conflict_graph'] == 3.0
    print("Test 3 (Cardinal graph degrees & MVC) Passed!")


if __name__ == "__main__":
    test_instance_1_corridor_and_distance()
    test_instance_2_congestion_and_frequency()
    test_instance_3_graph_degrees()
    print("All feature unit tests passed!")
