"""Regression tests for live full-frontier counterfactual rollouts."""
import unittest

from mapf.baselines import f_aware_residual_selector
from mapf.grid import GridMap
from mapf.icbs_h import ICBShSolver
from mapf.rollout_oracle import capture_full_search_snapshot, evaluate_conflict_with_rollout


def instance():
    passable = [[True] * 5 for _ in range(5)]
    starts = [(2, 0), (2, 4), (0, 2), (4, 2)]
    goals = [(2, 4), (2, 0), (4, 2), (0, 2)]
    return GridMap(5, 5, passable), starts, goals


def captured():
    grid, starts, goals = instance()
    snapshots = []
    def capture(node, classified, graph, candidates, state):
        if not snapshots:
            snap = capture_full_search_snapshot(node, classified, graph, candidates, state)
            snap.grid = grid
            snapshots.append(snap)
    normal = ICBShSolver(grid, conflict_selector=f_aware_residual_selector).solve(
        starts, goals, timeout=10, on_decision_callback=capture)
    assert snapshots and len(snapshots[0].candidates) >= 2
    return grid, starts, goals, snapshots[0], normal


def stable(result):
    result = dict(result)
    result.pop("solution_paths", None)
    result.pop("trajectory", None)
    checkpoints = []
    for row in result["checkpoints"]:
        row = dict(row)
        row.pop("wall_seconds", None)
        row.pop("cpu_seconds", None)
        checkpoints.append(row)
    result["checkpoints"] = checkpoints
    return result


class RolloutOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid, cls.starts, cls.goals, cls.snapshot, cls.normal = captured()

    def run_candidate(self, index, budgets=(20, 100, 250), **kwargs):
        return evaluate_conflict_with_rollout(
            self.grid, self.snapshot, self.snapshot.candidates[index], budgets,
            timeout=10, **kwargs)

    def test_first_conflict_intervention_only(self):
        result = self.run_candidate(1)
        self.assertTrue(result["chosen_conflicts"][0]["forced"])
        self.assertEqual(
            result["chosen_conflicts"][0]["conflict"],
            tuple([
                self.snapshot.candidates[1].conflict.agent_i,
                self.snapshot.candidates[1].conflict.agent_j,
                self.snapshot.candidates[1].conflict.time,
                self.snapshot.candidates[1].conflict.conflict_type,
                self.snapshot.candidates[1].conflict.loc1,
                self.snapshot.candidates[1].conflict.loc2,
            ]),
        )
        self.assertFalse(any(x["forced"] for x in result["chosen_conflicts"][1:]))

    def test_determinism(self):
        self.assertEqual(stable(self.run_candidate(0)), stable(self.run_candidate(0)))

    def test_candidate_isolation(self):
        before = stable(self.run_candidate(1))
        self.run_candidate(0)
        self.assertEqual(before, stable(self.run_candidate(1)))

    def test_prefix_budget_consistency(self):
        short = self.run_candidate(0, (100,))
        long = self.run_candidate(0, (100, 250))
        prefix = [x for x in long["chosen_conflicts"] if x["expansion"] <= 100]
        self.assertEqual(short["chosen_conflicts"], prefix)
        self.assertEqual(stable(short)["checkpoints"][0], stable(long)["checkpoints"][0])

    def test_residual_equivalence(self):
        node = self.snapshot.current
        graph = node.g_cf_cache
        context = dict(
            grid=self.grid, paths=node.paths, starts=self.starts, goals=self.goals,
            G_CF=graph, mvc_size=graph.exact_mvc_size(), effective_h=node.h,
            pair_counts=node.pair_counts, classified=self.snapshot.classified,
            constraints=node.flat_constraints, per_agent_constraints=node.constraints,
            h_tables=self.snapshot.h_tables)
        residual = f_aware_residual_selector(self.snapshot.candidates, context)
        index = next(i for i, c in enumerate(self.snapshot.candidates) if c is residual)
        forced = self.run_candidate(index)
        ordinary = self.run_candidate(index, force_first=False)
        a = stable(forced); b = stable(ordinary)
        for item in a["chosen_conflicts"]:
            item["forced"] = False
        self.assertEqual(a, b)

    def test_optimal_solution_preservation(self):
        costs = []
        for i in range(len(self.snapshot.candidates)):
            row = self.run_candidate(i, (500,))["checkpoints"][0]
            self.assertTrue(row["solved"])
            costs.append(row["solution_cost"])
        self.assertEqual(len(set(costs + [self.normal["cost"]])), 1)

    def test_censored_not_fake_completion(self):
        row = self.run_candidate(0, (1,))["checkpoints"][0]
        if not row["solved"]:
            self.assertTrue(row["censored"])
            self.assertIsNone(row["solution_cost"])


if __name__ == "__main__":
    unittest.main()
