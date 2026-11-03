"""Integrity checks for the committed family-transfer diagnostic."""
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rollout_oracle" / "model_c" / "family_transfer"


class ModelCFamilyTransferTests(unittest.TestCase):
    def test_family_counts_and_split_isolation(self):
        coverage = pd.read_csv(OUT / "family_data_coverage.csv")
        self.assertEqual(set(coverage.split), {"train", "validation"})
        self.assertEqual((coverage.split == "train").sum(), 19)
        self.assertEqual((coverage.split == "validation").sum(), 6)
        train = set(coverage.loc[coverage.split == "train", "family_id"])
        development = set(
            coverage.loc[coverage.split == "validation", "family_id"])
        self.assertTrue(train.isdisjoint(development))
        self.assertFalse(any("family-3" in family for family in train | development))

    def test_transfer_table_shapes(self):
        leave = pd.read_csv(OUT / "leave_one_training_family_out.csv")
        single = pd.read_csv(OUT / "single_family_transfer.csv")
        policy = pd.read_csv(OUT / "development_family_policy_metrics.csv")
        self.assertEqual(len(leave), 19 * 6)
        self.assertTrue((leave.groupby("omitted_training_family").size() == 6).all())
        self.assertEqual(len(single), 15 * 6)
        self.assertTrue((single.groupby("source_training_family").size() == 6).all())
        self.assertEqual(len(policy), 6 * 4)

    def test_summary_matches_original_full_model_and_keeps_test_sealed(self):
        summary = json.loads((OUT / "family_transfer_summary.json").read_text())
        policy = pd.read_csv(OUT / "development_family_policy_metrics.csv")
        model_c = policy[policy.policy == "model_c"]
        original = json.loads((OUT.parent / "model_c_metrics.json").read_text())
        expected = original["evaluations"]["validation"]["model_c"][
            "family_macro_pairwise"]
        self.assertAlmostEqual(model_c.family_macro_pairwise.mean(), expected)
        self.assertFalse(summary["sealed_test_used"])
        self.assertEqual(summary["development_families_where_model_c_beats_residual"], 0)


if __name__ == "__main__":
    unittest.main()
