"""Integrity checks for the fixed rollout-labeled Model C experiment."""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mapf.features import FEATURE_NAMES
from mapf.ml_revision_policy import SavedLinear


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rollout_oracle" / "model_c"
MODEL = ROOT / "models" / "model_c_rollout.json"


class ModelCPipelineTests(unittest.TestCase):
    def test_dataset_and_split_integrity(self):
        data = pd.read_csv(OUT / "rollout_feature_dataset.csv")
        self.assertEqual(data.query_id.nunique(), 60)
        self.assertEqual(len(data), 215)
        self.assertEqual(set(data.split), {"train", "validation"})
        self.assertTrue(np.isfinite(data[FEATURE_NAMES].to_numpy(float)).all())
        train = set(data.loc[data.split == "train", "family_id"])
        validation = set(data.loc[data.split == "validation", "family_id"])
        self.assertFalse(train & validation)
        self.assertFalse(data.family_id.str.contains("family-3[0-5]").any())

    def test_pair_semantics_and_equal_query_weight(self):
        data = pd.read_csv(OUT / "rollout_feature_dataset.csv")
        pairs = pd.read_csv(OUT / "rollout_pairs.csv")
        lookup = data.set_index(["query_id", "candidate_index"]).eventual_ct_expansions
        for row in pairs.itertuples():
            left = lookup.loc[(row.query_id, row.candidate_i)]
            right = lookup.loc[(row.query_id, row.candidate_j)]
            self.assertNotEqual(left, right)
            self.assertEqual(row.order, 1 if left < right else -1)
        sums = pairs.groupby("query_id").query_pair_weight.sum().to_numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums))

    def test_model_and_sealed_test_integrity(self):
        metadata = json.loads(Path(str(MODEL) + ".metadata.json").read_text())
        actual = hashlib.sha256(MODEL.read_bytes()).hexdigest()
        self.assertEqual(actual, metadata["model_sha256"])
        self.assertFalse(metadata["validation_used_for_fit"])
        self.assertFalse(metadata["sealed_test_used"])
        SavedLinear(MODEL)
        sealed = json.loads((OUT / "sealed_test_manifest.json").read_text())
        checksum = sealed.pop("manifest_sha256")
        encoded = json.dumps(sealed, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), checksum)
        self.assertTrue(sealed["generation_only_no_solver_execution"])


if __name__ == "__main__":
    unittest.main()
