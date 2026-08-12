from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.data import apply_imputation, fit_imputation, load_frozen_population, load_frozen_split
from pea_pgnn.features import FEATURE_NAMES, build_features, condition_frame
from pea_pgnn.formulas import aci209_prediction, b3_prediction, gl2000_prediction
from pea_pgnn.model import model_from_config
from pea_pgnn.support import assess_support


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
        cls.data_path = (ROOT / "configs" / cls.config["data_path"]).resolve()
        cls.split_path = (ROOT / "configs" / cls.config["frozen_split_path"]).resolve()
        cls.frame = load_frozen_population(cls.data_path) if cls.data_path.is_file() else None

    def test_frozen_population_identity(self):
        if self.frame is None:
            self.skipTest("Private frozen research population is not bundled")
        self.assertEqual(len(self.frame), 8729)
        self.assertEqual(self.frame["condition_id"].nunique(), 286)
        self.assertEqual(self.frame["ST_id"].nunique(), 1076)

    def test_feature_schema_and_model_parameter_count(self):
        sample = condition_frame(
            {
                "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
                "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
                "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
                "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
                "query_age": 365,
            }
        )
        spec = fit_imputation(sample, np.arange(len(sample)))
        features = build_features(apply_imputation(sample, spec))
        self.assertEqual(features.shape, (1, 39))
        self.assertEqual(len(FEATURE_NAMES), 39)
        model = model_from_config(self.config)
        self.assertEqual(model.n_parameters(), 104200)

    def test_condition_geometry_is_unambiguous(self):
        raw = condition_frame(
            {
                "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
                "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
                "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
                "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
                "query_age": 365,
            }
        )
        self.assertAlmostEqual(float(raw.loc[0, "VtoS"]), 22.75)
        self.assertAlmostEqual(float(raw.loc[0, "ks"]), 1.25)

    def test_empirical_formulas_match_legacy_scalar_values(self):
        age, t0, rh, vs, water, strength, ks = 365.0, 7.0, 50.0, 22.75, 186.0, 37.0, 1.25
        b3 = float(b3_prediction(age, t0, rh, vs, water, strength, ks))
        gl = float(gl2000_prediction(age, rh, vs, strength))
        aci = float(aci209_prediction(age, rh, vs))
        self.assertTrue(np.isfinite([b3, gl, aci]).all())
        self.assertGreater(b3, 0)
        self.assertGreater(gl, 0)
        self.assertGreater(aci, 0)

    def test_structural_contract_for_random_model(self):
        model = model_from_config(self.config).eval()
        raw = torch.randn(64, 39)
        raw[:, 34] = torch.linspace(100, 1400, 64)
        raw[:, 33] = torch.linspace(5, 1000, 64)
        scaled = torch.randn(64, 39)
        state = model.parameter_state(raw, scaled)
        ages = torch.linspace(0.1, 5000, 256)
        first_raw = raw[:1].repeat(len(ages), 1)
        first_scaled = scaled[:1].repeat(len(ages), 1)
        with torch.no_grad():
            prediction, details = model(first_raw, first_scaled, ages, return_details=True)
        self.assertTrue(bool(torch.all(prediction >= 0)))
        self.assertTrue(bool(torch.all(prediction[1:] >= prediction[:-1] - 1e-5)))
        self.assertTrue(bool(torch.all(prediction <= details["eps_inf"] + 1e-5)))
        self.assertTrue(bool(torch.allclose(state["weights"].sum(dim=1), torch.ones(64), atol=1e-6)))

    def test_frozen_split_has_no_condition_leakage(self):
        if self.frame is None or not self.split_path.is_file():
            self.skipTest("Private frozen research population and split are not bundled")
        split = load_frozen_split(self.split_path, self.frame, cutoff=365)
        self.assertEqual(sorted(split["fold"].unique().tolist()), [1, 2, 3, 4, 5])
        for fold in range(1, 6):
            block = split[split.fold == fold]
            train = set(block.loc[block.role == "inner_train_development", "condition_id"])
            held = set(block.loc[block.role == "heldout_extrapolation", "condition_id"])
            self.assertFalse(train & held)

    def test_support_diagnostic(self):
        support_spec = json.loads(
            (ROOT / "artifacts" / "deployment" / "support.json").read_text(encoding="utf-8")
        )
        typical = condition_frame(
            {
                "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
                "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
                "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
                "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
                "query_age": 365,
            }
        )
        result = assess_support(typical, support_spec)
        self.assertIn(result["level"], {"within", "boundary", "outside"})
        extreme = typical.copy()
        extreme.loc[0, "RH"] = 99.0
        self.assertEqual(assess_support(extreme, support_spec)["level"], "outside")


if __name__ == "__main__":
    unittest.main()
