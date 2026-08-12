from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.inference import Predictor


@unittest.skipUnless((ROOT / "artifacts" / "deployment" / "manifest.json").is_file(), "deployment artifact not trained")
class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(ROOT / "artifacts" / "deployment", device="cpu")
        cls.condition = {
            "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
            "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
            "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
            "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
            "query_age": 365,
        }

    def test_exact_key_ages_are_exactly_evaluated(self):
        result = self.predictor.predict(self.condition, [7, 14, 28, 56, 90, 180, 365])
        self.assertTrue(np.array_equal(result["ages"], np.array([7, 14, 28, 56, 90, 180, 365], dtype=np.float32)))
        self.assertTrue(np.isfinite(result["prediction"]).all())
        self.assertNotIn("EC2", result["references"])

    def test_curve_is_structurally_admissible(self):
        result = self.predictor.predict_curve(self.condition)
        prediction = result["prediction"]
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(np.diff(prediction) >= -1e-4))
        self.assertTrue(np.all(prediction <= result["eps_inf"] + 1e-4))

    def test_no_nominal_prediction_interval_claim(self):
        self.assertFalse(self.predictor.manifest["uncertainty"]["nominal_prediction_interval_available"])

    def test_unsaved_formula_preview_accepts_trial_condition_and_ages(self):
        document = {
            "schema_version": 1,
            "id": "preview_formula",
            "name": "Preview formula",
            "expression": "RH+t",
            "latex": r"\varepsilon_{\mathrm{sh}}=RH+t",
            "constants": {},
            "color": "#8B5E3C",
            "line_style": "--",
            "enabled": True,
        }
        condition = dict(self.condition)
        condition["RH"] = 70
        ages, values = self.predictor.preview_formula(document, condition, [1, 7, 28])
        np.testing.assert_allclose(ages, [1, 7, 28])
        np.testing.assert_allclose(values, [71, 77, 98])


if __name__ == "__main__":
    unittest.main()
