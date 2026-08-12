from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.trial_calculation import analyse_formula_curve, build_trial_age_grid, parse_number_list


class TrialCalculationTests(unittest.TestCase):
    def test_number_list_accepts_chinese_and_mixed_separators(self):
        values = parse_number_list("7， 28; 90 365")
        np.testing.assert_allclose(values, [7, 28, 90, 365])

    def test_age_grid_contains_key_ages(self):
        grid = build_trial_age_grid(400, [7, 28, 90, 365])
        self.assertTrue(np.all(grid > 0))
        for age in (7, 28, 90, 365, 400):
            self.assertTrue(np.any(np.isclose(grid, age)))

    def test_health_check_reports_monotonic_curve(self):
        report = analyse_formula_curve([1, 7, 28, 90], [10, 30, 70, 100], zero_value=0)
        self.assertEqual(report["failure_count"], 0)
        self.assertEqual(report["monotonicity_violations"], 0)

    def test_health_check_warns_for_decrease_and_nonzero_start(self):
        report = analyse_formula_curve([1, 7, 28, 90], [10, 30, 20, 100], zero_value=12)
        self.assertEqual(report["monotonicity_violations"], 1)
        self.assertGreaterEqual(report["warning_count"], 2)


if __name__ == "__main__":
    unittest.main()
