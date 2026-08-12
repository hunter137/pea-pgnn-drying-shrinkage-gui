from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.features import condition_frame
from pea_pgnn.formula_registry import FormulaRegistry, FormulaValidationError


class FormulaRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = FormulaRegistry(self.root)
        self.frame = condition_frame(
            {
                "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
                "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
                "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
                "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
                "query_age": 365,
            }
        )

    def tearDown(self):
        self.temp.cleanup()

    def _package(self, expression="eps_u*(1-(RH/100)**3)*sqrt(t/(t+k*VtoS**2))"):
        return {
            "schema_version": 1,
            "id": "test_formula",
            "name": "Test formula",
            "expression": expression,
            "latex": r"\varepsilon_{\mathrm{sh}}(t)",
            "constants": {"eps_u": 1000.0, "k": 0.15},
            "enabled": True,
        }

    def test_import_evaluate_toggle_archive_and_restore(self):
        source = self.root / "source.peaf"
        source.write_text(json.dumps(self._package()), encoding="utf-8")
        self.registry.import_package(source)
        values = self.registry.evaluate(self.frame.loc[self.frame.index.repeat(3)].reset_index(drop=True), np.array([7.0, 28.0, 365.0]))
        self.assertIn("Test formula", values)
        self.assertTrue(np.all(np.isfinite(values["Test formula"])))
        self.assertTrue(np.all(np.diff(values["Test formula"]) > 0))
        self.registry.set_enabled("test_formula", False)
        values = self.registry.evaluate(self.frame.loc[self.frame.index.repeat(3)].reset_index(drop=True), np.array([7.0, 28.0, 365.0]))
        self.assertNotIn("Test formula", values)
        archived = self.registry.remove("test_formula")
        self.assertNotIn("test_formula", self.registry.custom)
        self.assertTrue(archived.is_file())
        restored = self.registry.restore_archived(archived)
        self.assertEqual(restored["id"], "test_formula")
        self.assertIn("test_formula", self.registry.custom)

    def test_arbitrary_python_and_attribute_access_are_rejected(self):
        for expression in ("__import__('os').system('dir')", "t.__class__", "open('x')"):
            source = self.root / "bad.peaf"
            source.write_text(json.dumps(self._package(expression)), encoding="utf-8")
            with self.assertRaises(FormulaValidationError):
                self.registry.import_package(source, overwrite=True)

    def test_native_formulas_are_locked_and_ec2_is_not_exposed(self):
        definitions = {item["id"]: item for item in self.registry.definitions()}
        self.assertTrue(definitions["b3"]["model_prior"])
        self.assertEqual(set(definitions), {"b3", "gl2000", "aci209"})
        self.assertTrue(all(item["locked"] for item in definitions.values()))
        with self.assertRaises(FormulaValidationError):
            self.registry.remove("b3")
        with self.assertRaises(FormulaValidationError):
            self.registry.set_enabled("b3", False)

    def test_bad_runtime_formula_does_not_break_core_prediction(self):
        source = self.root / "runtime_bad.peaf"
        source.write_text(json.dumps(self._package("1/(t-t)")), encoding="utf-8")
        self.registry.import_package(source)
        frame = self.frame.loc[self.frame.index.repeat(2)].reset_index(drop=True)
        values = self.registry.evaluate(frame, np.array([7.0, 28.0]))
        self.assertIn("Model B3", values)
        self.assertNotIn("Test formula", values)
        self.assertIn("test_formula", self.registry.runtime_errors)

    def test_gui_style_save_and_rename(self):
        document = self._package()
        saved = self.registry.save_package(document)
        self.assertEqual(saved["id"], "test_formula")
        document["id"] = "renamed_formula"
        document["name"] = "Renamed formula"
        renamed = self.registry.save_package(document, original_id="test_formula")
        self.assertEqual(renamed["id"], "renamed_formula")
        self.assertNotIn("test_formula", self.registry.custom)
        self.assertIn("renamed_formula", self.registry.custom)
        self.assertGreaterEqual(len(list(self.registry.history_directory.glob("test_formula_*.peaf"))), 2)

    def test_manual_active_file_loss_can_be_recovered_from_history(self):
        self.registry.save_package(self._package())
        active = Path(self.registry.custom["test_formula"]["path"])
        snapshot = next(self.registry.history_directory.glob("test_formula_*.peaf"))
        active.unlink()
        self.registry.reload()
        self.assertNotIn("test_formula", self.registry.custom)
        listings = [item for item in self.registry.backups() if item["id"] == "test_formula"]
        self.assertTrue(listings)
        self.assertTrue(all(item["restorable"] for item in listings))
        restored = self.registry.restore_archived(snapshot)
        self.assertEqual(restored["id"], "test_formula")
        self.assertTrue(snapshot.exists())

    def test_invalid_active_file_is_quarantined_without_harming_natives(self):
        invalid = self.registry.custom_directory / "broken.peaf"
        invalid.write_text("not valid json", encoding="utf-8")
        self.registry.reload()
        self.assertFalse(invalid.exists())
        self.assertEqual(len(list(self.registry.quarantine_directory.glob("broken_*.peaf"))), 1)
        definitions = {item["id"] for item in self.registry.definitions()}
        self.assertEqual(definitions, {"b3", "gl2000", "aci209"})

    def test_legacy_packages_are_copied_once_and_original_is_preserved(self):
        legacy = self.root / "legacy"
        legacy.mkdir()
        source = legacy / "legacy_formula.peaf"
        document = self._package()
        document["id"] = "legacy_formula"
        source.write_text(json.dumps(document), encoding="utf-8")
        destination_root = self.root / "new_user_space"
        migrated = FormulaRegistry(destination_root, legacy_directory=legacy)
        self.assertIn("legacy_formula", migrated.custom)
        self.assertTrue(source.is_file())
        self.assertTrue((destination_root / ".legacy_migration_v1_complete").is_file())

    def test_unsaved_formula_preview(self):
        frame = self.frame.loc[self.frame.index.repeat(4)].reset_index(drop=True)
        ages = np.array([7.0, 28.0, 90.0, 365.0])
        values = self.registry.preview(self._package(), frame, ages)
        self.assertEqual(values.shape, ages.shape)
        self.assertTrue(np.all(np.diff(values) > 0))


if __name__ == "__main__":
    unittest.main()
