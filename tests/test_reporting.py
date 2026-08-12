from __future__ import annotations

from datetime import datetime, timezone
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.inference import Predictor
from pea_pgnn.reporting import build_pdf_report, make_report_id


def extracted_text(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


@unittest.skipUnless((ROOT / "artifacts" / "deployment" / "manifest.json").is_file(), "deployment artifact not trained")
class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.predictor = Predictor(
            ROOT / "artifacts" / "deployment",
            device="cpu",
            formula_directory=Path(cls.temporary.name) / "formula_data",
        )
        cls.condition = {
            "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
            "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
            "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
            "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
            "query_age": 365,
        }
        cls.result = cls.predictor.predict_curve(cls.condition)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_report_id_is_reproducible_for_a_fixed_snapshot_time(self):
        instant = datetime(2026, 8, 12, 8, 30, tzinfo=timezone.utc)
        first = make_report_id(self.condition, self.result, instant)
        second = make_report_id(self.condition, self.result, instant)
        self.assertEqual(first, second)
        self.assertRegex(first, r"^PEA-20260812-[A-F0-9]{8}$")

    def test_pdf_report_is_written_atomically_from_current_result(self):
        destination = Path(self.temporary.name) / "report.pdf"
        record = build_pdf_report(
            destination,
            self.predictor,
            self.condition,
            self.result,
            metadata={
                "title": "Drying-Shrinkage Calculation Report",
                "project": "Automated report test",
                "prepared_by": "Test operator",
                "notes": "Generated from a frozen test condition.",
            },
            generated_at=datetime(2026, 8, 12, 16, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(record.path, destination.resolve())
        self.assertTrue(destination.is_file())
        self.assertGreater(destination.stat().st_size, 10_000)
        self.assertEqual(destination.read_bytes()[:5], b"%PDF-")
        self.assertEqual(len(record.sha256), 64)
        self.assertFalse(destination.with_name(destination.name + ".tmp").exists())
        text = extracted_text(destination)
        if text is not None:
            self.assertIn("Standard engineering report", text)
            self.assertIn("Input-range verification", text)
            self.assertNotIn("Internal calculation coefficients", text)
            self.assertNotIn("Network size", text)
            self.assertNotIn("SHA-256 prefix", text)

    def test_technical_report_adds_audit_appendix(self):
        destination = Path(self.temporary.name) / "technical_report.pdf"
        build_pdf_report(
            destination,
            self.predictor,
            self.condition,
            self.result,
            metadata={"project": "Technical report test", "prepared_by": "Test operator"},
            generated_at=datetime(2026, 8, 12, 16, 30, tzinfo=timezone.utc),
            report_mode="technical",
        )
        text = extracted_text(destination)
        if text is not None:
            self.assertIn("Complete technical report", text)
            self.assertIn("Appendix B.  Model audit record", text)
            self.assertIn("Internal calculation coefficients", text)
            self.assertIn("Network size", text)

    def test_unknown_report_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Report mode"):
            build_pdf_report(
                Path(self.temporary.name) / "bad_mode.pdf",
                self.predictor,
                self.condition,
                self.result,
                report_mode="dashboard",
            )


if __name__ == "__main__":
    unittest.main()
