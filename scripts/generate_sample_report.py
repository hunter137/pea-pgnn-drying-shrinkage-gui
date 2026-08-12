"""Generate the stable PDF sample used by visual release QA."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.inference import Predictor
from pea_pgnn.reporting import build_pdf_report


def main():
    formula_space = ROOT / "tmp" / "pdfs" / "sample_formula_space"
    predictor = Predictor(ROOT / "artifacts" / "deployment", device="cpu", formula_directory=formula_space)
    condition = {
        "cement": 371,
        "water": 186,
        "aggregate": 1859,
        "wb": 0.48,
        "fc28": 37,
        "Ec28": 25958,
        "cement_type_code": 2,
        "agg_type_code": 1,
        "curing_type_code": 1,
        "t0": 7,
        "RH": 50,
        "T": 23,
        "h0": 45.5,
        "geometry": "Prism",
        "query_age": 365,
    }
    result = predictor.predict_curve(condition)
    destination = ROOT / "output" / "pdf" / "PEA_PGNN_V1.0.0_Sample_Calculation_Report.pdf"
    record = build_pdf_report(
        destination,
        predictor,
        condition,
        result,
        metadata={
            "title": "Drying-Shrinkage Calculation Report",
            "project": "Report export verification sample",
            "report_id": "PEA-V100-SAMPLE-001",
            "prepared_by": "Sample Engineer",
            "notes": "Standard verification condition used to check the report layout and exported values.",
        },
        generated_at=datetime.now().astimezone(),
        report_mode="standard",
    )
    print(record.path)
    print(record.report_id)
    print(record.sha256)

    technical_destination = ROOT / "output" / "pdf" / "PEA_PGNN_V1.0.0_Sample_Technical_Report.pdf"
    technical_record = build_pdf_report(
        technical_destination,
        predictor,
        condition,
        result,
        metadata={
            "title": "Drying-Shrinkage Technical Calculation Report",
            "project": "Technical report verification sample",
            "report_id": "PEA-V100-SAMPLE-TECH-001",
            "prepared_by": "Sample Engineer",
            "notes": "Verification condition used to check the optional model-audit appendix.",
        },
        generated_at=datetime.now().astimezone(),
        report_mode="technical",
    )
    print(technical_record.path)
    print(technical_record.report_id)
    print(technical_record.sha256)


if __name__ == "__main__":
    main()
