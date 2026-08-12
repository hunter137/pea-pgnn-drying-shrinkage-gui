"""Deterministic PDF calculation reports for PEA-PGNN predictions.

The report exporter deliberately uses the already-computed prediction result.
It does not call a language model, invent engineering conclusions, or mutate
the model/formula registry.  ReportLab is imported lazily so the prediction
workbench can still open and explain a missing optional PDF dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


REPORT_BLUE = "#35546E"
REPORT_BLUE_DARK = "#263B4D"
REPORT_BORDER = "#89939C"
REPORT_HEADER = "#E4E7E9"
REPORT_TEXT = "#151B20"
REPORT_MUTED = "#48525B"
MICROSTRAIN = "\u00b5\u03b5"
REPORT_MODE_STANDARD = "standard"
REPORT_MODE_TECHNICAL = "technical"
REPORT_MODES = (REPORT_MODE_STANDARD, REPORT_MODE_TECHNICAL)


class ReportDependencyError(RuntimeError):
    """Raised when the PDF runtime is unavailable."""


class ReportDataError(ValueError):
    """Raised when a stale or incomplete prediction is sent to the report."""


@dataclass(frozen=True)
class ReportBuildResult:
    path: Path
    report_id: str
    created_at: str
    sha256: str


def _reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Image,
            KeepTogether,
            LongTable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ReportDependencyError(
            "PDF export requires ReportLab. Install the project requirements "
            "with: python -m pip install -r requirements.txt"
        ) from exc
    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "TA_LEFT": TA_LEFT,
        "TA_RIGHT": TA_RIGHT,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "mm": mm,
        "pdfmetrics": pdfmetrics,
        "TTFont": TTFont,
        "Image": Image,
        "KeepTogether": KeepTogether,
        "LongTable": LongTable,
        "PageBreak": PageBreak,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _plain(value):
    """Normalise text for a PDF and avoid problematic Unicode dash glyphs."""
    text = str(value if value is not None else "")
    for character in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"):
        text = text.replace(character, "-")
    text = text.replace("\u00a0", " ")
    return "".join(character for character in text if character in "\n\t" or ord(character) >= 32)


def _xml(value):
    return escape(_plain(value)).replace("\n", "<br/>")


def _short(value, maximum):
    text = _plain(value).strip()
    return text if len(text) <= maximum else text[: maximum - 3].rstrip() + "..."


def _number(value, digits=1):
    return ("{:.%df}" % int(digits)).format(float(value))


def _query_index(condition, result):
    ages = np.asarray(result.get("ages", []), dtype=float).reshape(-1)
    prediction = np.asarray(result.get("prediction", []), dtype=float).reshape(-1)
    if ages.size == 0 or ages.size != prediction.size:
        raise ReportDataError("Prediction ages and values are missing or inconsistent")
    query_age = float(condition["query_age"])
    matches = np.flatnonzero(np.isclose(ages, query_age, atol=1.0e-6))
    if not len(matches):
        raise ReportDataError("The current result does not contain the requested query age")
    if not np.all(np.isfinite(ages)) or not np.all(np.isfinite(prediction)):
        raise ReportDataError("Prediction ages and values must be finite")
    return int(matches[0])


def make_report_id(condition, result, generated_at=None):
    """Create a compact traceable ID from the calculation snapshot."""
    generated_at = generated_at or datetime.now().astimezone()
    query_index = _query_index(condition, result)
    snapshot = {
        "condition": {key: condition[key] for key in sorted(condition)},
        "model_label": result.get("model_label", ""),
        "prediction": round(float(result["prediction"][query_index]), 8),
        "generated_at": generated_at.isoformat(),
    }
    digest = sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8].upper()
    return "PEA-{}-{}".format(generated_at.strftime("%Y%m%d"), digest)


def _register_fonts(pdf):
    pdfmetrics = pdf["pdfmetrics"]
    TTFont = pdf["TTFont"]
    regular_name = "PEAReportSans"
    bold_name = "PEAReportSansBold"
    if regular_name in pdfmetrics.getRegisteredFontNames():
        return regular_name, bold_name

    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        (windows_fonts / "msyh.ttc", windows_fonts / "msyhbd.ttc", 0),
        (windows_fonts / "arial.ttf", windows_fonts / "arialbd.ttf", 0),
        (
            Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf",
            Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf",
            0,
        ),
    )
    for regular, bold, subfont in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont(regular_name, str(regular), subfontIndex=subfont))
            pdfmetrics.registerFont(TTFont(bold_name, str(bold), subfontIndex=subfont))
            return regular_name, bold_name
    raise ReportDependencyError("No usable TrueType font was found for the PDF report")


def _curve_image(result, condition, formula_definitions):
    ages = np.asarray(result["ages"], dtype=float)
    prediction = np.asarray(result["prediction"], dtype=float)
    query_age = float(condition["query_age"])
    query_index = _query_index(condition, result)
    definition_map = {item["name"]: item for item in formula_definitions}
    style_map = {"-": "-", "--": "--", "-.": "-.", ":": ":", "long-dash": (0, (4, 2))}

    figure = Figure(figsize=(7.25, 3.75), dpi=190, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.plot(ages, prediction, color=REPORT_BLUE, linewidth=2.4, label="PEA-PGNN")
    for name, values in result.get("references", {}).items():
        definition = definition_map.get(name, {})
        axes.plot(
            ages,
            np.asarray(values, dtype=float),
            color=definition.get("color", "#7A5C3E"),
            linestyle=style_map.get(definition.get("line_style", "--"), "--"),
            linewidth=1.35,
            label=name,
        )
    axes.axvline(query_age, color="#98A2B3", linestyle=":", linewidth=1.0)
    axes.scatter([query_age], [prediction[query_index]], s=38, color=REPORT_BLUE, zorder=5)
    axes.annotate(
        "{:.1f} {}".format(prediction[query_index], MICROSTRAIN),
        (query_age, prediction[query_index]),
        xytext=(7, 7),
        textcoords="offset points",
            fontsize=9.0,
        color=REPORT_BLUE_DARK,
        fontweight="bold",
    )
    axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=10.0)
    axes.set_ylabel(r"Drying-shrinkage magnitude, $\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=10.0)
    axes.grid(True, linestyle=":", linewidth=0.55, color="#BEC4CB", alpha=0.8)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.tick_params(labelsize=9.0)
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    legend = axes.legend(loc="lower right", fontsize=8.0, frameon=True, ncol=2)
    legend.get_frame().set_edgecolor(REPORT_BORDER)
    legend.get_frame().set_linewidth(0.6)
    figure.tight_layout(pad=0.8)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=190, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    return buffer


def _formula_image(latex):
    notation = _plain(latex).strip()
    if not notation:
        return None
    notation = notation.strip("$")
    try:
        figure = Figure(figsize=(7.0, 0.52), dpi=190, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.axis("off")
        axes.text(0.01, 0.5, "$" + notation + "$", ha="left", va="center", fontsize=13.0, color=REPORT_TEXT)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=190, bbox_inches="tight", pad_inches=0.04, facecolor="white")
        buffer.seek(0)
        return buffer
    except Exception:
        return None


def _key_ages(condition, result):
    ages = np.asarray(result["ages"], dtype=float)
    requested = [7.0, 14.0, 28.0, 56.0, 90.0, 180.0, 365.0, float(condition["query_age"])]
    return sorted({age for age in requested if np.any(np.isclose(ages, age, atol=1.0e-6))})


def _normalise_metadata(metadata):
    source = dict(metadata or {})
    return {
        "title": _short(source.get("title") or "Drying-Shrinkage Calculation Report", 120),
        "project": _short(source.get("project") or "Unspecified project", 160),
        "report_id": _short(source.get("report_id") or "", 80),
        "prepared_by": _short(source.get("prepared_by") or "", 100),
        "notes": _short(source.get("notes") or "No project-specific notes were entered.", 1200),
    }


def _normalise_report_mode(report_mode):
    mode = str(report_mode or REPORT_MODE_STANDARD).strip().lower()
    if mode not in REPORT_MODES:
        raise ValueError("Report mode must be 'standard' or 'technical'")
    return mode


def build_pdf_report(
    destination,
    predictor,
    condition,
    result,
    metadata=None,
    generated_at=None,
    report_mode=REPORT_MODE_STANDARD,
):
    """Write a complete, atomic PDF report from an existing prediction.

    ``predictor`` supplies the frozen manifest and formula registry. ``result``
    must come from the current condition; the exact query age is checked before
    any output file is replaced.
    """
    pdf = _reportlab()
    regular_font, bold_font = _register_fonts(pdf)
    colors = pdf["colors"]
    Paragraph = pdf["Paragraph"]
    ParagraphStyle = pdf["ParagraphStyle"]
    Table = pdf["Table"]
    LongTable = pdf["LongTable"]
    TableStyle = pdf["TableStyle"]
    Spacer = pdf["Spacer"]
    PageBreak = pdf["PageBreak"]
    Image = pdf["Image"]
    KeepTogether = pdf["KeepTogether"]
    mm = pdf["mm"]

    condition = dict(condition)
    query_index = _query_index(condition, result)
    report_mode = _normalise_report_mode(report_mode)
    technical = report_mode == REPORT_MODE_TECHNICAL
    generated_at = generated_at or datetime.now().astimezone()
    meta = _normalise_metadata(metadata)
    if not meta["report_id"]:
        meta["report_id"] = make_report_id(condition, result, generated_at)
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != ".pdf":
        destination = destination.with_suffix(".pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    page_width, page_height = pdf["A4"]
    doc = pdf["SimpleDocTemplate"](
        str(temporary),
        pagesize=pdf["A4"],
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=24 * mm,
        bottomMargin=19 * mm,
        title=meta["title"],
        author=meta["prepared_by"] or "PEA-PGNN Research Software V1.0.0",
        subject="PEA-PGNN drying-shrinkage calculation report",
        creator="PEA-PGNN Research Software V1.0.0",
    )
    content_width = doc.width

    body = ParagraphStyle(
        "ReportBody", fontName=regular_font, fontSize=9.5, leading=14.0,
        textColor=colors.HexColor(REPORT_TEXT), spaceAfter=5,
    )
    small = ParagraphStyle(
        "ReportSmall", parent=body, fontSize=8.3, leading=11.3,
        textColor=colors.HexColor(REPORT_MUTED),
    )
    title_style = ParagraphStyle(
        "ReportTitle", parent=body, fontName=bold_font, fontSize=16, leading=20,
        textColor=colors.HexColor(REPORT_TEXT), alignment=pdf["TA_LEFT"], spaceAfter=3,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle", parent=body, fontSize=9.5, leading=13,
        textColor=colors.HexColor(REPORT_MUTED), spaceAfter=12,
    )
    section = ParagraphStyle(
        "ReportSection", parent=body, fontName=bold_font, fontSize=12.0, leading=16,
        textColor=colors.HexColor(REPORT_BLUE_DARK), spaceBefore=4, spaceAfter=7,
    )
    subsection = ParagraphStyle(
        "ReportSubsection", parent=body, fontName=bold_font, fontSize=10.2, leading=14,
        textColor=colors.HexColor(REPORT_TEXT), spaceBefore=7, spaceAfter=5,
    )
    table_head = ParagraphStyle(
        "ReportTableHead", parent=small, fontName=bold_font, fontSize=8.7, leading=11.5,
        textColor=colors.HexColor(REPORT_TEXT),
        alignment=pdf["TA_LEFT"],
    )
    table_cell = ParagraphStyle(
        "ReportTableCell", parent=small, fontSize=8.7, leading=11.8,
        textColor=colors.HexColor(REPORT_TEXT),
    )
    table_cell_right = ParagraphStyle("ReportTableCellRight", parent=table_cell, alignment=pdf["TA_RIGHT"])
    formula_name = ParagraphStyle(
        "ReportFormulaName", parent=body, fontName=bold_font, fontSize=10.0,
        textColor=colors.HexColor(REPORT_BLUE_DARK),
    )

    def p(text, style=body):
        return Paragraph(_xml(text), style)

    def rich(text, style=body):
        return Paragraph(_plain(text), style)

    def table(data, widths, header=True, right_columns=()):
        rows = []
        for row_index, row in enumerate(data):
            cells = []
            for column_index, value in enumerate(row):
                if hasattr(value, "wrap"):
                    cells.append(value)
                elif row_index == 0 and header:
                    cells.append(p(value, table_head))
                elif column_index in right_columns:
                    cells.append(p(value, table_cell_right))
                else:
                    cells.append(p(value, table_cell))
            rows.append(cells)
        result_table = Table(rows, colWidths=widths, repeatRows=(1 if header else 0), hAlign="LEFT")
        commands = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(REPORT_BORDER)),
        ]
        if header:
            commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(REPORT_HEADER)))
        for row_index in range(1 if header else 0, len(rows)):
            if row_index % 2 == 0:
                commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F8F8F8")))
        result_table.setStyle(TableStyle(commands))
        return result_table

    def section_heading(number, heading):
        return rich("{}.&nbsp;&nbsp;{}".format(number, _xml(heading)), section)

    def on_page(canvas, document):
        canvas.saveState()
        canvas.setTitle(meta["title"])
        canvas.setAuthor(meta["prepared_by"] or "PEA-PGNN Research Software V1.0.0")
        canvas.setSubject("PEA-PGNN drying-shrinkage calculation report")
        canvas.setCreator("PEA-PGNN Research Software V1.0.0")
        canvas.setStrokeColor(colors.HexColor(REPORT_BLUE_DARK))
        canvas.setLineWidth(0.65)
        canvas.line(doc.leftMargin, page_height - 16 * mm, page_width - doc.rightMargin, page_height - 16 * mm)
        canvas.setFont(bold_font, 8.3)
        canvas.setFillColor(colors.HexColor(REPORT_BLUE_DARK))
        header_label = "PEA-PGNN V1.0.0  |  {}".format(
            "TECHNICAL CALCULATION REPORT" if technical else "CALCULATION REPORT"
        )
        canvas.drawString(doc.leftMargin, page_height - 13.2 * mm, header_label)
        canvas.setFont(regular_font, 8.0)
        canvas.setFillColor(colors.HexColor(REPORT_MUTED))
        canvas.drawRightString(page_width - doc.rightMargin, page_height - 13.2 * mm, _short(meta["report_id"], 42))
        canvas.setStrokeColor(colors.HexColor(REPORT_BORDER))
        canvas.setLineWidth(0.45)
        canvas.line(doc.leftMargin, 13.5 * mm, page_width - doc.rightMargin, 13.5 * mm)
        canvas.setFont(regular_font, 7.2)
        canvas.setFillColor(colors.HexColor(REPORT_MUTED))
        canvas.drawString(doc.leftMargin, 9.4 * mm, "Research use only")
        canvas.drawRightString(page_width - doc.rightMargin, 9.4 * mm, "Page {}".format(document.page))
        canvas.restoreState()

    prediction = float(result["prediction"][query_index])
    seed_sd = float(result["optimization_sd"][query_index])
    support = result["support"]
    support_label = {
        "within": "Within recorded input range",
        "boundary": "Near recorded input-range limit",
        "outside": "Outside recorded input range",
    }.get(support.get("level"), support.get("label", "Input range not classified"))
    formulas = predictor.formula_definitions()
    manifest = predictor.manifest
    offset = generated_at.strftime("%z")
    offset = (offset[:3] + ":" + offset[3:]) if len(offset) == 5 else offset
    created_text = generated_at.strftime("%Y-%m-%d %H:%M:%S") + (" UTC" + offset if offset else "")
    buffers = []
    story = []

    # Page 1 - document identification and input data.
    story.append(rich(_xml(meta["title"]), title_style))
    story.append(p("PEA-PGNN drying-shrinkage prediction | Software version 1.0.0", subtitle))
    identity_data = [
        ("Project", meta["project"], "Report ID", meta["report_id"]),
        ("Prepared by", meta["prepared_by"] or "Not specified", "Generated", created_text),
        ("Calculation method", "PEA-PGNN V1.0.0", "Document status", "Calculation record"),
    ]
    identity = table(identity_data, [72, 172, 72, content_width - 316], header=False)
    identity.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(REPORT_HEADER)),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor(REPORT_HEADER)),
        ("FONTNAME", (0, 0), (0, -1), bold_font),
        ("FONTNAME", (2, 0), (2, -1), bold_font),
    ]))
    story.extend((identity, Spacer(1, 10), section_heading(1, "Input data")))

    left_inputs = [
        (rich("Cement content, <i>C</i>", table_cell), condition["cement"], "kg/m\u00b3"),
        (rich("Water content, <i>W</i>", table_cell), condition["water"], "kg/m\u00b3"),
        (rich("Aggregate content, <i>A</i>", table_cell), condition["aggregate"], "kg/m\u00b3"),
        (rich("Water-binder ratio, <i>w/b</i>", table_cell), condition["wb"], "-"),
        (rich("28-d strength, <i>f</i><sub>c,28</sub>", table_cell), condition["fc28"], "MPa"),
        (rich("28-d modulus, <i>E</i><sub>c,28</sub>", table_cell), condition["Ec28"], "MPa"),
        ("Cement type code", condition["cement_type_code"], "-"),
        ("Aggregate type code", condition["agg_type_code"], "-"),
    ]
    right_inputs = [
        (rich("Initial curing age, <i>t</i><sub>0</sub>", table_cell), condition["t0"], "d"),
        (rich("Relative humidity, <i>RH</i>", table_cell), condition["RH"], "%"),
        (rich("Temperature, <i>T</i>", table_cell), condition["T"], "deg C"),
        ("Curing type code", condition["curing_type_code"], "-"),
        (rich("Theoretical thickness, <i>h</i><sub>0</sub>", table_cell), condition["h0"], "mm"),
        (rich("Derived <i>V/S</i>", table_cell), float(condition["h0"]) / 2.0, "mm"),
        ("Geometry", condition["geometry"], "-"),
        (rich("Query drying age, <i>t</i>", table_cell), condition["query_age"], "d"),
    ]
    input_rows = [("Parameter", "Value", "Unit", "Parameter", "Value", "Unit")]
    for left, right in zip(left_inputs, right_inputs):
        input_rows.append((left[0], "{:g}".format(float(left[1])) if isinstance(left[1], (int, float)) else left[1], left[2], right[0], "{:g}".format(float(right[1])) if isinstance(right[1], (int, float)) else right[1], right[2]))
    story.append(table(input_rows, [126, 60, 60, 126, 60, content_width - 432], right_columns=(1, 4)))
    story.append(p("Project notes", subsection))
    notes_box = Table([[p(meta["notes"], table_cell)]], colWidths=[content_width])
    notes_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor(REPORT_BORDER)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(notes_box)

    story.extend((Spacer(1, 10), section_heading(2, "Calculation result")))
    result_rows = [
        ("Item", "Result", "Remarks"),
        (rich("Drying-shrinkage magnitude, <i>\u03b5</i><sub>sh</sub>({:g} d)".format(float(condition["query_age"])), table_cell), "{:.1f} {}".format(prediction, MICROSTRAIN), "Calculated value"),
        ("Input-range check", support_label, "See Section 4"),
    ]
    if technical:
        result_rows.insert(2, (
            "Ensemble-member SD",
            "{:.1f} {}".format(seed_sd, MICROSTRAIN),
            "Optimization-seed dispersion; not a prediction interval",
        ))
    story.append(table(result_rows, [205, 115, content_width - 320], right_columns=(1,)))
    story.append(p(
        "B3, GL2000 and ACI 209 are shown as reference calculations. Their equations are listed in Appendix A.",
        small,
    ))

    # Page 2 - curve and key ages.
    story.extend((PageBreak(), section_heading(3, "Prediction curve and key-age results")))
    curve_buffer = _curve_image(result, condition, formulas)
    buffers.append(curve_buffer)
    story.append(Image(curve_buffer, width=content_width, height=content_width * 0.50))
    story.append(p(
        "Figure 1 | Calculated PEA-PGNN curve and enabled reference equations. The vertical line marks the specified reporting age.",
        small,
    ))
    story.append(p("Key-age calculation table", subsection))
    reference_names = ("Model B3", "GL2000", "ACI 209")
    if technical:
        key_rows = [("Age (d)", "PEA-PGNN ({})".format(MICROSTRAIN), "Member SD ({})".format(MICROSTRAIN), "B3 ({})".format(MICROSTRAIN), "GL2000 ({})".format(MICROSTRAIN), "ACI 209 ({})".format(MICROSTRAIN))]
    else:
        key_rows = [("Age (d)", "PEA-PGNN ({})".format(MICROSTRAIN), "B3 ({})".format(MICROSTRAIN), "GL2000 ({})".format(MICROSTRAIN), "ACI 209 ({})".format(MICROSTRAIN))]
    ages_array = np.asarray(result["ages"], dtype=float)
    for age in _key_ages(condition, result):
        index = int(np.flatnonzero(np.isclose(ages_array, age, atol=1.0e-6))[0])
        row = [
            "{:g}".format(age),
            _number(result["prediction"][index]),
        ]
        if technical:
            row.append(_number(result["optimization_sd"][index]))
        row.extend((
            _number(result["references"][reference_names[0]][index]),
            _number(result["references"][reference_names[1]][index]),
            _number(result["references"][reference_names[2]][index]),
        ))
        key_rows.append(tuple(row))
    if technical:
        story.append(table(key_rows, [48, 90, 78, 74, 79, content_width - 369], right_columns=(0, 1, 2, 3, 4, 5)))
    else:
        story.append(table(key_rows, [58, 110, 98, 105, content_width - 371], right_columns=(0, 1, 2, 3, 4)))
    story.extend((PageBreak(), section_heading(4, "Reference calculations and input range")))
    story.append(p("Query-age reference comparison", subsection))
    difference_rows = [("Reference equation", "Reference result ({})".format(MICROSTRAIN), "Difference ({})".format(MICROSTRAIN))]
    for name, values in result.get("references", {}).items():
        reference_value = float(values[query_index])
        difference_rows.append((name, _number(reference_value), "{:+.1f}".format(prediction - reference_value)))
    story.append(table(difference_rows, [content_width * 0.42, content_width * 0.29, content_width * 0.29], right_columns=(1, 2)))
    story.append(p("Difference = PEA-PGNN result minus reference-equation result.", small))

    story.append(p("Input-range verification", subsection))
    detail_names = support.get("outside_variables") or support.get("boundary_variables") or []
    support_text = "{}{}".format(support_label, " | Variables: " + ", ".join(detail_names) if detail_names else "")
    support_note = {
        "within": "The entered values are within the recorded calculation range.",
        "boundary": "One or more entered values are close to a recorded range limit.",
        "outside": "One or more entered values are outside a recorded range; independent verification is required.",
    }.get(support.get("level"), support.get("note", ""))
    support_box = Table([[p(support_text, formula_name), p(support_note, small)]], colWidths=[content_width * 0.38, content_width * 0.62])
    support_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.55, colors.HexColor(REPORT_BORDER)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F3F3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend((support_box, Spacer(1, 5)))

    audit = manifest.get("strict_audit_evidence", {})
    story.append(p("Method verification data", subsection))
    validation_rows = [
        ("Item", "Recorded value", "Basis"),
        ("Verification records", "{:,}".format(int(audit.get("N", 0))), "Independent later-age verification subset"),
        ("Root-mean-square error", "{:.2f} {}".format(float(audit.get("RMSE", float("nan"))), MICROSTRAIN), "Recorded validation statistic"),
        ("Mean absolute error", "{:.2f} {}".format(float(audit.get("MAE", float("nan"))), MICROSTRAIN), "Recorded validation statistic"),
    ]
    story.append(table(validation_rows, [content_width * 0.34, content_width * 0.25, content_width * 0.41], right_columns=(1,)))

    story.append(p("Calculation constraints", subsection))
    story.append(rich(
        "&#8226; Calculated shrinkage is non-negative<br/>"
        "&#8226; Calculated shrinkage does not decrease with age<br/>"
        "&#8226; Calculated curve does not exceed the implemented upper bound<br/>"
        "The constraints define the implemented calculation response; they do not replace project-specific checks.",
        body,
    ))

    story.extend((Spacer(1, 10), section_heading(5, "Document control and sign-off")))
    record_rows = [
        ("Record item", "Value"),
        ("Report ID", meta["report_id"]),
        ("Generated", created_text),
        ("Software", "PEA-PGNN Research Software V1.0.0"),
        ("Report type", "Complete technical report" if technical else "Standard engineering report"),
        ("Calculation scope", "Research-use drying-shrinkage calculation"),
    ]
    story.append(table(record_rows, [135, content_width - 135]))
    story.append(Spacer(1, 8))
    disclaimer = Table([[p(
        "Limitation: this calculation report is not a design certificate. The result does not replace experimental assessment, "
        "applicable standard checks, project-specific verification, or independent engineering judgement.",
        table_cell,
    )]], colWidths=[content_width])
    disclaimer.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(REPORT_BORDER)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F3F3")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend((disclaimer, Spacer(1, 16)))
    signature = Table([
        (p("Prepared by: " + (meta["prepared_by"] or "________________________"), table_cell), p("Reviewed by: ________________________", table_cell)),
        (p("Date: ______________________________", table_cell), p("Date: ______________________________", table_cell)),
    ], colWidths=[content_width / 2.0, content_width / 2.0])
    signature.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(signature)

    # Keep the formula register as a conventional appendix. Standard reports
    # list only equations used in the calculation; technical reports also
    # preserve disabled user definitions for audit purposes.
    story.extend((PageBreak(), section_heading("Appendix A", "Formula register")))
    story.append(p(
        "The equations below are the reference calculations included in this report. Built-in definitions are read-only; user equations are stored in the separate user formula space."
    ))
    formula_blocks = []
    report_formulas = [definition for definition in formulas if technical or definition.get("enabled", True)]
    for position, definition in enumerate(report_formulas, start=1):
        status = "Enabled" if definition.get("enabled", True) else "Disabled"
        if definition.get("locked"):
            source_note = "Built-in reference equation - read-only"
        else:
            source_note = "User reference equation - editable"
        if technical:
            source_note += " - {}".format(status)
        parts = [rich("<b>A.{}. {}</b>".format(position, _xml(definition.get("name", "Unnamed"))), formula_name)]
        formula_buffer = _formula_image(definition.get("latex", ""))
        if formula_buffer is not None:
            buffers.append(formula_buffer)
            parts.append(Image(formula_buffer, width=min(content_width - 12, 455), height=40))
        else:
            parts.append(p("Calculation expression: " + definition.get("expression", "Not available"), table_cell))
        parts.append(p(source_note, small))
        if definition.get("description") and (technical or not definition.get("locked")):
            parts.append(p(_short(definition["description"], 600), small))
        if technical and not definition.get("locked"):
            parts.append(p("Restricted expression: " + _short(definition.get("expression", ""), 500), small))
        parts.append(Spacer(1, 9))
        formula_blocks.append(KeepTogether(parts))

    story.extend(formula_blocks)

    if technical:
        story.extend((PageBreak(), section_heading("Appendix B", "Model audit record")))
        story.append(p(
            "This appendix records implementation details for research review and reproducibility. It is not required for routine use of the standard engineering report.",
            body,
        ))
        story.append(p("B.1 Input-domain diagnostic", subsection))
        support_rows = [
            ("Diagnostic", "Value", "Reference"),
            ("Nearest-profile robust distance", _number(support.get("nearest_profile_distance", float("nan")), 3), "Combined-input proximity"),
            ("Recorded 95th-percentile distance", _number(support.get("distance_q95", float("nan")), 3), "Recorded threshold"),
            ("Boundary variables", ", ".join(support.get("boundary_variables", [])) or "None", "Individual recorded limits"),
            ("Outside-range variables", ", ".join(support.get("outside_variables", [])) or "None", "Individual recorded limits"),
        ]
        story.append(table(support_rows, [165, 120, content_width - 285]))

        story.append(p("B.2 Operational calculation quantities", subsection))
        quantity_rows = [("Quantity", "Value", "Definition")]
        for label, key, unit, meaning in (
            (rich("<i>\u03b5</i><sub>anchor</sub>", table_cell), "eps_anchor", MICROSTRAIN, "Reference-equation magnitude anchor"),
            (rich("<i>\u03b5</i><sub>inf</sub>", table_cell), "eps_inf", MICROSTRAIN, "Corrected upper magnitude"),
            (rich("<i>\u03c4</i><sub>anchor</sub>", table_cell), "tau_anchor", "d", "Reference-equation time anchor"),
            (rich("<i>\u03c4</i>", table_cell), "tau", "d", "Corrected characteristic time"),
        ):
            quantity_rows.append((label, "{} {}".format(_number(result[key][query_index], 2), unit), meaning))
        story.append(table(quantity_rows, [105, 105, content_width - 210], right_columns=(1,)))

        story.append(p("B.3 Internal calculation coefficients", subsection))
        weights = np.asarray(result["weights"][query_index], dtype=float)
        allocation_names = ("B3-type", "ACI-type", "GL-type", "Bounded logarithmic")
        allocation_rows = [("Component", "Coefficient")]
        for name, weight in zip(allocation_names, weights):
            allocation_rows.append((name, "{:.1%}".format(float(weight))))
        story.append(table(allocation_rows, [content_width * 0.68, content_width * 0.32], right_columns=(1,)))
        story.append(p("These coefficients are internal calculation values, not probabilities of physical mechanisms.", small))

        story.append(p("B.4 Model build record", subsection))
        provenance_rows = [
            ("Record", "Value", "Qualification"),
            ("Model version", manifest.get("model_version", "1.0.0"), result.get("model_label", "")),
            ("Feature definition", "{} variables".format(len(manifest.get("feature_schema", {}).get("names", []))), manifest.get("feature_schema", {}).get("version", "")),
            ("Network size", "{:,} parameters/member".format(int(manifest.get("parameter_count_per_member", 0))), "{} members".format(len(manifest.get("ensemble_seeds", [])))),
            ("Deployment fit", "All development records", "Not an independent validation set"),
            ("Validation record", "N={} | RMSE={:.2f} {} | MAE={:.2f} {}".format(int(audit.get("N", 0)), float(audit.get("RMSE", float("nan"))), MICROSTRAIN, float(audit.get("MAE", float("nan"))), MICROSTRAIN), "Independent later-age verification subset"),
            ("Data identity", _short(manifest.get("data", {}).get("sha256", ""), 16), "SHA-256 prefix; {} records".format(manifest.get("data", {}).get("records", ""))),
        ]
        story.append(table(provenance_rows, [106, 195, content_width - 301]))

    try:
        if temporary.exists():
            temporary.unlink()
        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
        os.replace(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    digest = sha256(destination.read_bytes()).hexdigest()
    return ReportBuildResult(
        path=destination,
        report_id=meta["report_id"],
        created_at=generated_at.isoformat(),
        sha256=digest,
    )
