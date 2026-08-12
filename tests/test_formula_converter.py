from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.formula_converter import FormulaConversionError, convert_formula, mathml_to_formula


class FormulaConverterTests(unittest.TestCase):
    def test_latex_formula_generates_safe_expression_parameters_and_notation(self):
        result = convert_formula(
            r"\varepsilon_{sh}(t)=\varepsilon_u[1-(RH/100)^n](t/(\tau+t))^m"
        )
        self.assertEqual(result["expression"], "eps_u*(1-(RH/100)**n)*(t/(tau+t))**m")
        self.assertEqual(set(result["parameters"]), {"eps_u", "n", "tau", "m"})
        self.assertIn(r"\varepsilon_{\mathrm{sh}}", result["latex"])
        ast.parse(result["expression"], mode="eval")

    def test_unicode_and_calculator_notation(self):
        result = convert_formula(
            "ε_sh(t) = eps_u × (1-(RH/100)³) × sqrt(t/(t+k×(V/S)²))"
        )
        self.assertIn("VtoS", result["expression"])
        self.assertEqual(set(result["parameters"]), {"eps_u", "k"})

    def test_latex_fraction_and_root(self):
        result = convert_formula(
            r"\varepsilon_{sh}(t)=\varepsilon_u\sqrt{\frac{t}{t+k(V/S)^2}}"
        )
        self.assertIn("sqrt", result["expression"])
        self.assertIn("VtoS", result["expression"])

    def test_implicit_multiplication_and_exponential(self):
        result = convert_formula(
            r"\varepsilon_{sh}(t)=1000[1-(RH/100)^3][1-e^{-t/55}]"
        )
        self.assertEqual(result["expression"], "1000*(1-(RH/100)**3)*(1-exp(-t/55))")
        self.assertEqual(result["parameters"], {})

    def test_rejects_code_and_ambiguous_equations(self):
        for source in (
            "eps=__import__('os').system('dir')",
            "eps=a=b+1",
            "eps=t.__class__",
            "eps=[x for x in t]",
        ):
            with self.assertRaises(FormulaConversionError):
                convert_formula(source)

    def test_mathtype_presentation_mathml_round_trip(self):
        mathml = """<?xml version='1.0'?>
        <math xmlns='http://www.w3.org/1998/Math/MathML'><mrow>
          <msub><mi>ε</mi><mrow><mi>s</mi><mi>h</mi></mrow></msub>
          <mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo>
          <msub><mi>ε</mi><mi>u</mi></msub>
          <msup><mrow><mo>(</mo><mfrac><mi>t</mi><mrow><mi>τ</mi><mo>+</mo><mi>t</mi></mrow></mfrac><mo>)</mo></mrow><mi>m</mi></msup>
        </mrow></math>"""
        notation = mathml_to_formula(mathml)
        result = convert_formula(notation)
        self.assertEqual(result["expression"], "eps_u*(t/(tau+t))**m")
        self.assertEqual(set(result["parameters"]), {"eps_u", "tau", "m"})


if __name__ == "__main__":
    unittest.main()
