"""Deterministic engineering-formula to safe-expression conversion.

The converter accepts common LaTeX, Unicode and calculator-style notation. It
does not execute the source text and does not invoke a general Python parser
until the text has been reduced to the formula registry's restricted grammar.
"""

from __future__ import annotations

import ast
import html
import re
import xml.etree.ElementTree as ET

from .formula_registry import FUNCTIONS, VARIABLES, FormulaValidationError, _ExpressionValidator


class FormulaConversionError(ValueError):
    """Raised when mathematical notation cannot be converted unambiguously."""


GREEK = {
    "varepsilon": "eps",
    "epsilon": "eps",
    "tau": "tau",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "lambda": "lambda_value",
    "mu": "mu_value",
    "rho": "rho",
    "sigma": "sigma",
    "phi": "phi",
}

UNICODE_REPLACEMENTS = {
    "ε": "eps",
    "ϵ": "eps",
    "τ": "tau",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "λ": "lambda_value",
    "ρ": "rho",
    "σ": "sigma",
    "φ": "phi",
    "∞": "inf",
    "×": "*",
    "·": "*",
    "⋅": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "²": "**2",
    "³": "**3",
    "⁴": "**4",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
}

ALIASES = {
    "eps_sh": "eps_sh",
    "epssh": "eps_sh",
    "eps_inf": "eps_u",
    "eps_infty": "eps_u",
    "eps_u": "eps_u",
    "epsu": "eps_u",
    "t_0": "t0",
    "h_0": "h0",
    "f_c_28": "fc28",
    "f_c28": "fc28",
    "fc_28": "fc28",
    "e_c_28": "Ec28",
    "ec_28": "Ec28",
    "relative_humidity": "RH",
    "temperature": "T",
}

DEFAULT_PARAMETERS = {
    "eps_u": 1000.0,
    "tau": 55.0,
    "n": 3.0,
    "m": 0.5,
    "k": 0.15,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "delta": 1.0,
    "rho": 1.0,
    "sigma": 1.0,
    "phi": 1.0,
}


def _mathml_tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _mathml_text(element):
    tag = _mathml_tag(element)
    children = list(element)
    if tag in {"math", "semantics", "mrow", "mstyle", "mpadded", "mphantom"}:
        usable = [child for child in children if _mathml_tag(child) != "annotation"]
        return "".join(_mathml_text(child) for child in usable)
    if tag in {"mi", "mn", "mo", "mtext"}:
        value = "".join(element.itertext()).strip()
        if tag == "mi":
            value = {
                "ε": r"\varepsilon", "ϵ": r"\epsilon", "τ": r"\tau",
                "α": r"\alpha", "β": r"\beta", "γ": r"\gamma",
                "δ": r"\delta", "ρ": r"\rho", "σ": r"\sigma", "φ": r"\phi",
            }.get(value, value)
        return value
    if tag == "msub" and len(children) >= 2:
        return "{}_{{{}}}".format(_mathml_text(children[0]), _mathml_text(children[1]))
    if tag == "msup" and len(children) >= 2:
        return "({})^{{{}}}".format(_mathml_text(children[0]), _mathml_text(children[1]))
    if tag == "msubsup" and len(children) >= 3:
        return "{}_{{{}}}^{{{}}}".format(_mathml_text(children[0]), _mathml_text(children[1]), _mathml_text(children[2]))
    if tag == "mfrac" and len(children) >= 2:
        return r"\frac{{{}}}{{{}}}".format(_mathml_text(children[0]), _mathml_text(children[1]))
    if tag == "msqrt":
        return r"\sqrt{{{}}}".format("".join(_mathml_text(child) for child in children))
    if tag == "mroot" and len(children) >= 2:
        return r"\sqrt[{}]{{{}}}".format(_mathml_text(children[1]), _mathml_text(children[0]))
    if tag == "mfenced":
        opening = element.attrib.get("open", "(")
        closing = element.attrib.get("close", ")")
        separator = element.attrib.get("separators", ",")
        return opening + separator.join(_mathml_text(child) for child in children) + closing
    if tag == "annotation":
        return ""
    if children:
        return "".join(_mathml_text(child) for child in children)
    return "".join(element.itertext()).strip()


def mathml_to_formula(mathml):
    """Convert MathType's Presentation MathML clipboard output to notation."""
    text = str(mathml).replace("\x00", "")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        # Some MathType clipboard variants include comments before <math> or
        # trailing transport bytes; retain only the document element.
        start = text.find("<math")
        end = text.rfind("</math>")
        if start < 0 or end < 0:
            raise FormulaConversionError("MathType did not provide readable MathML: {}".format(exc))
        root = ET.fromstring(text[start:end + len("</math>")])
    notation = html.unescape(_mathml_text(root))
    if not notation.strip():
        raise FormulaConversionError("The copied MathType equation is empty")
    return notation


def _extract_group(text, start):
    if start >= len(text) or text[start] != "{":
        raise FormulaConversionError("Expected a {...} group near position {}".format(start + 1))
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise FormulaConversionError("An opening brace has no matching closing brace")


def _replace_named_groups(text, command):
    marker = "\\" + command
    while marker in text:
        position = text.find(marker)
        cursor = position + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        content, end = _extract_group(text, cursor)
        text = text[:position] + content + text[end:]
    return text


def _replace_fractions(text):
    marker = "\\frac"
    while marker in text:
        position = text.find(marker)
        cursor = position + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        numerator, after_numerator = _extract_group(text, cursor)
        cursor = after_numerator
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        denominator, end = _extract_group(text, cursor)
        replacement = "(({})/({}))".format(_latex_fragment(numerator), _latex_fragment(denominator))
        text = text[:position] + replacement + text[end:]
    return text


def _replace_roots(text):
    marker = "\\sqrt"
    while marker in text:
        position = text.find(marker)
        cursor = position + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor < len(text) and text[cursor] == "[":
            close = text.find("]", cursor + 1)
            if close < 0:
                raise FormulaConversionError("Root degree is missing a closing bracket")
            degree = text[cursor + 1:close].strip()
            cursor = close + 1
        else:
            degree = "2"
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        content, end = _extract_group(text, cursor)
        inner = _latex_fragment(content)
        replacement = "sqrt({})".format(inner) if degree == "2" else "(({})**(1/({})))".format(inner, degree)
        text = text[:position] + replacement + text[end:]
    return text


def _replace_subscripts(text):
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9]*)_\{([^{}]+)\}")
    while pattern.search(text):
        text = pattern.sub(lambda match: match.group(1) + "_" + re.sub(r"[^A-Za-z0-9]+", "_", match.group(2)).strip("_"), text)
    text = re.sub(r"([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9]+)", r"\1_\2", text)
    return text


def _replace_superscripts(text):
    pattern = re.compile(r"\^\{([^{}]+)\}")
    while pattern.search(text):
        text = pattern.sub(lambda match: "**({})".format(_latex_fragment(match.group(1))), text)
    return text.replace("^", "**")


def _latex_fragment(text):
    text = text.replace("\\left", "").replace("\\right", "")
    for spacing in ("\\,", "\\!", "\\;", "\\:", "\\quad", "\\qquad"):
        text = text.replace(spacing, " ")
    text = _replace_named_groups(text, "mathrm")
    text = _replace_named_groups(text, "operatorname")
    text = _replace_named_groups(text, "text")
    text = _replace_fractions(text)
    text = _replace_roots(text)
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    for command in ("tanh", "exp", "log", "log10", "ln", "abs", "min", "max"):
        mapped = {"ln": "log", "min": "minimum", "max": "maximum"}.get(command, command)
        text = re.sub(r"\\{}\b".format(command), mapped, text)
    for command, mapped in GREEK.items():
        text = re.sub(r"\\{}(?=[_{{(+\-*/\s]|$)".format(command), mapped, text)
    text = _replace_subscripts(text)
    text = _replace_superscripts(text)
    text = text.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    return text


def _normalise_notation(source):
    text = str(source).strip()
    if not text:
        raise FormulaConversionError("Enter a mathematical formula first")
    text = text.replace("$", "").replace("&", "")
    text = re.sub(r"\\begin\{[^{}]+\}|\\end\{[^{}]+\}", "", text)
    text = text.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    if "=" in text:
        parts = text.split("=")
        if len(parts) != 2:
            raise FormulaConversionError("Use one equation sign only; put the predicted shrinkage on the left")
        text = parts[1]
    text = _latex_fragment(text)
    for source_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(source_char, replacement)
    text = re.sub(r"\bV\s*/\s*S\b", "VtoS", text)
    text = re.sub(r"\bw\s*/\s*b\b", "wb", text)
    text = re.sub(r"\be\s*\*\*\s*\(([^()]*)\)", r"exp(\1)", text)
    text = re.sub(r"\be\s*\*\*\s*([+-]?[A-Za-z0-9_./*]+)", r"exp(\1)", text)
    for alias, replacement in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
        text = re.sub(r"\b{}\b".format(re.escape(alias)), replacement, text, flags=re.IGNORECASE if alias.startswith("eps") else 0)
    text = re.sub(r"\s+", " ", text).strip()
    return text


TOKEN_PATTERN = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"[A-Za-z][A-Za-z0-9_]*|"
    r"\*\*|<=|>=|==|!=|[+\-*/(),<>]"
)


def _tokenize(text):
    tokens = []
    cursor = 0
    for match in TOKEN_PATTERN.finditer(text):
        skipped = text[cursor:match.start()]
        if skipped.strip():
            raise FormulaConversionError("Unsupported symbol '{}' near position {}".format(skipped.strip(), cursor + 1))
        tokens.append(match.group(0))
        cursor = match.end()
    if text[cursor:].strip():
        raise FormulaConversionError("Unsupported symbol '{}' near the end of the formula".format(text[cursor:].strip()))
    if not tokens:
        raise FormulaConversionError("No calculable expression was found")
    return tokens


def _is_number(token):
    return bool(re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token))


def _is_name(token):
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", token))


def _insert_multiplication(tokens):
    result = []
    for token in tokens:
        if result:
            previous = result[-1]
            left_atom = _is_number(previous) or _is_name(previous) or previous == ")"
            right_atom = _is_number(token) or _is_name(token) or token == "("
            function_call = _is_name(previous) and previous in FUNCTIONS and token == "("
            if left_atom and right_atom and not function_call:
                result.append("*")
        result.append(token)
    return result


def _canonical_name(name):
    if name in ALIASES:
        return ALIASES[name]
    lowered = name.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]
    canonical = {
        "rh": "RH",
        "vtoS": "VtoS",
        "vtos": "VtoS",
        "ec28": "Ec28",
    }.get(lowered)
    return canonical or name


def _canonicalize_tokens(tokens):
    return [_canonical_name(token) if _is_name(token) else token for token in tokens]


def _parameter_default(name):
    if name in DEFAULT_PARAMETERS:
        return DEFAULT_PARAMETERS[name]
    lowered = name.lower()
    if lowered.startswith("eps"):
        return 1000.0
    if lowered.startswith("tau") or "time" in lowered:
        return 55.0
    if lowered in {"n", "p", "q"} or "power" in lowered or "exponent" in lowered:
        return 1.0
    if lowered.startswith("k") or "factor" in lowered or "coefficient" in lowered:
        return 1.0
    return 1.0


LATEX_NAMES = {
    "t": "t",
    "t0": r"t_{0}",
    "RH": "RH",
    "T": "T",
    "h0": r"h_{0}",
    "VtoS": "V/S",
    "ks": r"k_{\mathrm{s}}",
    "wb": "w/b",
    "fc28": r"f_{\mathrm{c},28}",
    "Ec28": r"E_{\mathrm{c},28}",
    "eps_u": r"\varepsilon_{\mathrm{u}}",
    "tau": r"\tau",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "phi": r"\phi",
}


def _ast_latex(node, parent_precedence=0):
    precedence = 100
    if isinstance(node, ast.Constant):
        return "{:g}".format(float(node.value))
    if isinstance(node, ast.Name):
        if node.id in LATEX_NAMES:
            return LATEX_NAMES[node.id]
        if "_" in node.id:
            base, subscript = node.id.split("_", 1)
            return r"{}_{{\mathrm{{{}}}}}".format(base, subscript.replace("_", ","))
        return node.id
    if isinstance(node, ast.UnaryOp):
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        result = sign + _ast_latex(node.operand, 80)
        return r"\left({}\right)".format(result) if 80 < parent_precedence else result
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            precedence = 20
            result = r"{}+{}".format(_ast_latex(node.left, precedence), _ast_latex(node.right, precedence + 1))
            return r"\left({}\right)".format(result) if precedence < parent_precedence else result
        if isinstance(node.op, ast.Sub):
            precedence = 20
            result = r"{}-{}".format(_ast_latex(node.left, precedence), _ast_latex(node.right, precedence + 1))
            return r"\left({}\right)".format(result) if precedence < parent_precedence else result
        if isinstance(node.op, ast.Mult):
            precedence = 40
            result = r"{}\,{}".format(_ast_latex(node.left, precedence), _ast_latex(node.right, precedence))
            return r"\left({}\right)".format(result) if precedence < parent_precedence else result
        if isinstance(node.op, ast.Div):
            precedence = 40
            result = r"\frac{{{}}}{{{}}}".format(_ast_latex(node.left), _ast_latex(node.right))
            return r"\left({}\right)".format(result) if precedence < parent_precedence else result
        if isinstance(node.op, ast.Pow):
            precedence = 60
            result = r"\left({}\right)^{{{}}}".format(_ast_latex(node.left), _ast_latex(node.right))
            return r"\left({}\right)".format(result) if precedence < parent_precedence else result
    if isinstance(node, ast.Call):
        name = node.func.id
        arguments = [_ast_latex(argument) for argument in node.args]
        if name == "sqrt":
            return r"\sqrt{{{}}}".format(arguments[0])
        if name == "abs":
            return r"\left|{}\right|".format(arguments[0])
        mapped = {"minimum": "min", "maximum": "max"}.get(name, name)
        return r"\mathrm{{{}}}\left({}\right)".format(mapped, ",".join(arguments))
    if isinstance(node, ast.Compare):
        operators = {ast.Lt: "<", ast.LtE: r"\leq", ast.Gt: ">", ast.GtE: r"\geq", ast.Eq: "=", ast.NotEq: r"\ne"}
        pieces = [_ast_latex(node.left)]
        for operator, comparator in zip(node.ops, node.comparators):
            pieces.append(operators[type(operator)])
            pieces.append(_ast_latex(comparator))
        return "".join(pieces)
    raise FormulaConversionError("Cannot typeset {}".format(type(node).__name__))


def _ast_expression(node, parent_precedence=0):
    """Emit a compact canonical expression from an already validated AST."""
    if isinstance(node, ast.Constant):
        value = float(node.value)
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp):
        precedence = 40
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        result = sign + _ast_expression(node.operand, precedence)
        return "({})".format(result) if precedence < parent_precedence else result
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            precedence, operator = 10, "+"
            result = _ast_expression(node.left, precedence) + operator + _ast_expression(node.right, precedence + 1)
        elif isinstance(node.op, ast.Sub):
            precedence, operator = 10, "-"
            result = _ast_expression(node.left, precedence) + operator + _ast_expression(node.right, precedence + 1)
        elif isinstance(node.op, ast.Mult):
            precedence, operator = 20, "*"
            result = _ast_expression(node.left, precedence) + operator + _ast_expression(node.right, precedence)
        elif isinstance(node.op, ast.Div):
            precedence, operator = 20, "/"
            result = _ast_expression(node.left, precedence) + operator + _ast_expression(node.right, precedence + 1)
        elif isinstance(node.op, ast.Pow):
            precedence, operator = 30, "**"
            result = _ast_expression(node.left, precedence + 1) + operator + _ast_expression(node.right, precedence)
        else:
            raise FormulaConversionError("Unsupported binary operation")
        return "({})".format(result) if precedence < parent_precedence else result
    if isinstance(node, ast.Call):
        return "{}({})".format(node.func.id, ",".join(_ast_expression(argument) for argument in node.args))
    if isinstance(node, ast.Compare):
        operators = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
        pieces = [_ast_expression(node.left)]
        for operator, comparator in zip(node.ops, node.comparators):
            pieces.extend((operators[type(operator)], _ast_expression(comparator)))
        return "".join(pieces)
    raise FormulaConversionError("Cannot generate code for {}".format(type(node).__name__))


def convert_formula(source):
    """Convert mathematical notation to a validated registry expression."""
    normalized = _normalise_notation(source)
    tokens = _canonicalize_tokens(_tokenize(normalized))
    tokens = _insert_multiplication(tokens)
    expression = "".join(tokens)
    try:
        tree = ast.parse(expression, mode="eval")
        allowed = set(VARIABLES) | set(FUNCTIONS) | {"pi", "e"}
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed and node.id != "eps_sh" and node.id not in names:
                names.append(node.id)
        _ExpressionValidator(allowed | set(names)).visit(tree)
    except (SyntaxError, FormulaValidationError) as exc:
        raise FormulaConversionError("The formula could not be converted: {}".format(exc))
    if "eps_sh" in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}:
        raise FormulaConversionError("Put eps_sh(t) on the left of '='; the right side must contain only the calculation")
    parameters = {name: _parameter_default(name) for name in names}
    expression = _ast_expression(tree.body)
    latex = r"\varepsilon_{\mathrm{sh}}(t)=" + _ast_latex(tree.body)
    return {
        "expression": expression,
        "latex": latex,
        "parameters": parameters,
        "normalized": normalized,
    }
