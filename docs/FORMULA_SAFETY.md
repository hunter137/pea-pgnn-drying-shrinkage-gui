# Formula editing and safety

The formula library is designed for users who do not write Python code.

- Built-in B3, GL2000, and ACI 209 definitions are compiled, read-only model
  priors. They cannot be edited, disabled, archived, or removed.
- A built-in formula can be copied into an editable custom comparison curve.
- New formulas can start from a guided template or be converted from supported
  LaTeX, Unicode mathematics, calculator notation, or MathType MathML.
- Imported packages are validated against a restricted expression grammar.
  Python imports, attributes, file access, and shell execution are rejected.
- Unsaved formulas can be checked at key ages, compared with the deployed
  model and native references, and screened for invalid values, decrease,
  non-zero initial response, unusual magnitude, and missing age response.
- User formulas live under `%LOCALAPPDATA%\PEA-PGNN\V1.0.0\FormulaData`.
  Active files, archives, history, and quarantine are kept separately.

Custom formulas are comparison curves in V1.0.0. They do not change or retrain
the bundled neural network.

