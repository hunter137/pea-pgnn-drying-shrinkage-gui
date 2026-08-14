# PEA-PGNN Drying-Shrinkage Prediction GUI

[![Version](https://img.shields.io/badge/version-1.0.0-1F5A94?style=flat-square)](CHANGELOG.md)
[![Tests](https://github.com/hunter137/pea-pgnn-drying-shrinkage-gui/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/hunter137/pea-pgnn-drying-shrinkage-gui/actions/workflows/tests.yml)
[![GitHub release](https://img.shields.io/github/v/release/hunter137/pea-pgnn-drying-shrinkage-gui?style=flat-square)](https://github.com/hunter137/pea-pgnn-drying-shrinkage-gui/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-2E7D32?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D4?style=flat-square&logo=windows)](docs/INSTALLATION.md)
[![Research software](https://img.shields.io/badge/status-research%20software-6B7280?style=flat-square)](MODEL_CARD.md)

Windows desktop software for predicting concrete drying shrinkage with the
bundled PEA-PGNN model. The application compares reference equations, supports
custom formulas and batch calculations, and exports PDF calculation reports.

This repository contains the desktop application. The reusable Python package
is maintained separately at
[`hunter137/pea-pgnn`](https://github.com/hunter137/pea-pgnn).

> **Scope:** Results are research-use point predictions. V1.0.0 does not provide
> a calibrated prediction interval or a design certificate. Use applicable
> design codes, experiments, and project-specific engineering checks when
> making design decisions.

![PEA-PGNN engineering workbench](docs/images/workbench.png)

## Features

- Windows desktop interface for entering material, exposure, curing, and
  geometry conditions.
- Drying-shrinkage curves and key-age predictions from the bundled three-member
  PEA-PGNN model.
- Comparison with the B3, GL2000, and ACI 209 reference equations, together
  with checks against the recorded input ranges.
- Curve export, batch CSV prediction, and standard or technical PDF calculation
  reports.
- Protected built-in equations and a separate user formula library with
  editing, preview, trial calculation, backup, and recovery.
- CPU execution with automatic CUDA acceleration when a compatible PyTorch
  installation is available.

## Quick start on Windows

Clone the repository and install it in an isolated environment:

```powershell
git clone https://github.com/hunter137/pea-pgnn-drying-shrinkage-gui.git
cd pea-pgnn-drying-shrinkage-gui
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_gui.py
```

After the dependencies are installed, `launch_gui.bat` can also be
double-clicked. See [Installation and startup](docs/INSTALLATION.md) for CUDA,
MathType, and retraining notes.

## Main workflow

1. Enter the material, exposure, curing, geometry, and prediction age in
   the left panel. Use the section buttons or mouse wheel to move through the
   input groups.
2. Select **Run** or press **F5**.
3. Review the prediction curve, key-age table, reference comparisons,
   input-range check, and model diagnostics.
4. Export the curve, run a batch CSV calculation, or generate a standard or
   technical PDF calculation report.

The PEA-PGNN result at the requested age is a point prediction. The reported
ensemble-member standard deviation describes variation between optimization
seeds. It is not a calibrated prediction interval.

## PDF calculation reports

After running a prediction, select **Report**. The report uses the result shown
in the workbench and does not run the model again.

The **standard engineering report** includes the input condition, calculated
results, comparison curves, applicability checks, formula register, and
preparation/review fields. The **complete technical report** adds deployment
and validation details. The application assigns a report ID and writes the
saved PDF's SHA-256 digest to the message log.

## Formula editor and numerical trials

![Guided formula editor](docs/images/formula-editor.png)

Built-in B3, GL2000, and ACI 209 formulas are read-only. Select **Copy as
custom** to create an editable comparison curve without changing the trained
model.

Choose a template or enter a formula in supported LaTeX, Unicode, or calculator
notation. Conversion takes place locally, and desktop MathType is available as
an optional visual editor. The expression parser rejects Python imports,
attribute access, file operations, and shell commands.

![Formula trial calculation](docs/images/trial-calculation.png)

Before saving, the trial window compares the formula with PEA-PGNN and the
built-in reference equations at selected ages. It also checks the calculated
curve and exports the values. The sensitivity tab varies one quantity at a
time and is intended for screening, not a global sensitivity analysis or
experimental validation. See
[Formula editing and safety](docs/FORMULA_SAFETY.md).

## Formula-data protection

User formulas are stored outside the source tree under:

```text
%LOCALAPPDATA%\PEA-PGNN\V1.0.0\FormulaData
```

Editing a user formula creates a revision snapshot, and removing it moves the
file to an archive instead of deleting it permanently. Invalid packages are
quarantined so that they do not prevent the application from loading.

In V1.0.0, custom formulas are displayed only as comparison curves. Using a
custom equation inside the neural network would require a new feature schema,
model version, and training run.

## Reproducibility and artifact integrity

At startup, the application verifies every model, preprocessing, and support
file listed in `artifacts/deployment/manifest.json` against its SHA-256 digest.
The manifest also records:

- the ensemble seeds and training epoch count;
- the model architecture and feature schema;
- the dataset and split identifiers used for the recorded evaluation;
- the later-age audit metrics;
- Python, PyTorch, CUDA, and training-hardware information; and
- whether calibrated prediction intervals are available.

The deployment model was fitted using all development records. It is not an
independent test model. See [MODEL_CARD.md](MODEL_CARD.md) for intended use,
metrics, limitations, and interpretation.

## Testing

Run the repository tests and GUI smoke check with:

```powershell
python -m unittest discover -s tests -v
python run_gui.py --smoke
```

Two tests for data identity and split leakage require the private research data
and are skipped in the public CI. All deployment, inference, formula, GUI, and
reporting tests run from the public repository. GitHub Actions runs these checks
on every push and pull request.

## Retraining

The repository includes the trained artifacts required for inference. It does
not include the 8,729-record research database, private split files,
intermediate training outputs, manuscript files, or user formula data.

Authorized researchers can point `configs/training.json` to local copies and
run:

```powershell
python train.py --mode audit --cpu-threads 12
python train.py --mode deployment --cpu-threads 12
```

Training uses CUDA automatically when available and otherwise uses the CPU.

## Project files

- [Installation and startup](docs/INSTALLATION.md)
- [Formula editing and safety](docs/FORMULA_SAFETY.md)
- [Model card](MODEL_CARD.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## Acknowledgments

This work was supported by the National Key R&D Program of China (Grant Nos.
2024YFC38098 and 2024YFC3809803), the Liaoning Xingliao Talents Program for
Science and Technology Innovation Team (No. XLYC2404005), and the Technology
Research and Development Program of Shenyang Science and Technology Bureau
(Grant No. 24-213-3-33).

## License and citation

The source code and bundled deployment artifacts are released under the
[MIT License](LICENSE). If this software contributes to research, use the
repository's **Cite this repository** entry generated from `CITATION.cff`.
