# Security policy

## Supported version

Security and integrity fixes are currently provided for version 1.0.x.

## Formula-package boundary

Imported `.peaf` formulas are parsed as restricted mathematical expressions.
They do not execute Python source, imports, attribute access, shell commands,
or arbitrary functions. Built-in formulas and deployment model artifacts are
read-only. User formulas are stored separately under `%LOCALAPPDATA%` and
invalid packages are quarantined.

Model, preprocessing, and support files are checked against the SHA-256 values
recorded in the deployment manifest before inference starts.

## Reporting a vulnerability

Please use GitHub's private security-advisory workflow for this repository.
Do not include private research data, credentials, or unpublished manuscript
files in a public issue.

This project is research software. Its numerical checks do not replace design
codes, experimental verification, or independent engineering judgement.

