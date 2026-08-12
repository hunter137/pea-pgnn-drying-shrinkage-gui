"""Pure helpers for the formula trial-calculation workbench."""

from __future__ import annotations

import numpy as np


DEFAULT_TRIAL_CONDITION = {
    "cement": 371.0,
    "water": 186.0,
    "aggregate": 1859.0,
    "wb": 0.48,
    "fc28": 37.0,
    "Ec28": 25958.0,
    "cement_type_code": 2,
    "agg_type_code": 1,
    "curing_type_code": 1,
    "t0": 7.0,
    "RH": 50.0,
    "T": 23.0,
    "h0": 45.5,
    "geometry": "Prism",
    "query_age": 365.0,
}


def parse_number_list(text, *, positive=True, maximum_count=32):
    """Parse comma/semicolon/space separated trial values."""
    normalised = str(text).replace("，", ",").replace(";", ",").replace("；", ",")
    parts = []
    for chunk in normalised.split(","):
        parts.extend(item for item in chunk.split() if item)
    if not parts:
        raise ValueError("Enter at least one numerical value")
    if len(parts) > maximum_count:
        raise ValueError("At most {} values may be calculated at once".format(maximum_count))
    try:
        values = np.asarray([float(item) for item in parts], dtype=float)
    except ValueError as exc:
        raise ValueError("Values must be numbers separated by commas") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("Values must be finite numbers")
    if positive and np.any(values <= 0):
        raise ValueError("Values must be greater than zero")
    return np.unique(values)


def build_trial_age_grid(maximum_age, key_ages=(), points=220):
    """Return a dense positive grid that also contains all requested ages."""
    maximum_age = float(maximum_age)
    if not np.isfinite(maximum_age) or maximum_age <= 0:
        raise ValueError("Maximum age must be a finite positive number")
    if int(points) < 20:
        raise ValueError("At least 20 curve points are required")
    keys = np.asarray(key_ages, dtype=float).reshape(-1)
    keys = keys[(keys > 0) & (keys <= maximum_age)]
    if maximum_age <= 0.5:
        dense = np.linspace(maximum_age / float(points), maximum_age, int(points))
    else:
        dense = np.geomspace(0.5, maximum_age, int(points))
    return np.unique(np.r_[dense, keys])


def analyse_formula_curve(ages, values, zero_value=None):
    """Run transparent numerical/shape checks without claiming validation."""
    ages = np.asarray(ages, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if ages.shape != values.shape or ages.size < 2:
        raise ValueError("A formula health check requires matching curves with at least two points")
    order = np.argsort(ages)
    ages = ages[order]
    values = values[order]
    checks = []

    finite = bool(np.all(np.isfinite(values)))
    non_negative = bool(finite and np.all(values >= -1.0e-9))
    checks.append({
        "level": "pass" if finite and non_negative else "fail",
        "title": "Numerical output",
        "detail": "All sampled values are finite and non-negative." if finite and non_negative else "A non-finite or negative sampled value was found.",
    })

    tolerance = max(1.0e-6, float(np.nanmax(np.abs(values))) * 1.0e-7) if finite else 1.0e-6
    differences = np.diff(values)
    violations = int(np.sum(differences < -tolerance)) if finite else len(differences)
    checks.append({
        "level": "pass" if violations == 0 else "warn",
        "title": "Time monotonicity",
        "detail": "No decrease was detected on the sampled age grid." if violations == 0 else "{} decreasing interval(s) were detected; confirm whether this is intended.".format(violations),
    })

    if zero_value is None or not np.isfinite(float(zero_value)):
        checks.append({"level": "warn", "title": "Initial value", "detail": "The expression could not be evaluated at t = 0 d."})
    elif abs(float(zero_value)) <= 1.0:
        checks.append({"level": "pass", "title": "Initial value", "detail": "The value at t = 0 d is {:.3g} microstrain.".format(float(zero_value))})
    else:
        checks.append({"level": "warn", "title": "Initial value", "detail": "The value at t = 0 d is {:.1f} microstrain rather than approximately zero.".format(float(zero_value))})

    maximum = float(np.nanmax(values)) if finite else float("nan")
    checks.append({
        "level": "warn" if finite and maximum > 3000.0 else ("pass" if finite else "fail"),
        "title": "Magnitude screen",
        "detail": "Maximum sampled magnitude is {:.1f} microstrain{}".format(maximum, "; review units and coefficients." if maximum > 3000.0 else "."),
    })

    response_range = float(np.nanmax(values) - np.nanmin(values)) if finite else float("nan")
    checks.append({
        "level": "warn" if finite and response_range <= 1.0e-6 else ("pass" if finite else "fail"),
        "title": "Age response",
        "detail": "The curve changes across the sampled ages." if response_range > 1.0e-6 else "The output is effectively constant over time; check whether t is used.",
    })
    return {
        "checks": checks,
        "warning_count": sum(item["level"] == "warn" for item in checks),
        "failure_count": sum(item["level"] == "fail" for item in checks),
        "monotonicity_violations": violations,
        "maximum": maximum,
    }
