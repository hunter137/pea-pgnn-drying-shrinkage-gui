"""Transparent input-support diagnostics for the GUI.

The diagnostic is descriptive. It does not turn an in-range input into a
guarantee of predictive validity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SUPPORT_VARIABLES = [
    "t0", "RH", "T", "h0", "VtoS", "wb", "fc28_cyl",
    "cement", "water", "agg_total", "Ec28",
]


def build_support_spec(frame):
    # One row per leakage-safe condition profile prevents densely observed
    # trajectories from dominating the joint-distance reference.
    profiles = frame.sort_values("source_row_id").drop_duplicates("condition_id")
    values = profiles[SUPPORT_VARIABLES].to_numpy(float)
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    filled = np.where(np.isfinite(values), values, center)
    standardized = (filled - center) / scale
    # Chunked O(n^2) nearest-neighbour distance; only 286 profiles.
    difference = standardized[:, None, :] - standardized[None, :, :]
    distance = np.sqrt(np.sum(difference**2, axis=2))
    np.fill_diagonal(distance, np.inf)
    nearest = distance.min(axis=1)
    ranges = {}
    for index, name in enumerate(SUPPORT_VARIABLES):
        column = values[:, index]
        ranges[name] = {
            "min": float(np.nanmin(column)),
            "q01": float(np.nanpercentile(column, 1)),
            "q99": float(np.nanpercentile(column, 99)),
            "max": float(np.nanmax(column)),
        }
    return {
        "variables": SUPPORT_VARIABLES,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "reference_profiles": standardized.tolist(),
        "nearest_distance_q95": float(np.percentile(nearest, 95)),
        "nearest_distance_max": float(np.max(nearest)),
        "marginal_ranges": ranges,
        "n_reference_profiles": int(len(profiles)),
        "definition": "Marginal recorded ranges plus robust nearest-profile distance in 11 user-facing variables",
    }


def assess_support(condition_frame, spec):
    variables = spec["variables"]
    row = condition_frame.iloc[0]
    values = np.array([float(row[name]) for name in variables], dtype=float)
    center = np.asarray(spec["center"], dtype=float)
    scale = np.asarray(spec["scale"], dtype=float)
    reference = np.asarray(spec["reference_profiles"], dtype=float)
    standardized = (values - center) / scale
    nearest = float(np.sqrt(np.sum((reference - standardized) ** 2, axis=1)).min())
    outside = []
    near = []
    for name, value in zip(variables, values):
        limits = spec["marginal_ranges"][name]
        if value < limits["min"] or value > limits["max"]:
            outside.append(name)
        elif value < limits["q01"] or value > limits["q99"]:
            near.append(name)
    if outside or nearest > spec["nearest_distance_max"]:
        level = "outside"
        label = "Outside recorded support"
    elif near or nearest > spec["nearest_distance_q95"]:
        level = "boundary"
        label = "Near the recorded support boundary"
    else:
        level = "within"
        label = "Within recorded support"
    return {
        "level": level,
        "label": label,
        "outside_variables": outside,
        "boundary_variables": near,
        "nearest_profile_distance": nearest,
        "distance_q95": float(spec["nearest_distance_q95"]),
        "note": "Descriptive support status, not a guarantee of predictive validity.",
    }

