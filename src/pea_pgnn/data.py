"""Frozen population loading and leakage-controlled preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dt", "t0", "RH", "T", "h0", "VtoS", "cement", "water",
    "agg_total", "wb", "wc", "fc28_cyl", "Ec28", "cement_type_code",
    "agg_type_code", "curing_type_code", "shrinkage_strain", "geometry",
    "ks", "source_row_id", "ST_id", "leakage_safe_group_id",
}


@dataclass
class ImputationSpec:
    medians: Dict[str, float]
    categories: Dict[str, float]

    def as_dict(self):
        return {"medians": self.medians, "categories": self.categories}

    @classmethod
    def from_dict(cls, value):
        return cls(
            medians={str(k): float(v) for k, v in value["medians"].items()},
            categories={str(k): float(v) for k, v in value["categories"].items()},
        )


def load_frozen_population(path, expected_records=8729, expected_profiles=286, expected_trajectories=1076):
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("Frozen dataset is missing required columns: {}".format(missing))
    if len(frame) != int(expected_records):
        raise ValueError("Expected {} frozen records, found {}".format(expected_records, len(frame)))
    if frame["source_row_id"].duplicated().any():
        raise ValueError("source_row_id is not unique in the frozen population")
    if int(frame["leakage_safe_group_id"].nunique()) != int(expected_profiles):
        raise ValueError("Unexpected number of leakage-safe condition profiles")
    if int(frame["ST_id"].nunique()) != int(expected_trajectories):
        raise ValueError("Unexpected number of database trajectories")
    frame = frame.copy()
    frame["shrinkage_abs"] = frame["shrinkage_strain"].abs()
    valid = (frame["shrinkage_abs"] > 0) & (frame["dt"] > 0) & (frame["shrinkage_abs"] < 2000)
    if not bool(valid.all()):
        raise ValueError("Frozen population contains records outside the declared analysis filter")
    frame["condition_id"] = frame["leakage_safe_group_id"].astype(int)
    return frame.reset_index(drop=True)


def fit_imputation(frame, row_indices):
    train = frame.iloc[np.asarray(row_indices, dtype=int)]
    defaults = {
        "fc28_cyl": 30.0,
        "water": 180.0,
        "cement": 350.0,
        "agg_total": 1800.0,
        "Ec28": 30000.0,
        "T": 23.0,
    }
    medians = {}
    for name, default in defaults.items():
        value = train[name].median() if name in train else np.nan
        medians[name] = float(value) if np.isfinite(value) else float(default)
    categories = {}
    for name, default in (
        ("cement_type_code", 2.0),
        ("agg_type_code", 1.0),
        ("curing_type_code", 1.0),
    ):
        mode = train[name].dropna().mode()
        categories[name] = float(mode.iloc[0]) if len(mode) else float(default)
    return ImputationSpec(medians=medians, categories=categories)


def apply_imputation(frame, spec):
    if isinstance(spec, dict):
        spec = ImputationSpec.from_dict(spec)
    output = frame.copy()
    for name, value in spec.medians.items():
        output[name] = output[name].fillna(value)
    for name, value in spec.categories.items():
        output[name] = output[name].fillna(value)
    output["wb"] = output["wb"].fillna(
        output["wc"].fillna(output["water"] / np.maximum(output["cement"], 1.0))
    )
    output["ks"] = output["ks"].fillna(1.25)
    return output


def load_frozen_split(path, frame, cutoff=365):
    split = pd.read_csv(Path(path).resolve())
    required = {"row_index", "source_row_id", "ST_id", "condition_id", "threshold", "fold", "role"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError("Frozen split is missing columns: {}".format(missing))
    split = split.loc[split["threshold"] == int(cutoff)].copy()
    if split.empty:
        raise ValueError("No frozen split rows found for cutoff {}".format(cutoff))
    for fold in sorted(split["fold"].unique()):
        block = split.loc[split["fold"] == fold]
        train = set(block.loc[block["role"] == "inner_train_development", "condition_id"])
        val = set(block.loc[block["role"] == "inner_validation_development", "condition_id"])
        held = set(block.loc[block["role"].str.startswith("heldout"), "condition_id"])
        if train & val or train & held or val & held:
            raise ValueError("Condition-profile leakage detected in frozen fold {}".format(fold))
        idx = block["row_index"].to_numpy(int)
        if not np.array_equal(frame.iloc[idx]["source_row_id"].to_numpy(), block["source_row_id"].to_numpy()):
            raise ValueError("Frozen split row identity mismatch in fold {}".format(fold))
    return split


def indices_for_fold(split, fold):
    block = split.loc[split["fold"] == int(fold)]
    roles = {
        role: block.loc[block["role"] == role, "row_index"].to_numpy(dtype=int)
        for role in block["role"].unique()
    }
    return roles

