"""Evaluation metrics used by the audit and deployment reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 5:
        return {"N": int(len(y_true)), "RMSE": np.nan, "MAE": np.nan, "bias": np.nan, "R2": np.nan}
    residual = y_pred - y_true
    total = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "N": int(len(y_true)),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "R2": float(1.0 - np.sum(residual**2) / max(float(total), 1e-8)),
    }


def macro_rmse(y_true, y_pred, groups):
    frame = pd.DataFrame({"true": y_true, "pred": y_pred, "group": groups})
    values = frame.groupby("group", sort=False).apply(
        lambda block: np.sqrt(np.mean((block["pred"] - block["true"]) ** 2))
    )
    return float(values.mean())


def complete_metrics(y_true, y_pred, condition_ids=None, trajectory_ids=None):
    output = regression_metrics(y_true, y_pred)
    if condition_ids is not None:
        output["condition_macro_RMSE"] = macro_rmse(y_true, y_pred, condition_ids)
        output["n_condition_profiles"] = int(np.unique(condition_ids).size)
    if trajectory_ids is not None:
        output["trajectory_macro_RMSE"] = macro_rmse(y_true, y_pred, trajectory_ids)
        output["n_database_trajectories"] = int(np.unique(trajectory_ids).size)
    return output

