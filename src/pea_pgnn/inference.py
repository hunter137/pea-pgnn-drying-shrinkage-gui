"""Single shared inference API for desktop, batch and test consumers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import read_json, validate_manifest_files
from .data import apply_imputation
from .features import FEATURE_NAMES, build_features, condition_frame
from .formula_registry import FormulaRegistry, default_user_formula_directory
from .model import PEAPGNN
from .support import assess_support


class Predictor:
    def __init__(self, artifact_directory, device=None, formula_directory=None):
        self.root = Path(artifact_directory).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        validate_manifest_files(self.root, self.manifest)
        if self.manifest["feature_schema"]["names"] != FEATURE_NAMES:
            raise ValueError("Artifact feature schema does not match the frozen 39-variable implementation")
        if int(self.manifest["parameter_count_per_member"]) != 104200:
            raise ValueError("Artifact does not contain the frozen 104,200-parameter architecture")
        self.preprocessing = read_json(self.root / "preprocessing.json")
        self.support_spec = read_json(self.root / "support.json")
        project_root = self.root.parent.parent
        if formula_directory is None:
            self.formulas = FormulaRegistry(
                default_user_formula_directory(),
                legacy_directory=project_root / "formula_packages" / "custom",
            )
        else:
            self.formulas = FormulaRegistry(formula_directory)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        architecture = self.manifest["architecture"]
        self.models = []
        for seed in self.manifest["ensemble_seeds"]:
            model = PEAPGNN(
                n_features=39,
                eps_anchor_index=34,
                tau_anchor_index=33,
                hidden=tuple(architecture["hidden"]),
                dropout=architecture["dropout"],
                delta_eps_range=tuple(architecture["delta_eps_range"]),
                delta_tau_range=tuple(architecture["delta_tau_range"]),
                additive_scale=architecture["additive_scale"],
                eps_min=architecture["eps_min"],
                eps_max=architecture["eps_max"],
                tau_min=architecture["tau_min"],
                tau_max=architecture["tau_max"],
            )
            state = torch.load(
                self.root / "models" / "seed_{}.pt".format(seed),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
            self.models.append(model)
        self.scaler_center = np.asarray(self.preprocessing["scaler_center"], dtype=np.float32)
        self.scaler_scale = np.asarray(self.preprocessing["scaler_scale"], dtype=np.float32)

    @property
    def model_label(self):
        label = "PEA-PGNN V{} | {}-member ensemble".format(self.manifest["model_version"], len(self.models))
        return label

    def formula_definitions(self):
        return self.formulas.definitions()

    def reload_formulas(self):
        self.formulas.reload()
        return self.formula_definitions()

    def import_formula(self, path):
        return self.formulas.import_package(path)

    def save_formula(self, document, original_id=None):
        return self.formulas.save_package(document, original_id=original_id)

    def preview_formula(self, document, condition=None, ages=None):
        """Evaluate an unsaved formula for a caller-supplied trial condition.

        The editor uses the defaults for its instant four-point preview.  The
        trial-calculation workbench supplies the active GUI condition and an
        arbitrary age grid through the same safe formula evaluator.
        """
        if condition is None:
            condition = {
                "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48,
                "fc28": 37, "Ec28": 25958, "cement_type_code": 2,
                "agg_type_code": 1, "curing_type_code": 1, "t0": 7,
                "RH": 50, "T": 23, "h0": 45.5, "geometry": "Prism",
                "query_age": 365,
            }
        else:
            condition = dict(condition)
        if ages is None:
            ages = [7.0, 28.0, 90.0, 365.0]
        ages = np.asarray(ages, dtype=float).reshape(-1)
        if ages.size == 0 or not np.all(np.isfinite(ages)) or np.any(ages < 0):
            raise ValueError("Formula-preview ages must be finite and non-negative")
        condition.setdefault("query_age", float(max(float(np.max(ages)), 0.5)))
        self.validate_condition(condition)
        base = condition_frame(condition)
        base = apply_imputation(base, self.preprocessing["imputation"])
        repeated = pd.concat([base] * len(ages), ignore_index=True)
        return ages, self.formulas.preview(document, repeated, ages)

    def set_formula_enabled(self, formula_id, enabled):
        self.formulas.set_enabled(formula_id, enabled)

    def remove_formula(self, formula_id):
        return self.formulas.remove(formula_id)

    def restore_formula(self, path):
        return self.formulas.restore_archived(path)

    def validate_condition(self, condition):
        required_positive = ["t0", "h0", "wb", "fc28", "cement", "water", "aggregate", "Ec28", "query_age"]
        errors = []
        for name in required_positive:
            value = float(condition[name])
            if not np.isfinite(value) or value <= 0:
                errors.append("{} must be a finite positive value".format(name))
        rh = float(condition["RH"])
        if not np.isfinite(rh) or rh <= 0 or rh >= 100:
            errors.append("RH must lie strictly between 0 and 100%")
        temperature = float(condition["T"])
        if not np.isfinite(temperature):
            errors.append("T must be finite")
        maximum_age = float(self.manifest.get("max_query_age_days", 4832.92))
        if float(condition["query_age"]) > maximum_age:
            errors.append("query_age exceeds the implemented maximum of {:.2f} d".format(maximum_age))
        if errors:
            raise ValueError("; ".join(errors))

    def _prepare(self, condition, ages):
        self.validate_condition(condition)
        ages = np.asarray(ages, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(ages)) or np.any(ages <= 0):
            raise ValueError("All query ages must be finite and positive")
        base = condition_frame(condition)
        base = apply_imputation(base, self.preprocessing["imputation"])
        repeated = pd.concat([base] * len(ages), ignore_index=True)
        repeated["dt"] = ages
        raw = build_features(repeated)
        scaled = ((raw - self.scaler_center) / (self.scaler_scale + 1e-9)).astype(np.float32)
        return base, repeated, ages, raw, scaled

    @torch.no_grad()
    def predict(self, condition, ages, include_details=True):
        base, repeated, ages, raw, scaled = self._prepare(condition, ages)
        raw_tensor = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        scaled_tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        age_tensor = torch.as_tensor(ages, dtype=torch.float32, device=self.device)
        predictions = []
        detail_sets = []
        for model in self.models:
            prediction, details = model(raw_tensor, scaled_tensor, age_tensor, return_details=True)
            predictions.append(prediction.cpu().numpy())
            detail_sets.append({name: value.cpu().numpy() for name, value in details.items()})
        member_predictions = np.stack(predictions, axis=0)
        mean_prediction = member_predictions.mean(axis=0)
        output = {
            "ages": ages,
            "prediction": mean_prediction,
            "optimization_sd": member_predictions.std(axis=0, ddof=0),
            "member_predictions": member_predictions,
            "support": assess_support(base, self.support_spec),
            "model_label": self.model_label,
        }
        if include_details:
            for name in ("eps_anchor", "tau_anchor", "eps_inf", "tau", "alpha", "weights", "delta_eps", "delta_tau", "delta_add"):
                values = np.stack([details[name] for details in detail_sets], axis=0)
                output[name] = values.mean(axis=0)
                output[name + "_optimization_sd"] = values.std(axis=0, ddof=0)
        output["references"] = self.formulas.evaluate(repeated, ages)
        return output

    def predict_curve(self, condition, points=160):
        query_age = float(condition["query_age"])
        upper = max(400.0, query_age * 1.05)
        upper = min(upper, float(self.manifest.get("max_query_age_days", 4832.92)))
        ages = np.geomspace(0.5, upper, int(points))
        key = np.array([7.0, 14.0, 28.0, 56.0, 90.0, 180.0, 365.0, query_age])
        ages = np.unique(np.r_[ages, key[(key > 0) & (key <= upper)]])
        return self.predict(condition, ages, include_details=True)

    def predict_batch(self, frame):
        rows = []
        for index, source in frame.iterrows():
            try:
                condition = source.to_dict()
                result = self.predict(condition, [float(condition["query_age"])])
                row = dict(source)
                row.update(
                    {
                        "PEA_PGNN_microstrain": float(result["prediction"][0]),
                        "optimization_sd_microstrain": float(result["optimization_sd"][0]),
                        "support_status": result["support"]["label"],
                        "status": "OK",
                    }
                )
            except Exception as exc:
                row = dict(source)
                row.update({"status": "ERROR", "error": str(exc)})
            rows.append(row)
        return pd.DataFrame(rows)
