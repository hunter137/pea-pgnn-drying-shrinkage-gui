"""Safe, inspectable formula-package support for engineering references.

Formula packages deliberately use a restricted mathematical expression
language instead of executing Python files.  This gives the desktop software
an Abaqus-RPY-like extension point without granting an imported file access to
the filesystem, network, subprocesses or Python objects.
"""

from __future__ import annotations

import ast
from datetime import datetime
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

from .formulas import empirical_references


VARIABLES = (
    "t", "t0", "RH", "T", "h0", "VtoS", "ks", "wb", "fc28",
    "cement", "water", "aggregate", "Ec28",
)

FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "tanh": np.tanh,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
}

BUILTIN_FORMULAS = (
    {
        "id": "b3",
        "name": "Model B3",
        "source": "Built-in",
        "role": "Model prior and reference curve",
        "enabled": True,
        "locked": True,
        "model_prior": True,
        "output_unit": "microstrain",
        "color": "#2E7D32",
        "line_style": "--",
        "latex": r"\varepsilon_{\mathrm{sh}}^{\mathrm{B3}}(t)=\varepsilon_{\infty}^{\mathrm{B3}}\tanh\!\sqrt{t/\tau_{\mathrm{s}}}",
        "expression": "Frozen B3-compatible implementation; see formulas.py",
        "description": "Provides an ultimate-shrinkage anchor, a characteristic-time anchor and a comparison curve.",
    },
    {
        "id": "gl2000",
        "name": "GL2000",
        "source": "Built-in",
        "role": "Model prior and reference curve",
        "enabled": True,
        "locked": True,
        "model_prior": True,
        "output_unit": "microstrain",
        "color": "#C25400",
        "line_style": "-.",
        "latex": r"\varepsilon_{\mathrm{sh}}^{\mathrm{GL}}(t)=900\sqrt{30/f_{\mathrm{c}}}\,(1-1.18h^{4})\sqrt{\frac{t}{t+0.15(V/S)^{2}}}",
        "expression": "900*sqrt(30/fc28)*(1-1.18*(RH/100)^4)*sqrt(t/(t+0.15*VtoS^2))",
        "description": "GL2000-compatible database reference and ultimate-shrinkage prior.",
    },
    {
        "id": "aci209",
        "name": "ACI 209",
        "source": "Built-in",
        "role": "Model prior and reference curve",
        "enabled": True,
        "locked": True,
        "model_prior": True,
        "output_unit": "microstrain",
        "color": "#6B3FA0",
        "line_style": ":",
        "latex": r"\varepsilon_{\mathrm{sh}}^{\mathrm{ACI}}(t)=780\,\gamma_{h}\,(1.2)e^{-0.0047(V/S)}\frac{t}{35+t}",
        "expression": "780*humidity_factor*1.2*exp(-0.0047*VtoS)*t/(35+t)",
        "description": "ACI 209-compatible database reference and ultimate-shrinkage prior.",
    },
)


def default_user_formula_directory():
    """Return a per-user, versioned formula-data root outside the installation."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "PEA-PGNN" / "V1.0.0" / "FormulaData"


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _atomic_write_json(destination, document):
    """Write a package atomically so an interrupted save cannot truncate it."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(destination))


class FormulaValidationError(ValueError):
    """Raised when a formula package violates the safe schema."""


class _ExpressionValidator(ast.NodeVisitor):
    _operators = (
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    )
    _containers = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Compare)

    def __init__(self, allowed_names):
        self.allowed_names = set(allowed_names)
        self.nodes_seen = 0

    def generic_visit(self, node):
        self.nodes_seen += 1
        if self.nodes_seen > 512:
            raise FormulaValidationError("Expression is too complex (maximum 512 syntax nodes)")
        if not isinstance(node, self._containers + self._operators):
            raise FormulaValidationError("Unsupported syntax: {}".format(type(node).__name__))
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_") or node.id not in self.allowed_names:
            raise FormulaValidationError("Unknown or forbidden name: {}".format(node.id))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise FormulaValidationError("Only the documented mathematical functions may be called")
        if node.keywords:
            raise FormulaValidationError("Keyword arguments are not supported")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaValidationError("Only numeric constants are permitted")
        if not math.isfinite(float(node.value)) or abs(float(node.value)) > 1.0e12:
            raise FormulaValidationError("Numeric constant is outside the permitted range")


def _validate_package(document):
    if not isinstance(document, dict):
        raise FormulaValidationError("The package root must be a JSON object")
    if int(document.get("schema_version", -1)) != 1:
        raise FormulaValidationError("schema_version must be 1")
    formula_id = str(document.get("id", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,47}", formula_id):
        raise FormulaValidationError("id must contain 2-48 lowercase letters, digits or underscores")
    if formula_id in {item["id"] for item in BUILTIN_FORMULAS}:
        raise FormulaValidationError("id conflicts with a built-in formula")
    name = str(document.get("name", "")).strip()
    if not name or len(name) > 80:
        raise FormulaValidationError("name must contain 1-80 characters")
    expression = str(document.get("expression", "")).strip()
    if not expression or len(expression) > 4000:
        raise FormulaValidationError("expression must contain 1-4000 characters")
    constants = document.get("constants", {})
    if not isinstance(constants, dict) or len(constants) > 32:
        raise FormulaValidationError("constants must be an object with at most 32 entries")
    clean_constants = {}
    reserved = set(VARIABLES) | set(FUNCTIONS) | {"pi", "e"}
    for key, value in constants.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", str(key)) or key in reserved:
            raise FormulaValidationError("Invalid or reserved constant name: {}".format(key))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise FormulaValidationError("Constant {} must be a finite number".format(key))
        clean_constants[str(key)] = float(value)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FormulaValidationError("Invalid expression syntax: {}".format(exc.msg))
    _ExpressionValidator(set(VARIABLES) | set(FUNCTIONS) | set(clean_constants) | {"pi", "e"}).visit(tree)
    result = dict(document)
    result.update(
        {
            "schema_version": 1,
            "id": formula_id,
            "name": name,
            "expression": expression,
            "constants": clean_constants,
            "latex": str(document.get("latex", "")).strip()[:1000],
            "description": str(document.get("description", "")).strip()[:1000],
            "output_unit": "microstrain",
            "enabled": bool(document.get("enabled", True)),
            "source": "Custom package",
            "role": "Comparison curve only (model retraining required for use as a prior)",
            "locked": False,
            "model_prior": False,
            "color": str(document.get("color", "#8B5E3C")),
            "line_style": str(document.get("line_style", "--")),
        }
    )
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", result["color"]):
        raise FormulaValidationError("color must use #RRGGBB notation")
    if result["line_style"] not in {"-", "--", "-.", ":", "long-dash"}:
        raise FormulaValidationError("line_style must be one of -, --, -., :, long-dash")
    result["_code"] = compile(tree, "<PEA formula {}>".format(formula_id), "eval")
    return result


def _context(frame, ages):
    return {
        "t": np.asarray(ages, dtype=float),
        "t0": frame["t0"].to_numpy(float),
        "RH": frame["RH"].to_numpy(float),
        "T": frame["T"].to_numpy(float),
        "h0": frame["h0"].to_numpy(float),
        "VtoS": frame["VtoS"].to_numpy(float),
        "ks": frame["ks"].to_numpy(float),
        "wb": frame["wb"].to_numpy(float),
        "fc28": frame["fc28_cyl"].to_numpy(float),
        "cement": frame["cement"].to_numpy(float),
        "water": frame["water"].to_numpy(float),
        "aggregate": frame["agg_total"].to_numpy(float),
        "Ec28": frame["Ec28"].to_numpy(float),
    }


class FormulaRegistry:
    """Load built-ins and optional ``.peaf`` reference-curve packages."""

    def __init__(self, directory, legacy_directory=None):
        self.directory = Path(directory).resolve()
        self.custom_directory = self.directory / "custom"
        self.archive_directory = self.directory / "archive"
        self.history_directory = self.directory / "history"
        self.quarantine_directory = self.directory / "quarantine"
        self.custom_directory.mkdir(parents=True, exist_ok=True)
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        self.history_directory.mkdir(parents=True, exist_ok=True)
        self.quarantine_directory.mkdir(parents=True, exist_ok=True)
        self._write_data_space_guide()
        self._migrate_legacy_packages(legacy_directory)
        self.custom = {}
        self.errors = {}
        self.runtime_errors = {}
        self.reload()
        self._ensure_baseline_history()

    def _write_data_space_guide(self):
        guide = self.directory / "README.txt"
        if guide.exists():
            return
        guide.write_text(
            "PEA-PGNN V1.0.0 - User Formula Data\n\n"
            "custom     Active user formulas used by the software.\n"
            "archive    Formulas removed with the Archive button; these can be restored.\n"
            "history    Automatic snapshots created before edits and for initial saves/imports.\n"
            "quarantine Invalid packages moved out of the active library for safety.\n\n"
            "Native B3, GL2000 and ACI 209 formulas are not stored here. They are protected "
            "inside the application and cannot be damaged by changing files in this folder.\n",
            encoding="utf-8",
        )

    def _ensure_baseline_history(self):
        for formula_id, definition in self.custom.items():
            if any(self.history_directory.glob(formula_id + "_*.peaf")):
                continue
            source = Path(definition["path"])
            snapshot = self.history_directory / (formula_id + "_initial_" + _timestamp() + source.suffix)
            shutil.copyfile(str(source), str(snapshot))

    def _migrate_legacy_packages(self, legacy_directory):
        """Copy old project-local packages once, leaving their originals as backup."""
        if legacy_directory is None:
            return
        legacy = Path(legacy_directory).resolve()
        marker = self.directory / ".legacy_migration_v1_complete"
        if marker.exists():
            return
        migrated = []
        if legacy.is_dir() and legacy != self.custom_directory:
            for source in sorted(set(legacy.glob("*.peaf")) | set(legacy.glob("*.json"))):
                try:
                    document = json.loads(source.read_text(encoding="utf-8"))
                    definition = _validate_package(document)
                    destination = self.custom_directory / (definition["id"] + ".peaf")
                    if not destination.exists():
                        shutil.copyfile(str(source), str(destination))
                        migrated.append(source.name)
                except Exception:
                    # A malformed legacy file stays untouched in the legacy backup.
                    continue
        marker.write_text(
            "Legacy project-local formula migration completed.\nMigrated: {}\n".format(
                ", ".join(migrated) if migrated else "none"
            ),
            encoding="utf-8",
        )

    def reload(self):
        self.custom = {}
        self.errors = {}
        paths = sorted(set(self.custom_directory.glob("*.peaf")) | set(self.custom_directory.glob("*.json")))
        for path in paths:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                definition = _validate_package(document)
                definition["path"] = str(path)
                if definition["id"] in self.custom:
                    raise FormulaValidationError("Duplicate formula id: {}".format(definition["id"]))
                self.custom[definition["id"]] = definition
            except Exception as exc:
                quarantine = self.quarantine_directory / (path.stem + "_" + _timestamp() + path.suffix)
                try:
                    shutil.move(str(path), str(quarantine))
                    self.errors[str(quarantine)] = "Moved out of the active library: {}".format(exc)
                except Exception:
                    self.errors[str(path)] = str(exc)
        return self

    def definitions(self):
        builtins = [dict(item) for item in BUILTIN_FORMULAS]
        customs = []
        for item in self.custom.values():
            public = {key: value for key, value in item.items() if key != "_code"}
            customs.append(public)
        return builtins + sorted(customs, key=lambda item: item["name"].lower())

    def evaluate(self, frame, ages):
        references = empirical_references(frame, ages)
        context = _context(frame, ages)
        self.runtime_errors = {}
        for definition in self.custom.values():
            if not definition["enabled"]:
                continue
            environment = dict(FUNCTIONS)
            environment.update(context)
            environment.update(definition["constants"])
            environment.update({"pi": math.pi, "e": math.e})
            try:
                with np.errstate(all="raise"):
                    value = eval(definition["_code"], {"__builtins__": {}}, environment)
                array = np.asarray(value, dtype=float)
                array = np.broadcast_to(array, np.asarray(ages).shape).copy()
            except Exception as exc:
                self.runtime_errors[definition["id"]] = str(exc)
                continue
            if not np.all(np.isfinite(array)):
                self.runtime_errors[definition["id"]] = "produced non-finite values"
                continue
            if np.any(array < -1.0e-9) or np.any(np.abs(array) > 1.0e9):
                self.runtime_errors[definition["id"]] = "produced values outside the permitted range"
                continue
            references[definition["name"]] = np.maximum(array, 0.0)
        return references

    def preview(self, document, frame, ages):
        """Validate and evaluate one unsaved editor document."""
        definition = _validate_package(document)
        ages = np.asarray(ages, dtype=float)
        environment = dict(FUNCTIONS)
        environment.update(_context(frame, ages))
        environment.update(definition["constants"])
        environment.update({"pi": math.pi, "e": math.e})
        try:
            with np.errstate(all="raise"):
                value = eval(definition["_code"], {"__builtins__": {}}, environment)
            array = np.broadcast_to(np.asarray(value, dtype=float), ages.shape).copy()
        except Exception as exc:
            raise FormulaValidationError("Formula test failed: {}".format(exc))
        if not np.all(np.isfinite(array)):
            raise FormulaValidationError("Formula test produced non-finite values")
        if np.any(array < -1.0e-9) or np.any(np.abs(array) > 1.0e9):
            raise FormulaValidationError("Formula test produced values outside the permitted range")
        return np.maximum(array, 0.0)

    def import_package(self, source, overwrite=False):
        source = Path(source).resolve()
        document = json.loads(source.read_text(encoding="utf-8"))
        definition = _validate_package(document)
        destination = self.custom_directory / (definition["id"] + ".peaf")
        if destination.exists() and not overwrite and destination.resolve() != source:
            raise FileExistsError("A formula with id '{}' is already installed".format(definition["id"]))
        if destination.resolve() != source:
            shutil.copyfile(str(source), str(destination))
        self.reload()
        self._ensure_baseline_history()
        return next(item for item in self.definitions() if item["id"] == definition["id"])

    def save_package(self, document, original_id=None):
        """Validate and save a package created by the desktop editor."""
        definition = _validate_package(document)
        destination = self.custom_directory / (definition["id"] + ".peaf")
        if original_id:
            original = self.custom.get(original_id)
            if original is None:
                raise FormulaValidationError("The formula being edited is no longer installed")
            original_path = Path(original["path"])
            if destination.exists() and destination.resolve() != original_path.resolve():
                raise FileExistsError("A formula with id '{}' is already installed".format(definition["id"]))
            revision = self.history_directory / (original_id + "_" + _timestamp() + original_path.suffix)
            shutil.copyfile(str(original_path), str(revision))
        elif destination.exists():
            raise FileExistsError("A formula with id '{}' is already installed".format(definition["id"]))
        public = {key: value for key, value in document.items() if not str(key).startswith("_")}
        public["schema_version"] = 1
        public["id"] = definition["id"]
        public["name"] = definition["name"]
        public["expression"] = definition["expression"]
        public["latex"] = definition["latex"]
        public["description"] = definition["description"]
        public["constants"] = definition["constants"]
        public["color"] = definition["color"]
        public["line_style"] = definition["line_style"]
        public["enabled"] = definition["enabled"]
        _atomic_write_json(destination, public)
        if original_id and original_path.resolve() != destination.resolve():
            original_path.unlink()
        self.reload()
        self._ensure_baseline_history()
        return next(item for item in self.definitions() if item["id"] == definition["id"])

    def set_enabled(self, formula_id, enabled):
        definition = self.custom.get(formula_id)
        if definition is None:
            raise FormulaValidationError("Only custom formulas can be enabled or disabled")
        path = Path(definition["path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["enabled"] = bool(enabled)
        revision = self.history_directory / (formula_id + "_" + _timestamp() + path.suffix)
        shutil.copyfile(str(path), str(revision))
        _atomic_write_json(path, document)
        self.reload()

    def remove(self, formula_id):
        definition = self.custom.get(formula_id)
        if definition is None:
            raise FormulaValidationError("Native formulas are protected and cannot be archived or removed")
        source = Path(definition["path"])
        destination = self.archive_directory / (formula_id + "_" + _timestamp() + source.suffix)
        shutil.move(str(source), str(destination))
        self.reload()
        return destination

    def backups(self):
        """List valid recoverable user backups without exposing native formulas."""
        items = []
        for kind, directory in (("Archived", self.archive_directory), ("History", self.history_directory)):
            paths = sorted(set(directory.glob("*.peaf")) | set(directory.glob("*.json")))
            for path in paths:
                try:
                    definition = _validate_package(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
                items.append(
                    {
                        "id": definition["id"],
                        "name": definition["name"],
                        "kind": kind,
                        "saved_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "path": str(path),
                        "restorable": definition["id"] not in self.custom,
                    }
                )
        return sorted(items, key=lambda item: (item["saved_at"], item["kind"]), reverse=True)

    def restore_archived(self, source):
        """Restore a package from the archive or revision-history directories."""
        source = Path(source).resolve()
        in_archive = False
        in_history = False
        try:
            source.relative_to(self.archive_directory)
            in_archive = True
        except ValueError:
            pass
        try:
            source.relative_to(self.history_directory)
            in_history = True
        except ValueError:
            pass
        if not in_archive and not in_history:
            raise FormulaValidationError("Only files in the user formula archive or history can be restored")
        document = json.loads(source.read_text(encoding="utf-8"))
        definition = _validate_package(document)
        destination = self.custom_directory / (definition["id"] + ".peaf")
        if destination.exists():
            raise FileExistsError("An active formula with id '{}' already exists".format(definition["id"]))
        if in_archive:
            shutil.move(str(source), str(destination))
        else:
            shutil.copyfile(str(source), str(destination))
        self.reload()
        self._ensure_baseline_history()
        return next(item for item in self.definitions() if item["id"] == definition["id"])

    @staticmethod
    def write_template(destination):
        source = {
            "schema_version": 1,
            "id": "my_shrinkage_reference",
            "name": "My shrinkage reference",
            "description": "Example user-defined comparison curve.",
            "expression": "eps_u*(1-(RH/100)**3)*sqrt(t/(t+size_factor*VtoS**2))",
            "latex": r"\varepsilon_{\mathrm{sh}}(t)=\varepsilon_{\mathrm{u}}[1-(RH/100)^3]\sqrt{t/[t+k(V/S)^2]}",
            "constants": {"eps_u": 1000.0, "size_factor": 0.15},
            "color": "#8B5E3C",
            "line_style": "--",
            "enabled": True,
        }
        Path(destination).write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
