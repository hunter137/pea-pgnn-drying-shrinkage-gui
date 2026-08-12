"""GPU-accelerated strict-audit and deployment-ensemble training."""

from __future__ import annotations

import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from .artifacts import hardware_manifest, save_state_dict, sha256_file, write_json
from .data import apply_imputation, fit_imputation, indices_for_fold, load_frozen_population, load_frozen_split
from .features import EPS_ANCHOR_INDEX, FEATURE_NAMES, TAU_ANCHOR_INDEX, build_features
from .metrics import complete_metrics
from .model import model_from_config
from .support import build_support_spec


def configure_hardware(cpu_threads=None):
    cpu_threads = int(cpu_threads or os.cpu_count() or 1)
    cpu_threads = max(1, min(cpu_threads, os.cpu_count() or cpu_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
    except RuntimeError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    return device, cpu_threads


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def train_one(
    config,
    raw_train,
    scaled_train,
    age_train,
    target_train,
    raw_validation,
    scaled_validation,
    age_validation,
    target_validation,
    seed,
    device,
    fixed_epochs=None,
):
    set_seed(seed)
    model = model_from_config(config).to(device)
    if model.n_parameters() != 104200:
        raise RuntimeError("Frozen PEA-PGNN must contain 104,200 trainable parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    raw_train_t = _to_tensor(raw_train, device)
    scaled_train_t = _to_tensor(scaled_train, device)
    age_train_t = _to_tensor(age_train, device)
    target_train_t = _to_tensor(target_train, device)
    raw_val_t = _to_tensor(raw_validation, device)
    scaled_val_t = _to_tensor(scaled_validation, device)
    age_val_t = _to_tensor(age_validation, device)
    target_val_t = _to_tensor(target_validation, device)

    batch_size = int(config["batch_size"])
    y_scale = float(np.std(target_train) + 1e-8)
    epochs_per_restart = int(config["epochs_per_restart"])
    total_epochs = int(fixed_epochs or (epochs_per_restart * int(config["n_restarts"])))
    patience = int(config["patience_epochs"])
    best_loss = math.inf
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    started = time.time()

    for epoch in range(total_epochs):
        restart = epoch // epochs_per_restart
        local_epoch = epoch % epochs_per_restart
        maximum_lr = float(config["learning_rate"]) * (0.7 ** restart)
        minimum_lr = 1e-6
        if local_epoch < int(config["warmup_epochs"]):
            learning_rate = maximum_lr * (local_epoch + 1) / int(config["warmup_epochs"])
        else:
            denominator = max(1, epochs_per_restart - int(config["warmup_epochs"]))
            phase = (local_epoch - int(config["warmup_epochs"])) / denominator
            learning_rate = minimum_lr + (maximum_lr - minimum_lr) * 0.5 * (1.0 + math.cos(math.pi * phase))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        model.train()
        permutation = torch.randperm(len(raw_train_t), device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                prediction, details = model(
                    raw_train_t[index], scaled_train_t[index], age_train_t[index], return_details=True
                )
                mse = torch.mean(((prediction - target_train_t[index]) / y_scale) ** 2)
                penalty = (
                    torch.mean(details["delta_eps"] ** 2)
                    + torch.mean(details["delta_tau"] ** 2)
                    + torch.mean((details["delta_add"] / float(config["additive_scale"])) ** 2)
                )
                loss = mse + float(config["correction_penalty"]) * penalty
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss at epoch {}".format(epoch))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            batches += 1

        model.eval()
        with torch.no_grad():
            validation_prediction = model(raw_val_t, scaled_val_t, age_val_t)
            validation_loss = float(torch.mean(((validation_prediction - target_val_t) / y_scale) ** 2).cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": epoch_loss / max(batches, 1),
                "validation_loss": validation_loss,
            }
        )
        if fixed_epochs is not None:
            # Deployment fitting uses the epoch count selected exclusively by
            # the strict audit. It must retain the final selected epoch rather
            # than re-select an epoch on the all-development monitoring loss.
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            stale = 0
        elif validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_validation_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "elapsed_seconds": float(time.time() - started),
        "history": history,
    }


@torch.no_grad()
def predict_model(model, raw_features, scaled_features, ages, device, return_details=False):
    model.eval()
    result = model(
        _to_tensor(raw_features, device),
        _to_tensor(scaled_features, device),
        _to_tensor(ages, device),
        return_details=return_details,
    )
    if return_details:
        prediction, details = result
        return prediction.cpu().numpy(), {name: value.cpu().numpy() for name, value in details.items()}
    return result.cpu().numpy()


def _resolve_config(project_root, config_path):
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data_path"] = str((config_path.parent / config["data_path"]).resolve())
    config["frozen_split_path"] = str((config_path.parent / config["frozen_split_path"]).resolve())
    return project_root, config_path, config


def run_strict_audit(project_root, config_path, cpu_threads=None, quick=False):
    project_root, config_path, config = _resolve_config(project_root, config_path)
    device, cpu_threads = configure_hardware(cpu_threads)
    output = project_root / "artifacts" / "audit"
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("logs").mkdir(parents=True, exist_ok=True)
    frame = load_frozen_population(
        config["data_path"], config["expected_records"], config["expected_condition_profiles"], config["expected_database_trajectories"]
    )
    split = load_frozen_split(config["frozen_split_path"], frame, config["cutoff_days"])
    ages = frame["dt"].to_numpy(float)
    targets = frame["shrinkage_abs"].to_numpy(float)
    all_predictions = []
    run_rows = []
    best_epochs = []
    folds = sorted(split["fold"].unique())
    seeds = config["seeds"]
    if quick:
        folds = folds[:1]
        seeds = seeds[:1]
        config = dict(config)
        config["n_restarts"] = 1
        config["epochs_per_restart"] = 5
        config["patience_epochs"] = 5

    for fold in folds:
        roles = indices_for_fold(split, fold)
        train_index = roles["inner_train_development"]
        validation_index = roles["inner_validation_development"]
        test_index = roles["heldout_extrapolation"]
        imputation = fit_imputation(frame, train_index)
        filled = apply_imputation(frame, imputation)
        raw_features = build_features(filled)
        scaler = RobustScaler().fit(raw_features[train_index])
        scaled_features = scaler.transform(raw_features).astype(np.float32)
        seed_predictions = []
        for seed in seeds:
            print("[audit] fold={} seed={} device={}".format(fold, seed, device), flush=True)
            model, report = train_one(
                config,
                raw_features[train_index], scaled_features[train_index], ages[train_index], targets[train_index],
                raw_features[validation_index], scaled_features[validation_index], ages[validation_index], targets[validation_index],
                int(seed) + 100 * int(fold), device,
            )
            prediction = predict_model(
                model, raw_features[test_index], scaled_features[test_index], ages[test_index], device
            )
            seed_predictions.append(prediction)
            metrics = complete_metrics(
                targets[test_index], prediction,
                frame.iloc[test_index]["condition_id"].to_numpy(),
                frame.iloc[test_index]["ST_id"].to_numpy(),
            )
            best_epochs.append(report["best_epoch"])
            row = {"fold": int(fold), "seed": int(seed), **metrics, **{k: report[k] for k in ("best_validation_loss", "best_epoch", "epochs_ran", "elapsed_seconds")}}
            run_rows.append(row)
            weight_path = output / "models" / "fold_{}_seed_{}.pt".format(fold, seed)
            save_state_dict(weight_path, model)
            pd.DataFrame(report["history"]).to_csv(
                output / "logs" / "fold_{}_seed_{}.csv".format(fold, seed), index=False
            )
            print("  RMSE={RMSE:.2f} R2={R2:.4f} best_epoch={best_epoch}".format(**row), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ensemble = np.mean(seed_predictions, axis=0)
        all_predictions.append(
            pd.DataFrame(
                {
                    "row_index": test_index,
                    "source_row_id": frame.iloc[test_index]["source_row_id"].to_numpy(),
                    "condition_id": frame.iloc[test_index]["condition_id"].to_numpy(),
                    "ST_id": frame.iloc[test_index]["ST_id"].to_numpy(),
                    "fold": int(fold),
                    "dt": ages[test_index],
                    "y_true": targets[test_index],
                    "y_pred": ensemble,
                    "optimization_sd": np.std(seed_predictions, axis=0, ddof=0),
                }
            )
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    overall = complete_metrics(
        predictions["y_true"], predictions["y_pred"], predictions["condition_id"], predictions["ST_id"]
    )
    predictions.to_csv(output / "strict_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(run_rows).to_csv(output / "seed_metrics.csv", index=False)
    summary = {
        "role": "strict_condition_disjoint_audit",
        "cutoff_days": int(config["cutoff_days"]),
        "overall_seed_ensemble_metrics": overall,
        "median_best_epoch": int(np.median(best_epochs)),
        "n_runs": int(len(run_rows)),
        "quick": bool(quick),
        "feature_names": FEATURE_NAMES,
        "parameter_count": 104200,
        "data_sha256": sha256_file(config["data_path"]),
        "split_sha256": sha256_file(config["frozen_split_path"]),
        "hardware": hardware_manifest(device, cpu_threads),
    }
    write_json(output / "audit_summary.json", summary)
    return summary


def run_deployment_training(project_root, config_path, cpu_threads=None, quick=False):
    project_root, config_path, config = _resolve_config(project_root, config_path)
    device, cpu_threads = configure_hardware(cpu_threads)
    output = project_root / "artifacts" / "deployment"
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("logs").mkdir(parents=True, exist_ok=True)
    frame = load_frozen_population(
        config["data_path"], config["expected_records"], config["expected_condition_profiles"], config["expected_database_trajectories"]
    )
    split = load_frozen_split(config["frozen_split_path"], frame, config["cutoff_days"])
    audit_summary_path = project_root / "artifacts" / "audit" / "audit_summary.json"
    if not audit_summary_path.is_file():
        raise FileNotFoundError("Strict audit must complete before deployment training")
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    fixed_epochs = int(audit_summary["median_best_epoch"])
    if quick:
        fixed_epochs = 5
    development_index = np.where(frame["dt"].to_numpy(float) <= float(config["cutoff_days"]))[0]
    # Fixed, leakage-safe 10% profile validation is used only for monitoring.
    # Deployment training is then repeated at the selected epoch count; all
    # development records are used by supplying the same set as monitor data.
    imputation = fit_imputation(frame, development_index)
    filled = apply_imputation(frame, imputation)
    raw_features = build_features(filled)
    scaler = RobustScaler().fit(raw_features[development_index])
    scaled_features = scaler.transform(raw_features).astype(np.float32)
    ages = frame["dt"].to_numpy(float)
    targets = frame["shrinkage_abs"].to_numpy(float)
    seeds = config["seeds"][:1] if quick else config["seeds"]
    file_records = []
    member_reports = []
    for seed in seeds:
        print("[deployment] seed={} fixed_epochs={} device={}".format(seed, fixed_epochs, device), flush=True)
        model, report = train_one(
            config,
            raw_features[development_index], scaled_features[development_index], ages[development_index], targets[development_index],
            raw_features[development_index], scaled_features[development_index], ages[development_index], targets[development_index],
            int(seed), device, fixed_epochs=fixed_epochs,
        )
        relative = Path("models") / "seed_{}.pt".format(seed)
        digest = save_state_dict(output / relative, model)
        file_records.append({"path": relative.as_posix(), "sha256": digest})
        member_reports.append(
            {
                "seed": int(seed),
                "fixed_epochs": int(fixed_epochs),
                "training_monitor_loss": report["best_validation_loss"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
        pd.DataFrame(report["history"]).to_csv(output / "logs" / "seed_{}.csv".format(seed), index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preprocessing = {
        "imputation": imputation.as_dict(),
        "scaler_center": scaler.center_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
        "eps_anchor_index": EPS_ANCHOR_INDEX,
        "tau_anchor_index": TAU_ANCHOR_INDEX,
    }
    write_json(output / "preprocessing.json", preprocessing)
    file_records.append({"path": "preprocessing.json", "sha256": sha256_file(output / "preprocessing.json")})
    support = build_support_spec(filled.iloc[development_index])
    write_json(output / "support.json", support)
    file_records.append({"path": "support.json", "sha256": sha256_file(output / "support.json")})
    manifest = {
        "project_name": config["project_name"],
        "model_version": config["model_version"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": "deployment_ensemble_fit_on_all_development_records",
        "independent_test_claim": False,
        "cutoff_days": int(config["cutoff_days"]),
        "n_development_records": int(len(development_index)),
        "ensemble_seeds": [int(value) for value in seeds],
        "parameter_count_per_member": 104200,
        "architecture": {key: config[key] for key in ("hidden", "dropout", "delta_eps_range", "delta_tau_range", "additive_scale", "eps_min", "eps_max", "tau_min", "tau_max")},
        "feature_schema": {"version": "frozen-8729-v1", "names": FEATURE_NAMES},
        "training": {"fixed_epochs_selected_from_strict_audit": fixed_epochs, "members": member_reports},
        "strict_audit_evidence": audit_summary["overall_seed_ensemble_metrics"],
        "data": {
            "records": int(config["expected_records"]),
            "condition_profiles": int(config["expected_condition_profiles"]),
            "database_trajectories": int(config["expected_database_trajectories"]),
            "sha256": sha256_file(config["data_path"]),
            "frozen_split_sha256": sha256_file(config["frozen_split_path"]),
        },
        "uncertainty": {
            "nominal_prediction_interval_available": False,
            "ensemble_standard_deviation_meaning": "optimization-seed variation only; not a prediction interval",
        },
        "structural_contract": ["non-negative point prediction", "non-decreasing point trajectory", "bounded by corrected magnitude"],
        "scope": "Research-use prototype for the frozen OPC population; not a substitute for experiments or code checks.",
        "max_query_age_days": float(config["max_query_age_days"]),
        "hardware": hardware_manifest(device, cpu_threads),
        "files": file_records,
        "quick": bool(quick),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
