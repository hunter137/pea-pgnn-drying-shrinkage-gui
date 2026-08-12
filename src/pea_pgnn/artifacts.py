"""Checksummed, versioned model artifact helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hardware_manifest(device, cpu_threads):
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": "{}.{}".format(properties.major, properties.minor),
        }
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_logical_threads_requested": int(cpu_threads),
        "torch_device": str(device),
        "gpu": gpu,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def save_state_dict(path, model):
    """Save tensors only, so inference can use torch.load(weights_only=True)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state, path)
    return sha256_file(path)


def validate_manifest_files(root, manifest):
    root = Path(root)
    failures = []
    for record in manifest.get("files", []):
        path = root / record["path"]
        if not path.is_file():
            failures.append("missing: {}".format(record["path"]))
            continue
        actual = sha256_file(path)
        if actual != record["sha256"]:
            failures.append("hash mismatch: {}".format(record["path"]))
    if failures:
        raise ValueError("Artifact validation failed: " + "; ".join(failures))

