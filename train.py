from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.training import run_deployment_training, run_strict_audit


def main():
    parser = argparse.ArgumentParser(description="Train and audit the modular PEA-PGNN research software")
    parser.add_argument("--mode", choices=("audit", "deployment", "all"), default="all")
    parser.add_argument("--config", default=str(ROOT / "configs" / "training.json"))
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Five-epoch smoke run, not publication evidence")
    args = parser.parse_args()
    if args.mode in ("audit", "all"):
        audit = run_strict_audit(ROOT, args.config, args.cpu_threads, quick=args.quick)
        print("Strict audit:", audit["overall_seed_ensemble_metrics"])
    if args.mode in ("deployment", "all"):
        manifest = run_deployment_training(ROOT, args.config, args.cpu_threads, quick=args.quick)
        print("Deployment artifact:", ROOT / "artifacts" / "deployment" / "manifest.json")
        print("Ensemble members:", manifest["ensemble_seeds"])


if __name__ == "__main__":
    main()

