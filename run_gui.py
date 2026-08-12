from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.gui import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(ROOT / "artifacts" / "deployment"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--auto-predict", action="store_true", help="Run the default condition after the window opens")
    parser.add_argument("--show-formulas", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--show-formula-editor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--show-formula-trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--show-report", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    main(
        args.artifact,
        smoke=args.smoke,
        auto_predict=args.auto_predict,
        show_formulas=args.show_formulas,
        show_formula_editor=args.show_formula_editor,
        show_formula_trial=args.show_formula_trial,
        show_report=args.show_report,
    )
